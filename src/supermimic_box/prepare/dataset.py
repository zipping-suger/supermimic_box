"""Reference clips as one compact archive in the robot's joint order.

Sources are OmniContact motion files (one clip per ``.npz``, Isaac Lab joint order), resampled to
50 Hz where they are recorded at another rate (the dataset is at 90 Hz). Either
the clips shipped with OmniContact_sim2sim (``data/<task>box/``) and, optionally, any under an
OmniContact checkout's ``assets/data``; or the box clips of the OmniContact dataset
(``npz_clips/box``), with its train/val/test split. Apart from the OmniContact_sim2sim clips, the
file name gives the task and the box size. A clip keeps its root pose,
joint positions, object pose, the four contact flags (left foot, right foot, left hand, right hand
on the object) and where its two support platforms stand; velocities and body poses are recomputed
when the archive is loaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .objects import box_name

# Isaac Lab's breadth-first joint order of the G1, in which the source files store joints.
ISAAC_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)  # fmt: skip
# The first four recorded end effectors (``ee_pos_w``), used to check this repository's kinematics.
EE_BODIES = ("left_palm_link", "right_palm_link", "left_ankle_roll_link", "right_ankle_roll_link")
TASKS = ("carry", "push", "slide")  # tasks the sources hold
DEFAULT_TASKS = ("carry", "push")  # tasks that are prepared unless asked otherwise: the hand tasks
# Boxes of the OmniContact_sim2sim clips: half extents [m] in the recorded object frame (y is up at rest).
SIM2SIM_BOXES = {"carry": (0.15, 0.15, 0.15), "slide": (0.15, 0.15, 0.15), "push": (0.23, 0.25, 0.26)}
MAX_LIMIT_EXCESS = 0.2  # a joint further outside its limit marks a corrupted reference
# ``<task>_box_<x>-<y>-<z>_<capture>_<time>_<clip>.npz``: box extents in cm along the recorded object
# frame; capture and time are the tails of the capture id that the dataset's split is keyed by.
CLIP_NAME = re.compile(r"(carry|push|slide)_box_(\d+)-(\d+)-(\d+)_(\d+)_(\d+)_.*\.npz")
LOCAL_SPLIT = "local"  # clips that come with no split


def resample(arrays: dict, fps: float, target: float = 50.0) -> dict:
    """Clip arrays (frames first) at ``target`` Hz: positions and joint angles interpolated linearly,
    quaternions along the shorter arc, contact flags taken from the nearest source frame."""
    frames = len(arrays["joint_pos"])
    position = np.arange(int(np.floor((frames - 1) * target / fps + 1e-6)) + 1) * fps / target
    lower = np.minimum(position.astype(np.int64), frames - 2)
    weight = (position - lower).astype(np.float32)
    out = {}
    for key, value in arrays.items():
        if key == "contact":
            out[key] = value[np.minimum(np.rint(position).astype(np.int64), frames - 1)]
            continue
        a, b = value[lower], value[lower + 1]
        w = weight.reshape(-1, *([1] * (value.ndim - 1)))
        if key.endswith("quat"):
            b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
            mixed = (1 - w) * a + w * b
            out[key] = mixed / np.linalg.norm(mixed, axis=-1, keepdims=True)
        else:
            out[key] = (1 - w) * a + w * b
    return out


def named_sources(folder: Path, tasks, splits: dict | None = None) -> list[dict]:
    """Motion files below ``folder`` whose name gives the task and box size."""
    sources = []
    for file in sorted(Path(folder).rglob("*.npz")):
        match = CLIP_NAME.fullmatch(file.name)
        if match and match.group(1) in tasks:
            half = tuple(int(x) / 200 for x in match.groups()[1:4])
            split = LOCAL_SPLIT if splits is None else splits[(int(match.group(5)), match.group(6))]
            sources.append(dict(file=file, name=file.stem, task=match.group(1), half=half, split=split))
    return sources


def read_splits(dataset: Path) -> dict:
    """(capture number, tail of its time stamp) -> split, from the dataset's ``metadata/splits.csv``.
    A capture id is ``<date and number>_<take>_<time stamp>``; clip names keep the last three digits of
    the first field (without leading zeros) and the last seven of the time stamp."""
    splits = {}
    with open(Path(dataset) / "metadata" / "splits.csv") as file:
        for row in csv.DictReader(file):
            head, _, stamp = row["capture_id"].split("_")
            key = (int(head[-3:]), stamp[-7:])
            if splits.setdefault(key, row["split"]) != row["split"]:
                raise ValueError(f"Captures with different splits share the clip key {key}")
    return splits


def find_sources(sim2sim: Path, omnicontact: Path | None, tasks, dataset: Path | None = None) -> list[dict]:
    """The motion files of the requested tasks: path, clip name, task, split, box half extents and kind.
    With ``dataset`` (an OmniContact dataset folder) its box clips; otherwise the local clips."""
    if dataset is not None:
        sources = named_sources(Path(dataset) / "npz_clips" / "box", tasks, read_splits(dataset))
    else:
        sources = []
        for task in tasks:
            for file in sorted((Path(sim2sim) / "data" / f"{task}box").glob("*.npz")):
                name = file.stem.removesuffix("_with_contact").replace("box_motion", "")
                sources.append(dict(file=file, name=name, task=task, half=SIM2SIM_BOXES[task], split=LOCAL_SPLIT))
        if omnicontact is not None:
            sources += named_sources(Path(omnicontact) / "assets" / "data", tasks)
    for source in sources:
        source["kind"] = "push" if source["task"] == "push" else "box"
        source["object"] = box_name(source["half"], source["kind"])
    if len({s["name"] for s in sources}) != len(sources):
        raise ValueError("Clip names are not unique")
    return sources


def write_dataset(output: Path, robot, clips: list[dict], manifest: dict) -> dict:
    """Write ``motion.npz`` and ``manifest.json``.

    Every clip gives its ``name``, ``object``, ``task``, ``split``, ``tables`` (2, 3) and per frame:
    root pose (``root_pos``, ``root_quat``), ``joint_pos`` in ``robot.joint_names`` order, object pose
    (``object_pos``, ``object_quat``), ``contact`` (4), the reference's contact flags, and
    ``overlap_depth`` (3), how deep the reference interpenetrates per kind of collision pair.
    """
    keys = ("root_pos", "root_quat", "joint_pos", "object_pos", "object_quat", "contact", "overlap_depth")
    names, lengths = [clip["name"] for clip in clips], [len(clip["joint_pos"]) for clip in clips]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "motion.npz", **{k: np.concatenate([clip[k] for clip in clips]) for k in keys},
        clip_names=np.array(names), clip_objects=np.array([clip["object"] for clip in clips]),
        clip_tasks=np.array([clip["task"] for clip in clips]), clip_tables=np.stack([clip["tables"] for clip in clips]),
        clip_splits=np.array([clip["split"] for clip in clips]),
        clip_lengths=np.array(lengths), joint_names=np.array(robot.joint_names), fps=np.array(50))  # fmt: skip
    frames = sum(lengths)
    manifest = dict(format_version=1, **manifest, clips=len(clips), frames=frames, minutes=frames / 50 / 60)
    manifest = dict(manifest, motion_sha256=hashlib.sha256((output / "motion.npz").read_bytes()).hexdigest(), clip_names=names,
                    clip_splits=[clip["split"] for clip in clips])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_dataset(sources: list[dict], output: Path, robot, object_assets: Path) -> dict:
    """Read, check and screen the clips of ``sources`` and write them to ``output``.

    A clip is rejected when a joint is far outside its limit, when it is not a clean clip with static
    platforms, or when none of its frames can start an episode because all of them overlap
    the ground or the object by more than the reset screen admits.
    """
    import torch

    from ..config import TrainConfig
    from ..kinematics import Kinematics
    from ..reset_guard import admissible_frames, overlap_depths
    from ..rotations import quat_to_mat
    from ..sim import load_registry

    registry, _ = load_registry(object_assets)
    object_names = [o["name"] for o in registry["objects"]]
    order = [ISAAC_JOINTS.index(n) for n in robot.joint_names]
    kin = Kinematics(robot, bodies=EE_BODIES)
    lo, hi = robot.joint_limits.T
    clips, rejected, hashes = [], {}, {}
    fk_error, limit_excess = 0.0, 0.0
    for source in sources:
        with np.load(source["file"], allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            frames = len(data["joint_pos"])
            raw = dict(joint_pos=data["joint_pos"][:, order].astype(np.float32), root_pos=data["body_pos_w"][:, 0],
                       root_quat=data["body_quat_w"][:, 0], object_pos=data["object_pos_w"], object_quat=data["object_quat_w"],
                       contact=data["contact_info"].reshape(frames, -1) > 0)  # fmt: skip
            tables = np.stack((data["table1_pos_w"], data["table2_pos_w"]), 1)
            ee = data["ee_pos_w"][:, : len(EE_BODIES)]
        excess = float(np.maximum(lo - raw["joint_pos"], raw["joint_pos"] - hi).max()) if frames else 0.0
        reason = ("too short" if frames * 50 < 2 * fps + 50 else "contact flags" if raw["contact"].shape[1] != 4
                  else "nonfinite" if not all(np.isfinite(x).all() for x in (*raw.values(), tables))
                  else "moving platforms" if np.ptp(tables, 0).max() > 1e-4
                  else "joint limits" if excess > MAX_LIMIT_EXCESS else None)  # fmt: skip
        if reason:
            rejected[source["name"]] = reason
            continue
        for key in ("root_quat", "object_quat"):
            raw[key] = raw[key] / np.linalg.norm(raw[key], axis=-1, keepdims=True)
        # The recorded end effectors must be reproduced by this repository's model and joint order.
        zero = torch.zeros(frames, 3)
        fk = kin.forward(torch.from_numpy(raw["root_pos"]), quat_to_mat(torch.from_numpy(raw["root_quat"])), zero, zero,
                         torch.from_numpy(raw["joint_pos"]), torch.zeros(frames, robot.num_joints))[0]  # fmt: skip
        fk_error = max(fk_error, float((fk - torch.from_numpy(ee)).norm(dim=-1).max()))
        limit_excess = max(limit_excess, excess)
        hashes[source["file"].name] = hashlib.sha256(source["file"].read_bytes()).hexdigest()
        clip = raw if abs(fps - 50) < 1e-6 else resample(raw, fps)
        clips.append(dict(name=source["name"], object=source["object"], task=source["task"], split=source["split"],
                          tables=tables[0].astype(np.float32), contact=clip.pop("contact").astype(np.uint8),
                          **{key: value.astype(np.float32) for key, value in clip.items()}))  # fmt: skip
    if fk_error > 2e-3:
        raise ValueError(f"FK disagrees with the recorded end-effector positions by {fk_error:.4f} m")

    # The state an episode would start from at every frame: the reference with its joints inside their limits.
    lengths = [len(clip["joint_pos"]) for clip in clips]
    starts = np.concatenate(([0], np.cumsum(lengths)))
    stack = lambda key: torch.from_numpy(np.concatenate([clip[key] for clip in clips]))  # noqa: E731
    poses = (stack("root_pos"), quat_to_mat(stack("root_quat")), stack("joint_pos").clamp(torch.from_numpy(lo), torch.from_numpy(hi)),
             stack("object_pos"), quat_to_mat(stack("object_quat")))  # fmt: skip

    def start_state(frames):
        p, R, q, op, oR = (x[frames] for x in poses)
        still = torch.zeros_like(p)
        return p, R, still, still, q, torch.zeros_like(q), op, oR, still, still

    depths = overlap_depths(start_state, [object_names.index(clip["object"]) for clip in clips], [clip["tables"].flatten() for clip in clips],
                            starts[:-1].tolist(), (starts[1:] - 1).tolist(), robot, registry, object_assets)  # fmt: skip
    admissible = admissible_frames(depths, TrainConfig().reset_overlap_tolerance).numpy()
    kept = []
    for clip, a, b in zip(clips, starts[:-1], starts[1:]):
        # A clip's last frame has no successor, so it cannot start an episode.
        if admissible[a : b - 1].any():
            kept.append(dict(clip, overlap_depth=depths[a:b].numpy().astype(np.float32)))
        else:
            rejected[clip["name"]] = "no admissible reset frame"
    count = lambda key: {v: sum(clip[key] == v for clip in kept) for v in sorted({clip[key] for clip in kept})}  # noqa: E731
    return write_dataset(output, robot, kept, dict(
        source="OmniContact motions", per_task=count("task"), per_split=count("split"), source_sha256=hashes,
        rejected_clips=rejected, fk_max_error_m=fk_error, joint_limit_max_excess_rad=max(limit_excess, 0.0)))  # fmt: skip
