"""Reference library: synchronized robot/object clips at 50 Hz with FK of the tracked bodies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .kinematics import Kinematics
from .robot import CONTACT_CHANNELS, Robot
from .rotations import angular_velocity, central_difference, mat_to_6d, mm3, mv3, quat_to_mat, yaw_mat

CONTROL_DT = 0.02
ROTATION_LENGTH = 0.15  # metres that one radian of object rotation counts for in an object pose step
# Control steps ahead at which the goal samples the reference: dense for the next moves, sparse out to one second.
FUTURE_OFFSETS = (1, 2, 3, 4, 8, 12, 16, 24, 32, 50)
KEY_SCALE = 0.2  # box half extents are given to the networks in units of this length [m]
GOAL_MODES = ("flow", "full")


def select_clips(dataset: str | Path, clips=(), split: str = "") -> list[str]:
    """Names of the clips a run uses: ``clips`` as given, else the clips of ``split``, else every clip."""
    manifest = json.loads((Path(dataset) / "manifest.json").read_text())
    if clips:
        return list(clips)
    if not split:
        return manifest["clip_names"]
    names = [name for name, s in zip(manifest["clip_names"], manifest.get("clip_splits", [])) if s == split]
    if not names:
        raise ValueError(f"The dataset has no clips of the split {split!r}")
    return names


def pose_step(p0, R0, p1, R1):
    """(..., 6) change of an object pose over one control step: translation, then the world-frame
    rotation scaled to a length. The rotation is the antisymmetric part of R1 R0^T (axis times the
    sine of the angle), which is smooth at rest and within 1 % of the rotation vector for a step."""
    r = mm3(R1, R0.transpose(-1, -2))
    turn = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1) * 0.5
    return torch.cat((p1 - p0, ROTATION_LENGTH * turn), -1)


class Motion:
    """All selected clips concatenated along the frame axis.

    Robot and object share one world frame and are never re-anchored independently. Joint
    references are projected onto the joint limits before velocities and FK are derived, so every
    reference state is a feasible reset. ``contact`` is reference intent, not a measurement: whether
    the left foot, right foot, left hand and right hand are on the object.
    """

    def __init__(self, dataset: str | Path, robot: Robot, objects: list[dict], device: str = "cpu", clips=(),
                 goal_mode: str = "flow", split: str = ""):  # fmt: skip
        """``objects``: the registry's object entries; their order defines the object keys. ``clips``
        names the clips to load; without it they are the clips of ``split``, or all of them."""
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}")
        dataset = Path(dataset)
        manifest = json.loads((dataset / "manifest.json").read_text())
        with np.load(dataset / "motion.npz", allow_pickle=False) as data:
            if int(data["fps"]) != round(1 / CONTROL_DT) or tuple(data["joint_names"]) != robot.joint_names:
                raise ValueError("Dataset must be 50 Hz in the robot's joint order; rerun supermimic_box.prepare")
            all_names = data["clip_names"].tolist()
            names = select_clips(dataset, clips, split)
            if len(set(names)) != len(names) or set(names) - set(all_names):
                raise ValueError("Unknown or duplicate clips")
            starts = np.concatenate(([0], np.cumsum(data["clip_lengths"])))
            ids = [all_names.index(n) for n in names]
            rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in ids])
            lengths = data["clip_lengths"][ids].tolist()
            clip_objects = data["clip_objects"][ids].tolist()
            self.tasks = data["clip_tasks"][ids].tolist()
            tables = torch.from_numpy(data["clip_tables"][ids])
            arrays = {k: torch.from_numpy(data[k][rows]) for k in
                      ("root_pos", "root_quat", "joint_pos", "object_pos", "object_quat", "contact")}  # fmt: skip
            # How deep every frame interpenetrates, where preparation has measured it.
            self.overlap_depth = torch.from_numpy(data["overlap_depth"][rows]) if "overlap_depth" in data.files else None
        if any(not torch.isfinite(x.float()).all() for x in arrays.values()) or min(lengths) < 2:
            raise ValueError("Reference clips must be finite and at least two frames long")

        self.device, self.names, self.goal_mode = device, names, goal_mode
        self.object_names = [o["name"] for o in objects]
        self.kin = Kinematics(robot, device)
        self.clip_len = torch.tensor(lengths, device=device)
        self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1  # last frame, inclusive
        self.clip_id = torch.repeat_interleave(torch.arange(len(names), device=device), self.clip_len)
        self.clip_object = torch.tensor([self.object_names.index(o) for o in clip_objects], device=device)
        self.clip_tables = tables.to(device).flatten(-2)  # (clips, 6): the two platform centres
        self.half = torch.tensor([o["half"] for o in objects], dtype=torch.float32, device=device)
        self.num_frames = int(self.clip_len.sum())
        self.future_offsets = torch.tensor(FUTURE_OFFSETS, device=device)

        q_raw = arrays["joint_pos"]
        q = q_raw.clamp(self.kin.q_lo.cpu(), self.kin.q_hi.cpu())
        self.projection = dict(frames_changed=int(((q - q_raw).abs() > 1e-6).any(-1).sum()),
                               max_correction_rad=float((q - q_raw).abs().max()))  # fmt: skip
        root_R, object_R = quat_to_mat(arrays["root_quat"]), quat_to_mat(arrays["object_quat"])
        per_clip = lambda fn, x: torch.cat([fn(part, CONTROL_DT) for part in x.split(lengths)])  # noqa: E731
        to = lambda x: x.to(device)  # noqa: E731
        self.root_p, self.root_R, self.q = to(arrays["root_pos"]), to(root_R), to(q)
        self.root_v, self.qd = to(per_clip(central_difference, arrays["root_pos"])), to(per_clip(central_difference, q))
        self.root_w = to(per_clip(angular_velocity, root_R))
        self.object_p, self.object_R = to(arrays["object_pos"]), to(object_R)
        self.object_v = to(per_clip(central_difference, arrays["object_pos"]))
        self.object_w = to(per_clip(angular_velocity, object_R))
        # Reference object step into every frame; zero into a clip's first frame.
        before = (torch.arange(self.num_frames, device=device) - 1).clamp_min(self.clip_start[self.clip_id])
        self.object_step = pose_step(self.object_p[before], self.object_R[before], self.object_p, self.object_R)

        index = lambda bodies: [robot.tracked_bodies.index(n) for n in bodies]  # noqa: E731
        self.num_tracked = len(robot.tracked_bodies)
        self.anchor = robot.tracked_bodies.index(robot.anchor_body)
        self.hands, self.feet, self.flow = index(robot.hand_bodies), index(robot.foot_bodies), index(robot.flow_bodies)
        parts = [[], [], [], []]
        with torch.no_grad():
            for chunk in torch.arange(self.num_frames, device=device).split(65536):
                for store, value in zip(parts, self.kin.forward(*self.ref_robot_state(chunk))):
                    store.append(value)
        self.body_p, self.body_R, self.body_v, self.body_w = (torch.cat(x) for x in parts)
        # Indexing with a Python list inside a captured CUDA graph would copy the index from the host.
        self.flow_index = torch.tensor(self.flow, device=device)

        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local, self.root_w_local = mv3(Rt, self.root_v), mv3(Rt, self.root_w)
        self.contact = to(arrays["contact"].float())
        if self.contact.shape[-1] != len(CONTACT_CHANNELS):
            raise ValueError("Contact flags must cover the four contact channels")
        digest = hashlib.sha256(manifest["motion_sha256"].encode())
        digest.update(json.dumps([names, goal_mode]).encode() + robot.joint_limits.tobytes())
        self.signature = digest.hexdigest()

    def ref_robot_state(self, f):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def ref_state(self, f):
        return (*self.ref_robot_state(f), self.object_p[f], self.object_R[f], self.object_v[f], self.object_w[f])

    def next_frame(self, f):
        return torch.minimum(f + 1, self.clip_end[self.clip_id[f]])

    def object_of(self, f):
        """Object key of the clip that frame ``f`` belongs to."""
        return self.clip_object[self.clip_id[f]]

    def key(self, f):
        """(..., 3) what the networks know about the object of frame ``f``: its scaled half extents."""
        return self.half[self.object_of(f)] / KEY_SCALE

    def tables(self, f):
        """(..., 6) centres of the two support platforms of the clip that frame ``f`` belongs to."""
        return self.clip_tables[self.clip_id[f]]

    def goal_dim(self, K: int) -> int:
        if K > len(FUTURE_OFFSETS):
            raise ValueError(f"The goal has at most {len(FUTURE_OFFSETS)} future frames")
        per_frame = 9 * len(self.flow) + 4 if self.goal_mode == "flow" else 15 + 2 * self.q.shape[-1] + 15 + 4
        return K * per_frame

    def goal(self, f, K: int):
        """Reference-only goal at the first K of FUTURE_OFFSETS, relative to the current reference root.

        ``flow``: the contact flow, as in OmniContact: pose of both hands, the torso and both feet,
        and the four contact flags. ``full``: root, joints and object as well.
        """
        end = self.clip_end[self.clip_id[f]]
        future = torch.minimum(f.unsqueeze(-1) + self.future_offsets[:K], end.unsqueeze(-1))
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        origin = self.root_p[f].unsqueeze(1)
        if self.goal_mode == "flow":
            bodies = (future.unsqueeze(-1), self.flow_index)
            position = mv3(Ht.unsqueeze(2), self.body_p[bodies] - origin.unsqueeze(2))
            rotation = mat_to_6d(mm3(Ht.unsqueeze(2), self.body_R[bodies]))
            features = torch.cat((position.flatten(-2), rotation.flatten(-2), self.contact[future]), -1)
        else:
            features = torch.cat((
                mv3(Ht, self.root_p[future] - origin), mat_to_6d(mm3(Ht, self.root_R[future])),
                self.root_v_local[future], self.root_w_local[future], self.q[future], self.qd[future] * 0.1,
                mv3(Ht, self.object_p[future] - origin), mat_to_6d(mm3(Ht, self.object_R[future])),
                mv3(Ht, self.object_v[future]), mv3(Ht, self.object_w[future]), self.contact[future]), -1)  # fmt: skip
        return features.flatten(-2)

    def precontact_frames(self, lead: int = 10, min_steps: int = 10):
        """Per clip, the frames ``lead`` steps before each sustained contact phase of a channel begins.

        Returns a padded (clips, max_count) frame table and the per-clip counts; a clip without a
        sustained phase gets its first frame.
        """
        intent = self.contact.cpu().numpy() > 0
        choices = []
        for first, last in zip(self.clip_start.tolist(), self.clip_end.tolist()):
            frames = set()
            for channel in range(intent.shape[1]):
                edges = np.diff(np.concatenate(([False], intent[first : last + 1, channel], [False])).astype(int))
                for begin, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if stop - begin >= min_steps:
                        frames.add(max(first, first + int(begin) - lead))
            choices.append(sorted(frames) or [first])
        table = torch.zeros(len(choices), max(map(len, choices)), dtype=torch.long, device=self.device)
        for i, frames in enumerate(choices):
            table[i, : len(frames)] = torch.tensor(frames, device=self.device)
        return table, torch.tensor([len(c) for c in choices], device=self.device)
