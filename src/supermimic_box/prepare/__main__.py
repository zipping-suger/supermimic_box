"""Build the robot MJCF, the box and platform assets and the reference clips that training reads.

    python -m supermimic_box.prepare --sim2sim /path/to/OmniContact_sim2sim [--omnicontact /path/to/OmniContact]
    python -m supermimic_box.prepare --sim2sim /path/to/OmniContact_sim2sim --dataset /path/to/OmniContact-Dataset

The robot always comes from the OmniContact_sim2sim checkout. Carrying and pushing are prepared;
``--tasks`` can add sliding the box with the feet. Without ``--dataset`` the clips are
the ones under its ``data/{carry,push}box`` (plus, with ``--omnicontact``, the motion files
under that checkout's ``assets/data``), written to ``data/box`` and ``data/objects``. With
``--dataset`` they are the box clips of the OmniContact dataset (``npz_clips/box``) with its
train/val/test split, written to ``data/omnicontact/clips`` and ``data/omnicontact/objects``.
"""

import argparse
from pathlib import Path

from ..robot import load_robot
from .dataset import DEFAULT_TASKS, TASKS, find_sources, prepare_dataset
from .objects import write_objects
from .robot import prepare_robot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim2sim", required=True, type=Path, help="OmniContact_sim2sim checkout")
    p.add_argument("--omnicontact", type=Path, help="OmniContact checkout with motion files under assets/data")
    p.add_argument("--dataset", type=Path, help="OmniContact dataset folder with npz_clips/box and metadata/splits.csv")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=TASKS, help="Tasks to prepare (default: carry push)")
    p.add_argument("--output", type=Path, default=Path("data"))
    a = p.parse_args()

    prepare_robot(a.sim2sim, a.output / "robot")
    robot = load_robot(a.output / "robot")
    print(f"Robot: {robot.num_joints} joints, {robot.model.body_mass.sum():.2f} kg")
    sources = find_sources(a.sim2sim, a.omnicontact, a.tasks, a.dataset)
    clips, objects = (a.output / "omnicontact" / "clips", a.output / "omnicontact" / "objects") if a.dataset else (a.output / "box", a.output / "objects")
    registry = write_objects(objects, [(s["half"], s["kind"]) for s in sources])
    print(f"Objects: {len(registry['objects'])} boxes")
    manifest = prepare_dataset(sources, clips, robot, objects)
    print(f"Dataset: {manifest['clips']} clips, {manifest['frames']} frames ({manifest['minutes']:.1f} min), "
          f"FK error {manifest['fk_max_error_m'] * 1000:.3f} mm, "
          f"largest joint-limit excess {manifest['joint_limit_max_excess_rad']:.4f} rad")  # fmt: skip
    print(f"Per task: {manifest['per_task']}; per split: {manifest['per_split']}; rejected: {len(manifest['rejected_clips'])}")
    for name, reason in manifest["rejected_clips"].items():
        print(f"  rejected {name}: {reason}")


if __name__ == "__main__":
    main()
