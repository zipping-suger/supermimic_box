"""Audit the references on the training physics by open-loop playback.

Every clip is started at its first admissible frame and driven with the action that asks for the
reference joint positions. This is not a tracking benchmark, since open-loop PD is not expected to
follow a clip; it shows whether the scene is consistent with the references: an untouched box must
stay where the reference has it, and the hands and feet should meet the box where the reference
says they do.

    python -m supermimic_box.audit --config configs/box.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import torch

from .config import TrainConfig
from .env import make_task
from .mdp import FAILURE_REASONS
from .motion import select_clips
from .robot import CONTACT_CHANNELS


@torch.no_grad()
def audit(cfg: TrainConfig, steps: int = 300) -> list[dict]:
    """Per clip: how far playback gets, why it ends, the object's drift while the reference leaves
    it alone, and how often each contact channel is measured against how often it is intended."""
    cfg = replace(cfg, randomize=False, cuda_graph=False, compile=False)
    cfg.num_envs = len(select_clips(cfg.dataset, cfg.clips, cfg.split))
    task = make_task(cfg)
    env, m = task.env, task.motion
    N = env.N
    everyone = torch.ones(N, dtype=torch.bool, device=cfg.device)
    start = env.reset_map[m.clip_start]
    env.reset(everyone, start)
    active = everyone.clone()
    survived = torch.zeros(N, dtype=torch.long, device=cfg.device)
    reasons = torch.zeros(N, len(FAILURE_REASONS), dtype=torch.bool, device=cfg.device)
    drift = torch.zeros(N, device=cfg.device)  # largest object error before the first intended contact
    untouched = everyone.clone()
    measured, intended = torch.zeros(N, 4, device=cfg.device), torch.zeros(N, 4, device=cfg.device)
    for _ in range(steps):
        row = env.step(task.controller.reference_action(m.q[m.next_frame(env.f)]))
        env.physics.check()
        state, f = env.physics.state(), env.f
        untouched &= m.contact[f].sum(-1) == 0
        error = (state[6] - m.object_p[f]).norm(dim=-1)
        drift = torch.where(active & untouched, torch.maximum(drift, error), drift)
        measured += torch.where(active[:, None], row["contact"], 0.0)
        intended += torch.where(active[:, None], m.contact[f], 0.0)
        survived += active
        reasons |= row["reasons"] & active[:, None]
        active &= ~(row["term"] | row["clip_end"])
        if not bool(active.any()):
            break
        env.reset(~active, start)
    results = []
    for i, name in enumerate(m.names):
        results.append(dict(
            clip=name, steps=int(survived[i]), reference_steps=int(m.clip_end[i] - start[i]),
            ended_by=[r for r, flag in zip(FAILURE_REASONS, reasons[i].tolist()) if flag],
            untouched_object_drift_m=round(float(drift[i]), 4),
            contact_steps={c: [int(measured[i, k]), int(intended[i, k])] for k, c in enumerate(CONTACT_CHANNELS)}))  # fmt: skip
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file of TrainConfig fields")
    p.add_argument("--device")
    p.add_argument("--split")
    p.add_argument("--steps", type=int, default=300)
    a = p.parse_args()
    cfg = TrainConfig.from_json(a.config, device=a.device, split=a.split)
    if cfg.device == "cpu":
        cfg.backend = "cpu"
    for result in audit(cfg, a.steps):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
