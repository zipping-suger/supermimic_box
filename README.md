# supermimic_box

A low-level contact-flow tracker for box loco-manipulation on the **Unitree G1 (29 DoF, rubber
hands)**: the task of OmniContact's CFtrack, trained the [SuperMimic](https://arxiv.org/abs/2610.28480)
way. Instead of PPO in the simulator, one policy is trained with a learned world model of the
robot and the box and with differentiable actor rollouts inside that model. Physics is
MuJoCo-Warp through mjlab.

The code is adapted from `supermimic_hoi_vibed`; the task follows
[OmniContact](https://github.com/Ingrid789/OmniContact) (`CFBaseEnvCfg`).

## Results

Results for carrying and pushing are being regenerated: the run reported here earlier also
trained on the dataset's slide clips (boxes moved with the feet), which are no longer prepared.
That run (`logs/omnicontact_with_slide`, 923 clips, 20,000 iterations, 1 h 32 min on one RTX 4080)
reached, on held-out carry and push clips, 93 % success on the 139 test clips (130, 129) and
92 % on the 145 validation clips; and 50 % on the 20 held-out slide clips.

## What is the same as OmniContact, and what is not

| | OmniContact CFtrack | here |
|---|---|---|
| Robot | G1 29 DoF, rubber hands, PD at 10 Hz / damping ratio 2 | the same, from OmniContact_sim2sim's MJCF |
| Goal given to the policy | contact flow: poses of both hands, torso and both feet at 11 future frames, and 4 contact flags | the same bodies and flags at 10 future frames (`goal_mode: flow`) |
| Action | joint targets around the default pose, scale 0.25 effort / kp | the same (`action_mode: absolute`) |
| Rewards | anchor, relative body, hand, object, hand-object interaction, contact; action rate, arm torque, joint limits, foot slip | the same terms and weights |
| Scene | box, two kinematic platforms, ground | the same |
| Training | PPO (+ AMP), 4096 worlds, Isaac Lab | world model + differentiable rollouts, 1024 worlds, MuJoCo-Warp |
| Policy | MLP or Transformer on a 5-step history | goal encoder + MLP decoder on a 10-step history |

Deliberate differences:

- **Layout of the goal.** OmniContact expresses the contact flow in the robot's current torso
  frame. Here the goal is reference-only (relative to the reference root), so it is encoded once
  for a whole imagined rollout, and the tracking error is part of the observation instead. The
  information is the same; the exported network is **not** a drop-in for OmniContact_sim2sim.
- **Terminations** are tighter (torso 0.3 m, object 0.3 m / 1 rad, hand or foot height 0.25 m)
  than OmniContact's 0.5 m height checks; short episodes near the reference are what the world
  model learns from.
- **Resets** put robot and box on the same reference frame without pose noise. Frames whose
  reference interpenetrates too deeply are replaced by the nearest admissible one.
- **Randomization**: robot friction, torso CoM, encoder offsets, observation noise, box mass
  (0.5 to 1.5 times nominal) and box friction (per task). Not implemented: pushes, PD gain
  scaling, torso orientation bias, AMP.
- **Self-collision is off**, and each foot has an extra box collider that touches only the object.
  As in OmniContact_sim2sim's scenes, a platform carries the box but does not collide with the robot.
- The actor also observes the object's velocity and the distance of hands and feet to the box.

`goal_mode: full` with `action_mode: residual` is supermimic_hoi's interface (full joint reference,
residual actions). It cannot follow CFgen references, which have no joint angles; it is there to
separate problems of the task from problems of the sparse interface.

## Setup

Python 3.12 and a CUDA GPU. Versions are pinned in `uv.lock` (the same as supermimic_hoi_vibed):

```bash
uv sync --all-extras
```

Assets and references are built from two checkouts:

```bash
uv run python -m supermimic_box.prepare --sim2sim /path/to/OmniContact_sim2sim --omnicontact /path/to/OmniContact
```

For the OmniContact dataset, download its `npz_clips/box` and `metadata` folders (gated, 4.6 GB,
CC BY-NC 4.0) and point `--dataset` at the folder:

```bash
hf download lightcone02/OmniContact-Dataset --repo-type dataset --include "npz_clips/box/*" --include "metadata/*" --local-dir data/source/OmniContact-Dataset
uv run python -m supermimic_box.prepare --sim2sim /path/to/OmniContact_sim2sim --dataset data/source/OmniContact-Dataset
```

This writes `data/omnicontact/`: 1,070 clips (5.5 h; 617 carry, 453 push) resampled from 90 to
50 Hz, 90 box sizes, and the split (786 train, 145 val, 139 test). Six carry clips with joint
angles far outside the robot's limits are left out. The dataset's 190 slide clips (the box is
moved with the feet) are not prepared unless `--tasks carry push slide` asks for them.

Without `--dataset`, the local clips are prepared instead. That
writes to `data/` (git-ignored): the robot MJCF, three boxes, the platform and 16 clips
(4.9 min): 13 carry and 2 push clips from `OmniContact_sim2sim/data`, and the carry clip
under `OmniContact/assets/data`. `--omnicontact` is optional; any motion file below its
`assets/data` whose name gives the task and box size
(`carry_box_15-30-22_*.npz`) is added, so the Hugging Face dataset can be dropped in there.

Box sizes are not stored in the OmniContact_sim2sim clips. They are taken from its scenes:
a 30 cm cube for carry, and 46 x 50 x 52 cm for push. Every box weighs 2 kg, as in
OmniContact's training; pushed boxes slide more easily (friction 0.15 to 0.35 instead of 0.4 to 0.7).

## Usage

```bash
# Check the scene against the references: open-loop playback of every clip
uv run python -m supermimic_box.audit --config configs/box.json

# Train: the dataset's 786 training clips, or the 16 local clips; 1,024 worlds
uv run python -m supermimic_box.train --config configs/omnicontact.json
uv run python -m supermimic_box.train --config configs/box.json

# Evaluate: completion and success rate per task, on held-out clips or on the training clips
uv run python -m supermimic_box.evaluate logs/omnicontact/last.pt --split test
uv run python -m supermimic_box.evaluate logs/box/last.pt

# Record one clip and render it next to its reference
uv run python -m supermimic_box.evaluate logs/box/last.pt --record carry_1
uv run python -m supermimic_box.render --rollout results/evaluation_carry_1.npz --output results/rollout.mp4

# Tests
uv run pytest
```

A clip is **completed** when the episode reaches its last frame, and a **success** when in
addition the box stayed within 0.2 m and 0.8 rad of its reference, covered at least half of the
reference box's path, and was raised if the reference raises it.

## Layout

| File | Content |
|---|---|
| `prepare/` | robot MJCF, boxes and platform, clips |
| `robot.py` | joints, drives, tracked bodies, contact channels |
| `motion.py` | reference library and the goal |
| `sim.py`, `contacts.py` | MuJoCo CPU and batched MuJoCo-Warp physics; measured contact forces |
| `mdp.py` | observations, rewards, terminations (analytic, differentiable) |
| `env.py`, `reset_guard.py`, `randomization.py` | episodes, reset screening, randomization |
| `networks.py`, `runner.py`, `learning.py` | world model, contact head, actor, critic and their training |
| `evaluate.py`, `render.py`, `audit.py` | evaluation, videos, scene audit |
