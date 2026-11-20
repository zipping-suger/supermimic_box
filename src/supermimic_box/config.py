"""Serializable configuration shared by training, evaluation and checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields


@dataclass
class RewardConfig:
    """Reward weights/scales and termination thresholds. Every reward is scaled by the 0.02 s control step.

    The bonuses up to ``contact`` and the costs up to ``joint_limit`` are OmniContact's CFtrack
    rewards with its weights. OmniContact shapes the gait with an adversarial motion prior; here
    the gait terms do it (foot tracking, the stance pattern, and costs on sliding feet and short
    swings): without them the policy shuffles in short, low steps.
    """

    # Tracking bonuses: weight * exp(-error^2 / std^2).
    anchor_pos: float = 0.2  # torso, world frame
    anchor_ori: float = 0.2
    body_pos: float = 0.2  # tracked bodies relative to the torso
    body_ori: float = 0.2
    hand_pos: float = 0.4  # both hands, world frame
    hand_ori: float = 0.4
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    object_pos: float = 1.0
    object_pos_std: float = 0.3
    object_ori: float = 1.0
    object_ori_std: float = 0.3
    # Hands relative to the object; the tolerance grows with the reference hand-object distance.
    interact_pos: float = 0.4
    interact_ori: float = 0.4
    interact_std: float = 0.3
    # exp(-mean squared difference between the measured (simulation) or predicted (imagination)
    # contact of each channel with the object and the reference's contact flags / std^2).
    contact: float = 1.0
    contact_std: float = 1.0
    hand_contact_force: float = 5.0  # [N] above which a hand counts as on the object
    foot_object_force: float = 1.0  # [N] above which a foot counts as on the object
    # Gait. Both feet in the world frame, with a tolerance that resolves a step's height.
    foot_pos: float = 0.5
    foot_pos_std: float = 0.1
    # exp(-mean squared difference between each foot's measured or predicted contact with the
    # ground and the reference's stance / std^2): in the air when the reference swings, planted
    # when it stands.
    stance: float = 0.5
    stance_std: float = 0.5
    # Velocity tracking; not part of OmniContact's rewards.
    body_velocity: float = 0.0
    body_angular_velocity: float = 0.0
    body_angvel_std: float = 3.14
    object_velocity: float = 0.0
    # Costs.
    action_rate: float = 0.1
    torque: float = 5e-3  # squared torque of the elbow and wrist drives
    joint_limit: float = 10.0
    # Feet: horizontal speed of a foot on the ground, and the time by which a swing that just
    # ended fell short of ``feet_air_time_threshold`` seconds. A foot is on the ground while the
    # ground pushes on it with more than ``foot_contact_force`` newtons.
    feet_slide: float = 1.0  # OmniContact: 0.1
    feet_air_time: float = 20.0
    feet_air_time_threshold: float = 0.3
    foot_contact_force: float = 1.0
    # Termination.
    term_anchor_pos: float = 0.3
    term_tilt: float = 0.8
    term_object_pos: float = 0.3
    term_object_rot: float = 1.0  # rotation angle between the object and its reference [rad]
    term_end_height: float = 0.25  # height error of a foot or hand

    BONUSES = ("anchor_pos", "anchor_ori", "body_pos", "body_ori", "hand_pos", "hand_ori", "object_pos", "object_ori",
               "interact_pos", "interact_ori", "contact", "foot_pos", "stance", "body_velocity", "body_angular_velocity",
               "object_velocity")  # fmt: skip
    COSTS = ("action_rate", "torque", "joint_limit", "feet_slide", "feet_air_time")

    def validate(self):
        # The critic's finite return bound assumes nonnegative bounded bonuses and costs.
        for name in (*self.BONUSES, *self.COSTS):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reward.{name} must be nonnegative and finite")
        for name in ("body_pos_std", "body_ori_std", "body_angvel_std", "object_pos_std", "object_ori_std", "interact_std",
                     "contact_std", "foot_pos_std", "stance_std", "feet_air_time_threshold", "foot_contact_force", "hand_contact_force", "foot_object_force"):  # fmt: skip
            if not getattr(self, name) > 0:
                raise ValueError(f"reward.{name} must be positive")


@dataclass
class TrainConfig:
    # Data produced by `python -m supermimic_box.prepare`.
    dataset: str = "data/box"
    robot_assets: str = "data/robot"
    object_assets: str = "data/objects"
    clips: list[str] = field(default_factory=list)  # empty: the clips of ``split``
    split: str = ""  # "train", "val" or "test" of a dataset that has a split; empty: every clip

    # What the policy is given and what it commands.
    # goal: "flow" (OmniContact's contact flow: future poses of hands, torso and feet, and contact
    # flags) or "full" (root, joints and object as well). action: "absolute" (joint targets around
    # the default pose) or "residual" (around the reference pose, which only a full reference has).
    goal_mode: str = "flow"
    action_mode: str = "absolute"
    # A frozen pre-trained tracker whose action the policy adds to: "" (none) or "sonic" (GEAR-SONIC,
    # which follows the joint reference; the policy then learns only a residual). SONIC needs its
    # own drive set, "sonic", in place of "omnicontact".
    base_policy: str = ""
    base_checkpoint: str = "data/sonic/last_ported.pt"
    # With a base policy: scale of the residual on the leg and waist joints (1: as free as on the
    # arms). A small value leaves locomotion and balance to the base; the arms handle the box.
    leg_residual_scale: float = 1.0
    drives: str = "omnicontact"

    # Simulation. A world draws a clip at every reset; num_envs is at least the number of objects.
    device: str = "cuda:0"
    backend: str = "warp"
    num_envs: int = 600
    max_episode_steps: int = 1000
    randomize: bool = True  # robot friction, torso CoM, encoder offsets, object mass and friction, observation noise
    object_mass_range: list[float] = field(default_factory=lambda: [0.5, 1.5])  # scale of the nominal mass
    nconmax: int = 64
    njmax: int = 400
    nccdmax: int | None = None
    # Numerical admission limits, not MDP termination: a world beyond them is rejected and reset.
    max_linear_speed: float = 20.0
    max_angular_speed: float = 200.0
    max_joint_speed: float = 500.0

    # Episode starts: frame zero, shortly before a contact phase, otherwise uniform over the clip.
    reset_frame_zero_probability: float = 0.15
    reset_precontact_probability: float = 0.25
    # Deepest reference overlap [m] admitted in a reset state, per kind of collision pair: a hand or
    # foot in the object, the robot in the ground or a platform, anything else.
    reset_overlap_tolerance: dict[str, float] = field(default_factory=lambda: {"limb_object": 0.08, "robot_ground": 0.1, "other": 0.06})

    # Collection and replay.
    iterations: int = 30000
    collect_steps: int = 5
    buffer_steps: int = 1000
    warmup_steps: int = 64
    exploration: float = 0.1

    # World model and contact-reward head.
    hidden: list[int] = field(default_factory=lambda: [1024, 1024, 1024])
    reward_head_hidden: list[int] = field(default_factory=lambda: [256, 128])
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 5

    # Actor and critic.
    policy_hidden: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    critic_hidden: list[int] = field(default_factory=lambda: [256, 128])
    history: int = 10
    future: int = 10
    pi_batch: int = 2048
    horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    target_critic_ema: float = 0.4  # fraction of the bootstrap critic retained per policy update
    gamma: float = 0.99
    lam: float = 0.95
    ema: float = 0.995

    # Optimization.
    lr: float = 4e-4
    lr_decay_iters: int = 250000
    pi_lr_mult: float = 4.0  # actor multiplier until the return trigger, then ramped to 1
    lr_decay_ret: float = 0.75
    lr_decay_by: int = 1400
    lr_decay_len: int = 300
    pi_beta1: float = 0.9
    pi_clip_value: float = 0.1
    grad_clip_norm: float = 1.0
    max_gradient_norm: float = 1e4  # an update with a larger pre-clipping norm is skipped
    max_skipped_updates: int = 20  # consecutive skipped updates of one model that stop training
    norm_max_count: float = 2e8  # running normalizers freeze after this many rows
    action_clip: float = 20.0  # bound of the normalized joint command

    # Execution.
    amp: str = "bf16"
    compile: bool = True
    cuda_graph: bool = True

    # Bookkeeping.
    seed: int = 0
    log_dir: str = "logs/box"
    log_every: int = 10
    save_every: int = 1000
    resume: str = ""
    swanlab_mode: str = "disabled"
    swanlab_project: str = "supermimic_box"
    swanlab_name: str | None = None

    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)

    @classmethod
    def from_json(cls, path, **overrides):
        values = json.loads(open(path).read()) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def validate(self):
        self.reward.validate()
        for name in ("num_envs", "max_episode_steps", "nconmax", "njmax", "iterations", "collect_steps", "buffer_steps",
                     "warmup_steps", "wm_window", "wm_batch", "wm_updates", "history", "future", "pi_batch", "horizon",
                     "critic_epochs", "critic_minibatches", "lr_decay_iters", "lr_decay_len", "log_every", "save_every",
                     "max_skipped_updates"):  # fmt: skip
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_linear_speed", "max_angular_speed", "max_joint_speed", "lr", "pi_lr_mult", "grad_clip_norm",
                     "max_gradient_norm", "norm_max_count", "action_clip"):  # fmt: skip
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.backend not in ("cpu", "warp") or self.amp not in ("none", "bf16"):
            raise ValueError("Unsupported backend or AMP mode")
        if (self.backend == "cpu") != (self.device == "cpu"):
            raise ValueError("backend='cpu' requires device='cpu'; backend='warp' requires a CUDA device")
        if self.backend == "cpu" and (self.randomize or self.cuda_graph):
            raise ValueError("The CPU backend supports neither randomization nor CUDA graphs")
        if self.nccdmax is not None and not 0 < self.nccdmax <= self.nconmax:
            raise ValueError("nccdmax must be positive and at most nconmax")
        if self.goal_mode not in ("flow", "full") or self.action_mode not in ("absolute", "residual"):
            raise ValueError("Unsupported goal or action mode")
        if self.action_mode == "residual" and self.goal_mode != "full":
            raise ValueError("Residual actions need the full reference as the goal")
        if self.base_policy not in ("", "sonic") or self.drives not in ("omnicontact", "sonic"):
            raise ValueError("Unsupported base policy or drive set")
        if self.base_policy == "sonic" and (self.action_mode != "absolute" or self.drives != "sonic" or self.history < 10):
            raise ValueError("The SONIC base needs absolute actions, the sonic drive set and a history of at least 10 steps")
        if not 0 <= self.leg_residual_scale <= 1:
            raise ValueError("leg_residual_scale must be in [0, 1]")
        if not 0 < self.object_mass_range[0] <= self.object_mass_range[1]:
            raise ValueError("Invalid object mass range")
        probabilities = (self.reset_frame_zero_probability, self.reset_precontact_probability)
        if any(not 0 <= p <= 1 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError("Invalid reset mixture probabilities")
        if set(self.reset_overlap_tolerance) != {"limb_object", "robot_ground", "other"} or any(
                not (math.isfinite(v) and v >= 0) for v in self.reset_overlap_tolerance.values()):  # fmt: skip
            raise ValueError("reset_overlap_tolerance needs finite nonnegative limb_object, robot_ground and other depths")
        if self.buffer_steps < self.wm_window or self.warmup_steps < self.wm_window:
            raise ValueError("Replay and warmup must cover wm_window")
        if (self.pi_batch * self.horizon) % self.critic_minibatches:
            raise ValueError("pi_batch * horizon must be divisible by critic_minibatches")
        if not (0 < self.gamma < 1 and 0 <= self.lam <= 1 and 0 <= self.ema < 1 and 0 <= self.pi_beta1 < 1):
            raise ValueError("Invalid discount, lambda, EMA or beta1")
        if not 0 <= self.target_critic_ema < 1 or not 0 <= self.exploration:
            raise ValueError("Invalid target critic retention or exploration scale")
        if self.swanlab_mode not in ("disabled", "offline", "online"):
            raise ValueError("Unsupported SwanLab mode")
