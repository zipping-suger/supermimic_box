"""Policy actions -> joint PD targets.

``absolute`` (OmniContact): the target is the default pose plus the scaled action, so the policy
needs no joint reference. ``residual``: the target is the next reference pose plus the scaled
action. Every joint is commanded; the scale is 0.25 effort / stiffness per joint.
"""

from __future__ import annotations

import torch
from torch import nn

from .robot import Robot

ACTION_MODES = ("absolute", "residual")


class Controller(nn.Module):
    def __init__(self, robot: Robot, device: str = "cpu", action_mode: str = "absolute", action_clip: float = 20.0):
        super().__init__()
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        value = lambda x: torch.tensor(x, device=device, dtype=torch.float32)  # noqa: E731
        self.register_buffer("limits", value(robot.joint_limits))
        self.register_buffer("default", value(robot.default_joint_pos))
        self.register_buffer("scale", value(robot.control_scale))
        self.register_buffer("gains", value(robot.gains))
        self.register_buffer("torque_joints", torch.tensor(robot.torque_idx, device=device, dtype=torch.long))
        self.residual, self.action_clip, self.num_actions = action_mode == "residual", action_clip, robot.num_actions

    def skip(self, qref_next):
        """The part of the normalized command that the reference supplies."""
        return (qref_next - self.default) / self.scale if self.residual else torch.zeros_like(qref_next)

    def command(self, action, qref_next):
        """Full normalized command: what the drives are asked for, in units of the action."""
        return self.skip(qref_next) + action

    def bound_action(self, action, qref_next):
        """Clip the full command and return the bounded action."""
        skip = self.skip(qref_next)
        return (action + skip).clamp(-self.action_clip, self.action_clip) - skip

    def reference_action(self, qref):
        """The action that asks for the reference pose ``qref``."""
        return (qref - self.default) / self.scale - self.skip(qref)

    def forward(self, action, qref_next, offset):
        """PD targets (..., J). ``action`` must already be bounded. ``offset`` is the per-joint encoder
        offset: the policy observes q - offset, so every target moves by +offset."""
        return self.default + self.scale * self.command(action, qref_next) + offset

    def effective(self, target):
        """Normalized physical PD target, the world model's action input."""
        return (target - self.default) / self.scale

    def torque(self, target, q, qd):
        """PD torque (..., J) that the drives apply for ``target`` in the state (q, qd)."""
        stiffness, damping, effort = self.gains.unbind(-1)
        return torch.maximum(torch.minimum(stiffness * (target - q) - damping * qd, effort), -effort)
