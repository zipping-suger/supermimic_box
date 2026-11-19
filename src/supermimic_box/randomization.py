"""Domain randomization and the privileged simulation context.

The context seen by the world model, contact head and critic is, in order: the object's mass and
sliding friction, the robot parameters xi = [friction, encoder offsets, torso CoM offset] (all
normalized), and the world positions of the two support platforms. The actor never sees it.
Robot friction and CoM are drawn once per world; encoder offsets, object mass and object friction
once per episode.

Friction in MuJoCo is the larger one of a contact pair. The ground and the platforms get almost
none, so the robot's friction is that of its feet on the ground, the object's friction that of the
object on the ground, and a hand on the object gets the larger of the two.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import COM_RANGE, FRICTION_RANGE, JOINT_OFFSET, Robot

OBJECT_CONTEXT = 2  # mass, sliding friction
TABLES = 6
SUPPORT_FRICTION = 0.01  # ground and platforms
MASS_SCALE, FRICTION_CENTER, FRICTION_SCALE = 4.0, 0.5, 0.3


class ContextLayout:
    def __init__(self, robot: Robot, device: str):
        J = robot.num_joints
        self.friction, self.joint_offset, self.com = slice(0, 1), slice(1, 1 + J), slice(1 + J, 4 + J)
        self.xi_dim = 4 + J
        self.model_dim = OBJECT_CONTEXT + self.xi_dim
        self.dim = self.model_dim + TABLES
        lo, hi = FRICTION_RANGE
        self.center = torch.tensor([(lo + hi) / 2] + [0.0] * (J + 3), device=device)
        self.scale = torch.tensor([(hi - lo) / 2] + [JOINT_OFFSET] * J + [0.05] * 3, device=device)

    def normalize(self, xi):
        return (xi - self.center) / self.scale

    def offsets(self, context):
        """Encoder offsets in radians from a full context row."""
        xi = context[..., OBJECT_CONTEXT : self.model_dim] * self.scale + self.center
        return xi[..., self.joint_offset]

    def model(self, context):
        """The normalized physical parameters, without the platform positions."""
        return context[..., : self.model_dim]

    def tables(self, context):
        """(..., 2, 3) world positions of the two platform centres."""
        return context[..., self.model_dim :].unflatten(-1, (2, 3))


class Randomizer:
    """Samples the randomized parameters and applies the robot's part to the simulator. Disabled:
    robot friction 1, no offsets, nominal object mass and the middle of its friction range."""

    def __init__(self, robot: Robot, physics, layout: ContextLayout, objects: list[dict], enabled: bool,
                 mass_range=(0.5, 1.5)):  # fmt: skip
        self.layout, self.enabled, self.physics, self.mass_range = layout, enabled, physics, mass_range
        N, device = physics.N, physics.device
        self.xi = torch.zeros(N, layout.xi_dim, device=device)
        self.xi[:, layout.friction] = 1.0
        self.joint_offset = torch.full((robot.num_joints,), JOINT_OFFSET, device=device)
        self.object_mass = torch.tensor([o["mass"] for o in objects], device=device)
        self.object_friction = torch.tensor([o["friction"] for o in objects], device=device)
        self.mass_scale = torch.ones(N, device=device)
        self.friction = torch.zeros(N, device=device)
        self.mass = torch.zeros(N, device=device)
        if hasattr(physics, "sim"):
            self._apply(robot, physics.sim, N, device)
        elif enabled:
            raise ValueError("Randomization needs the MuJoCo-Warp backend")

    def _apply(self, robot, sim, N, device):
        model = sim.mj_model
        body_names = [model.body(b).name for b in model.geom_bodyid]
        robot_geoms = torch.tensor([i for i, n in enumerate(body_names) if n.startswith("robot/")], device=device)
        support = torch.tensor(support_geoms(model), device=device)
        # The physics backend has already made friction and inertial frames per-world.
        if self.enabled:
            self.xi[:, self.layout.friction] = torch.empty(N, 1, device=device).uniform_(*FRICTION_RANGE)
            com = torch.stack([torch.empty(N, device=device).uniform_(lo, hi) for lo, hi in COM_RANGE], -1)
            self.xi[:, self.layout.com] = com
            torso = model.body("robot/" + robot.com_body).id
            default = sim.get_default_field("body_ipos")
            default = default[:, torso] if default.ndim == 3 else default[torso]
            sim.model.body_ipos[:, torso] = default + com
        sim.model.geom_friction[:, support, 0] = SUPPORT_FRICTION
        sim.model.geom_friction[:, robot_geoms, 0] = self.xi[:, self.layout.friction]

    def resample(self, mask, objects):
        """New encoder offsets, object mass and object friction for the worlds that start an episode
        with the objects ``objects`` (N,)."""
        lo, hi = self.object_friction[objects].unbind(-1)
        if self.enabled:
            new = (2 * torch.rand_like(self.xi[:, self.layout.joint_offset]) - 1) * self.joint_offset
            self.xi[:, self.layout.joint_offset] = torch.where(mask[:, None], new, self.xi[:, self.layout.joint_offset])
            scale = torch.empty_like(self.mass_scale).uniform_(*self.mass_range)
            friction = lo + torch.rand_like(lo) * (hi - lo)
        else:
            scale, friction = torch.ones_like(self.mass_scale), (lo + hi) / 2
        self.mass_scale.copy_(torch.where(mask, scale, self.mass_scale))
        self.friction.copy_(torch.where(mask, friction, self.friction))
        self.mass.copy_(torch.where(mask, self.object_mass[objects] * scale, self.mass))

    def context(self, tables):
        """Full context rows for platform positions ``tables`` (N, 6)."""
        obj = torch.stack((self.mass / MASS_SCALE, (self.friction - FRICTION_CENTER) / FRICTION_SCALE), -1)
        return torch.cat((obj, self.layout.normalize(self.xi), tables), -1)


def support_geoms(model: mujoco.MjModel) -> np.ndarray:
    """Geoms that things rest on: the ground plane and the platforms (which only the object touches)."""
    body = [model.body(b).name for b in model.geom_bodyid]
    support = (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) | np.array([n.startswith("table") for n in body])
    if not (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any():
        raise ValueError("The scene has no ground plane")
    return np.flatnonzero(support)
