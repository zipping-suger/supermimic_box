"""Measured contact forces. Simulator only: imagination never queries contact geometry.

Six force totals per world: the object on the [left foot, right foot, left hand, right hand]
(OmniContact's contact channels), then the [left foot, right foot] on what they stand on (the
ground or a platform). Only the normal force counts.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

from .randomization import support_geoms

NUM_GROUPS = 6
OBJECT, FEET = slice(0, 4), slice(4, 6)


class ContactGroups:
    """Per-geom lookup tables for a compiled scene with ``robot/``, ``object/`` and ``table*/`` entities."""

    def __init__(self, model: mujoco.MjModel, robot):
        self.object_geom = np.zeros(model.ngeom, dtype=np.int32)
        self.channel = np.full(model.ngeom, -1, dtype=np.int32)
        self.foot = np.full(model.ngeom, -1, dtype=np.int32)
        self.support = np.zeros(model.ngeom, dtype=np.int32)
        self.support[support_geoms(model)] = 1
        for geom in range(model.ngeom):
            body = model.body(model.geom_bodyid[geom]).name
            self.object_geom[geom] = body.startswith("object/")
            if body.startswith("robot/"):
                link = body.removeprefix("robot/")
                self.channel[geom] = robot.contact_channel(link)
                if link in robot.foot_bodies:
                    self.foot[geom] = robot.foot_bodies.index(link)

    def group(self, g0: int, g1: int) -> int:
        if self.object_geom[g0] != self.object_geom[g1]:
            return int(self.channel[g0 if self.object_geom[g1] else g1])
        if self.support[g0] != self.support[g1]:
            side = self.foot[g0 if self.support[g1] else g1]
            return int(4 + side) if side >= 0 else -1
        return -1

    def cpu_forces(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        forces, wrench = np.zeros(NUM_GROUPS), np.zeros(6)
        for i in range(data.ncon):
            g0, g1 = data.contact[i].geom
            group = self.group(g0, g1) if min(g0, g1) >= 0 else -1
            if group >= 0:
                mujoco.mj_contactForce(model, data, i, wrench)
                forces[group] += max(0.0, wrench[0])
        return forces


@wp.kernel(enable_backward=False)
def reduce_contacts(cone: int, njmax: int, nacon: wp.array(dtype=wp.int32),
                    geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=wp.int32),
                    frame: wp.array(dtype=wp.mat33), friction: wp.array(dtype=vec5),
                    dim: wp.array(dtype=wp.int32), efc_address: wp.array2d(dtype=wp.int32),
                    efc_force: wp.array2d(dtype=wp.float32), object_geom: wp.array(dtype=wp.int32),
                    channel: wp.array(dtype=wp.int32), support: wp.array(dtype=wp.int32),
                    foot: wp.array(dtype=wp.int32), result: wp.array2d(dtype=wp.float32)):  # fmt: skip
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    if g[0] < 0 or g[1] < 0:
        return
    group = int(-1)
    if object_geom[g[0]] != object_geom[g[1]]:
        other = g[0]
        if object_geom[g[0]] != 0:
            other = g[1]
        group = channel[other]
    elif support[g[0]] != support[g[1]]:
        other = g[0]
        if support[g[0]] != 0:
            other = g[1]
        if foot[other] >= 0:
            group = 4 + foot[other]
    if group < 0:
        return
    world = worldid[i]
    force = contact_force_fn(cone, frame, friction, dim, efc_address, efc_force, njmax, nacon, world, i, False)
    wp.atomic_add(result, world, group, wp.max(force[0], 0.0))


def kernel_inputs(model, data):
    """The MuJoCo-Warp arrays ``reduce_contacts`` reads (its dataclasses are host objects)."""
    c = data.contact
    return [int(model.opt.cone), data.njmax, data.nacon, c.geom, c.worldid, c.frame, c.friction, c.dim,
            c.efc_address, data.efc.force]  # fmt: skip
