"""MuJoCo CPU and batched MuJoCo-Warp physics behind one state contract.

Both backends expose ``state()``, ``reset(mask, state, objects, tables, mass_scale, friction)``,
``step(target)`` and ``check()``. A state is the tuple ``(p, R, v, w, q, qd, op, oR, ov, ow)``: root
position, rotation matrix, world linear/angular velocity, joint positions/velocities, and the same
four object quantities. The two support platforms are static within an episode and not part of
the state. One control step is four 5 ms physics steps with the PD targets held.
"""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import torch

from .contacts import FEET, NUM_GROUPS, OBJECT, ContactGroups
from .randomization import SUPPORT_FRICTION, support_geoms
from .rotations import mat_to_quat, mv3, quat_to_mat

DECIMATION = 4
PHYSICS_DT = 0.02 / DECIMATION
CCD_ITERATIONS = 35
# Fatal fault bits: these stop training. A finite speed excursion is not fatal; that world is
# rejected and reset.
NONFINITE_STATE, CAPACITY, NONFINITE_TARGET = 1, 2, 4
FAULTS = {NONFINITE_STATE: "nonfinite state", CAPACITY: "contact/constraint capacity exceeded",
          NONFINITE_TARGET: "nonfinite controller target"}  # fmt: skip


def load_registry(folder: str | Path) -> tuple[dict, str]:
    """Object registry and its signature; every generated file must match its recorded checksum."""
    folder = Path(folder)
    raw = (folder / "registry.json").read_bytes()
    registry = json.loads(raw)
    for name, expected in registry["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Object asset changed since preparation: {name}")
    return registry, hashlib.sha256(raw).hexdigest()


def robot_signature(robot) -> str:
    """Fingerprint of the robot MJCF, the meshes it references and its other asset files."""
    digest = hashlib.sha256(robot.mjcf_path.read_bytes())
    meshes = sorted(path for path in (robot.asset_dir / "meshes").rglob("*") if path.is_file())
    for file in meshes:
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    if robot.drives != "omnicontact":
        digest.update(robot.drives.encode())
    return digest.hexdigest()


def simulation_cfg(nconmax: int, njmax: int):
    import warp as wp
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mujoco_warp._src import collision_convex, collision_flex, types

    class Cfg(SimulationCfg):
        def apply_wp_opt(self, wp_opt) -> None:
            super().apply_wp_opt(wp_opt)
            if not wp.is_conditional_graph_supported():
                # Drivers before CUDA 12.4: run the solver's fixed maximum loop instead of a
                # conditional graph node. Convergence checks and tolerances are unchanged.
                wp_opt.graph_conditional = False

    # MuJoCo-Warp 3.11 sizes its EPA horizon by a module constant (24) that mesh-mesh contacts
    # overflow. Each of the 6 + 5 * ccd_iterations faces contributes at most three horizon edges.
    horizon = 3 * (6 + 5 * CCD_ITERATIONS) + 1
    for module in (types, collision_convex, collision_flex):
        module.MJ_MAX_EPAHORIZON = max(horizon, module.MJ_MAX_EPAHORIZON)
    return Cfg(nconmax=nconmax, njmax=njmax, mujoco=MujocoCfg(
        timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal", solver="newton", iterations=100,
        ls_iterations=50, tolerance=1e-6, ccd_iterations=CCD_ITERATIONS))  # fmt: skip


TABLES = ("table1", "table2")
# Model rows that make a world's box what it is: its size, bounds and inertial.
OBJECT_GEOM_FIELDS = ("geom_size", "geom_rbound", "geom_aabb")
OBJECT_BODY_FIELDS = ("body_mass", "body_inertia", "body_invweight0", "body_subtreemass")


def make_scene(robot, registry: dict, folder: str | Path, obj: int, num_envs: int, device: str):
    """Robot, ground plane, the object with registry index ``obj`` and the two support platforms.
    A platform is a fixed-base entity, which mjlab makes a mocap body that can be placed per world."""
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    spec = lambda xml: partial(mujoco.MjSpec.from_file, str(Path(folder) / xml))  # noqa: E731
    entities = {"robot": robot.entity_cfg(), "object": EntityCfg(spec_fn=spec(registry["objects"][obj]["xml"]))}
    entities.update({name: EntityCfg(spec_fn=spec(registry["table"]["xml"])) for name in TABLES})
    cfg = SceneCfg(num_envs=num_envs, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"), entities=entities)
    return Scene(cfg, device)


def table_mocap_ids(model: mujoco.MjModel) -> list[int]:
    """Mocap index of each platform."""
    ids = []
    for name in TABLES:
        bodies = [b for b in range(model.nbody) if model.body(b).name.startswith(name + "/") and model.body_mocapid[b] >= 0]
        if len(bodies) != 1:
            raise ValueError(f"Expected one mocap body for {name}")
        ids.append(int(model.body_mocapid[bodies[0]]))
    return ids


class Indexing:
    """qpos/qvel/ctrl addresses of the robot joints and of both free bodies in a compiled scene."""

    def __init__(self, model: mujoco.MjModel, robot):
        joints = [model.joint("robot/" + name).id for name in robot.joint_names]
        self.jq = model.jnt_qposadr[joints].copy()
        self.jv = model.jnt_dofadr[joints].copy()
        self.ctrl = np.array([next(i for i in range(model.nu) if model.actuator_trnid[i, 0] == j) for j in joints])
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        root = next(j for j in free if model.body(model.jnt_bodyid[j]).name.startswith("robot/"))
        obj = next(j for j in free if model.body(model.jnt_bodyid[j]).name.startswith("object/"))
        self.rq, self.rv = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
        self.oq, self.ov = int(model.jnt_qposadr[obj]), int(model.jnt_dofadr[obj])

    def to(self, device):
        for name in ("jq", "jv", "ctrl"):
            setattr(self, name, torch.as_tensor(getattr(self, name), device=device, dtype=torch.long))
        return self


def decode(qpos, qvel, idx: Indexing):
    """MuJoCo coordinates -> state tuple (free-joint angular velocity is body-local in qvel)."""
    R = quat_to_mat(qpos[..., idx.rq + 3 : idx.rq + 7])
    oR = quat_to_mat(qpos[..., idx.oq + 3 : idx.oq + 7])
    return (qpos[..., idx.rq : idx.rq + 3], R, qvel[..., idx.rv : idx.rv + 3], mv3(R, qvel[..., idx.rv + 3 : idx.rv + 6]),
            qpos[..., idx.jq], qvel[..., idx.jv],
            qpos[..., idx.oq : idx.oq + 3], oR, qvel[..., idx.ov : idx.ov + 3], mv3(oR, qvel[..., idx.ov + 3 : idx.ov + 6]))  # fmt: skip


def encode(state, qpos, qvel, idx: Indexing):
    """Write a state tuple into MuJoCo coordinates in place."""
    p, R, v, w, q, qd, op, oR, ov, ow = state
    qpos[..., idx.rq : idx.rq + 7] = torch.cat((p, mat_to_quat(R)), -1)
    qpos[..., idx.oq : idx.oq + 7] = torch.cat((op, mat_to_quat(oR)), -1)
    qpos[..., idx.jq] = q
    qvel[..., idx.rv : idx.rv + 6] = torch.cat((v, mv3(R.transpose(-1, -2), w)), -1)
    qvel[..., idx.ov : idx.ov + 6] = torch.cat((ov, mv3(oR.transpose(-1, -2), ow)), -1)
    qvel[..., idx.jv] = qd


def numerical_status(qpos, qvel, idx: Indexing, cfg):
    """(valid, finite, speeds): numerical admission of a state, separate from MDP termination."""
    finite = torch.isfinite(qpos).all(-1) & torch.isfinite(qvel).all(-1)
    linear = torch.maximum(qvel[..., idx.rv : idx.rv + 3].norm(dim=-1), qvel[..., idx.ov : idx.ov + 3].norm(dim=-1))
    angular = torch.maximum(qvel[..., idx.rv + 3 : idx.rv + 6].norm(dim=-1), qvel[..., idx.ov + 3 : idx.ov + 6].norm(dim=-1))
    joint = qvel[..., idx.jv].abs().amax(-1)
    valid = finite & (linear <= cfg.max_linear_speed) & (angular <= cfg.max_angular_speed) & (joint <= cfg.max_joint_speed)
    return valid, finite, torch.stack((linear, angular, joint), -1)


class CPUPhysics:
    """One MuJoCo model per object and one MjData per world. Deterministic and nominal (no
    randomization: robot friction 1, objects at their nominal mass and friction); for tests."""

    def __init__(self, robot, registry, folder, objects: list[int], num_envs: int, cfg):
        """``objects``: registry indices of the objects the worlds may hold."""
        self.cfg, self.device, self.N = cfg, "cpu", num_envs
        self.models, self.indices, self.groups, self.mocap = {}, {}, {}, {}
        sim_cfg = simulation_cfg(cfg.nconmax, cfg.njmax)
        for k in objects:
            model = make_scene(robot, registry, folder, k, 1, "cpu").compile()
            sim_cfg.mujoco.apply(model)
            self.groups[k] = ContactGroups(model, robot)
            on_robot = np.array([model.body(b).name.startswith("robot/") for b in model.geom_bodyid])
            model.geom_friction[on_robot, 0] = 1.0
            model.geom_friction[support_geoms(model), 0] = SUPPORT_FRICTION
            self.models[k], self.indices[k], self.mocap[k] = model, Indexing(model, robot), table_mocap_ids(model)
        self.assignment = [objects[e % len(objects)] for e in range(num_envs)]
        self.data = [mujoco.MjData(self.models[k]) for k in self.assignment]
        self.contact_force, self.foot_force = torch.zeros(self.N, 4), torch.zeros(self.N, 2)

    def state(self):
        states = [decode(torch.tensor(d.qpos, dtype=torch.float32), torch.tensor(d.qvel, dtype=torch.float32), self.indices[k])
                  for k, d in zip(self.assignment, self.data)]  # fmt: skip
        return tuple(torch.stack([s[i] for s in states]) for i in range(10))

    def reset(self, mask, state, objects, tables, mass_scale=None, friction=None, gains=None):
        for e in mask.nonzero().flatten().tolist():
            k = int(objects[e])
            if k != self.assignment[e]:
                self.assignment[e], self.data[e] = k, mujoco.MjData(self.models[k])
            d = self.data[e]
            mujoco.mj_resetData(self.models[k], d)
            encode(tuple(x[e].double() for x in state), torch.from_numpy(d.qpos), torch.from_numpy(d.qvel), self.indices[k])
            d.mocap_pos[self.mocap[k]] = tables[e].reshape(2, 3).numpy()
            d.ctrl[self.indices[k].ctrl] = state[4][e].numpy()
            mujoco.mj_forward(self.models[k], d)
            self.contact_force[e], self.foot_force[e] = 0, 0

    def step(self, target):
        if not bool(torch.isfinite(target).all()):
            raise RuntimeError("Nonfinite controller target")
        for e, (k, d) in enumerate(zip(self.assignment, self.data)):
            model, idx = self.models[k], self.indices[k]
            d.ctrl[idx.ctrl] = target[e].numpy()
            warnings = d.warning.number.copy()
            force = np.zeros(NUM_GROUPS)
            for substep in range(DECIMATION):
                mujoco.mj_step(model, d)
                valid = numerical_status(torch.from_numpy(d.qpos), torch.from_numpy(d.qvel), idx, self.cfg)[0]
                if not bool(valid) or (d.warning.number > warnings).any():
                    raise RuntimeError(f"Invalid physics in world {e}, substep {substep}")
                force += self.groups[k].cpu_forces(model, d) / DECIMATION
            force = torch.from_numpy(force).float()
            self.contact_force[e], self.foot_force[e] = force[OBJECT], force[FEET]
        return self.state(), self.contact_force.clone(), self.foot_force.clone(), torch.ones(self.N, dtype=torch.bool)

    @property
    def ok(self):
        return torch.tensor(True)

    def check(self) -> int:
        return 0


class WarpPhysics:
    """Batched MuJoCo-Warp worlds, each holding one box that a reset may exchange.

    All worlds share one compiled structure. A box is a set of per-world model rows (its size,
    bounds and inertial); a reset writes the rows of the new box, scaled to the world's mass, its
    friction, and the positions of the two platforms. Every method is fixed-shape and free of host
    synchronization, so a whole control step can be captured in a CUDA graph. Invalid worlds are
    detected at the first bad substep and restored to their last valid state, which keeps the
    remaining kernels safe; ``step`` reports them as invalid and ``check`` (host side, after the
    step) raises on fatal faults.
    """

    def __init__(self, robot, registry, folder, objects: list[int], num_envs: int, cfg):
        """``objects``: registry indices of the objects the worlds may hold."""
        import warp as wp
        from mjlab.sim import Simulation

        if not torch.cuda.is_available():
            raise RuntimeError("The MuJoCo-Warp backend needs CUDA; use backend='cpu' with device='cpu' for tests")
        wp.config.quiet = True
        self.cfg, self.device, self.N = cfg, cfg.device, num_envs
        scene = make_scene(robot, registry, folder, objects[0], self.N, cfg.device)
        self.sim = Simulation(num_envs=self.N, cfg=simulation_cfg(cfg.nconmax, cfg.njmax), spec=scene.spec,
                              variant_info=scene.collect_variant_info(), device=cfg.device)  # fmt: skip
        if cfg.nccdmax is not None:
            self.sim.wp_data.naccdmax = cfg.nccdmax * self.N  # convex-collision workspace only
        self.sim.expand_model_fields((*OBJECT_GEOM_FIELDS, *OBJECT_BODY_FIELDS, "geom_friction", "body_ipos",
                                      "actuator_gainprm", "actuator_biasprm"))
        model = self.sim.mj_model
        self.idx = Indexing(model, robot).to(cfg.device)
        # The rows that make a world's box what it is, read from a host model compiled with each box.
        body = model.body("object/object").id
        geoms = np.flatnonzero(model.geom_bodyid == body)
        self._object_geoms = torch.tensor(geoms, device=cfg.device)
        hosts = [make_scene(robot, registry, folder, k, 1, "cpu").compile() for k in objects]
        if any(h.ngeom != model.ngeom or h.nbody != model.nbody for h in hosts):
            raise ValueError("Objects must compile to the same model structure")
        self._object_rows = {}
        for name in (*OBJECT_GEOM_FIELDS, *OBJECT_BODY_FIELDS):
            # The world body's subtree mass includes the object.
            columns = geoms if name.startswith("geom_") else np.array([0, body] if name == "body_subtreemass" else [body])
            rows = wp.to_torch(getattr(self.sim.wp_model, name))
            table = torch.tensor(np.stack([np.asarray(getattr(h, name))[columns] for h in hosts]), dtype=rows.dtype, device=cfg.device)
            self._object_rows[name] = (rows, torch.tensor(columns, device=cfg.device), table.reshape(len(hosts), len(columns), *rows.shape[2:]))
        self._friction = wp.to_torch(self.sim.wp_model.geom_friction)
        self._mass = torch.tensor([float(h.body_mass[body]) for h in hosts], device=cfg.device)
        self._position = torch.full((len(registry["objects"]),), -1, device=cfg.device)  # registry index -> row of the tables
        self._position[torch.tensor(objects, device=cfg.device)] = torch.arange(len(objects), device=cfg.device)
        # Position actuators: force = kp * ctrl - kp * q - kd * qd, so kp sits in gainprm[0] and -kp, -kd in biasprm[1:3].
        self._gainprm, self._biasprm = wp.to_torch(self.sim.wp_model.actuator_gainprm), wp.to_torch(self.sim.wp_model.actuator_biasprm)
        self._kp, self._kd = self._gainprm[0, self.idx.ctrl, 0].clone(), -self._biasprm[0, self.idx.ctrl, 2].clone()
        self.gain_scale = torch.ones(self.N, len(robot.joint_names), 2, device=cfg.device)
        self.held = torch.zeros(self.N, dtype=torch.long, device=cfg.device)  # table row of the box every world holds
        self.mass_scale = torch.ones(self.N, device=cfg.device)
        self.friction = torch.full((self.N,), float(model.geom_friction[geoms[0], 0]), device=cfg.device)

        data = self.sim.wp_data
        self.tensors = {name: wp.to_torch(getattr(data, name)) for name in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time")}
        self._mocap_pos = wp.to_torch(data.mocap_pos)
        self._mocap = torch.tensor(table_mocap_ids(model), device=cfg.device)
        self._counts = [wp.to_torch(x) for x in (data.ncollision, data.nacon, data.nefc)]
        self._overflow = wp.to_torch(data.overflow)
        self._row_adr, self._row_nnz = wp.to_torch(data.efc.J_rowadr), wp.to_torch(data.efc.J_rownnz)
        groups = ContactGroups(model, robot)
        self._groups = [wp.array(x, dtype=wp.int32, device=cfg.device)
                        for x in (groups.object_geom, groups.channel, groups.support, groups.foot)]
        self._force_step = torch.zeros(self.N, NUM_GROUPS, device=cfg.device)
        self._force_wp = wp.from_torch(self._force_step)
        self.contact_force = torch.zeros(self.N, 4, device=cfg.device)
        self.foot_force = torch.zeros(self.N, 2, device=cfg.device)
        self.capacity_peak = torch.zeros(3, device=cfg.device)  # collision candidates, contacts, constraints/world
        # Guard state: worlds rejected in the current step, their fatal bits, the speeds at the
        # first rejection, and the last valid physics state of every world.
        self.bad = torch.zeros(self.N, dtype=torch.bool, device=cfg.device)
        self.fatal = torch.zeros(self.N, dtype=torch.long, device=cfg.device)
        self.bad_speeds = torch.zeros(self.N, 3, device=cfg.device)
        self._safe = {name: value.clone() for name, value in self.tensors.items()}

    def state(self):
        return tuple(x.clone() for x in decode(self.tensors["qpos"], self.tensors["qvel"], self.idx))

    @property
    def ok(self):
        """0-dim bool: no fatal fault in the current step."""
        return ~self.fatal.any()

    def _reject(self, invalid, fatal_bit: int = 0):
        self.fatal.bitwise_or_(invalid.long() * fatal_bit)
        self.bad.logical_or_(invalid)

    def _restore_rejected(self):
        for name, value in self.tensors.items():
            mask = self.bad.reshape(-1, *([1] * (value.ndim - 1)))
            value.copy_(torch.where(mask, self._safe[name], value))

    def _check_capacity(self):
        ncollision, nacon, nefc = self._counts
        data = self.sim.wp_data
        self.capacity_peak.copy_(torch.maximum(self.capacity_peak, torch.stack((ncollision[0], nacon[0], nefc.max()))))
        # forward() does not raise the step-time overflow flags, so compare the counts as well.
        valid = (nefc <= data.njmax).all() & (ncollision <= data.naconmax).all() & (nacon <= data.naconmax).all()
        valid = valid & (self._overflow == 0).all()
        if self.sim.wp_model.is_sparse:
            last = (nefc.clamp(1, data.njmax) - 1).long().unsqueeze(-1)
            nnz = (self._row_adr.gather(1, last) + self._row_nnz.gather(1, last)).squeeze(-1)
            valid = valid & ((nefc == 0) | (nnz <= data.njmax_nnz)).all()
        self._reject(~valid.expand(self.N), CAPACITY)

    def _write_objects(self):
        """Write every world's box rows for the box it holds, its mass scale and its friction."""
        scale = self.mass_scale
        for name, (rows, columns, table) in self._object_rows.items():
            value = table[self.held]
            shape = (-1, *([1] * (value.ndim - 1)))
            if name in ("body_mass", "body_inertia"):
                value = value * scale.reshape(shape)
            elif name == "body_invweight0":
                value = value / scale.reshape(shape)
            elif name == "body_subtreemass":
                extra = (scale - 1) * self._mass[self.held]
                value = value + extra[:, None].expand_as(value)
            rows[:, columns] = value
        self._friction[:, self._object_geoms, 0] = self.friction[:, None]

    def reset(self, mask, state, objects, tables, mass_scale, friction, gains=None):
        """Set the worlds where ``mask`` is true to ``state`` and give them the boxes ``objects`` (N,)
        (registry indices) with mass scale ``mass_scale`` (N,) and sliding friction ``friction`` (N,),
        the platforms at ``tables`` (N, 6) and the stiffness and damping scales ``gains`` (N, J, 2);
        the others keep theirs."""
        import mujoco_warp as mjwarp
        import warp as wp

        sim = self.sim
        self.held.copy_(torch.where(mask, self._position[objects], self.held))
        self.mass_scale.copy_(torch.where(mask, mass_scale, self.mass_scale))
        self.friction.copy_(torch.where(mask, friction, self.friction))
        self._write_objects()
        if gains is not None:
            self.gain_scale.copy_(torch.where(mask[:, None, None], gains, self.gain_scale))
            kp, kd = self._kp * self.gain_scale[..., 0], self._kd * self.gain_scale[..., 1]
            self._gainprm[:, self.idx.ctrl, 0] = kp
            self._biasprm[:, self.idx.ctrl, 1] = -kp
            self._biasprm[:, self.idx.ctrl, 2] = -kd
        wp.to_torch(sim._reset_mask_wp).copy_(mask)
        with wp.ScopedDevice(sim.wp_device):
            if sim.use_cuda_graph and sim.reset_graph is not None:
                wp.capture_launch(sim.reset_graph)
            else:
                mjwarp.reset_data(sim.wp_model, sim.wp_data, reset=sim._reset_mask_wp)
        qpos, qvel, ctrl = (self.tensors[k] for k in ("qpos", "qvel", "ctrl"))
        new_qpos, new_qvel, new_ctrl = qpos.clone(), qvel.clone(), ctrl.clone()
        encode(state, new_qpos, new_qvel, self.idx)
        new_ctrl[:, self.idx.ctrl] = state[4]
        for value, new in ((qpos, new_qpos), (qvel, new_qvel), (ctrl, new_ctrl)):
            value.copy_(torch.where(mask[:, None], new, value))
        self._mocap_pos[:, self._mocap] = torch.where(mask[:, None, None], tables.unflatten(-1, (2, 3)), self._mocap_pos[:, self._mocap])
        self.contact_force.masked_fill_(mask[:, None], 0)
        self.foot_force.masked_fill_(mask[:, None], 0)
        sim.forward()
        self._check_capacity()

    def step(self, target):
        """Advance one control step. Returns (state, object contact force (N, 4), foot support force
        (N, 2), valid); forces are means over the physics steps."""
        import warp as wp

        from .contacts import kernel_inputs, reduce_contacts

        # Inside an external capture Warp must keep using the registered stream object.
        if torch.cuda.is_current_stream_capturing():
            stream = wp.get_stream(self.device)
        else:
            stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        self.bad.zero_()
        self.fatal.zero_()
        admitted = numerical_status(self.tensors["qpos"], self.tensors["qvel"], self.idx, self.cfg)[0]
        for value in self.tensors.values():
            admitted = admitted & torch.isfinite(value.reshape(self.N, -1)).all(-1)
        for name, value in self.tensors.items():
            self._safe[name].copy_(torch.where(admitted.reshape(-1, *([1] * (value.ndim - 1))), value, self._safe[name]))
        finite_target = torch.isfinite(target).all(-1)
        self._reject(~finite_target, NONFINITE_TARGET)
        ctrl = self.tensors["ctrl"]
        ctrl[:, self.idx.ctrl] = torch.where(finite_target[:, None], target, ctrl[:, self.idx.ctrl])
        self.contact_force.zero_()
        self.foot_force.zero_()
        for _ in range(DECIMATION):
            self.sim.step()
            valid, finite, speeds = numerical_status(self.tensors["qpos"], self.tensors["qvel"], self.idx, self.cfg)
            for value in self.tensors.values():
                finite = finite & torch.isfinite(value.reshape(self.N, -1)).all(-1)
            self.bad_speeds.copy_(torch.where((~valid & ~self.bad)[:, None], speeds, self.bad_speeds))
            self._reject(~finite, NONFINITE_STATE)
            self._reject(~valid)
            self._check_capacity()
            self._restore_rejected()
            self._force_step.zero_()
            wp.launch(reduce_contacts, dim=self.sim.wp_data.naconmax, device=self.device, stream=stream,
                      inputs=[*kernel_inputs(self.sim.wp_model, self.sim.wp_data), *self._groups], outputs=[self._force_wp])  # fmt: skip
            self.contact_force.add_(self._force_step[:, OBJECT], alpha=1 / DECIMATION)
            self.foot_force.add_(self._force_step[:, FEET], alpha=1 / DECIMATION)
        return self.state(), self.contact_force.clone(), self.foot_force.clone(), ~self.bad

    def check(self) -> int:
        """Host-side check after a step: the number of rejected worlds. Raises on a fatal fault."""
        if not bool(self.bad.any()):
            return 0
        if bool(self.fatal.any()):
            worlds = self.fatal.nonzero().flatten()[:8].tolist()
            detail = "; ".join(f"world {w}: " + ", ".join(label for bit, label in FAULTS.items() if int(self.fatal[w]) & bit)
                               for w in worlds)  # fmt: skip
            raise RuntimeError(f"Fatal physics fault in {int((self.fatal != 0).sum())} worlds ({detail})")
        return int(self.bad.sum())
