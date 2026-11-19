"""Analytic observation, reward and termination heads.

Everything here is tensor arithmetic on a state tuple and a reference frame, so the same code
labels real simulator transitions and differentiates through imagined ones. Contacts enter only
as numbers supplied by the caller: whether each contact channel (left foot, right foot, left hand,
right hand) is on the object and whether each foot is on the ground, measured in simulation and
predicted by the contact head in imagination. Distances to the box come from its analytic shape.

The reward terms and their defaults are OmniContact's CFtrack rewards.
"""

from __future__ import annotations

import torch

from . import robot as R
from .motion import CONTROL_DT
from .rotations import angle_squared, mat_to_6d, mm3, mv3, yaw_mat

FAILURE_REASONS = ("anchor_position", "anchor_tilt", "object_position", "object_rotation", "end_height")
SURFACE_SCALE = 0.02  # distance [m] on which the surface features resolve the contact band
TABLE_FEATURES = 12
SIGHT, REACH = 4.0, 1.5  # lengths [m] beyond which a relative position is not resolved


def feet_step(feet, contact, threshold: float):
    """Air-time bookkeeping of one control step.

    ``feet`` (..., 4): whether the left and right foot were on the ground before the step, then the
    seconds each has been in the air. ``contact`` (..., 2): whether they are on the ground after it;
    a probability works as well. Returns (shortfall (..., 2), next feet): the time by which a swing
    that ended in this step fell short of ``threshold`` seconds.
    """
    before, air = feet[..., :2], feet[..., 2:]
    shortfall = torch.relu(threshold - air) * contact * (1 - before)
    return shortfall, torch.cat((contact, (air + CONTROL_DT) * (1 - contact)), -1)


def chain_product(value, dim):
    """Product over a short dimension as a chain of multiplications (torch.prod's backward reads a
    zero count on the host, which a CUDA graph cannot capture)."""
    factors = value.unbind(dim)
    result = factors[0]
    for factor in factors[1:]:
        result = result * factor
    return result


def limit_norm(v, length: float):
    """``v`` (..., 3) shortened to at most ``length``."""
    norm = v.square().sum(-1, keepdim=True).add(1e-12).sqrt()
    return v * (length / norm).clamp(max=1.0)


def angle(a, b):
    """Rotation angle between two rotation matrices, differentiable at zero."""
    return angle_squared(a, b).add(1e-8).sqrt()


class HOIMDP:
    def __init__(self, motion, controller, layout, cfg, noise: bool, geometry, table_half):
        self.motion, self.controller, self.layout, self.cfg, self.noisy = motion, controller, layout, cfg, noise
        self.geometry, self.kin, self.robot = geometry, motion.kin, motion.kin.robot
        device = motion.device
        J, A = controller.limits.shape[0], controller.num_actions
        index = lambda x: torch.tensor(x, device=device)  # noqa: E731
        self.anchor, self.hands, self.feet = motion.anchor, index(motion.hands), index(motion.feet)
        self.surface_body = index([self.robot.tracked_bodies.index(body) for body, _ in self.robot.surface_points])
        self.surface_offset = torch.tensor([offset for _, offset in self.robot.surface_points], device=device)
        self.surface_dim = 5 * len(self.robot.surface_points)
        self.table_top = torch.tensor([0.0, 0.0, table_half[2]], device=device)
        margin = 0.05 * (self.kin.q_hi - self.kin.q_lo)
        self.soft_lo, self.soft_hi = self.kin.q_lo + margin, self.kin.q_hi - margin
        # Uniform noise half-widths on the head of the observation.
        self.noise = torch.cat((
            torch.full((J,), R.NOISE_JOINT_POS), torch.full((J,), R.NOISE_JOINT_VEL), torch.full((3,), R.NOISE_ANG_VEL),
            torch.full((3,), R.NOISE_GRAVITY), torch.zeros(A), torch.full((6,), R.NOISE_REF_ORI),
            torch.full((3,), R.NOISE_OBJECT_POS), torch.full((6,), R.NOISE_OBJECT_ORI))).to(device)  # fmt: skip
        self.ends = torch.cat((self.feet, self.hands))
        self.thresholds = torch.tensor([cfg.term_anchor_pos, cfg.term_tilt, cfg.term_object_pos, cfg.term_object_rot,
                                        cfg.term_end_height], device=device)  # fmt: skip
        self.temperatures = torch.tensor([0.02, 0.04, 0.02, 0.08, 0.02], device=device)
        self.max_step_reward = CONTROL_DT * sum(getattr(cfg, name) for name in cfg.BONUSES)

    def fk(self, s):
        return self.kin.forward(*s[:6])

    # ------------------------------------------------------------------ features
    def surface_features(self, s, f, fk=None):
        """Per surface point of the hands and feet: distance to the box, the same on a 2 cm scale that
        resolves the contact band, and the direction away from the box in the root's heading frame."""
        bp, bR = (self.fk(s) if fk is None else fk)[:2]
        points = bp[..., self.surface_body, :] + mv3(bR[..., self.surface_body, :, :], self.surface_offset)
        local = mv3(s[7].transpose(-1, -2).unsqueeze(-3), points - s[6].unsqueeze(-2))
        field = self.geometry.surface(self.motion.object_of(f), local)
        distance = field[..., :1]
        heading_from_object = mm3(yaw_mat(s[1]).transpose(-1, -2), s[7])
        return torch.cat((distance.clamp(-0.05, 0.3), torch.tanh(distance / SURFACE_SCALE),
                          mv3(heading_from_object.unsqueeze(-3), field[..., 1:])), -1).flatten(-2)  # fmt: skip

    def box_corners(self, s, f):
        """(..., 24) corners of the box relative to the root, in the root's heading frame."""
        corners = s[6].unsqueeze(-2) + mv3(s[7].unsqueeze(-3), self.geometry.corners(self.motion.object_of(f)))
        Ht = yaw_mat(s[1]).transpose(-1, -2).unsqueeze(-3)
        return limit_norm(mv3(Ht, corners - s[0].unsqueeze(-2)), SIGHT).flatten(-2)

    def table_features(self, s, context):
        """(..., 12) per platform, in the root's heading frame: its centre relative to the root and the
        middle of its top relative to the object."""
        tables = self.layout.tables(context)
        Ht = yaw_mat(s[1]).transpose(-1, -2).unsqueeze(-3)
        from_root = limit_norm(mv3(Ht, tables - s[0].unsqueeze(-2)), SIGHT)
        from_object = limit_norm(mv3(Ht, tables + self.table_top - s[6].unsqueeze(-2)), REACH)
        return torch.cat((from_root.flatten(-2), from_object.flatten(-2)), -1)

    def model_features(self, s, f, context, fk=None):
        """What the world model and the contact head know besides the state and the PD targets: the box's
        size, the physical parameters, the platforms and the hand/foot distances to the box."""
        return torch.cat((self.motion.key(f), self.layout.model(context), self.table_features(s, context),
                          self.surface_features(s, f, fk)), -1)  # fmt: skip

    @property
    def model_feature_dim(self) -> int:
        return 3 + self.layout.model_dim + TABLE_FEATURES + self.surface_dim

    # ------------------------------------------------------------------ observations
    def add_noise(self, obs):
        if not self.noisy:
            return obs
        n = self.noise.numel()
        return torch.cat((obs[..., :n] + (2 * torch.rand_like(obs[..., :n]) - 1) * self.noise, obs[..., n:]), -1)

    def obs(self, s, f, previous_action, context, fk=None):
        """Noise-free actor observation of state ``s`` at reference frame ``f``."""
        p, Rm, v, w, q, qd, op, oR, ov, ow = s
        m = self.motion
        fk = self.fk(s) if fk is None else fk
        Ht, Rt = yaw_mat(Rm).transpose(-1, -2), Rm.transpose(-1, -2)
        return torch.cat((
            q - self.layout.offsets(context) - self.controller.default, qd, mv3(Rt, w), -Rm[..., 2, :],
            previous_action, mat_to_6d(mm3(Rt, m.root_R[f])),
            limit_norm(mv3(Ht, op - p), SIGHT), mat_to_6d(mm3(Ht, oR)), mv3(Ht, ov - v), mv3(Ht, ow) * 0.25,
            self.box_corners(s, f), m.key(f), m.contact[f],
            # Global tracking error: observable even when robot and object drift together.
            mv3(Ht, m.root_p[f] - p), self.surface_features(s, f, fk)), -1)  # fmt: skip

    def initial_feet(self, like):
        """Feet of a fresh episode, shaped like ``like`` (..., 4): on the ground, and with enough air
        time behind them that a swing the reset interrupts counts as complete."""
        on_ground = torch.ones_like(like[..., :2])
        return torch.cat((on_ground, on_ground * self.cfg.feet_air_time_threshold), -1)

    def critic_input(self, s, f, previous_action, context, goal, feet, fk=None):
        m = self.motion
        fk = self.fk(s) if fk is None else fk
        Ht = yaw_mat(s[1]).transpose(-1, -2)
        body_error = mv3(Ht.unsqueeze(-3), m.body_p[f] - fk[0]).flatten(-2)
        feet = torch.cat((feet[..., :2], feet[..., 2:].clamp(max=2 * self.cfg.feet_air_time_threshold)), -1)
        object_error = torch.cat((mv3(Ht, m.object_p[f] - s[6]), mat_to_6d(mm3(s[7].transpose(-1, -2), m.object_R[f]))), -1)
        return torch.cat((self.obs(s, f, previous_action, context, fk), mv3(Ht, s[2]), m.root_p[f] - s[0], body_error,
                          object_error, self.layout.model(context), self.table_features(s, context), goal, feet), -1)  # fmt: skip

    # ------------------------------------------------------------------ reward
    def reward(self, s, f, action, previous_action, contact, foot_contact, swing_shortfall, episode_start, context, fk=None):
        """Reward of arriving in ``s`` at reference frame ``f`` (the post-transition frame).

        ``contact`` (..., 4): measured or predicted contact of each channel with the object,
        ``foot_contact`` (..., 2): whether each foot is on the ground in ``s``, ``swing_shortfall``
        (..., 2) from ``feet_step``. ``episode_start`` marks the first transition after a reset,
        which has no previous action to compare with.
        """
        m, c = self.motion, self.cfg
        p, Rm, v, w, q, qd, op, oR, ov, ow = s
        bp, bR, bv, bw = self.fk(s) if fk is None else fk
        rp, rR = m.body_p[f], m.body_R[f]
        A, H = self.anchor, self.hands

        # Bodies relative to the anchor: the reference is moved to the robot's anchor in the ground
        # plane and turned by the anchor's yaw error, so only the pose around the torso is scored.
        turn = yaw_mat(mm3(bR[..., A, :, :], rR[..., A, :, :].transpose(-1, -2)))
        base = torch.cat((bp[..., A, :2], rp[..., A, 2:]), -1)
        relative_p = base.unsqueeze(-2) + mv3(turn.unsqueeze(-3), rp - rp[..., A : A + 1, :])
        relative_R = mm3(turn.unsqueeze(-3), rR)
        e_body_pos = (bp - relative_p).square().sum(-1).mean(-1)
        e_body_ori = angle_squared(bR, relative_R).mean(-1)

        # Hands relative to the object, compared in the heading frame of each anchor. The tolerance
        # widens with the reference distance: far from the object the hands are free.
        hand_from_object = mv3(turn.transpose(-1, -2).unsqueeze(-3), bp[..., H, :] - op.unsqueeze(-2))
        ref_from_object = rp[..., H, :] - m.object_p[f].unsqueeze(-2)
        reach = ref_from_object.square().sum(-1).add(1e-12).sqrt()
        tolerance = c.interact_std * torch.exp(2 * reach)
        e_interact_pos = (hand_from_object - ref_from_object).square().sum(-1)
        e_interact_ori = (angle(oR.unsqueeze(-3), bR[..., H, :, :]) - angle(m.object_R[f].unsqueeze(-3), rR[..., H, :, :])).square()

        qref = m.q[f]
        previous_frame = torch.maximum(f - 1, m.clip_start[m.clip_id[f]])
        command = self.controller.command(action, qref)
        previous_command = self.controller.command(previous_action, m.q[previous_frame])
        rate = torch.where(episode_start, 0.0, (command - previous_command).square().sum(-1))
        target = self.controller(action, qref, self.layout.offsets(context))
        torque = self.controller.torque(target, q, qd)[..., self.controller.torque_joints]
        limits = torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)
        foot_speed = bv[..., self.feet, :2].square().sum(-1).add(1e-8).sqrt()
        terms = {
            "anchor_pos": c.anchor_pos * torch.exp(-(bp[..., A, :] - rp[..., A, :]).square().sum(-1) / c.body_pos_std**2),
            "anchor_ori": c.anchor_ori * torch.exp(-angle_squared(bR[..., A, :, :], rR[..., A, :, :]) / c.body_ori_std**2),
            "body_pos": c.body_pos * torch.exp(-e_body_pos / c.body_pos_std**2),
            "body_ori": c.body_ori * torch.exp(-e_body_ori / c.body_ori_std**2),
            "hand_pos": c.hand_pos * torch.exp(-(bp[..., H, :] - rp[..., H, :]).square().sum(-1).mean(-1) / c.body_pos_std**2),
            "hand_ori": c.hand_ori * torch.exp(-angle_squared(bR[..., H, :, :], rR[..., H, :, :]).mean(-1) / c.body_ori_std**2),
            "object_pos": c.object_pos * torch.exp(-(op - m.object_p[f]).square().sum(-1) / c.object_pos_std**2),
            "object_ori": c.object_ori * torch.exp(-angle_squared(oR, m.object_R[f]) / c.object_ori_std**2),
            "interact_pos": c.interact_pos * torch.exp(-e_interact_pos / tolerance**2).mean(-1),
            "interact_ori": c.interact_ori * torch.exp(-e_interact_ori / tolerance**2).mean(-1),
            "contact": c.contact * torch.exp(-(contact - m.contact[f]).square().mean(-1) / c.contact_std**2),
            "body_velocity": c.body_velocity * torch.exp(-(bv - m.body_v[f]).square().sum(-1).mean(-1)),
            "body_angular_velocity": c.body_angular_velocity * torch.exp(
                -(bw - m.body_w[f]).square().sum(-1).mean(-1) / c.body_angvel_std**2),
            "object_velocity": c.object_velocity * torch.exp(
                -(ov - m.object_v[f]).square().sum(-1) - (ow - m.object_w[f]).square().sum(-1) / 3.14**2),
            "action_rate": -c.action_rate * rate,
            "torque": -c.torque * torque.square().sum(-1),
            "joint_limit": -c.joint_limit * limits.sum(-1),
            "feet_slide": -c.feet_slide * (foot_speed * foot_contact).sum(-1),
            "feet_air_time": -c.feet_air_time * swing_shortfall.sum(-1),
        }  # fmt: skip
        terms = {k: v * CONTROL_DT for k, v in terms.items()}
        return sum(terms.values()), terms

    # ------------------------------------------------------------------ termination
    def errors(self, s, f, fk=None):
        """(..., 5) distances from the reference that end an episode: anchor position and tilt, object
        position and rotation, and the largest height error of a foot or hand."""
        m, A = self.motion, self.anchor
        bp, bR = (self.fk(s) if fk is None else fk)[:2]
        rp, rR = m.body_p[f], m.body_R[f]
        height = (bp[..., self.ends, 2] - rp[..., self.ends, 2]).abs().amax(-1)
        return torch.stack((
            (bp[..., A, :] - rp[..., A, :]).square().sum(-1).add(1e-12).sqrt(), (bR[..., A, 2, 2] - rR[..., A, 2, 2]).abs(),
            (s[6] - m.object_p[f]).square().sum(-1).add(1e-12).sqrt(), angle(s[7], m.object_R[f]), height), -1)  # fmt: skip

    def hard_termination(self, s, f, fk=None):
        """(failed, reasons) in simulation; reasons follow FAILURE_REASONS."""
        flags = self.errors(s, f, fk) > self.thresholds
        return flags.any(-1), flags

    def soft_termination(self, s, f, fk=None):
        """Differentiable surrogate of ``hard_termination`` for imagination."""
        return 1 - chain_product(1 - torch.sigmoid((self.errors(s, f, fk) - self.thresholds) / self.temperatures), -1)
