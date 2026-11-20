"""Unitree G1 (29 DoF) with rubber hands: joint layout, drives and task bodies.

The robot is OmniContact's: no fingers, every joint is commanded. Joint order, limits and
kinematics come from the compiled MJCF, so kinematics, control, physics and the MDP only use the
names listed here.

Contact with the object is counted on four channels, in OmniContact's order: left foot, right
foot, left hand, right hand. A hand is the wrist-yaw link, which carries the rubber hand.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

# Body drives: implicit PD at 10 Hz natural frequency, damping ratio 2, on the reflected rotor
# inertia (BeyondMimic / OmniContact). Reflected inertias [kg m^2] of the Unitree actuators.
NATURAL_FREQ = 10 * 2 * math.pi
DAMPING_RATIO = 2.0
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# Domain randomization and observation noise half-widths. The ground has (almost) no friction of
# its own and MuJoCo takes the larger friction of a pair, so the robot's friction is the friction
# of its feet on the ground.
FRICTION_RANGE = (0.6, 1.2)
COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
JOINT_OFFSET = 0.01
NOISE_JOINT_POS, NOISE_JOINT_VEL, NOISE_ANG_VEL, NOISE_GRAVITY, NOISE_REF_ORI = 0.01, 0.5, 0.2, 0.05, 0.05
# Localization of the robot [m], and the box pose as mocap or vision would give it: position [m] and
# a rotation about each axis [rad]. Every box feature is computed from the same noisy pose.
NOISE_ROOT_POS, NOISE_OBJECT_POS, NOISE_OBJECT_ORI = 0.05, 0.05, 0.1
# PD gains: stiffness and damping of every joint scaled log-uniformly in this range, once per
# episode (OmniContact). Orientation estimate: a fixed roll, pitch, yaw error [rad] of the robot's
# root orientation per episode, as OmniContact's torso orientation bias (5, 5, 8 degrees). It
# affects what the robot knows of its pose relative to the reference and the box, not the IMU's
# gravity and angular velocity.
GAIN_RANGE = (0.75, 1.5)
ORIENTATION_BIAS = (0.0873, 0.0873, 0.1396)

CONTACT_CHANNELS = ("left_foot", "right_foot", "left_hand", "right_hand")

# Motors: (effort limit [Nm], reflected inertia [kg m^2]).
MOTORS = {
    "hip_pitch": (88.0, ARMATURE_7520_14),
    "hip_roll": (139.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, ARMATURE_7520_14),
    "knee": (139.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 2 * ARMATURE_5020),
    "ankle_roll": (50.0, 2 * ARMATURE_5020),
    "waist_yaw": (88.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 2 * ARMATURE_5020),
    "waist_pitch": (50.0, 2 * ARMATURE_5020),
    "shoulder_pitch": (25.0, ARMATURE_5020),
    "shoulder_roll": (25.0, ARMATURE_5020),
    "shoulder_yaw": (25.0, ARMATURE_5020),
    "elbow": (25.0, ARMATURE_5020),
    "wrist_roll": (25.0, ARMATURE_5020),
    "wrist_pitch": (5.0, ARMATURE_4010),
    "wrist_yaw": (5.0, ARMATURE_4010),
}


# Drive sets. "sonic": the hip pitch joints use the larger actuator, as the SONIC tracker was trained.
DRIVES = {"omnicontact": {}, "sonic": {"hip_pitch": (139.0, ARMATURE_7520_22)}}


@dataclass(frozen=True)
class Drive:
    """PD position drive of one joint."""

    stiffness: float
    damping: float
    effort: float
    armature: float


def motor_drive(effort: float, armature: float) -> Drive:
    return Drive(armature * NATURAL_FREQ**2, 2 * DAMPING_RATIO * armature * NATURAL_FREQ, effort, armature)


def side_joint(name: str) -> str:
    """``left_knee_joint`` -> ``knee``: the key of the motor table."""
    return re.sub(r"^(left|right)_", "", name).removesuffix("_joint")


class Robot:
    name = "g1_rubber"
    mjcf = "g1_rubber.xml"
    # OmniContact's tracked bodies; the first one is the root. A palm link is the rubber-hand frame.
    tracked_bodies = (
        "pelvis",
        "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_palm_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_palm_link",
    )  # fmt: skip
    anchor_body = "torso_link"
    hand_bodies = ("left_palm_link", "right_palm_link")
    foot_bodies = ("left_ankle_roll_link", "right_ankle_roll_link")
    # The contact flow: the bodies whose future poses are the policy's goal.
    flow_bodies = ("left_palm_link", "right_palm_link", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
    # Links whose contacts with the object are counted, in the order of CONTACT_CHANNELS.
    contact_links = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    # Points whose distance to the object's surface the models are told: (tracked body, offset in its frame).
    surface_points = (
        ("left_palm_link", (0.05, 0.0, 0.0)), ("left_palm_link", (0.11, 0.0, 0.0)),
        ("right_palm_link", (0.05, 0.0, 0.0)), ("right_palm_link", (0.11, 0.0, 0.0)),
        ("left_ankle_roll_link", (0.12, 0.0, -0.02)), ("left_ankle_roll_link", (-0.05, 0.0, -0.02)),
        ("right_ankle_roll_link", (0.12, 0.0, -0.02)), ("right_ankle_roll_link", (-0.05, 0.0, -0.02)),
    )  # fmt: skip
    torque_joints = ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")  # joints whose torque is penalized
    com_body = "torso_link"
    root_height = 0.76
    default_pose = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }

    def __init__(self, asset_dir: str | Path, drives: str = "omnicontact"):
        if drives not in DRIVES:
            raise ValueError(f"drives must be one of {tuple(DRIVES)}")
        self.asset_dir, self.drives = Path(asset_dir), drives
        self.motors = {**MOTORS, **DRIVES[drives]}

    def drive(self, joint: str) -> Drive:
        return motor_drive(*self.motors[side_joint(joint)])

    @cached_property
    def arm_idx(self) -> list[int]:
        """Joints of the arms: shoulders, elbows and wrists."""
        return [i for i, n in enumerate(self.joint_names) if side_joint(n).split("_")[0] in ("shoulder", "elbow", "wrist")]

    def contact_channel(self, body: str) -> int:
        """Channel of CONTACT_CHANNELS that a link's contacts with the object count for, or -1."""
        return self.contact_links.index(body) if body in self.contact_links else -1

    @property
    def mjcf_path(self) -> Path:
        return self.asset_dir / self.mjcf

    @cached_property
    def model(self) -> mujoco.MjModel:
        if not self.mjcf_path.is_file():
            raise FileNotFoundError(f"Missing {self.mjcf_path}; run `python -m supermimic_box.prepare`")
        return mujoco.MjModel.from_xml_path(str(self.mjcf_path))

    @cached_property
    def joint_names(self) -> tuple[str, ...]:
        m = self.model
        return tuple(m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actions(self) -> int:
        return self.num_joints

    @cached_property
    def joint_limits(self) -> np.ndarray:
        """(J, 2) position limits in joint order."""
        m = self.model
        return np.array([m.jnt_range[m.joint(n).id] for n in self.joint_names], dtype=np.float32)

    @cached_property
    def default_joint_pos(self) -> np.ndarray:
        q = np.zeros(self.num_joints, dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            for pattern, value in self.default_pose.items():
                if re.fullmatch(pattern, name):
                    q[i] = value
        return q

    @cached_property
    def control_scale(self) -> np.ndarray:
        """Radians per unit of action or normalized PD target: 0.25 effort / kp."""
        return np.asarray([0.25 * self.drive(n).effort / self.drive(n).stiffness for n in self.joint_names], dtype=np.float32)

    @cached_property
    def gains(self) -> np.ndarray:
        """(J, 3) stiffness, damping and effort limit in joint order."""
        return np.asarray([[self.drive(n).stiffness, self.drive(n).damping, self.drive(n).effort] for n in self.joint_names],
                          dtype=np.float32)  # fmt: skip

    @cached_property
    def torque_idx(self) -> list[int]:
        return [i for i, n in enumerate(self.joint_names) if side_joint(n) in self.torque_joints]

    @property
    def fk_bodies(self) -> tuple[str, ...]:
        return self.tracked_bodies

    def entity_cfg(self):
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

        actuators = []
        for name in self.joint_names:
            d = self.drive(name)
            actuators.append(BuiltinPositionActuatorCfg(
                target_names_expr=(name,), stiffness=d.stiffness, damping=d.damping, effort_limit=d.effort,
                armature=d.armature))  # fmt: skip
        return EntityCfg(
            init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, self.root_height), joint_pos=self.default_pose, joint_vel={".*": 0.0}),
            spec_fn=lambda: mujoco.MjSpec.from_file(str(self.mjcf_path)),
            articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9),
        )


def load_robot(asset_dir: str | Path, drives: str = "omnicontact") -> Robot:
    return Robot(asset_dir, drives)
