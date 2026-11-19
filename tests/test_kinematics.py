import mujoco
import numpy as np
import torch

from supermimic_box.kinematics import Kinematics
from supermimic_box.rotations import mat_to_quat, so3_exp


def test_robot_layout(robot):
    assert robot.num_joints == robot.num_actions == 29 and len(robot.tracked_bodies) == 14
    assert set(robot.hand_bodies + robot.foot_bodies + robot.flow_bodies + (robot.anchor_body,)) <= set(robot.tracked_bodies)
    assert {body for body, _ in robot.surface_points} <= set(robot.tracked_bodies)
    # Contacts with the object are counted on both feet and both hands, in that order.
    channels = [robot.contact_channel(robot.model.body(b).name) for b in range(robot.model.nbody)]
    assert sorted(c for c in channels if c >= 0) == [0, 1, 2, 3]
    assert [robot.contact_links[c].split("_")[0] for c in range(4)] == ["left", "right", "left", "right"]
    assert len(robot.torque_idx) == 8 and (robot.control_scale > 0).all()
    # No hinge joint has a reference offset, which the torch FK relies on.
    hinge = robot.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    assert not robot.model.qpos0[robot.model.jnt_qposadr[hinge]].any()


def test_fk_matches_mujoco_positions_rotations_and_velocities(robot):
    torch.manual_seed(0)
    model, data = robot.model, mujoco.MjData(robot.model)
    kin = Kinematics(robot)
    lo, hi = kin.q_lo.double(), kin.q_hi.double()
    for _ in range(5):
        p, rot = torch.randn(3, dtype=torch.float64), so3_exp(torch.randn(3, dtype=torch.float64))
        v, w = torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
        q, qd = lo + torch.rand(len(lo), dtype=torch.float64) * (hi - lo), torch.randn(len(lo), dtype=torch.float64)
        data.qpos[:] = torch.cat((p, mat_to_quat(rot), q)).numpy()
        data.qvel[:] = torch.cat((v, rot.T @ w, qd)).numpy()
        mujoco.mj_forward(model, data)
        kin64 = Kinematics(robot)
        for name in ("axes", "anchors", "off_p", "off_R"):
            setattr(kin64, name, getattr(kin64, name).double())
        bp, bR, bv, bw = kin64.forward(p, rot, v, w, q, qd)
        for i, name in enumerate(robot.fk_bodies):
            body = model.body(name).id
            assert np.allclose(bp[i].numpy(), data.xpos[body], atol=1e-6), name
            assert np.allclose(bR[i].numpy(), data.xmat[body].reshape(3, 3), atol=1e-6), name
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            assert np.allclose(bw[i].numpy(), velocity[:3], atol=1e-5), name
            assert np.allclose(bv[i].numpy(), velocity[3:], atol=1e-5), name
