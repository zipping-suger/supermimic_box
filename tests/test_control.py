import pytest
import torch

from supermimic_box.control import Controller


@pytest.mark.parametrize("mode", ["absolute", "residual"])
def test_reference_action_asks_for_the_reference_and_commands_are_bounded(robot, mode):
    c = Controller(robot, action_mode=mode, action_clip=20.0)
    q = torch.tensor(robot.default_joint_pos) + 0.1 * torch.randn(5, robot.num_joints)
    offset = 0.01 * torch.randn(5, robot.num_joints)
    action = c.reference_action(q)
    assert torch.allclose(c(action, q, offset), q + offset, atol=1e-5)
    assert torch.allclose(c.effective(c(action, q, torch.zeros_like(q))), c.command(action, q), atol=1e-4)
    if mode == "residual":
        assert action.abs().max() < 1e-5
    wild = c.bound_action(1000 * torch.randn(5, robot.num_joints), q)
    assert c.command(wild, q).abs().max() <= 20.0 + 1e-3
    # Torque is the PD law within the motor's effort limit.
    torque = c.torque(q + 10.0, q, torch.zeros_like(q))
    assert torch.allclose(torque, c.gains[:, 2].expand_as(torque))
    assert len(c.torque_joints) == 8
