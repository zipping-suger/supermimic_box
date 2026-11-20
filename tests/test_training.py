import json

import pytest
import torch

from supermimic_box.config import TrainConfig
from supermimic_box.runner import Runner


@pytest.mark.parametrize("modes", [("flow", "absolute"), ("full", "residual")])
def test_two_iterations_train_save_and_evaluate(tiny_config, modes):
    from supermimic_box.evaluate import evaluate, load_runner

    tiny_config.goal_mode, tiny_config.action_mode = modes
    runner = Runner(tiny_config)
    before = [p.detach().clone() for p in runner.actor.parameters()]
    runner.train()
    assert any((a - b).abs().sum() > 0 for a, b in zip(before, runner.actor.parameters()))
    log = [json.loads(line) for line in open(f"{tiny_config.log_dir}/metrics.jsonl")]
    assert log[-1]["iteration"] == 2 and all(torch.isfinite(torch.tensor(v)) for v in log[-1].values())
    report, _ = evaluate(load_runner(f"{tiny_config.log_dir}/last.pt", device="cpu"))
    assert report["clips"] == 3 and 0 <= report["success_rate"] <= report["completion_rate"] <= 1


def test_sonic_base_with_residual_trains_and_starts_as_sonic(tiny_config, sonic_checkpoint):
    from supermimic_box.evaluate import evaluate, load_runner

    tiny_config.base_policy, tiny_config.base_checkpoint, tiny_config.drives = "sonic", str(sonic_checkpoint), "sonic"
    runner = Runner(tiny_config)
    env, m = runner.env, runner.motion
    # The hip pitch drives are SONIC's, and an untrained residual leaves SONIC's action nearly unchanged.
    hip = runner.task.robot.joint_names.index("left_hip_pitch_joint")
    assert abs(float(runner.control.gains[hip, 2]) - 139.0) < 1e-6
    root_R = env.physics.state()[1]
    with torch.no_grad():
        base = runner.base(env.history, root_R, env.f)
        action = runner.policy_action(runner.actor, env.history, m.goal(env.f, tiny_config.future), None, root_R, env.f, env.context)
    assert base.shape == (env.N, 29) and torch.isfinite(base).all() and 0.01 < base.abs().max() < 20
    assert (action - base).abs().max() < 0.05 * base.abs().max()
    # A leg residual scale shrinks the residual on legs and waist only.
    runner.residual_scale[:] = 1.0
    runner.residual_scale[[i for i in range(29) if i not in runner.task.robot.arm_idx]] = 0.1
    with torch.no_grad():
        scaled = runner.policy_action(runner.actor, env.history, m.goal(env.f, tiny_config.future), None, root_R, env.f, env.context)
        residual = runner.actor(env.history, m.goal(env.f, tiny_config.future))
    arms = runner.task.robot.arm_idx
    legs = [i for i in range(29) if i not in arms]
    assert len(arms) == 14 and torch.allclose(scaled[:, arms] - base[:, arms], residual[:, arms], atol=1e-5)
    assert torch.allclose(scaled[:, legs] - base[:, legs], 0.1 * residual[:, legs], atol=1e-5)
    assert not any(p.requires_grad for p in runner.base.parameters())
    # The tokenizer's first rows are reference joint positions, its last columns the relative pelvis rotation.
    tokens = runner.base.tokenizer(root_R, env.f).unflatten(-1, (10, 64))
    assert torch.allclose(tokens[:, 0, :29], m.q[env.f]) and torch.allclose(tokens[:, 0, 29:58], m.q[m.next_frame(env.f + 4)])
    relative = root_R.transpose(-1, -2) @ m.root_R[env.f]
    assert torch.allclose(tokens[:, 0, 58:], relative[:, :, :2].flatten(1), atol=1e-6)
    before = {k: v.clone() for k, v in runner.base.state_dict().items()}
    runner.train()
    assert all(torch.equal(v, runner.base.state_dict()[k]) for k, v in before.items())
    report, _ = evaluate(load_runner(f"{tiny_config.log_dir}/last.pt", device="cpu"))
    assert report["clips"] == 3 and 0 <= report["success_rate"] <= 1


def test_config_rejects_inconsistent_modes():
    with pytest.raises(ValueError):
        TrainConfig(goal_mode="flow", action_mode="residual", device="cpu", backend="cpu", randomize=False, cuda_graph=False).validate()
    with pytest.raises(ValueError):
        TrainConfig(base_policy="sonic", device="cpu", backend="cpu", randomize=False, cuda_graph=False).validate()
    with pytest.raises(ValueError):
        TrainConfig.from_json(None, unknown_key=1)
