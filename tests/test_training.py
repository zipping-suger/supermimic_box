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


def test_config_rejects_inconsistent_modes():
    with pytest.raises(ValueError):
        TrainConfig(goal_mode="flow", action_mode="residual", device="cpu", backend="cpu", randomize=False, cuda_graph=False).validate()
    with pytest.raises(ValueError):
        TrainConfig.from_json(None, unknown_key=1)
