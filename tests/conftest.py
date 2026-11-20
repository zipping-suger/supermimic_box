"""Shared fixtures. Tests that need prepared data skip when ``python -m supermimic_box.prepare`` has not run."""

import json
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def data():
    if not (DATA / "box" / "manifest.json").is_file():
        pytest.skip("Prepared data not found; run python -m supermimic_box.prepare")
    return DATA


@pytest.fixture(scope="session")
def robot(data):
    from supermimic_box.robot import load_robot

    return load_robot(data / "robot")


@pytest.fixture(scope="session")
def registry(data):
    from supermimic_box.sim import load_registry

    return load_registry(data / "objects")[0]


@pytest.fixture(scope="session")
def clips(data):
    """Two carry clips and a push clip."""
    names = json.loads((data / "box" / "manifest.json").read_text())["clip_names"]
    carry = [n for n in names if n.startswith("carry")]
    return [carry[0], next(n for n in names if n.startswith("push")), carry[1]]


@pytest.fixture(scope="session")
def sonic_checkpoint(data):
    path = data / "sonic" / "last_ported.pt"
    if not path.is_file():
        pytest.skip("SONIC checkpoint not found at data/sonic/last_ported.pt")
    return path


@pytest.fixture
def tiny_config(data, clips, tmp_path):
    """Three clips, four CPU worlds and very small networks."""
    from supermimic_box.config import TrainConfig

    torch.manual_seed(0)
    return TrainConfig(
        dataset=str(data / "box"), robot_assets=str(data / "robot"), object_assets=str(data / "objects"),
        clips=clips, num_envs=4, device="cpu", backend="cpu", randomize=False, cuda_graph=False, compile=False,
        amp="none", iterations=2, warmup_steps=12, buffer_steps=64, wm_batch=16, pi_batch=8, horizon=6, wm_updates=2,
        hidden=[64, 64], policy_hidden=[64, 64], reward_head_hidden=[32], critic_hidden=[32], log_every=1,
        save_every=100, log_dir=str(tmp_path / "run"))  # fmt: skip
