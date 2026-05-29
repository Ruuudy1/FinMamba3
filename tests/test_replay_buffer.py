# region imports
from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from finmamba3.config import DotDict
from finmamba3.replay_buffer import ReplayBuffer
# endregion


def _make_config(buffer_max_length: int = 64, feature_dim: int = 4, on_gpu: bool = False) -> DotDict:
    return DotDict({
        "BasicSettings": {
            "ReplayBufferOnGPU": on_gpu,
            "ObsMode": "features",
            "FeatureDim": feature_dim,
        },
        "JointTrainAgent": {
            "BufferMaxLength": buffer_max_length,
            "WorldModelWarmUp": 4,
            "BehaviourWarmUp": 1_000_000,
            "Tau": 10.0,
            "ImaginationTau": 10.0,
            "Alpha": 1.0,
            "Beta": 1.0,
            "BatchSize": 4,
            "ImagineBatchSize": 4,
        },
    })


def test_outcome_buffer_initialized_to_nan():
    cfg = _make_config()
    buf = ReplayBuffer(cfg, device="cpu")
    assert np.isnan(buf.outcome_buffer).all()


def test_append_records_outcome():
    cfg = _make_config(buffer_max_length=16, feature_dim=3)
    buf = ReplayBuffer(cfg, device="cpu")
    obs = np.zeros(3, dtype=np.float32)
    buf.append(obs=obs, action=0, reward=0.0, termination=0.0, outcome=1.0)
    buf.append(obs=obs, action=0, reward=0.0, termination=0.0, outcome=0.0)
    buf.append(obs=obs, action=0, reward=0.0, termination=0.0)
    assert buf.outcome_buffer[0] == 1.0
    assert buf.outcome_buffer[1] == 0.0
    assert math.isnan(buf.outcome_buffer[2])


def test_sample_returns_outcome_tuple():
    cfg = _make_config(buffer_max_length=32, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    obs = np.zeros(2, dtype=np.float32)
    for _ in range(20):
        buf.append(obs=obs, action=0, reward=0.0, termination=0.0, outcome=1.0)
    sampled = buf.sample(batch_size=4, batch_length=8, imagine=False)
    assert len(sampled) == 5
    sample_obs, sample_action, sample_reward, sample_termination, sample_outcome = sampled
    assert sample_outcome.shape == (4, 8)
    assert torch.all(sample_outcome == 1.0)


def test_sample_propagates_nan_for_unresolved_market():
    cfg = _make_config(buffer_max_length=32, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    obs = np.zeros(2, dtype=np.float32)
    for _ in range(20):
        buf.append(obs=obs, action=0, reward=0.0, termination=0.0)
    _, _, _, _, sample_outcome = buf.sample(batch_size=4, batch_length=8, imagine=False)
    assert torch.isnan(sample_outcome).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for store_on_gpu=True path")
def test_gpu_path_outcome_buffer():
    cfg = _make_config(buffer_max_length=16, feature_dim=2, on_gpu=True)
    buf = ReplayBuffer(cfg, device="cuda")
    assert torch.isnan(buf.outcome_buffer).all()
    obs = np.zeros(2, dtype=np.float32)
    buf.append(obs=obs, action=0, reward=0.0, termination=0.0, outcome=1.0)
    assert buf.outcome_buffer[0].item() == 1.0
