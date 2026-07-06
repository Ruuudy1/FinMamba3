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


def test_supervision_channels_roundtrip_and_legacy_arity():
    cfg = _make_config(buffer_max_length=32, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    assert np.isnan(buf.tte_frac_buffer).all()
    assert np.isnan(buf.spot_dist_buffer).all()
    assert np.isnan(buf.event_count_buffer).all()
    obs = np.zeros(2, dtype=np.float32)
    for i in range(20):
        buf.append(
            obs=obs, action=0, reward=0.0, termination=0.0, outcome=1.0,
            event_counts=np.array([float(i % 3), float((i + 1) % 4)], dtype=np.float32),
            tte_frac=1.0 - i / 19.0, spot_signed_distance=0.001 * (i - 10),
        )
    # Default sample stays the legacy five-tuple so imagination and the existing callers are untouched.
    assert len(buf.sample(batch_size=4, batch_length=8, imagine=False)) == 5
    sampled = buf.sample(batch_size=4, batch_length=8, imagine=False, with_supervision=True)
    assert len(sampled) == 12
    _, _, _, _, _, tte_frac, spot_dist, event_counts, yes_ask, no_ask, yes_mid, book_depth = sampled
    assert tte_frac.shape == (4, 8)
    assert spot_dist.shape == (4, 8)
    assert event_counts.shape == (4, 8, 2)
    assert torch.isfinite(tte_frac).all()
    assert torch.isfinite(spot_dist).all()
    assert torch.isfinite(event_counts).all()
    assert torch.all((tte_frac >= 0.0) & (tte_frac <= 1.0))
    # `yes_ask`/`no_ask`/`yes_mid`/`book_depth` were not provided in append(), so NaN-initialized.
    assert torch.isnan(yes_ask).all()
    assert torch.isnan(no_ask).all()
    assert torch.isnan(yes_mid).all()
    assert torch.isnan(book_depth).all()


def test_supervision_channels_default_nan_when_unpopulated():
    cfg = _make_config(buffer_max_length=16, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    obs = np.zeros(2, dtype=np.float32)
    for _ in range(12):
        buf.append(obs=obs, action=0, reward=0.0, termination=0.0)
    _, _, _, _, _, tte_frac, spot_dist, event_counts, yes_ask, no_ask, yes_mid, book_depth = buf.sample(batch_size=4, batch_length=4, with_supervision=True)
    assert torch.isnan(tte_frac).all()
    assert torch.isnan(spot_dist).all()
    assert torch.isnan(event_counts).all()
    assert torch.isnan(yes_ask).all()
    assert torch.isnan(no_ask).all()
    assert torch.isnan(yes_mid).all()
    assert torch.isnan(book_depth).all()


def test_append_rejects_bad_event_count_shape():
    cfg = _make_config(buffer_max_length=16, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    obs = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="event_counts"):
        buf.append(obs=obs, action=0, reward=0.0, termination=0.0, event_counts=np.ones(3, dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for store_on_gpu=True path")
def test_gpu_path_outcome_buffer():
    cfg = _make_config(buffer_max_length=16, feature_dim=2, on_gpu=True)
    buf = ReplayBuffer(cfg, device="cuda")
    assert torch.isnan(buf.outcome_buffer).all()
    obs = np.zeros(2, dtype=np.float32)
    buf.append(obs=obs, action=0, reward=0.0, termination=0.0, outcome=1.0)
    assert buf.outcome_buffer[0].item() == 1.0


def _populate_two_markets(buf: ReplayBuffer, sizes=(20, 20)) -> None:
    """Encode each tick's obs as [global_index, market_id] so a sampled window can be checked
    for boundary straddling: the market id must stay constant within a window.
    """
    global_index = 0
    for market_id, size in enumerate(sizes):
        for t in range(size):
            obs = np.array([float(global_index), float(market_id)], dtype=np.float32)
            buf.append(obs=obs, action=0, reward=0.0, termination=0.0, segment_end=(t == size - 1))
            global_index += 1


def test_valid_start_mask_matches_hand_computation():
    """Market A is indices 0..4, market B is 5..9; the boundary sits at index 4. With window
    length 3, starts 3 and 4 straddle that boundary and all others are valid.
    """
    cfg = _make_config(buffer_max_length=16, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    _populate_two_markets(buf, sizes=(5, 5))
    mask = buf._valid_start_mask(batch_length=3)
    expected = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=np.float64)
    assert np.array_equal(mask, expected)


def test_sample_never_straddles_a_market_boundary():
    cfg = _make_config(buffer_max_length=64, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    _populate_two_markets(buf)
    sample_obs, _, _, _, _ = buf.sample(batch_size=64, batch_length=8, imagine=False)
    market_ids = sample_obs[..., 1]
    assert torch.all(market_ids[:, 0:1] == market_ids)


def test_imagine_sample_never_straddles_a_market_boundary():
    cfg = _make_config(buffer_max_length=64, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    _populate_two_markets(buf)
    sample_obs, _, _, _, _ = buf.sample(batch_size=32, batch_length=8, imagine=True)
    market_ids = sample_obs[..., 1]
    assert torch.all(market_ids[:, 0:1] == market_ids)


def test_single_market_sampling_stays_contiguous():
    """With no boundaries the mask is all-True, so each window is a contiguous run of indices,
    exactly as the legacy single-market path produced.
    """
    cfg = _make_config(buffer_max_length=64, feature_dim=2)
    buf = ReplayBuffer(cfg, device="cpu")
    for i in range(40):
        obs = np.array([float(i), 0.0], dtype=np.float32)
        buf.append(obs=obs, action=0, reward=0.0, termination=0.0)
    sample_obs, _, _, _, _ = buf.sample(batch_size=8, batch_length=8, imagine=False)
    global_indices = sample_obs[..., 0]
    diffs = global_indices[:, 1:] - global_indices[:, :-1]
    assert torch.all(diffs == 1.0)
