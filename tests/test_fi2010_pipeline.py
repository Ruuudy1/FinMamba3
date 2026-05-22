"""Tests for the FI-2010 LOB pipeline: loader, feature engineering, normalization, encoder, and world model."""
# region imports
from __future__ import annotations
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from envs.fi2010_loader import (
    FI2010Sequence,
    FI2010_FEATURE_DIM,
    FI2010_F_LEVEL,
    FI2010_F_TICK,
    FI2010_K_LEVELS,
    FLAT_FEATURE_NAMES_FI2010,
    _load_raw_matrix,
    _remap_labels,
    load_fi2010_split,
)
from envs.lob_features import (
    LOBSequence,
    apply_normalization,
    compute_basic_tick_features,
    fit_normalization,
    make_aggregate_only,
)
# endregion

# Events written to synthetic split files used by the split_dir fixture.
_N_EVENTS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_raw_matrix(n_events: int = _N_EVENTS, seed: int = 0) -> np.ndarray:
    """Return a valid synthetic (n_events, 149) FI-2010 raw matrix."""
    rng = np.random.default_rng(seed)
    mat = rng.uniform(0.1, 10.0, size=(n_events, 149)).astype(np.float32)
    for col in range(144, 149):
        mat[:, col] = rng.choice([1.0, 2.0, 3.0], size=n_events)
    return mat


def _write_split_file(path: Path, n_events: int = _N_EVENTS) -> None:
    np.savetxt(path, _synthetic_raw_matrix(n_events), fmt="%.6f")


def _make_lobsequence(n_events: int = 80) -> LOBSequence:
    """Build a minimal LOBSequence with FI-2010 dimensions from synthetic data."""
    rng = np.random.default_rng(7)
    per_level = rng.uniform(0.1, 5.0, (n_events, FI2010_K_LEVELS, FI2010_F_LEVEL)).astype(np.float32)
    per_tick = compute_basic_tick_features(per_level)
    mid = 0.5 * (per_level[:, 0, 0] + per_level[:, 0, 2])
    return LOBSequence(
        market_slug="fi2010_test",
        per_level=per_level,
        per_tick=per_tick,
        midprice=mid.astype(np.float32),
        ts_sec=np.arange(n_events, dtype=np.int64),
        yes_outcome=None,
    )


# ---------------------------------------------------------------------------
# Fake Mamba installation (used by WorldModel tests).
# ---------------------------------------------------------------------------

def _install_fake_mamba() -> None:
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]

    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = []
    mods = types.ModuleType("mamba_ssm.modules")
    mods.__path__ = []

    class _FakeBlock(torch.nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.proj = torch.nn.Linear(d_model, d_model)

        def forward(self, x, **kwargs):
            return self.proj(x)

    for sub in ("mamba3", "mamba2", "mamba_simple"):
        m = types.ModuleType(f"mamba_ssm.modules.{sub}")
        m.Mamba3 = _FakeBlock
        m.Mamba2 = _FakeBlock
        m.Mamba = _FakeBlock
        sys.modules[f"mamba_ssm.modules.{sub}"] = m
    sys.modules["mamba_ssm"] = pkg
    sys.modules["mamba_ssm.modules"] = mods


def _ensure_pytorch_warmup() -> None:
    try:
        import pytorch_warmup  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("pytorch_warmup")

        class _LW:
            def __init__(self, optimizer, warmup_period):
                pass

            def dampen(self):
                pass

        stub.LinearWarmup = _LW
        sys.modules["pytorch_warmup"] = stub


@pytest.fixture(scope="module", autouse=True)
def _module_setup():
    """Install fake Mamba and warmup stubs once before all tests in this module."""
    _ensure_pytorch_warmup()
    _install_fake_mamba()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def split_dir(tmp_path):
    """Temporary directory containing one train and one validation split file."""
    _write_split_file(tmp_path / "Train_Dst_NoAuction_DecPre_CF_7.txt")
    _write_split_file(tmp_path / "Test_Dst_NoAuction_DecPre_CF_7.txt")
    return tmp_path


def _fi2010_world_model_config():
    """Minimal WorldModel config for FI-2010 (CPU, fake Mamba, 46-dim obs)."""
    from types import SimpleNamespace

    def _ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
        return d

    return _ns({
        "BasicSettings": {
            "ObsMode": "features",
            "FeatureDim": FI2010_FEATURE_DIM,
            "ReplayBufferOnGPU": False,
            "Use_amp": False,
            "Use_cg": False,
            "Compile": False,
            "NormClip": 8.0,
        },
        "JointTrainAgent": {
            "BatchLength": 8,
            "ImagineContextLength": 4,
            "ImagineBatchLength": 4,
            "RealityContextLength": 4,
            "SaveEverySteps": 1000,
            "SampleMaxSteps": 100,
        },
        "Models": {
            "WorldModel": {
                "dtype": torch.float32,
                "Backbone": "Mamba3",
                "Act": "SiLU",
                "CategoricalDim": 4,
                "ClassDim": 4,
                "HiddenStateDim": 32,
                "Optimiser": "Adam",
                "LRSchedule": "constant",
                "LRMinRatio": 0.1,
                "Dropout": 0.0,
                "Unimix_ratio": 0.01,
                "Weight_decay": 0.0,
                "Max_grad_norm": 100.0,
                "Warmup_steps": 1,
                "UseActionInput": False,
                "DirectionThresholds": None,
                "NaNGuardSteps": 0,
                "RepresentationLossWeight": 0.1,
                "FreeBits": 1.0,
                "Adam": {"LearningRate": 1e-4},
                "Encoder": {
                    "Type": "lob",
                    "K": FI2010_K_LEVELS,
                    "FeatureDimLevel": FI2010_F_LEVEL,
                    "FeatureDimTick": FI2010_F_TICK,
                    "DModel": 32,
                    "NumLayers": 1,
                    "NumHeads": 4,
                    "DimFeedforward": 64,
                    "OutputFlattenDim": 64,
                    "AggregateOnly": False,
                    "GradientCheckpointing": False,
                },
                "Decoder": {
                    "Kind": "mse",
                    "HiddenDim": 32,
                    "NumLayers": 1,
                    "NuInit": 5.0,
                    "LearnableNu": True,
                    "SizeWeight": 1.0,
                    "LevelSizeIndices": [1, 3],
                    "TickSizeIndices": [5],
                },
                "Reward": {"Enabled": False, "HiddenUnits": 32, "LayerNum": 1},
                "Termination": {"Enabled": False, "HiddenUnits": 32, "LayerNum": 1},
                "Transformer": {"FinalFeatureWidth": 4, "NumLayers": 1, "NumHeads": 2},
                "Mamba": {"n_layer": 1, "d_intermediate": 0, "ssm_cfg": {"d_state": 16}},
                "Mamba3": {
                    "Enabled": True,
                    "n_layer": 1,
                    "is_mimo": True,
                    "mimo_rank": 1,
                    "d_state": 8,
                    "headdim": 16,
                    "chunk_size": 4,
                    "is_outproj_norm": False,
                    "rope_fraction": 0.0,
                },
                "Direction": {
                    "Enabled": False,
                    "NumClasses": 3,
                    "Threshold": 0.01,
                    "LossWeight": 0.5,
                    "Dropout": 0.0,
                },
                "Regime": {"Enabled": False},
                "EpisodicMemory": {"Enabled": False},
                "Hawkes": {"Enabled": False},
                "Settlement": {"Enabled": False},
            }
        },
    })


# ===========================================================================
# 1. Constants
# ===========================================================================

def test_feature_dim_is_46():
    assert FI2010_FEATURE_DIM == 46, "FI-2010 flat dim must be 10 * 4 + 6 = 46."


def test_k_levels_f_level_f_tick():
    assert FI2010_K_LEVELS == 10
    assert FI2010_F_LEVEL == 4
    assert FI2010_F_TICK == 6


def test_flat_feature_names_length():
    assert len(FLAT_FEATURE_NAMES_FI2010) == FI2010_FEATURE_DIM


def test_flat_feature_names_first_last():
    assert FLAT_FEATURE_NAMES_FI2010[0] == "level0.ask_price"
    assert FLAT_FEATURE_NAMES_FI2010[FI2010_K_LEVELS * FI2010_F_LEVEL] == "tick.mid"


# ===========================================================================
# 2. Label remapping
# ===========================================================================

def test_remap_up():
    out = _remap_labels(np.array([1.0, 1.0], dtype=np.float32))
    assert list(out) == [2, 2], "FI-2010 label 1 (up) must remap to 2."


def test_remap_flat():
    out = _remap_labels(np.array([2.0, 2.0], dtype=np.float32))
    assert list(out) == [1, 1], "FI-2010 label 2 (stationary) must remap to 1."


def test_remap_down():
    out = _remap_labels(np.array([3.0, 3.0], dtype=np.float32))
    assert list(out) == [0, 0], "FI-2010 label 3 (down) must remap to 0."


def test_remap_all_classes():
    out = _remap_labels(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert list(out) == [2, 1, 0]


def test_remap_output_dtype_is_int64():
    out = _remap_labels(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert out.dtype == np.int64


# ===========================================================================
# 3. Raw matrix loading
# ===========================================================================

def test_load_raw_matrix_row_major(tmp_path):
    mat = _synthetic_raw_matrix(n_events=30)
    path = tmp_path / "row.txt"
    np.savetxt(path, mat, fmt="%.6f")
    loaded = _load_raw_matrix(path)
    assert loaded.shape == (30, 149), "Row-major file must load as (N, 149)."


def test_load_raw_matrix_col_major_transposed(tmp_path):
    # Write in (149, N) orientation; loader should auto-transpose.
    mat = _synthetic_raw_matrix(n_events=25)
    path = tmp_path / "col.txt"
    np.savetxt(path, mat.T, fmt="%.6f")
    loaded = _load_raw_matrix(path)
    assert loaded.shape == (25, 149), "Column-major file must be transposed to (N, 149)."


def test_load_raw_matrix_wrong_columns_raises(tmp_path):
    bad = np.zeros((10, 100), dtype=np.float32)
    path = tmp_path / "bad.txt"
    np.savetxt(path, bad, fmt="%.1f")
    with pytest.raises(ValueError, match="149"):
        _load_raw_matrix(path)


# ===========================================================================
# 4. compute_basic_tick_features
# ===========================================================================

def test_compute_basic_tick_features_output_shape():
    per_level = np.random.default_rng(1).uniform(0.1, 10.0, (20, 10, 4)).astype(np.float32)
    out = compute_basic_tick_features(per_level)
    assert out.shape == (20, 6)


def test_compute_basic_tick_features_wrong_shape_raises():
    with pytest.raises(ValueError):
        compute_basic_tick_features(np.zeros((5, 10, 8), dtype=np.float32))


def test_compute_basic_tick_features_mid_equals_half_sum():
    per_level = np.zeros((3, 10, 4), dtype=np.float32)
    per_level[:, 0, 0] = 1.1  # best ask
    per_level[:, 0, 2] = 0.9  # best bid
    out = compute_basic_tick_features(per_level)
    np.testing.assert_allclose(out[:, 0], 0.5 * (1.1 + 0.9), rtol=1e-5)


def test_compute_basic_tick_features_spread():
    per_level = np.zeros((3, 10, 4), dtype=np.float32)
    per_level[:, 0, 0] = 1.2
    per_level[:, 0, 2] = 0.8
    out = compute_basic_tick_features(per_level)
    np.testing.assert_allclose(out[:, 1], 0.4, rtol=1e-5)


def test_compute_basic_tick_features_imbalance_defaults_to_half_when_zero_size():
    per_level = np.zeros((2, 10, 4), dtype=np.float32)
    per_level[:, 0, 0] = 1.0  # ask price
    per_level[:, 0, 2] = 0.9  # bid price, sizes are zero
    out = compute_basic_tick_features(per_level)
    np.testing.assert_allclose(out[:, 3], 0.5, rtol=1e-5,
                               err_msg="Imbalance must default to 0.5 when top-of-book size sum is zero.")


def test_compute_basic_tick_features_all_finite():
    per_level = np.random.default_rng(2).uniform(0.01, 5.0, (100, 10, 4)).astype(np.float32)
    out = compute_basic_tick_features(per_level)
    assert np.isfinite(out).all()


# ===========================================================================
# 5. load_fi2010_split
# ===========================================================================

def test_load_returns_fi2010sequence_type(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    assert isinstance(bundle, FI2010Sequence)


def test_load_invalid_split_raises(split_dir):
    with pytest.raises(ValueError, match="split"):
        load_fi2010_split(split_dir, split="test")


def test_load_invalid_horizon_raises(split_dir):
    with pytest.raises(ValueError, match="horizon"):
        load_fi2010_split(split_dir, split="train", horizon=15)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fi2010_split(tmp_path, split="train", horizon=10)


def test_load_per_level_shape(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    assert bundle.sequence.per_level.shape == (_N_EVENTS, FI2010_K_LEVELS, FI2010_F_LEVEL)


def test_load_per_tick_shape(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    assert bundle.sequence.per_tick.shape == (_N_EVENTS, FI2010_F_TICK)


def test_load_flat_dim_is_46(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    flat = bundle.sequence.to_flat()
    assert flat.shape == (_N_EVENTS, FI2010_FEATURE_DIM), "to_flat() must produce (N, 46) for FI-2010."


def test_load_direction_labels_in_valid_range(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    labels = bundle.direction_labels
    assert labels.shape == (_N_EVENTS,)
    assert labels.dtype == np.int64
    assert set(labels).issubset({0, 1, 2})


def test_load_label_remap_applied_correctly(split_dir):
    mat = _synthetic_raw_matrix(n_events=6)
    mat[:, 144] = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    np.savetxt(split_dir / "Train_Dst_NoAuction_DecPre_CF_7.txt", mat, fmt="%.6f")
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    expected = np.array([2, 1, 0, 2, 1, 0], dtype=np.int64)
    np.testing.assert_array_equal(bundle.direction_labels, expected)


def test_load_max_events_caps_to_n(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10, max_events=20)
    assert bundle.sequence.per_level.shape[0] == 20


def test_load_max_events_larger_than_file_returns_all(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10, max_events=9999)
    assert bundle.sequence.per_level.shape[0] == _N_EVENTS


def test_load_max_events_zero_returns_all(split_dir):
    # 0 means "no cap" per the config convention.
    bundle = load_fi2010_split(split_dir, split="train", horizon=10, max_events=0)
    assert bundle.sequence.per_level.shape[0] == _N_EVENTS


def test_load_zscore_filename_fallback(tmp_path):
    # Only the ZScore variant is present; loader must accept it.
    _write_split_file(tmp_path / "Train_Dst_NoAuction_ZScore_CF_7.txt")
    bundle = load_fi2010_split(tmp_path, split="train", horizon=10)
    assert bundle.sequence.per_level.shape[0] == _N_EVENTS


def test_load_all_valid_horizons(split_dir):
    for h in (10, 20, 30, 50, 100):
        bundle = load_fi2010_split(split_dir, split="train", horizon=h)
        assert bundle.horizon == h


def test_load_market_slug_contains_horizon(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=20)
    assert bundle.sequence.market_slug == "fi2010_h20"


def test_load_ts_sec_is_arange(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    np.testing.assert_array_equal(bundle.sequence.ts_sec, np.arange(_N_EVENTS, dtype=np.int64))


def test_load_midprice_finite(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    assert np.isfinite(bundle.sequence.midprice).all()


def test_load_validation_split(split_dir):
    bundle = load_fi2010_split(split_dir, split="validation", horizon=10)
    assert bundle.sequence.per_level.shape[0] == _N_EVENTS


def test_load_per_level_dtype_float32(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    assert bundle.sequence.per_level.dtype == np.float32


def test_load_flat_all_finite(split_dir):
    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    assert np.isfinite(bundle.sequence.to_flat()).all()


# ===========================================================================
# 6. Normalization with FI-2010 schema
# ===========================================================================

def test_fit_normalization_stats_finite():
    seq = _make_lobsequence()
    stats = fit_normalization(seq)
    assert np.isfinite(stats.per_level_mean).all()
    assert np.isfinite(stats.per_level_std).all()
    assert np.isfinite(stats.per_tick_mean).all()
    assert np.isfinite(stats.per_tick_std).all()


def test_fit_normalization_std_positive():
    seq = _make_lobsequence()
    stats = fit_normalization(seq)
    assert (stats.per_level_std > 0).all(), "Per-level std must be strictly positive."
    assert (stats.per_tick_std > 0).all(), "Per-tick std must be strictly positive."


def test_apply_normalization_preserves_shape():
    seq = _make_lobsequence()
    stats = fit_normalization(seq)
    norm = apply_normalization(seq, stats)
    assert norm.per_level.shape == seq.per_level.shape
    assert norm.per_tick.shape == seq.per_tick.shape


def test_apply_normalization_within_clip():
    seq = _make_lobsequence()
    stats = fit_normalization(seq, clip_value=8.0)
    norm = apply_normalization(seq, stats)
    flat = norm.to_flat()
    assert np.isfinite(flat).all()
    assert float(np.abs(flat).max()) <= 8.0 + 1e-4


def test_apply_normalization_flat_dim_is_46():
    seq = _make_lobsequence()
    stats = fit_normalization(seq)
    norm = apply_normalization(seq, stats)
    assert norm.to_flat().shape[1] == FI2010_FEATURE_DIM


def test_make_aggregate_only_zeros_per_level():
    seq = _make_lobsequence()
    agg = make_aggregate_only(seq)
    assert np.all(agg.per_level == 0.0), "aggregate_only must zero all depth tokens."
    np.testing.assert_array_equal(agg.per_tick, seq.per_tick)


# ===========================================================================
# 7. LOBEncoder with FI-2010 config
# ===========================================================================

def _fi2010_encoder():
    from sub_models.lob_encoder import LOBEncoder
    return LOBEncoder(
        k_levels=FI2010_K_LEVELS,
        f_level=FI2010_F_LEVEL,
        f_tick=FI2010_F_TICK,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        output_flatten_dim=64,
    )


def test_lob_encoder_fi2010_output_shape():
    enc = _fi2010_encoder()
    x = torch.randn(2, 8, FI2010_FEATURE_DIM)
    out = enc(x)
    assert out.shape == (2, 8, 64)


def test_lob_encoder_fi2010_output_finite():
    enc = _fi2010_encoder()
    x = torch.randn(2, 4, FI2010_FEATURE_DIM)
    out = enc(x)
    assert torch.isfinite(out).all()


def test_lob_encoder_fi2010_batch_independence():
    enc = _fi2010_encoder()
    enc.eval()
    x = torch.randn(4, 6, FI2010_FEATURE_DIM)
    with torch.no_grad():
        out_full = enc(x)
        out_first = enc(x[:2])
    torch.testing.assert_close(out_full[:2], out_first, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 8. LOBReconstructionLoss and StudentTReconstructionLoss with FI-2010 indices
# ===========================================================================

def test_lob_reconstruction_loss_fi2010_scalar_finite():
    from sub_models.lob_encoder import LOBReconstructionLoss
    loss_fn = LOBReconstructionLoss(
        k_levels=FI2010_K_LEVELS,
        f_level=FI2010_F_LEVEL,
        f_tick=FI2010_F_TICK,
        size_weight=2.0,
        level_size_indices=(1, 3),
        tick_size_indices=(5,),
    )
    obs_hat = torch.randn(4, 8, FI2010_FEATURE_DIM)
    obs = torch.randn(4, 8, FI2010_FEATURE_DIM)
    loss = loss_fn(obs_hat, obs)
    assert loss.shape == (), "Loss must be a scalar."
    assert torch.isfinite(loss)


def test_lob_reconstruction_loss_fi2010_feature_weight_shape():
    from sub_models.lob_encoder import LOBReconstructionLoss
    loss_fn = LOBReconstructionLoss(
        k_levels=FI2010_K_LEVELS,
        f_level=FI2010_F_LEVEL,
        f_tick=FI2010_F_TICK,
        size_weight=2.0,
        level_size_indices=(1, 3),
        tick_size_indices=(5,),
    )
    assert loss_fn.feature_weight.shape == (FI2010_FEATURE_DIM,)


def test_lob_reconstruction_loss_fi2010_size_indices_upweighted():
    from sub_models.lob_encoder import LOBReconstructionLoss
    loss_fn = LOBReconstructionLoss(
        k_levels=FI2010_K_LEVELS,
        f_level=FI2010_F_LEVEL,
        f_tick=FI2010_F_TICK,
        size_weight=3.0,
        level_size_indices=(1, 3),
        tick_size_indices=(5,),
    )
    w = loss_fn.feature_weight
    # Level index 1 and 3 (ask_size and bid_size) per level must be upweighted.
    for k in range(FI2010_K_LEVELS):
        base = k * FI2010_F_LEVEL
        assert w[base + 1] > w[base + 0], "ask_size weight must exceed ask_price weight."
        assert w[base + 3] > w[base + 0], "bid_size weight must exceed ask_price weight."
    # Tick index 5 (log_total_vol) must be upweighted.
    tick_start = FI2010_K_LEVELS * FI2010_F_LEVEL
    assert w[tick_start + 5] > w[tick_start + 0], "log_total_vol weight must exceed mid weight."


def test_studentt_reconstruction_loss_fi2010_scalar_finite():
    from sub_models.lob_encoder import StudentTLOBDecoder, StudentTReconstructionLoss
    decoder = StudentTLOBDecoder(
        stoch_dim=16,
        hidden_dim=32,
        num_layers=1,
        k_levels=FI2010_K_LEVELS,
        f_level=FI2010_F_LEVEL,
        f_tick=FI2010_F_TICK,
    )
    loss_fn = StudentTReconstructionLoss(
        k_levels=FI2010_K_LEVELS,
        f_level=FI2010_F_LEVEL,
        f_tick=FI2010_F_TICK,
        level_size_indices=(1, 3),
        tick_size_indices=(5,),
    )
    x = torch.randn(2, 4, 16)
    mean, log_scale = decoder(x)
    obs = torch.randn(2, 4, FI2010_FEATURE_DIM)
    loss = loss_fn(decoder, mean, log_scale, obs)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# ===========================================================================
# 9. WorldModel with FI-2010 config (CPU, fake Mamba, 46-dim obs)
# ===========================================================================

def test_world_model_fi2010_constructs():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    assert wm.encoder_type == "lob"
    assert wm.model == "Mamba3"


def test_world_model_fi2010_stoch_flattened_dim():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    # CategoricalDim=4, ClassDim=4 → stoch_flattened_dim=16.
    assert wm.stoch_flattened_dim == 16


def test_world_model_fi2010_update_returns_twelve_losses():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    B, L = 2, 8
    obs = torch.randn(B, L, FI2010_FEATURE_DIM)
    action = torch.zeros(B, L, dtype=torch.long)
    reward = torch.zeros(B, L)
    termination = torch.zeros(B, L)
    losses = wm.update(obs, action, reward, termination, global_step=1, epoch_step=0)
    assert len(losses) == 12
    for t in losses:
        assert torch.is_tensor(t) and torch.isfinite(t).all()


def test_world_model_fi2010_encode_obs_shape():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    obs = torch.randn(2, 4, FI2010_FEATURE_DIM)
    with torch.no_grad():
        encoded = wm.encode_obs(obs)
    assert encoded.shape == (2, 4, wm.stoch_flattened_dim)


def test_world_model_fi2010_direction_head_enabled_loss_finite():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    cfg.Models.WorldModel.Direction.Enabled = True
    cfg.Models.WorldModel.Direction.LossWeight = 0.5
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    assert wm.use_direction_head
    B, L = 2, 8
    obs = torch.randn(B, L, FI2010_FEATURE_DIM)
    action = torch.zeros(B, L, dtype=torch.long)
    losses = wm.update(obs, action, torch.zeros(B, L), torch.zeros(B, L),
                       global_step=1, epoch_step=0)
    assert len(losses) == 12
    direction_loss = losses[7]
    assert torch.isfinite(direction_loss)


def test_world_model_fi2010_regime_head_enabled_loss_finite():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    cfg.Models.WorldModel.Regime.Enabled = True
    cfg.Models.WorldModel.Regime.NumRegimes = 4
    cfg.Models.WorldModel.Regime.EmbedDim = 8
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    assert wm.use_regime
    B, L = 2, 8
    obs = torch.randn(B, L, FI2010_FEATURE_DIM)
    action = torch.zeros(B, L, dtype=torch.long)
    losses = wm.update(obs, action, torch.zeros(B, L), torch.zeros(B, L),
                       global_step=1, epoch_step=0)
    assert all(torch.isfinite(t).all() for t in losses)


def test_world_model_fi2010_studentt_decoder_loss_finite():
    from sub_models.world_models import WorldModel
    cfg = _fi2010_world_model_config()
    cfg.Models.WorldModel.Decoder.Kind = "studentt"
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    B, L = 2, 8
    obs = torch.randn(B, L, FI2010_FEATURE_DIM)
    action = torch.zeros(B, L, dtype=torch.long)
    losses = wm.update(obs, action, torch.zeros(B, L), torch.zeros(B, L),
                       global_step=1, epoch_step=0)
    assert all(torch.isfinite(t).all() for t in losses)


# ===========================================================================
# 10. End-to-end normalization pipeline (loader → fit → apply → flat)
# ===========================================================================

def test_e2e_loader_to_normalized_flat(split_dir, tmp_path):
    from envs.lob_features import save_normalization, load_normalization

    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    seq = bundle.sequence

    stats = fit_normalization(seq, clip_value=8.0)
    norm_path = tmp_path / "norm.json"
    save_normalization(stats, norm_path)
    stats_reloaded = load_normalization(norm_path)

    seq_norm = apply_normalization(seq, stats_reloaded)
    flat = seq_norm.to_flat()

    assert flat.shape == (_N_EVENTS, FI2010_FEATURE_DIM)
    assert np.isfinite(flat).all()
    assert float(np.abs(flat).max()) <= 8.0 + 1e-4


def test_e2e_normalization_stats_survive_serialization(split_dir, tmp_path):
    from envs.lob_features import save_normalization, load_normalization

    bundle = load_fi2010_split(split_dir, split="train", horizon=10)
    stats = fit_normalization(bundle.sequence)

    norm_path = tmp_path / "norm.json"
    save_normalization(stats, norm_path)
    stats2 = load_normalization(norm_path)

    np.testing.assert_allclose(stats.per_level_mean, stats2.per_level_mean, rtol=1e-6)
    np.testing.assert_allclose(stats.per_level_std, stats2.per_level_std, rtol=1e-6)
    np.testing.assert_allclose(stats.per_tick_mean, stats2.per_tick_mean, rtol=1e-6)
    np.testing.assert_allclose(stats.per_tick_std, stats2.per_tick_std, rtol=1e-6)
