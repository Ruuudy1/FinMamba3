"""Tests for the multi-metric FI-2010 generalization-gap evaluator.

Two layers: the feature-mask index math is a pure function and is checked exhaustively against
the FI-2010 flat layout, and the Student-t prediction-NLL scoring function is exercised on a CPU
world model (fake Mamba, studentt decoder) so the (sum_nll, count) contract and the channel
restriction are verified without a GPU. The full main() flow needs trained checkpoints and runs
in the A/B step, not here.
"""
# region imports
from __future__ import annotations
import sys
import types
import numpy as np
import pytest
import torch
from finmamba3.envs.fi2010_loader import (
    FI2010_FEATURE_DIM,
    FI2010_F_LEVEL,
    FI2010_F_TICK,
    FI2010_K_LEVELS,
)
from finmamba3.eval.eval_regime_generalization_fi2010 import _METRIC_SPEC, _feature_indices
# endregion
_LEVEL_WIDTH = FI2010_K_LEVELS * FI2010_F_LEVEL


def _install_fake_mamba() -> None:
    """Replace mamba_ssm with a linear stand-in so the world model builds on CPU."""
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
    _ensure_pytorch_warmup()
    _install_fake_mamba()


def _studentt_world_model():
    """A minimal CPU FI-2010 world model with a Student-t decoder for scoring tests."""
    from finmamba3.config import DotDict
    from finmamba3.models.world_model import WorldModel
    cfg = DotDict({
        "BasicSettings": {
            "ObsMode": "features", "FeatureDim": FI2010_FEATURE_DIM, "ReplayBufferOnGPU": False,
            "Use_amp": False, "Use_cg": False, "Compile": False, "NormClip": 8.0,
        },
        "JointTrainAgent": {
            "BatchLength": 8, "ImagineContextLength": 4, "ImagineBatchLength": 4,
            "RealityContextLength": 4, "SaveEverySteps": 1000, "SampleMaxSteps": 100,
        },
        "Models": {"WorldModel": {
            "dtype": torch.float32, "Backbone": "Mamba3", "Act": "SiLU",
            "CategoricalDim": 4, "ClassDim": 4, "HiddenStateDim": 32, "Optimiser": "Adam",
            "LRSchedule": "constant", "LRMinRatio": 0.1, "Dropout": 0.0, "Unimix_ratio": 0.01,
            "Weight_decay": 0.0, "Max_grad_norm": 100.0, "Warmup_steps": 1, "UseActionInput": False,
            "DirectionThresholds": None, "NaNGuardSteps": 0, "RepresentationLossWeight": 0.1,
            "FreeBits": 1.0, "Adam": {"LearningRate": 1e-4},
            "Encoder": {
                "Type": "lob", "K": FI2010_K_LEVELS, "FeatureDimLevel": FI2010_F_LEVEL,
                "FeatureDimTick": FI2010_F_TICK, "DModel": 32, "NumLayers": 1, "NumHeads": 4,
                "DimFeedforward": 64, "OutputFlattenDim": 64, "AggregateOnly": False,
                "GradientCheckpointing": False,
            },
            "Decoder": {
                "Kind": "studentt", "HiddenDim": 32, "NumLayers": 1, "NuInit": 5.0,
                "LearnableNu": True, "SizeWeight": 1.0, "LevelSizeIndices": [1, 3], "TickSizeIndices": [5],
            },
            "Reward": {"Enabled": False, "HiddenUnits": 32, "LayerNum": 1},
            "Termination": {"Enabled": False, "HiddenUnits": 32, "LayerNum": 1},
            "Transformer": {"FinalFeatureWidth": 4, "NumLayers": 1, "NumHeads": 2},
            "Mamba": {"n_layer": 1, "d_intermediate": 0, "ssm_cfg": {"d_state": 16}},
            "Mamba3": {
                "Enabled": True, "n_layer": 1, "is_mimo": True, "mimo_rank": 1, "d_state": 8,
                "headdim": 16, "chunk_size": 4, "is_outproj_norm": False, "rope_fraction": 0.0,
            },
            "Direction": {"Enabled": False, "NumClasses": 3, "Threshold": 0.01, "LossWeight": 0.5, "Dropout": 0.0},
            "Regime": {"Enabled": False},
            "EpisodicMemory": {"Enabled": False},
            "Hawkes": {"Enabled": False},
            "Settlement": {"Enabled": False},
        }},
    })
    return WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))


def _fi2010_sequence(n_events: int = 200):
    from finmamba3.envs.lob_features import LOBSequence, compute_basic_tick_features
    rng = np.random.default_rng(11)
    per_level = rng.uniform(0.1, 5.0, (n_events, FI2010_K_LEVELS, FI2010_F_LEVEL)).astype(np.float32)
    per_tick = compute_basic_tick_features(per_level)
    mid = 0.5 * (per_level[:, 0, 0] + per_level[:, 0, 2])
    return LOBSequence(
        market_slug="fi2010_test", per_level=per_level, per_tick=per_tick,
        midprice=mid.astype(np.float32), ts_sec=np.arange(n_events, dtype=np.int64), yes_outcome=None,
    )


# ===========================================================================
# 1. Feature-mask index math (pure function)
# ===========================================================================

def test_price_scale_mask_contains_per_level_price_indices():
    idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "price_scale")
    for k in range(FI2010_K_LEVELS):
        assert k * FI2010_F_LEVEL + 0 in idx, "ask_price channel of every level must be in the price mask."
        assert k * FI2010_F_LEVEL + 2 in idx, "bid_price channel of every level must be in the price mask."


def test_price_scale_mask_contains_tick_mid_and_microprice():
    idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "price_scale")
    assert _LEVEL_WIDTH + 0 in idx, "tick mid must be in the price/scale mask."
    assert _LEVEL_WIDTH + 4 in idx, "tick microprice must be in the price/scale mask."


def test_price_scale_mask_excludes_size_channels():
    idx = set(_feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "price_scale"))
    for k in range(FI2010_K_LEVELS):
        assert k * FI2010_F_LEVEL + 1 not in idx, "ask_size must not be in the price/scale mask."
        assert k * FI2010_F_LEVEL + 3 not in idx, "bid_size must not be in the price/scale mask."
    assert _LEVEL_WIDTH + 5 not in idx, "log_total_vol must not be in the price/scale mask."


def test_volume_mask_contains_per_level_size_indices_and_total_vol():
    idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "volume")
    for k in range(FI2010_K_LEVELS):
        assert k * FI2010_F_LEVEL + 1 in idx, "ask_size channel of every level must be in the volume mask."
        assert k * FI2010_F_LEVEL + 3 in idx, "bid_size channel of every level must be in the volume mask."
    assert _LEVEL_WIDTH + 5 in idx, "tick log_total_vol must be in the volume mask."


def test_price_and_volume_masks_are_disjoint():
    price = set(_feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "price_scale"))
    volume = set(_feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "volume"))
    assert price.isdisjoint(volume), "price/scale and volume masks must not share channels."


def test_mask_indices_in_range():
    for mask_name in ("price_scale", "volume"):
        idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, mask_name)
        assert min(idx) >= 0 and max(idx) < FI2010_FEATURE_DIM


def test_unknown_mask_name_raises():
    with pytest.raises(AssertionError):
        _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "bogus")


# ===========================================================================
# 2. Metric spec table
# ===========================================================================

def test_metric_spec_studentt_metrics_need_studentt_and_a_mask():
    for metric in ("studentt_nll", "volume_nll"):
        assert _METRIC_SPEC[metric]["needs_studentt"]
        assert _METRIC_SPEC[metric]["mask"] is not None
        assert not _METRIC_SPEC[metric]["higher_is_better"]


def test_metric_spec_direction_is_higher_is_better_without_studentt():
    spec = _METRIC_SPEC["direction_macro_f1"]
    assert spec["higher_is_better"]
    assert not spec["needs_studentt"]
    assert spec["mask"] is None


# ===========================================================================
# 3. Student-t prediction-NLL scoring on a CPU world model
# ===========================================================================

def test_prediction_nll_count_matches_mask_width_and_windows():
    from finmamba3.eval.compare_direction import world_model_prediction_nll
    wm = _studentt_world_model()
    seq = _fi2010_sequence(n_events=200)
    idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "price_scale")
    feature_indices = torch.tensor(idx, dtype=torch.long)
    windows_per_market, window_len = 16, 64
    sum_nll, count = world_model_prediction_nll(
        wm, seq, torch.device("cpu"), feature_indices,
        windows_per_market=windows_per_market, window_len=window_len,
    )
    # count = windows * (window_len - 1 next-tick targets) * masked channels.
    assert count == windows_per_market * (window_len - 1) * len(idx)
    assert np.isfinite(sum_nll)


def test_prediction_nll_volume_mask_has_fewer_channels_than_price():
    from finmamba3.eval.compare_direction import world_model_prediction_nll
    wm = _studentt_world_model()
    seq = _fi2010_sequence(n_events=160)
    price_idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "price_scale")
    volume_idx = _feature_indices(FI2010_K_LEVELS, FI2010_F_LEVEL, FI2010_F_TICK, "volume")
    _, price_count = world_model_prediction_nll(
        wm, seq, torch.device("cpu"), torch.tensor(price_idx, dtype=torch.long),
        windows_per_market=8, window_len=64,
    )
    _, volume_count = world_model_prediction_nll(
        wm, seq, torch.device("cpu"), torch.tensor(volume_idx, dtype=torch.long),
        windows_per_market=8, window_len=64,
    )
    assert volume_count < price_count, "volume mask (21 ch) is narrower than price/scale (22 ch)."


def test_prediction_nll_rejects_gaussian_decoder():
    from finmamba3.eval.compare_direction import world_model_prediction_nll
    wm = _studentt_world_model()
    wm.decoder_kind = "mse"  # Force the precondition off to confirm the guard fires.
    seq = _fi2010_sequence(n_events=80)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(AssertionError, match="Student-t decoder"):
        world_model_prediction_nll(wm, seq, torch.device("cpu"), idx, windows_per_market=2, window_len=64)
