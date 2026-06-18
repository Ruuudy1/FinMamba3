"""Tests for the Kaggle crypto spot LOB loader: schema, feature math, and direction labels."""
# region imports
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas
import pytest
from finmamba3.envs.kaggle_lob_loader import (
    KAGGLE_BOOK_DEPTH,
    KAGGLE_F_LEVEL,
    KAGGLE_F_TICK,
    KAGGLE_FEATURE_DIM,
    KAGGLE_K_LEVELS,
    KAGGLE_LEVELS_PER_SIDE,
    KaggleLOBSequence,
    _direction_labels,
    _encode_side,
    _rolling_std,
    load_kaggle_lob,
    max_rows_for_hours,
)
from finmamba3.envs.lob_features import (
    apply_normalization,
    fit_normalization,
    LEVEL_FEATURE_NAMES,
    TICK_FEATURE_NAMES,
)
# endregion

# A controlled midprice path so direction labels can be hand-verified.
_MID_PATH = np.array([100.0, 101.0, 100.5, 100.5, 102.0, 101.5, 103.0], dtype=np.float64)


def _synthetic_kaggle_frame(mid: np.ndarray, seed: int = 0) -> pandas.DataFrame:
    """Build a synthetic frame carrying every column the loader reads via usecols."""
    rng = np.random.default_rng(seed)
    num_ticks = mid.shape[0]
    column_by_name = {"midpoint": mid, "spread": np.full(num_ticks, 0.02)}
    column_by_name["buys"] = rng.uniform(1.0e3, 1.0e4, num_ticks)
    column_by_name["sells"] = rng.uniform(1.0e3, 1.0e4, num_ticks)
    for k in range(KAGGLE_LEVELS_PER_SIDE):
        column_by_name[f"bids_distance_{k}"] = -1.0e-4 * (k + 1) * np.ones(num_ticks)
        column_by_name[f"asks_distance_{k}"] = 1.0e-4 * (k + 1) * np.ones(num_ticks)
    for k in range(KAGGLE_BOOK_DEPTH):
        column_by_name[f"bids_notional_{k}"] = rng.uniform(100.0, 1000.0, num_ticks)
        column_by_name[f"asks_notional_{k}"] = rng.uniform(100.0, 1000.0, num_ticks)
        column_by_name[f"bids_limit_notional_{k}"] = np.zeros(num_ticks)
        column_by_name[f"asks_limit_notional_{k}"] = np.zeros(num_ticks)
        column_by_name[f"bids_market_notional_{k}"] = np.zeros(num_ticks)
        column_by_name[f"asks_market_notional_{k}"] = np.zeros(num_ticks)
    column_by_name["bids_limit_notional_0"][1::2] = 10.0
    column_by_name["asks_market_notional_0"][2::2] = 5.0
    column_by_name["asks_limit_notional_1"][3::2] = 7.0
    column_by_name["bids_market_notional_1"][4::2] = 3.0
    return pandas.DataFrame(column_by_name)


@pytest.fixture
def kaggle_csv(tmp_path) -> Path:
    """Write a synthetic BTC 1min CSV with the real file's leading index column."""
    frame = _synthetic_kaggle_frame(_MID_PATH)
    path = tmp_path / "BTC_1min.csv"
    frame.to_csv(path, index=True)
    return path


# ===========================================================================
# 1. Schema constants
# ===========================================================================

def test_feature_dim_is_94():
    assert KAGGLE_FEATURE_DIM == 94, "Kaggle flat dim must be 10 * 8 + 14 = 94 (Polymarket base schema)."


def test_constants_match_polymarket_schema():
    assert KAGGLE_K_LEVELS == 10
    assert KAGGLE_F_LEVEL == 8
    assert KAGGLE_F_TICK == 14
    assert KAGGLE_LEVELS_PER_SIDE == 5


def test_schema_reuses_lob_feature_names():
    # The loader must not invent a per-level or per-tick schema; the widths come from the shared names.
    assert len(LEVEL_FEATURE_NAMES) == KAGGLE_F_LEVEL
    assert len(TICK_FEATURE_NAMES) == KAGGLE_F_TICK


# ===========================================================================
# 2. Load shapes and finiteness
# ===========================================================================

def test_load_returns_kaggle_sequence_type(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert isinstance(bundle, KaggleLOBSequence)
    assert bundle.asset == "BTC" and bundle.resolution == "1min"


def test_load_per_level_shape(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert bundle.sequence.per_level.shape == (_MID_PATH.shape[0], KAGGLE_K_LEVELS, KAGGLE_F_LEVEL)


def test_load_per_tick_shape(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert bundle.sequence.per_tick.shape == (_MID_PATH.shape[0], KAGGLE_F_TICK)


def test_load_flat_dim_is_94(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert bundle.sequence.to_flat().shape == (_MID_PATH.shape[0], KAGGLE_FEATURE_DIM)


def test_load_flat_all_finite(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert np.isfinite(bundle.sequence.to_flat()).all()


def test_load_event_counts_shape_and_values(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    counts = bundle.sequence.event_counts
    assert counts.shape == (_MID_PATH.shape[0], 2)
    assert counts.dtype == np.float32
    assert counts[0, 0] == 0.0
    assert counts[1, 0] == 1.0
    assert counts[2, 0] == 1.0
    assert counts[3, 1] == 1.0
    assert counts[4, 1] == 1.0


def test_load_per_level_dtype_float32(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert bundle.sequence.per_level.dtype == np.float32
    assert bundle.sequence.per_tick.dtype == np.float32


def test_load_market_slug(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert bundle.sequence.market_slug == "kaggle_btc_1min"


# ===========================================================================
# 3. Feature semantics
# ===========================================================================

def test_mid_channel_equals_midpoint(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    np.testing.assert_allclose(bundle.sequence.per_tick[:, 0], _MID_PATH.astype(np.float32), rtol=1e-5)
    np.testing.assert_allclose(bundle.sequence.midprice, _MID_PATH.astype(np.float32), rtol=1e-5)


def test_side_indicator_bids_then_asks(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    side = bundle.sequence.per_level[:, :, 6]
    assert (side[:, :KAGGLE_LEVELS_PER_SIDE] == 1.0).all(), "Rows 0-4 must be the bid side (+1)."
    assert (side[:, KAGGLE_LEVELS_PER_SIDE:] == -1.0).all(), "Rows 5-9 must be the ask side (-1)."


def test_relative_price_sign(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    rel_price = bundle.sequence.per_level[:, :, 0]
    assert (rel_price[:, :KAGGLE_LEVELS_PER_SIDE] < 0.0).all(), "Bid relative prices sit below mid."
    assert (rel_price[:, KAGGLE_LEVELS_PER_SIDE:] > 0.0).all(), "Ask relative prices sit above mid."


def test_staleness_channel_is_zero(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert (bundle.sequence.per_level[:, :, 7] == 0.0).all(), "Kaggle snapshots carry no book staleness."


def test_level_index_channel_is_symmetric(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    expected = np.linspace(-1.0, 1.0, KAGGLE_LEVELS_PER_SIDE, dtype=np.float32)
    np.testing.assert_allclose(bundle.sequence.per_level[0, :KAGGLE_LEVELS_PER_SIDE, 5], expected)
    np.testing.assert_allclose(bundle.sequence.per_level[0, KAGGLE_LEVELS_PER_SIDE:, 5], expected)


# ===========================================================================
# 4. Direction labels
# ===========================================================================

def test_direction_labels_match_hand_computed(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min", flat_threshold=0.0)
    # mid = [100, 101, 100.5, 100.5, 102, 101.5, 103] -> up, down, flat, up, down, up, flat-filled.
    expected = np.array([2, 0, 1, 2, 0, 2, 1], dtype=np.int64)
    np.testing.assert_array_equal(bundle.direction_labels, expected)


def test_direction_labels_threshold_buckets_small_moves_flat():
    mid = np.array([100.0, 100.05, 99.97, 100.02, 100.0], dtype=np.float64)
    labels = _direction_labels(mid, flat_threshold=0.01)
    assert (labels == 1).all(), "A 1% flat band must bucket sub-1% moves as flat."


def test_direction_labels_length_matches(kaggle_csv):
    bundle = load_kaggle_lob(kaggle_csv, "BTC", "1min")
    assert bundle.direction_labels.shape == (_MID_PATH.shape[0],)
    assert bundle.direction_labels.dtype == np.int64


# ===========================================================================
# 5. Helper functions
# ===========================================================================

def test_rolling_std_first_is_zero_and_matches_numpy():
    series = np.array([1.0, 3.0, 2.0, 8.0, 5.0], dtype=np.float64)
    rolling = _rolling_std(series, window=3)
    assert rolling[0] == 0.0
    np.testing.assert_allclose(rolling[2], np.std(series[0:3]), rtol=1e-6)
    np.testing.assert_allclose(rolling[4], np.std(series[2:5]), rtol=1e-6)


def test_encode_side_volume_share_is_monotone_to_one():
    distance = (-1.0e-4 * np.arange(1, KAGGLE_LEVELS_PER_SIDE + 1))[None, :]
    notional = np.full((1, KAGGLE_LEVELS_PER_SIDE), 100.0)
    tokens = _encode_side(distance, notional, 1.0)
    volume_share = tokens[0, :, 3]
    assert volume_share[-1] == pytest.approx(1.0)
    assert (np.diff(volume_share) >= 0.0).all()


def test_encode_side_gap_first_level_equals_distance():
    distance = np.array([[-1.0e-4, -3.0e-4, -6.0e-4, -1.0e-3, -1.5e-3]])
    notional = np.full((1, KAGGLE_LEVELS_PER_SIDE), 100.0)
    tokens = _encode_side(distance, notional, 1.0)
    assert tokens[0, 0, 4] == pytest.approx(-1.0e-4), "Level-0 price gap is the distance from mid."
    assert tokens[0, 1, 4] == pytest.approx(-2.0e-4), "Level-1 gap is the step from level 0."


def test_max_rows_for_hours():
    assert max_rows_for_hours(1.0, "1min") == 60
    assert max_rows_for_hours(2.0, "5min") == 24
    assert max_rows_for_hours(0.5, "1sec") == 1800
    assert max_rows_for_hours(None, "1min") is None
    assert max_rows_for_hours(0.0, "1min") is None


def test_max_rows_for_hours_invalid_resolution_raises():
    with pytest.raises(ValueError, match="resolution"):
        max_rows_for_hours(1.0, "2min")


# ===========================================================================
# 6. Validation and slicing
# ===========================================================================

def test_invalid_asset_raises(kaggle_csv):
    with pytest.raises(ValueError, match="asset"):
        load_kaggle_lob(kaggle_csv, "DOGE", "1min")


def test_invalid_resolution_raises(kaggle_csv):
    with pytest.raises(ValueError, match="resolution"):
        load_kaggle_lob(kaggle_csv, "BTC", "2min")


def test_non_positive_midpoint_raises(tmp_path):
    mid = np.array([100.0, -1.0, 101.0, 102.0], dtype=np.float64)
    frame = _synthetic_kaggle_frame(mid)
    path = tmp_path / "BTC_1min.csv"
    frame.to_csv(path, index=True)
    with pytest.raises(ValueError, match="midpoint"):
        load_kaggle_lob(path, "BTC", "1min")


def test_max_rows_caps_loaded_ticks(tmp_path):
    mid = np.linspace(100.0, 110.0, 50, dtype=np.float64)
    frame = _synthetic_kaggle_frame(mid)
    path = tmp_path / "BTC_1min.csv"
    frame.to_csv(path, index=True)
    bundle = load_kaggle_lob(path, "BTC", "1min", max_rows=10)
    assert bundle.sequence.per_level.shape[0] == 10


def test_dropna_removes_gap_rows(tmp_path):
    mid = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    frame = _synthetic_kaggle_frame(mid)
    frame.loc[2, "bids_notional_0"] = np.nan
    path = tmp_path / "BTC_1min.csv"
    frame.to_csv(path, index=True)
    bundle = load_kaggle_lob(path, "BTC", "1min")
    assert bundle.sequence.per_level.shape[0] == 4, "The NaN snapshot must be dropped."


# ===========================================================================
# 7. End-to-end normalization (loader -> fit -> apply -> within clip)
# ===========================================================================

def test_e2e_normalization_within_clip(tmp_path):
    mid = 100.0 + np.cumsum(np.random.default_rng(1).normal(0.0, 0.5, 400))
    mid = np.clip(mid, 1.0, None)
    frame = _synthetic_kaggle_frame(mid, seed=2)
    path = tmp_path / "BTC_1min.csv"
    frame.to_csv(path, index=True)
    bundle = load_kaggle_lob(path, "BTC", "1min")
    stats = fit_normalization(bundle.sequence, clip_value=8.0)
    norm = apply_normalization(bundle.sequence, stats)
    flat = norm.to_flat()
    assert flat.shape == (mid.shape[0], KAGGLE_FEATURE_DIM)
    assert np.isfinite(flat).all()
    assert float(np.abs(flat).max()) <= 8.0 + 1e-4


# ===========================================================================
# 8. build_kaggle_sequences chronological train/val split
# ===========================================================================

def _write_kaggle_data_dir(tmp_path, num_rows: int = 300) -> Path:
    mid = 100.0 + np.cumsum(np.random.default_rng(3).normal(0.0, 0.4, num_rows))
    mid = np.clip(mid, 1.0, None)
    frame = _synthetic_kaggle_frame(mid, seed=4)
    frame.to_csv(tmp_path / "BTC_1min.csv", index=True)
    return tmp_path


def test_build_kaggle_split_shapes(tmp_path):
    from finmamba3.sequence_builder import build_kaggle_sequences
    data_dir = _write_kaggle_data_dir(tmp_path, num_rows=300)
    norm_path = tmp_path / "kaggle_norm.json"
    train_seq, slug, _stats = build_kaggle_sequences(
        data_dir, asset="BTC", resolution="1min", split="train",
        hours_train=2.0, hours_val=1.0, norm_path=norm_path, fit_stats=True, norm_clip=8.0,
    )
    val_seq, _, _ = build_kaggle_sequences(
        data_dir, asset="BTC", resolution="1min", split="validation",
        hours_train=2.0, hours_val=1.0, norm_path=norm_path, fit_stats=False, norm_clip=8.0,
    )
    assert train_seq.per_level.shape[0] == 120, "2h of 1min bars is 120 train ticks."
    assert val_seq.per_level.shape[0] == 60, "The next 1h is 60 val ticks carved from the same stream."
    assert slug == "kaggle_btc_1min"


def test_build_kaggle_val_is_chronological_tail(tmp_path):
    from finmamba3.sequence_builder import build_kaggle_sequences
    from finmamba3.envs.kaggle_lob_loader import load_kaggle_lob
    data_dir = _write_kaggle_data_dir(tmp_path, num_rows=300)
    norm_path = tmp_path / "kaggle_norm.json"
    build_kaggle_sequences(
        data_dir, asset="BTC", resolution="1min", split="train",
        hours_train=2.0, hours_val=1.0, norm_path=norm_path, fit_stats=True, norm_clip=8.0,
    )
    val_seq, _, _ = build_kaggle_sequences(
        data_dir, asset="BTC", resolution="1min", split="validation",
        hours_train=2.0, hours_val=1.0, norm_path=norm_path, fit_stats=False, norm_clip=8.0,
    )
    full = load_kaggle_lob(data_dir / "BTC_1min.csv", "BTC", "1min")
    # The val midprice (raw) must equal the stream tail starting one train-length in, proving no leakage.
    np.testing.assert_allclose(val_seq.midprice, full.sequence.midprice[120:180], rtol=1e-5)


def test_build_kaggle_validation_reuses_train_normalization(tmp_path):
    from finmamba3.sequence_builder import build_kaggle_sequences
    data_dir = _write_kaggle_data_dir(tmp_path, num_rows=300)
    norm_path = tmp_path / "kaggle_norm.json"
    build_kaggle_sequences(
        data_dir, asset="BTC", resolution="1min", split="train",
        hours_train=2.0, hours_val=1.0, norm_path=norm_path, fit_stats=True, norm_clip=8.0,
    )
    assert norm_path.exists(), "The train split must fit and save normalization stats."
    val_seq, _, _ = build_kaggle_sequences(
        data_dir, asset="BTC", resolution="1min", split="validation",
        hours_train=2.0, hours_val=1.0, norm_path=norm_path, fit_stats=False, norm_clip=8.0,
    )
    assert float(np.abs(val_seq.to_flat()).max()) <= 8.0 + 1e-4


def test_build_kaggle_rejects_oversized_train_slice(tmp_path):
    from finmamba3.sequence_builder import build_kaggle_sequences
    data_dir = _write_kaggle_data_dir(tmp_path, num_rows=100)
    norm_path = tmp_path / "kaggle_norm.json"
    with pytest.raises(RuntimeError, match="validation tail"):
        build_kaggle_sequences(
            data_dir, asset="BTC", resolution="1min", split="validation",
            hours_train=5.0, hours_val=1.0, norm_path=norm_path, fit_stats=False, norm_clip=8.0,
        )
