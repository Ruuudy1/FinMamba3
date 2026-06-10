"""Unit tests for the FI-2010 fixed-length window volatility split.

Model-free: covers the routing of high-vol windows into the shifted regime, the
non-overlapping contiguous tiling of the stream, the fail-fast on a too-short stream,
and the sub-sequence slicing the evaluator feeds the scoring path.
"""
# region imports
from __future__ import annotations
import numpy as np
import pytest
from finmamba3.envs.lob_features import LOBSequence
from finmamba3.eval.eval_regime_generalization_fi2010 import _slice_sequence
from finmamba3.eval.regime_split import window_bounds, window_volatility_split
# endregion


def test_window_bounds_tiles_exactly():
    assert window_bounds(40, 10) == [(0, 10), (10, 20), (20, 30), (30, 40)]


def test_window_bounds_drops_ragged_tail():
    assert window_bounds(45, 10) == [(0, 10), (10, 20), (20, 30), (30, 40)]


def test_window_bounds_fails_fast_on_single_window():
    with pytest.raises(AssertionError):
        window_bounds(12, 10)


def _alternating_vol_mid(window_len: int, num_windows: int) -> np.ndarray:
    """Mid stream whose even windows are calm and odd windows are wild.

    Each window oscillates with an amplitude set by the window's index parity, so the
    within-window first-difference std scales with the amplitude and cleanly separates the
    halves at the median. A plain linear ramp would not: its first difference is constant,
    so its std is zero regardless of step size.
    """
    segments = []
    oscillation = (-1.0) ** np.arange(window_len, dtype=np.float64)
    for w in range(num_windows):
        amplitude = 0.001 if w % 2 == 0 else 1.0
        base = 100.0 + 10.0 * w
        segments.append(base + amplitude * oscillation)
    return np.concatenate(segments)


def test_split_routes_wild_windows_to_the_high_vol_regime():
    window_len = 8
    mid = _alternating_vol_mid(window_len, num_windows=6)
    result = window_volatility_split(mid, window_len=window_len, quantile=0.5)
    # The three calm (even-index) windows are the low-vol reference; the three wild ones shift.
    assert result.low_vol_windows == [(0, 8), (16, 24), (32, 40)]
    assert result.high_vol_windows == [(8, 16), (24, 32), (40, 48)]
    assert result.description.startswith("win8-vol>")


def test_split_tiles_the_stream_without_overlap_or_gaps():
    window_len = 10
    mid = _alternating_vol_mid(window_len, num_windows=4)
    result = window_volatility_split(mid, window_len=window_len, quantile=0.5)
    all_windows = sorted(result.low_vol_windows + result.high_vol_windows)
    # Every window is window_len wide and starts where the previous ended: a clean tiling.
    assert all_windows == [(0, 10), (10, 20), (20, 30), (30, 40)]


def test_split_drops_the_ragged_tail_shorter_than_one_window():
    window_len = 10
    mid = _alternating_vol_mid(window_len, num_windows=3)
    # Append five trailing events that cannot fill a fourth window; they must be excluded.
    mid = np.concatenate([mid, np.linspace(200.0, 201.0, 5)])
    result = window_volatility_split(mid, window_len=window_len, quantile=0.5)
    covered = result.low_vol_windows + result.high_vol_windows
    assert max(end for _, end in covered) == 30


def test_split_fails_fast_when_stream_has_fewer_than_two_windows():
    mid = np.linspace(100.0, 101.0, 12)
    with pytest.raises(AssertionError):
        window_volatility_split(mid, window_len=10, quantile=0.5)


def test_slice_sequence_keeps_rows_aligned_and_drops_outcome():
    T = 40
    seq = LOBSequence(
        market_slug="fi2010_h10",
        per_level=np.arange(T * 10 * 4, dtype=np.float32).reshape(T, 10, 4),
        per_tick=np.arange(T * 6, dtype=np.float32).reshape(T, 6),
        midprice=np.arange(T, dtype=np.float32),
        ts_sec=np.arange(T, dtype=np.int64),
        yes_outcome=None,
    )
    sub = _slice_sequence(seq, 8, 16)
    assert sub.per_level.shape == (8, 10, 4)
    assert sub.per_tick.shape == (8, 6)
    assert sub.midprice.tolist() == list(range(8, 16))
    assert sub.to_flat().shape == (8, 46)
    assert sub.yes_outcome is None
