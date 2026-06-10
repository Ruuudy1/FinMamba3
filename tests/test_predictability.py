"""Phase 0.7: the predictability estimators behave on known signals and rank a trending market
above a mean-reverting one, so the per-market split separates forecastable from random-walk windows.
"""
# region imports
import numpy as np
from finmamba3.backtester.data_loader import BacktestData, TickData
from finmamba3.backtester.strategy import MarketLifecycle
from finmamba3.eval.predictability import (
    efficiency_ratio,
    hurst_exponent,
    permutation_entropy,
    predictability_from_timeline,
    window_predictability_split,
)
# endregion


def _alternating_predictability_mid(window_len: int, num_windows: int) -> np.ndarray:
    """Mid stream whose even windows trend cleanly (ER near 1) and odd windows chop (ER near 0).

    The Efficiency Ratio of a monotone ramp is 1 and of a zig-zag is near 0, so the median ER split
    cleanly routes forecastable windows to the reference regime and random-walk windows to the shifted
    regime. A plain linear ramp across the whole stream would give every window ER 1 and not separate.
    """
    ramp = np.linspace(0.0, 1.0, window_len)
    oscillation = (-1.0) ** np.arange(window_len, dtype=np.float64)
    segments = []
    for w in range(num_windows):
        base = 100.0 + 10.0 * w
        segments.append(base + 5.0 * ramp if w % 2 == 0 else base + 0.5 * oscillation)
    return np.concatenate(segments)


def test_window_predictability_split_routes_trends_to_reference():
    window_len = 8
    mid = _alternating_predictability_mid(window_len, num_windows=6)
    result = window_predictability_split(mid, window_len=window_len, quantile=0.5)
    # The three trending (even-index) windows are the forecastable reference; the three chop windows shift.
    assert result.reference_windows == [(0, 8), (16, 24), (32, 40)]
    assert result.shifted_windows == [(8, 16), (24, 32), (40, 48)]
    assert result.description.startswith("win8-ER>=")


def test_window_predictability_split_tiles_without_overlap():
    window_len = 10
    mid = _alternating_predictability_mid(window_len, num_windows=4)
    result = window_predictability_split(mid, window_len=window_len, quantile=0.5)
    all_windows = sorted(result.reference_windows + result.shifted_windows)
    assert all_windows == [(0, 10), (10, 20), (20, 30), (30, 40)]


def test_window_predictability_split_fails_fast_on_short_stream():
    import pytest
    mid = np.linspace(100.0, 101.0, 12)
    with pytest.raises(AssertionError):
        window_predictability_split(mid, window_len=10, quantile=0.5)


def test_window_predictability_split_non_degenerate_on_random_walk():
    # A real-ish random walk must still produce two non-empty regimes at the median, so the Kaggle
    # eval never silently scores an empty regime.
    rng = np.random.default_rng(7)
    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.5, 4000))
    result = window_predictability_split(mid, window_len=128, quantile=0.5)
    assert len(result.reference_windows) >= 1
    assert len(result.shifted_windows) >= 1


def test_efficiency_ratio_trend_versus_zigzag():
    assert efficiency_ratio(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 1.0
    assert efficiency_ratio(np.array([1.0, 2.0, 1.0, 2.0, 1.0])) == 0.0


def test_permutation_entropy_monotone_versus_noise():
    monotone = np.arange(64, dtype=np.float64)
    assert permutation_entropy(monotone, order=4) == 0.0
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(2000)
    assert permutation_entropy(noise, order=4) > 0.9


def test_hurst_random_walk_near_half_and_finite_on_short_window():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.standard_normal(4000))
    h = hurst_exponent(walk)
    assert abs(h - 0.5) < 0.1
    short = hurst_exponent(np.array([1.0, 2.0, 1.5, 3.0, 2.0]))
    assert np.isfinite(short)


def _spot_timeline():
    # One BTC market trends (ER near 1), the other mean-reverts (ER near 0); both span disjoint windows.
    trend = [50_000.0 + 10.0 * i for i in range(40)]
    revert = [50_000.0 + (50.0 if i % 2 else -50.0) for i in range(40)]
    ticks = []
    for i in range(40):
        ticks.append(TickData(ts_sec=i, chainlink_btc=trend[i]))
    for i in range(40):
        ticks.append(TickData(ts_sec=100 + i, chainlink_btc=revert[i]))
    lifecycles = [
        MarketLifecycle("btc-updown-5m-0", "5m", start_ts=0, end_ts=39),
        MarketLifecycle("btc-updown-5m-100", "5m", start_ts=100, end_ts=139),
    ]
    return BacktestData(timeline=ticks, lifecycles=lifecycles, settlements={}, start_ts=0, end_ts=139)


def test_predictability_ranks_trend_above_reverting_market():
    bt = _spot_timeline()
    er_by_slug = predictability_from_timeline(bt.timeline, bt.lifecycles, metric="efficiency_ratio")
    assert er_by_slug["btc-updown-5m-0"] > er_by_slug["btc-updown-5m-100"]
    assert er_by_slug["btc-updown-5m-0"] > 0.9
