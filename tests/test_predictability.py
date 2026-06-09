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
)
# endregion


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
