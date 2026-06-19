"""Parity: the opt-in native C++ engine must reproduce the pure-Python engine's PnL and trade count.

Skipped when the extension has not been built (``python -m finmamba3.backtester.engine_cpp.build``).
Each test drives one synthetic scoped timeline through both BacktestEngine (the reference) and
run_backtest_cpp and asserts the headline numbers match to float tolerance.
"""
# region imports
import pytest
from finmamba3.backtester import BacktestData, OrderBookLevel, OrderBookSnapshot, StoredBook, TickData
from finmamba3.backtester.engine import BacktestEngine
from finmamba3.backtester.engine_cpp import run_backtest_cpp
from finmamba3.backtester.engine_cpp import _engine as _native_engine
from finmamba3.backtester.strategy import MarketLifecycle, Settlement, Token
from finmamba3.eval.pnl_backtest import NaiveLagStrategy, WorldModelStrategy, _data_for_slugs
# endregion
_SLUG = "btc-updown-5m-0"
_PARAMS = {
    "edge_threshold": 0.05, "order_size": 50.0, "max_tte_frac": 1.0, "min_tte_frac": 0.0,
    "use_cusum": False, "cusum_threshold": 0.003, "sizing": "fixed", "calibration_temperature": 1.0,
    "use_predictability_gate": False, "min_book_depth": 0.0, "max_book_depth": 0.0, "ev_by_slug_ts": None,
}
pytestmark = pytest.mark.skipif(_native_engine is None, reason="native C++ engine not built")


def _stored(yes_mid):
    return StoredBook(
        yes_book=OrderBookSnapshot(
            bids=(OrderBookLevel(round(yes_mid - 0.01, 4), 100.0),),
            asks=(OrderBookLevel(round(yes_mid + 0.01, 4), 100.0),),
        ),
        no_book=OrderBookSnapshot(
            bids=(OrderBookLevel(round(0.99 - yes_mid, 4), 100.0),),
            asks=(OrderBookLevel(round(1.01 - yes_mid, 4), 100.0),),
        ),
        book_ts=0,
    )


def _scenario(yes_mid, chainlink_series, outcome):
    # Build a 1-market scoped BacktestData: a fresh book every tick (so it is never stale), the given
    # Chainlink path, and a market that settles to `outcome`. start_ts=0, end_ts=5 over 7 ticks.
    timeline = []
    for tick_index, spot in enumerate(chainlink_series):
        tick = TickData(ts_sec=tick_index, btc_mid=50_000.0, chainlink_btc=spot)
        tick.order_books[_SLUG] = _stored(yes_mid)
        tick.book_timestamps[_SLUG] = tick_index
        timeline.append(tick)
    bt = BacktestData(
        timeline=timeline,
        lifecycles=[MarketLifecycle(_SLUG, "5m", start_ts=0, end_ts=5)],
        settlements={_SLUG: Settlement(_SLUG, "5m", outcome, 0, 5, 50_000.0, 50_500.0)},
        start_ts=0, end_ts=len(chainlink_series) - 1,
    )
    return _data_for_slugs(bt, [_SLUG])


def test_cpp_matches_python_world_model_winning_yes_trade():
    bt = _scenario(0.40, [50_000.0] * 7, Token.YES)
    prob_by_slug_ts = {(_SLUG, tick_index): 0.90 for tick_index in range(7)}
    python_result = BacktestEngine(
        bt, WorldModelStrategy(prob_by_slug_ts, edge_threshold=0.05, order_size=50.0),
        starting_cash=10_000.0, snapshot_interval=1,
    ).run()
    cpp_result = run_backtest_cpp(bt, [_SLUG], prob_by_slug_ts, 0, _PARAMS, 10_000.0, 1)
    assert python_result.total_pnl > 0.0
    assert cpp_result.total_trades == python_result.total_trades
    assert abs(cpp_result.total_pnl - python_result.total_pnl) < 1e-6


def test_cpp_matches_python_world_model_no_trade_on_agreement():
    bt = _scenario(0.40, [50_000.0] * 7, Token.YES)
    # Model agrees with the 0.40 book, so neither engine should trade.
    prob_by_slug_ts = {(_SLUG, tick_index): 0.42 for tick_index in range(7)}
    python_result = BacktestEngine(
        bt, WorldModelStrategy(prob_by_slug_ts, edge_threshold=0.05, order_size=50.0),
        starting_cash=10_000.0, snapshot_interval=1,
    ).run()
    cpp_result = run_backtest_cpp(bt, [_SLUG], prob_by_slug_ts, 0, _PARAMS, 10_000.0, 1)
    assert python_result.total_trades == 0
    assert cpp_result.total_trades == 0
    assert abs(cpp_result.total_pnl - python_result.total_pnl) < 1e-6


def test_cpp_matches_python_naive_lag_on_spot_drift():
    chainlink_series = [50_000.0 * (1.0015 ** tick_index) for tick_index in range(7)]
    bt = _scenario(0.40, chainlink_series, Token.YES)
    python_result = BacktestEngine(
        bt, NaiveLagStrategy(cusum_threshold=0.003, order_size=50.0, max_tte_frac=1.0),
        starting_cash=10_000.0, snapshot_interval=1,
    ).run()
    cpp_result = run_backtest_cpp(bt, [_SLUG], {}, 1, _PARAMS, 10_000.0, 1)
    assert cpp_result.total_trades == python_result.total_trades
    assert abs(cpp_result.total_pnl - python_result.total_pnl) < 1e-6
