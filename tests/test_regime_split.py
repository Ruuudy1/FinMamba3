"""Unit tests for the volatility regime split.

Covers realized_vol_from_timeline (per-market vol from the loaded books, with
forward-fill repeats collapsed), the volatility_split routing, and the CLI
filter that keeps the high-vol tail for the regime-shift A/B.
"""
# region imports
from __future__ import annotations
from finmamba3.backtester.data_loader import BacktestData, TickData
from finmamba3.backtester.strategy import (
    MarketLifecycle,
    OrderBookLevel,
    OrderBookSnapshot,
    StoredBook,
)
from finmamba3.eval.regime_split import realized_vol_from_timeline, volatility_split
from finmamba3.eval.run_backtest_cli import _filter_backtest_data
# endregion
_CALM = "btc-updown-5m-0"
_WILD = "btc-updown-5m-300"
_CALM_MIDS = [0.50, 0.50, 0.51, 0.50, 0.51]
_WILD_MIDS = [0.50, 0.70, 0.40, 0.80, 0.30]


def _book(mid):
    return OrderBookSnapshot(
        bids=(OrderBookLevel(mid - 0.01, 100.0),),
        asks=(OrderBookLevel(mid + 0.01, 100.0),),
    )


def _stored(mid, book_ts):
    return StoredBook(yes_book=_book(mid), no_book=_book(1.0 - mid), book_ts=book_ts)


def _timeline_two_markets():
    ticks = []
    for ts in range(len(_CALM_MIDS)):
        tick = TickData(ts_sec=ts)
        tick.order_books[_CALM] = _stored(_CALM_MIDS[ts], ts)
        tick.order_books[_WILD] = _stored(_WILD_MIDS[ts], ts)
        ticks.append(tick)
    return ticks


def _backtest_data():
    return BacktestData(
        timeline=_timeline_two_markets(),
        lifecycles=[
            MarketLifecycle(_CALM, "5m", start_ts=0, end_ts=300),
            MarketLifecycle(_WILD, "5m", start_ts=300, end_ts=600),
        ],
        settlements={},
        start_ts=0,
        end_ts=4,
    )


def test_realized_vol_ranks_volatile_market_higher():
    vol = realized_vol_from_timeline(_timeline_two_markets())
    assert vol[_WILD] > vol[_CALM]


def test_realized_vol_collapses_forward_filled_repeats():
    # Three ticks share book_ts 0 (forward-fill), then one update at book_ts 1.
    ticks = []
    for ts, (mid, book_ts) in enumerate([(0.5, 0), (0.5, 0), (0.5, 0), (0.6, 1)]):
        tick = TickData(ts_sec=ts)
        tick.order_books[_CALM] = _stored(mid, book_ts)
        ticks.append(tick)
    vol = realized_vol_from_timeline(ticks)
    # Only two distinct snapshots (0.5 -> 0.6) survive, so one 20% return, std 0.
    assert vol[_CALM] == 0.0


def test_realized_vol_omits_single_observation_markets():
    tick = TickData(ts_sec=0)
    tick.order_books[_CALM] = _stored(0.5, 0)
    vol = realized_vol_from_timeline([tick])
    assert _CALM not in vol


def test_volatility_split_routes_high_vol_to_test():
    bt = _backtest_data()
    realized_vol = realized_vol_from_timeline(bt.timeline)
    result = volatility_split(bt.lifecycles, realized_vol=realized_vol, quantile=0.5)
    test_slugs = [m.market_slug for m in result.test_markets]
    train_slugs = [m.market_slug for m in result.train_markets]
    assert test_slugs == [_WILD]
    assert train_slugs == [_CALM]


def test_filter_backtest_data_volatility_keeps_high_vol():
    bt = _backtest_data()
    bt2, desc = _filter_backtest_data(bt, "volatility:0.5")
    kept = [m.market_slug for m in bt2.lifecycles]
    assert kept == [_WILD]
    assert desc.startswith("vol>")
