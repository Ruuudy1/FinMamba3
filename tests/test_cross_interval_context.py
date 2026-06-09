"""Phase 0.4: the cross-interval context block links contemporaneous short-interval markets to
the hourly market that contains them, exposing the already-settled short outcomes as direct
evidence about the hourly direction.
"""
# region imports
import math
import numpy as np
from finmamba3.backtester.data_loader import BacktestData, TickData
from finmamba3.backtester.strategy import (
    MarketLifecycle,
    OrderBookLevel,
    OrderBookSnapshot,
    Settlement,
    StoredBook,
    Token,
)
from finmamba3.sequence_builder import CROSS_INTERVAL_WIDTH, cross_interval_context_block
# endregion
_HOURLY = "bitcoin-up-or-down-test"
_SHORT1 = "btc-updown-5m-0"
_SHORT2 = "btc-updown-5m-300"


def _book(mid):
    return StoredBook(
        yes_book=OrderBookSnapshot(
            bids=(OrderBookLevel(mid - 0.01, 100.0),),
            asks=(OrderBookLevel(mid + 0.01, 100.0),),
        ),
        no_book=OrderBookSnapshot(),
        book_ts=0,
    )


def _backtest_data(with_shorts: bool = True):
    books_by_ts = {50: {}, 250: {}, 450: {}, 650: {}}
    lifecycles = [MarketLifecycle(_HOURLY, "hourly", start_ts=0, end_ts=3600)]
    settlements = {}
    if with_shorts:
        books_by_ts = {50: {_SHORT1: _book(0.6)}, 250: {_SHORT1: _book(0.6)}, 450: {_SHORT2: _book(0.3)}, 650: {}}
        lifecycles += [
            MarketLifecycle(_SHORT1, "5m", start_ts=0, end_ts=300),
            MarketLifecycle(_SHORT2, "5m", start_ts=300, end_ts=600),
        ]
        settlements = {
            _SHORT1: Settlement(_SHORT1, "5m", Token.YES, 0, 300),
            _SHORT2: Settlement(_SHORT2, "5m", Token.NO, 300, 600),
        }
    timeline = []
    for ts, books in books_by_ts.items():
        tick = TickData(ts_sec=ts)
        for slug, stored in books.items():
            tick.order_books[slug] = stored
        timeline.append(tick)
    return BacktestData(timeline=timeline, lifecycles=lifecycles, settlements=settlements, start_ts=50, end_ts=650)


def test_context_tracks_active_mid_and_resolved_evidence():
    bt = _backtest_data(with_shorts=True)
    ts = np.array([50, 250, 450, 650], dtype=np.int64)
    block = cross_interval_context_block(bt, _HOURLY, ts)
    assert block.shape == (4, CROSS_INTERVAL_WIDTH)
    # Active mean short mid: short1 (0.60) while it is live, short2 (0.30) next, none live last (0.5 default).
    np.testing.assert_allclose(block[:, 0], [0.60, 0.60, 0.30, 0.50], atol=1e-6)
    # Resolved-YES fraction: nothing settled, nothing settled, short1=YES, then 1 YES of 2 settled.
    np.testing.assert_allclose(block[:, 1], [0.5, 0.5, 1.0, 0.5], atol=1e-6)
    # Resolved count grows monotonically as short markets expire inside the hour (log1p of the count).
    np.testing.assert_allclose(block[:, 2], [0.0, 0.0, math.log1p(1), math.log1p(2)], atol=1e-6)


def test_context_defaults_when_no_short_markets():
    bt = _backtest_data(with_shorts=False)
    ts = np.array([50, 250, 650], dtype=np.int64)
    block = cross_interval_context_block(bt, _HOURLY, ts)
    # With no contemporaneous short markets every channel falls back to its neutral default.
    np.testing.assert_allclose(block[:, 0], 0.5)
    np.testing.assert_allclose(block[:, 1], 0.5)
    np.testing.assert_allclose(block[:, 2], 0.0)
