"""Smoke tests for the backtest CLI helpers.

Exercises the regime-split filtering and the run_backtest path with a tiny
synthetic timeline plus a fake policy. The full CLI main() requires a trained
world model checkpoint, which is out of scope for unit tests; that path is
exercised manually after a Phase A pretrain.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


try:
    from envs.polymarket_lob_env import PolymarketLOBEnv
except ModuleNotFoundError as exc:
    PolymarketLOBEnv = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from eval.backtest import run_backtest  # noqa: E402
from eval.run_backtest_cli import _filter_backtest_data  # noqa: E402
from lob.backtester.data_loader import BacktestData, TickData  # noqa: E402
from lob.backtester.strategy import (  # noqa: E402
    MarketLifecycle,
    OrderBookLevel,
    OrderBookSnapshot,
    Settlement,
    StoredBook,
    Token,
)


def _book(bid, ask):
    return OrderBookSnapshot(
        bids=(OrderBookLevel(float(bid), 100.0),),
        asks=(OrderBookLevel(float(ask), 100.0),),
    )


def _data():
    slug = "btc-updown-5m-0"
    ticks = []
    for ts, yes_bid, yes_ask in [(0, 0.58, 0.60), (1, 0.60, 0.62), (2, 1.0, 1.0), (3, 1.0, 1.0)]:
        tick = TickData(ts_sec=ts)
        tick.order_books[slug] = StoredBook(
            yes_book=_book(yes_bid, yes_ask),
            no_book=_book(1.0 - yes_ask, 1.0 - yes_bid),
            book_ts=ts,
        )
        tick.book_timestamps[slug] = ts
        tick.btc_mid = 50_000.0 + ts
        tick.chainlink_btc = 50_000.0 + ts
        ticks.append(tick)
    return BacktestData(
        timeline=ticks,
        lifecycles=[MarketLifecycle(slug, "5m", start_ts=0, end_ts=2)],
        settlements={
            slug: Settlement(slug, "5m", outcome=Token.YES, start_ts=0, end_ts=2)
        },
        start_ts=0,
        end_ts=3,
    )


class _NoopPolicy:
    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> int:
        return 0


def test_filter_backtest_data_none_keeps_all():
    bt = _data()
    bt2, desc = _filter_backtest_data(bt, "none")
    assert desc == "all"
    assert len(bt2.lifecycles) == 1


def test_filter_backtest_data_time_split_keeps_post_cutoff():
    bt = _data()
    bt2, desc = _filter_backtest_data(bt, "time:1")
    # The single market has end_ts=2, so it is in the post-cutoff (test) bucket.
    assert "time<" in desc
    assert len(bt2.lifecycles) == 1


@unittest.skipIf(PolymarketLOBEnv is None, f"Gymnasium unavailable: {IMPORT_ERROR}")
def test_run_backtest_returns_finite_metrics_with_noop_policy():
    env = PolymarketLOBEnv(_data(), initial_cash=100.0, max_markets=1)
    metrics = run_backtest(env, _NoopPolicy(), max_steps=4)
    assert np.isfinite(metrics.total_return)
    assert np.isfinite(metrics.sharpe)
    assert np.isfinite(metrics.max_drawdown)
    assert metrics.num_trades == 0
