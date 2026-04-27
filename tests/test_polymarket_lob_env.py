import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from envs.polymarket_lob_env import PolymarketLOBEnv
except ModuleNotFoundError as exc:
    PolymarketLOBEnv = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

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


@unittest.skipIf(PolymarketLOBEnv is None, f"Gymnasium unavailable: {IMPORT_ERROR}")
class PolymarketLOBEnvTest(unittest.TestCase):
    def test_reset_step_latency_and_settlement(self):
        env = PolymarketLOBEnv(_data(), initial_cash=100.0, max_markets=1)
        obs, info = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertFalse(info["invalid_action"])

        obs, reward, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNotNone(info["fill"])
        self.assertEqual(info["fill"].timestamp, 1)
        self.assertAlmostEqual(info["fill"].avg_price, 0.62)
        self.assertTrue(math.isfinite(reward))

        _, _, _, _, info = env.step(0)
        self.assertEqual(len(info["settlements"]), 1)
        self.assertAlmostEqual(env.cash, 100.0 - 6.2 + 10.0)

    def test_invalid_sell_does_not_change_cash(self):
        env = PolymarketLOBEnv(_data(), initial_cash=100.0, max_markets=1)
        env.reset()
        sell_no_small = 1 + 3 * 3
        _, _, _, _, info = env.step(sell_no_small)
        self.assertTrue(info["invalid_action"])
        self.assertIsNone(info["fill"])
        self.assertAlmostEqual(env.cash, 100.0)


if __name__ == "__main__":
    unittest.main()
