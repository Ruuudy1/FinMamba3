# region imports
import math
import sys
import unittest
from pathlib import Path
try:
    from finmamba3.envs.lob_env import PolymarketLOBEnv
except ModuleNotFoundError as exc:
    PolymarketLOBEnv = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None
from finmamba3.backtester.data_loader import BacktestData, TickData  # noqa: E402
from finmamba3.backtester.strategy import (  # noqa: E402
    MarketLifecycle,
    OrderBookLevel,
    OrderBookSnapshot,
    Settlement,
    StoredBook,
    Token,
)
# endregion


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
    def test_settlement_calibrated_reward_adds_realized_profit_term(self):
        # Same trajectory under "default" and "settlement_calibrated"; the only reward
        # difference on the settling step is the realized-profit shaping term, which the
        # old getattr-on-Settlement path silently zeroed out (Settlement has no payoff field).
        default_env = PolymarketLOBEnv(
            _data(), initial_cash=100.0, max_markets=1, reward_kind="default"
        )
        calibrated_env = PolymarketLOBEnv(
            _data(), initial_cash=100.0, max_markets=1,
            reward_kind="settlement_calibrated", reward_settlement_weight=1.0,
        )
        default_env.reset()
        calibrated_env.reset()
        default_env.step(1)
        calibrated_env.step(1)
        _, default_reward, _, _, _ = default_env.step(0)
        _, calibrated_reward, _, _, calibrated_info = calibrated_env.step(0)
        self.assertEqual(len(calibrated_info["settlements"]), 1)
        # 10 YES shares bought at 0.62 redeem at $1.00 each: realized profit 10.0 - 6.2.
        expected_term = math.tanh((10.0 - 6.2) / calibrated_env.vol_scale)
        self.assertGreater(expected_term, 0.0)
        self.assertAlmostEqual(calibrated_reward - default_reward, expected_term, places=5)
if __name__ == "__main__":
    unittest.main()
