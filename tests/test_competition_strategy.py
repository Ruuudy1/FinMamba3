# region imports
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envs.lob_features import F_TICK_BINARY, NormalizationStats  # noqa: E402
from eval.competition_strategy import FinDramaCompetitionStrategy  # noqa: E402
from lob.backtester.strategy import (  # noqa: E402
    MarketState,
    MarketView,
    OrderBookLevel,
    OrderBookSnapshot,
)
# endregion


def _make_view(slug, mid, end_ts):
    half_spread = 0.01
    bids = tuple(OrderBookLevel(round(mid - half_spread - 0.001 * k, 4), 100.0 + k) for k in range(10))
    asks = tuple(OrderBookLevel(round(mid + half_spread + 0.001 * k, 4), 100.0 + k) for k in range(10))
    book = OrderBookSnapshot(bids=bids, asks=asks)
    return MarketView(
        market_slug=slug, interval="5m", start_ts=0, end_ts=end_ts,
        time_remaining_s=float(end_ts), time_remaining_frac=0.5,
        yes_book=book, no_book=book,
    )


class CompetitionStrategyFeatureTest(unittest.TestCase):
    def test_feature_vector_is_100_dim_and_finite(self):
        stats = NormalizationStats(
            per_level_mean=np.zeros(8, dtype=np.float32),
            per_level_std=np.ones(8, dtype=np.float32),
            per_tick_mean=np.zeros(F_TICK_BINARY, dtype=np.float32),
            per_tick_std=np.ones(F_TICK_BINARY, dtype=np.float32),
            clip_value=8.0,
        )
        strategy = FinDramaCompetitionStrategy(
            world_model=SimpleNamespace(direction_head=object()),
            stats=stats, mid_index=80, device="cpu",
        )
        slug = "btc-updown-5m-0"
        vector = None
        for t in range(40):
            mid = 0.5 + 0.001 * float(np.sin(t / 5.0))
            view = _make_view(slug, mid, end_ts=10_000)
            state = MarketState(timestamp=1000 + t, timestamp_utc="", markets={slug: view})
            vector = strategy._feature_vector(slug, view, state.timestamp)
        self.assertEqual(vector.shape, (80 + F_TICK_BINARY,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertLessEqual(float(np.abs(vector).max()), 8.0 + 1e-4)
    def test_missing_direction_head_raises(self):
        stats = NormalizationStats(
            per_level_mean=np.zeros(8, dtype=np.float32),
            per_level_std=np.ones(8, dtype=np.float32),
            per_tick_mean=np.zeros(F_TICK_BINARY, dtype=np.float32),
            per_tick_std=np.ones(F_TICK_BINARY, dtype=np.float32),
            clip_value=8.0,
        )
        with self.assertRaises(ValueError):
            FinDramaCompetitionStrategy(
                world_model=SimpleNamespace(direction_head=None),
                stats=stats, mid_index=80, device="cpu",
            )
if __name__ == "__main__":
    unittest.main()
