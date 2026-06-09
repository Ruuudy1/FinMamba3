# region imports
import sys
import unittest
from pathlib import Path
import numpy as np
from finmamba3.backtester import OrderBookLevel, OrderBookSnapshot, StoredBook, TickData
from finmamba3.envs.lob_features import (  # noqa: E402
    F_LEVEL,
    F_TICK,
    F_TICK_BINARY,
    F_TICK_BINARY_SPOT,
    F_TICK_SPOT,
    K_LEVELS,
    LOBSequence,
    NormalizationStats,
    SPOT_BLOCK_WIDTH,
    append_binary_market_features,
    apply_normalization,
    extract_features,
    fit_normalization,
    normalized_feature_diagnostics,
    tick_feature_names_for_width,
)
# endregion


def _spot_timeline(asset: str, attr: str, start_ts: int, end_ts: int, n: int, rising: bool):
    # A tiny single-market timeline whose Chainlink price for the chosen asset rises linearly,
    # so the spot block's signed distance is a known monotone ramp the test can check exactly.
    slug = "btc-updown-5m-1000"
    ticks = []
    for i in range(n):
        ts = start_ts + i
        spot = 50_000.0 * (1.0 + 0.0001 * i) if rising else 50_000.0
        yes_book = OrderBookSnapshot(
            bids=(OrderBookLevel(0.49, 100.0), OrderBookLevel(0.48, 50.0)),
            asks=(OrderBookLevel(0.51, 100.0), OrderBookLevel(0.52, 50.0)),
        )
        tick = TickData(ts_sec=ts, **{attr: spot})
        tick.order_books[slug] = StoredBook(yes_book=yes_book, no_book=OrderBookSnapshot(), book_ts=ts)
        tick.book_timestamps[slug] = ts
        ticks.append(tick)
    return ticks, slug


class LOBFeatureNormalizationTest(unittest.TestCase):
    def test_normalization_is_finite_clipped_and_preserves_token_metadata(self):
        t = 8
        per_level = np.zeros((t, K_LEVELS, F_LEVEL), dtype=np.float32)
        per_level[..., 0] = 0.001
        per_level[..., 1] = 2.0
        per_level[..., 2] = 3.0
        per_level[..., 3] = 0.5
        per_level[..., 4] = 0.01
        per_level[..., 5] = np.linspace(-1.0, 1.0, K_LEVELS, dtype=np.float32)
        per_level[..., 6] = np.where(np.arange(K_LEVELS) < 5, 1.0, -1.0)
        per_level[..., 7] = 6.0
        per_tick = np.zeros((t, F_TICK), dtype=np.float32)
        per_tick[:, 0] = 0.5
        per_tick[:, 1] = 0.02
        per_tick[:, 2] = np.log1p(0.02)
        per_tick[:, 3] = 0.5
        per_tick[:, 4] = 0.5
        per_tick[:, 6] = 4.0
        per_tick[:, 7] = 4.0
        per_tick[-1, 11] = 10_000.0
        per_tick[-1, 12] = 10_000.0
        seq = LOBSequence(
            market_slug="btc-updown-5m-0",
            per_level=per_level,
            per_tick=per_tick,
            midprice=np.full(t, 0.5, dtype=np.float32),
            ts_sec=np.arange(t, dtype=np.int64),
        )
        stats = fit_normalization(seq, clip_value=4.0)
        norm = apply_normalization(seq, stats)
        diag = normalized_feature_diagnostics(norm, stats.clip_value)
        self.assertTrue(diag["finite"])
        self.assertTrue(diag["within_clip"])
        self.assertLessEqual(diag["max_abs"], 4.0 + 1e-5)
        np.testing.assert_allclose(norm.per_level[..., 5], per_level[..., 5])
        np.testing.assert_allclose(norm.per_level[..., 6], per_level[..., 6])


class BinaryMarketFeatureTest(unittest.TestCase):
    def test_append_binary_features_shapes_and_values(self):
        num_ticks = 12
        per_tick = np.zeros((num_ticks, F_TICK), dtype=np.float32)
        per_tick[:, 6] = 4.0
        per_tick[:, 7] = 5.0
        midprice = np.linspace(0.2, 0.8, num_ticks).astype(np.float32)
        out = append_binary_market_features(per_tick, midprice, vol_window=5)
        self.assertEqual(out.shape, (num_ticks, F_TICK_BINARY))
        mid = np.clip(midprice.astype(np.float64), 1e-6, 1.0 - 1e-6)
        expected_boundary = np.minimum(mid, 1.0 - mid)
        np.testing.assert_allclose(out[:, F_TICK + 0], expected_boundary, rtol=1e-4, atol=1e-6)
        logit_mid = np.log(mid / (1.0 - mid))
        expected_velocity = np.zeros_like(logit_mid)
        expected_velocity[1:] = logit_mid[1:] - logit_mid[:-1]
        np.testing.assert_allclose(out[:, F_TICK + 2], expected_velocity, rtol=1e-4, atol=1e-5)
        self.assertEqual(float(out[0, F_TICK + 2]), 0.0)
        self.assertEqual(float(out[0, F_TICK + 3]), 0.0)
        self.assertTrue(np.isfinite(out).all())
    def test_binary_features_normalization_roundtrip(self):
        num_ticks = 16
        per_level = np.zeros((num_ticks, K_LEVELS, F_LEVEL), dtype=np.float32)
        per_level[..., 1] = 2.0
        per_level[..., 5] = np.linspace(-1.0, 1.0, K_LEVELS, dtype=np.float32)
        per_level[..., 6] = np.where(np.arange(K_LEVELS) < 5, 1.0, -1.0)
        base_tick = np.zeros((num_ticks, F_TICK), dtype=np.float32)
        base_tick[:, 0] = 0.5
        base_tick[:, 6] = 4.0
        base_tick[:, 7] = 4.0
        midprice = np.full(num_ticks, 0.5, dtype=np.float32)
        per_tick = append_binary_market_features(base_tick, midprice, vol_window=8)
        seq = LOBSequence(
            market_slug="btc-updown-5m-0",
            per_level=per_level,
            per_tick=per_tick,
            midprice=midprice,
            ts_sec=np.arange(num_ticks, dtype=np.int64),
        )
        stats = fit_normalization(seq, clip_value=8.0)
        self.assertEqual(stats.per_tick_mean.shape[0], F_TICK_BINARY)
        norm = apply_normalization(seq, stats)
        diag = normalized_feature_diagnostics(norm, stats.clip_value)
        self.assertTrue(diag["finite"])
        self.assertTrue(diag["within_clip"])
        restored = NormalizationStats.from_json(stats.to_json())
        self.assertEqual(restored.per_tick_std.shape[0], F_TICK_BINARY)
        self.assertEqual(len(stats.to_json()["tick_feature_names"]), F_TICK_BINARY)
class SpotFeatureTest(unittest.TestCase):
    def test_spot_block_shape_channels_and_raw_arrays(self):
        ticks, slug = _spot_timeline("BTC", "chainlink_btc", 1000, 1300, 11, rising=True)
        seq = extract_features(
            ticks, slug, vol_window=5, yes_outcome=1.0,
            asset="BTC", start_ts=1000, end_ts=1300, include_spot_features=True,
        )
        self.assertEqual(seq.per_tick.shape[1], F_TICK_SPOT)
        self.assertEqual(SPOT_BLOCK_WIDTH, 8)
        spot_start = F_TICK
        # Phase 0.1: signed distance is the known monotone ramp 0.0001 * i and matches the raw array.
        dist = seq.per_tick[:, spot_start + 2]
        self.assertTrue(np.all(np.diff(dist) > 0.0))
        self.assertAlmostEqual(float(dist[0]), 0.0, places=6)
        self.assertAlmostEqual(float(dist[1]), 0.0001, places=6)
        np.testing.assert_allclose(dist, seq.spot_signed_distance, rtol=1e-5, atol=1e-7)
        # Phase 0.2: tte_frac is monotone decreasing in [0, 1], starts at 1.0, matches the raw array.
        tte = seq.per_tick[:, spot_start + 3]
        self.assertTrue(np.all(np.diff(tte) < 0.0))
        self.assertAlmostEqual(float(tte[0]), 1.0, places=5)
        self.assertLessEqual(float(tte.max()), 1.0)
        self.assertGreaterEqual(float(tte.min()), 0.0)
        np.testing.assert_allclose(tte, seq.tte_frac, rtol=1e-5, atol=1e-7)
        # Phase 0.6: BTC one-hot is [1, 0, 0] across the market.
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 5], 1.0)
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 6], 0.0)
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 7], 0.0)
        self.assertTrue(np.isfinite(seq.per_tick).all())
        self.assertTrue((seq.per_tick[:, spot_start + 1] >= 0.0).all())
    def test_spot_block_after_binary_and_asset_onehot_eth(self):
        ticks, slug = _spot_timeline("ETH", "chainlink_eth", 2000, 2900, 9, rising=False)
        seq = extract_features(
            ticks, slug, vol_window=4, yes_outcome=0.0, include_binary_features=True,
            asset="ETH", start_ts=2000, end_ts=2900, include_spot_features=True,
        )
        self.assertEqual(seq.per_tick.shape[1], F_TICK_BINARY_SPOT)
        spot_start = F_TICK_BINARY
        # A flat spot path leaves distance and return at zero but the one-hot still flags ETH.
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 2], 0.0, atol=1e-7)
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 5], 0.0)
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 6], 1.0)
        np.testing.assert_allclose(seq.per_tick[:, spot_start + 7], 0.0)
    def test_spot_features_require_lifecycle_and_normalize_finite(self):
        ticks, slug = _spot_timeline("BTC", "chainlink_btc", 1000, 1300, 12, rising=True)
        with self.assertRaises(ValueError):
            extract_features(ticks, slug, include_spot_features=True)
        seq = extract_features(
            ticks, slug, asset="BTC", start_ts=1000, end_ts=1300, include_spot_features=True,
        )
        stats = fit_normalization(seq, clip_value=8.0)
        self.assertEqual(stats.per_tick_mean.shape[0], F_TICK_SPOT)
        self.assertEqual(len(stats.to_json()["tick_feature_names"]), F_TICK_SPOT)
        self.assertEqual(len(tick_feature_names_for_width(F_TICK_BINARY_SPOT)), F_TICK_BINARY_SPOT)
        norm = apply_normalization(seq, stats)
        diag = normalized_feature_diagnostics(norm, stats.clip_value)
        self.assertTrue(diag["finite"])
        self.assertTrue(diag["within_clip"])
        np.testing.assert_allclose(norm.tte_frac, seq.tte_frac)


class PickTopMarketsAssetFilterTest(unittest.TestCase):
    def test_assets_filter_restricts_market_pool_to_requested_symbols(self):
        from finmamba3.backtester import BacktestData
        from finmamba3.envs.lob_features import pick_top_markets
        book = OrderBookSnapshot(bids=(OrderBookLevel(0.49, 100.0),), asks=(OrderBookLevel(0.51, 100.0),))
        ticks = []
        for i in range(200):
            tick = TickData(ts_sec=i)
            for slug in ("btc-updown-5m-1", "eth-updown-5m-2", "sol-updown-5m-3"):
                tick.order_books[slug] = StoredBook(yes_book=book, no_book=OrderBookSnapshot(), book_ts=i)
            ticks.append(tick)
        bt = BacktestData(timeline=ticks, lifecycles=[], settlements={}, start_ts=0, end_ts=200)
        self.assertEqual(pick_top_markets(bt, 10, assets=["ETH"]), ["eth-updown-5m-2"])
        self.assertEqual(set(pick_top_markets(bt, 10)), {"btc-updown-5m-1", "eth-updown-5m-2", "sol-updown-5m-3"})


if __name__ == "__main__":
    unittest.main()
