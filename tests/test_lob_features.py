# region imports
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envs.lob_features import (  # noqa: E402
    F_LEVEL,
    F_TICK,
    F_TICK_BINARY,
    K_LEVELS,
    LOBSequence,
    NormalizationStats,
    append_binary_market_features,
    apply_normalization,
    fit_normalization,
    normalized_feature_diagnostics,
)
# endregion


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
if __name__ == "__main__":
    unittest.main()
