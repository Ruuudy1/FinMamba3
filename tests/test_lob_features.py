import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.lob_features import (  # noqa: E402
    F_LEVEL,
    F_TICK,
    K_LEVELS,
    LOBSequence,
    apply_normalization,
    fit_normalization,
    normalized_feature_diagnostics,
)


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


if __name__ == "__main__":
    unittest.main()
