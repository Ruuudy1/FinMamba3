"""Linear vector autoregression baseline for LOB direction prediction.

Establishes the floor every Mamba/Transformer model must beat. Fits an
ordinary-least-squares regression of next-tick midprice on a window of
recent flat features. Direction labels are then read off the sign of the
predicted return at the configured threshold.

The fit is closed-form via numpy.linalg.lstsq so this baseline trains in
seconds and gives a sanity check that any neural model is actually learning
something beyond a linear lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearARConfig:
    lookback: int = 16
    threshold: float = 0.001
    midprice_index: int = 80


class LinearAR:
    """Fits next-tick midprice on a flat window of past features."""

    def __init__(self, config: LinearARConfig | None = None) -> None:
        self.config = config or LinearARConfig()
        self.coef: np.ndarray | None = None
        self.feature_dim: int | None = None

    def _build_design_matrix(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, F = features.shape
        L = self.config.lookback
        if T <= L + 1:
            raise ValueError(f"Need at least {L + 2} ticks; got {T}")
        rows = []
        targets = []
        for t in range(L, T - 1):
            window = features[t - L + 1 : t + 1].reshape(-1)
            rows.append(np.concatenate([[1.0], window]))
            targets.append(features[t + 1, self.config.midprice_index])
        return np.asarray(rows), np.asarray(targets)

    def fit(self, features: np.ndarray) -> None:
        X, y = self._build_design_matrix(features)
        self.coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.feature_dim = int(features.shape[1])

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coef is None or self.feature_dim is None:
            raise RuntimeError("Call LinearAR.fit before predict.")
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: trained on {self.feature_dim}, got {features.shape[1]}"
            )
        X, _ = self._build_design_matrix(features)
        return X @ self.coef

    def direction_labels(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted and actual three-class direction labels.

        Returns (predicted_labels, actual_labels) both of length T - lookback - 1.
        """
        if self.coef is None:
            raise RuntimeError("Call LinearAR.fit before direction_labels.")
        L = self.config.lookback
        thr = self.config.threshold
        mid_idx = self.config.midprice_index
        preds = self.predict(features)
        T = features.shape[0]
        actual_diff = np.diff(features[L:T, mid_idx])
        last_mid = features[L : T - 1, mid_idx]
        pred_diff = preds[: T - L - 1] - last_mid
        actual = np.full_like(actual_diff, 1, dtype=np.int64)
        actual[actual_diff > thr] = 2
        actual[actual_diff < -thr] = 0
        predicted = np.full_like(pred_diff, 1, dtype=np.int64)
        predicted[pred_diff > thr] = 2
        predicted[pred_diff < -thr] = 0
        return predicted, actual
