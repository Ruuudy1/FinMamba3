"""Smoke tests for the LOB direction-prediction baselines."""
# region imports
from __future__ import annotations
import os
import sys
import numpy as np
import torch
from finmamba3.baselines.deeplob import DeepLOB  # noqa: E402
from finmamba3.baselines.linear_ar import LinearAR, LinearARConfig  # noqa: E402
# endregion


def test_deeplob_forward_shape():
    model = DeepLOB(k_levels=10, f_level=8)
    x = torch.randn(2, 16, 80)
    logits = model(x)
    assert logits.shape == (2, 16, 3)


def test_linear_ar_fits_and_predicts():
    rng = np.random.default_rng(0)
    feat_dim = 94
    T = 200
    feats = rng.normal(size=(T, feat_dim)).astype(np.float32)
    ar = LinearAR(LinearARConfig(lookback=8, threshold=0.01, midprice_index=80))
    ar.fit(feats)
    preds = ar.predict(feats)
    assert preds.shape[0] > 0
    pred_labels, actual_labels = ar.direction_labels(feats)
    assert pred_labels.shape == actual_labels.shape
