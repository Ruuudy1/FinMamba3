"""Unit tests for the helpers in eval/compare_direction.

Exercises the pure-Python helpers (label generation, classification scoring,
markdown formatting) that the CLI uses internally. The full main() flow needs
a trained checkpoint and a real LOBSequence; that path is exercised manually
during the diagnose/eval workflow rather than under unit tests.
"""
# region imports
from __future__ import annotations
import numpy as np
from finmamba3.eval.compare_direction import (
    _format_table,
    _label_directions,
    classification_metrics,
)
# endregion


def test_label_directions_three_buckets():
    mid = np.array([0.0, 0.05, 0.0, -0.05, -0.05], dtype=np.float64)
    labels = _label_directions(mid, threshold=0.01)
    # Up, down, down, flat.
    assert labels.tolist() == [2, 0, 0, 1]


def test_classification_metrics_perfect_predictor():
    # One-hot probabilities that match the labels yield accuracy 1, macro-F1 1, brier 0.
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    probs = np.eye(3)[labels]
    metrics = classification_metrics(probs, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["brier"] == 0.0


def test_classification_metrics_uniform_predictor():
    # Uniform probabilities give accuracy near 1/3 and a positive Brier score.
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probs = np.full((labels.size, 3), 1.0 / 3.0)
    metrics = classification_metrics(probs, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["brier"] > 0.0


def test_macro_f1_penalizes_majority_class_collapse():
    # A predictor that always picks the dominant flat class scores high accuracy but low
    # macro-F1; that gap is exactly why the benchmark reports macro-F1, not accuracy.
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 2], dtype=np.int64)
    always_flat = np.eye(3)[np.ones(labels.size, dtype=np.int64)]
    metrics = classification_metrics(always_flat, labels)
    assert metrics["accuracy"] == 6.0 / 8.0
    assert metrics["macro_f1"] < metrics["accuracy"]


def test_format_table_includes_macro_f1():
    rows = [
        {"method": "linear_ar", "threshold": 0.001, "accuracy": 0.45, "macro_f1": 0.30, "brier": 0.55},
        {"method": "deeplob", "threshold": 0.005, "accuracy": 0.52, "macro_f1": 0.41, "brier": 0.48},
    ]
    out = _format_table(rows)
    assert "| method | threshold | accuracy | macro_f1 | brier |" in out
    assert "| linear_ar | 0.0010 | 0.4500 | 0.3000 | 0.5500 |" in out
    assert "| deeplob | 0.0050 | 0.5200 | 0.4100 | 0.4800 |" in out
