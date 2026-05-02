"""Unit tests for the helpers in eval/compare_direction.

Exercises the pure-Python helpers (label generation, accuracy/Brier scoring,
markdown formatting) that the CLI uses internally. The full main() flow needs
a trained checkpoint and a real LOBSequence; that path is exercised manually
during the diagnose/eval workflow rather than under unit tests.
"""

from __future__ import annotations

import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from eval.compare_direction import (  # noqa: E402
    _accuracy_brier,
    _format_table,
    _label_directions,
)


def test_label_directions_three_buckets():
    mid = np.array([0.0, 0.05, 0.0, -0.05, -0.05], dtype=np.float64)
    labels = _label_directions(mid, threshold=0.01)
    # Up, down, down, flat.
    assert labels.tolist() == [2, 0, 0, 1]


def test_accuracy_brier_perfect_predictor():
    # One-hot probabilities that match the labels should yield accuracy=1, brier=0.
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    probs = np.eye(3)[labels]
    acc, brier = _accuracy_brier(probs, labels)
    assert acc == 1.0
    assert brier == 0.0


def test_accuracy_brier_uniform_predictor():
    # Uniform probabilities across 3 classes should yield accuracy ~ 1/3 and brier > 0.
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    probs = np.full((labels.size, 3), 1.0 / 3.0)
    acc, brier = _accuracy_brier(probs, labels)
    assert 0.0 <= acc <= 1.0
    assert brier > 0.0


def test_format_table_sorts_rows_in_input_order():
    rows = [
        {"method": "linear_ar", "threshold": 0.001, "accuracy": 0.45, "brier": 0.55},
        {"method": "deeplob", "threshold": 0.005, "accuracy": 0.52, "brier": 0.48},
    ]
    out = _format_table(rows)
    assert "| method | threshold | accuracy | brier |" in out
    assert "| linear_ar | 0.0010 | 0.4500 | 0.5500 |" in out
    assert "| deeplob | 0.0050 | 0.5200 | 0.4800 |" in out
