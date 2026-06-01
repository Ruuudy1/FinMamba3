"""Unit tests for the pure helpers in eval/benchmark_fi2010.

The DeepLOB and world-model-probe arms need the FI-2010 files and a checkpoint, so
they are exercised manually on the 4080/Colab; here we cover the windowing and the
majority-class floor, which are pure NumPy.
"""
# region imports
from __future__ import annotations
import numpy as np
from finmamba3.eval.benchmark_fi2010 import _eval_majority, _windows_and_labels
# endregion


def test_windows_and_labels_uses_label_at_window_end():
    # per_level[t] is filled with t so each window is identifiable; the label for a window
    # is the published label at its last observation, per the FI-2010 convention.
    per_level = np.zeros((10, 2, 2), dtype=np.float32)
    for t in range(10):
        per_level[t] = t
    labels = np.arange(10, dtype=np.int64) % 3
    windows, window_labels = _windows_and_labels(per_level, labels, window=3)
    assert windows.shape == (7, 3, 4)
    assert window_labels.tolist() == (np.arange(2, 9) % 3).tolist()
    # The first window spans observations 0,1,2 flattened to width K*F = 4.
    assert np.allclose(windows[0, :, 0], np.array([0.0, 1.0, 2.0]))


def test_majority_floor_predicts_train_majority_class():
    # Train is dominated by class 1, so the floor predicts class 1 on every val row.
    ytr = np.array([1, 1, 1, 1, 0, 2], dtype=np.int64)
    yva = np.array([1, 1, 0, 2], dtype=np.int64)
    metrics = _eval_majority(ytr, yva)
    assert metrics["accuracy"] == 0.5
    # Only the flat class scores, so macro-F1 is well below accuracy: the floor is weak.
    assert metrics["macro_f1"] < metrics["accuracy"]
