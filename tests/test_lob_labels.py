"""Tests for triple-barrier and multi-threshold labeling helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from envs.lob_labels import (  # noqa: E402
    TripleBarrierConfig,
    multi_threshold_direction_labels,
    triple_barrier_labels_numpy,
    triple_barrier_labels_torch,
)


def test_triple_barrier_uniformly_rising_hits_profit():
    mid = np.linspace(1.0, 1.1, 32, dtype=np.float64)
    cfg = TripleBarrierConfig(profit_threshold=0.01, stop_threshold=0.01, horizon=8)
    labels, mask = triple_barrier_labels_numpy(mid, cfg)
    early = mask & (np.arange(len(mid)) < len(mid) - cfg.horizon)
    assert np.all(labels[early] == 2)


def test_triple_barrier_uniformly_falling_hits_stop():
    mid = np.linspace(1.0, 0.9, 32, dtype=np.float64)
    cfg = TripleBarrierConfig(profit_threshold=0.01, stop_threshold=0.01, horizon=8)
    labels, mask = triple_barrier_labels_numpy(mid, cfg)
    early = mask & (np.arange(len(mid)) < len(mid) - cfg.horizon)
    assert np.all(labels[early] == 0)


def test_triple_barrier_flat_is_time_barrier():
    mid = np.full(32, 1.0, dtype=np.float64)
    cfg = TripleBarrierConfig(profit_threshold=0.01, stop_threshold=0.01, horizon=8)
    labels, mask = triple_barrier_labels_numpy(mid, cfg)
    early = mask & (np.arange(len(mid)) < len(mid) - cfg.horizon)
    assert np.all(labels[early] == 1)


def test_torch_implementation_matches_numpy():
    rng = np.random.default_rng(42)
    mid = 1.0 + np.cumsum(rng.normal(scale=0.005, size=64))
    cfg = TripleBarrierConfig(profit_threshold=0.005, stop_threshold=0.005, horizon=8)
    np_labels, _ = triple_barrier_labels_numpy(mid, cfg)
    torch_labels, _ = triple_barrier_labels_torch(
        torch.from_numpy(mid).unsqueeze(0), cfg
    )
    assert torch_labels.shape == (1, mid.shape[0])
    assert torch_labels.dtype == torch.long


def test_multi_threshold_direction_labels_shape():
    mid_diff = torch.randn(2, 5)
    labels = multi_threshold_direction_labels(mid_diff, [0.01, 0.05, 0.1])
    assert labels.shape == (2, 5, 3)
    assert labels.dtype == torch.long
