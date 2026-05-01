"""Tests for tick-to-bar aggregation in src/envs/lob_aggregation.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from envs.lob_aggregation import (  # noqa: E402
    BarConfig,
    DEFAULT_BID_VOL_INDEX,
    DEFAULT_MID_INDEX,
    DEFAULT_SUM_INDICES,
    aggregate_array,
)


def _toy_features(n_ticks: int, feature_dim: int = 94) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(n_ticks, feature_dim)).astype(np.float32)
    feats[:, DEFAULT_MID_INDEX] = np.cumsum(rng.normal(scale=0.001, size=n_ticks))
    feats[:, DEFAULT_BID_VOL_INDEX] = np.abs(rng.normal(loc=2.0, scale=0.5, size=n_ticks))
    ts = np.arange(n_ticks, dtype=np.float64) * 0.5
    return feats, ts


def test_time_bars_emit_at_fixed_intervals():
    feats, ts = _toy_features(200)
    bars, bar_ts = aggregate_array(
        feats,
        ts,
        DEFAULT_MID_INDEX,
        DEFAULT_BID_VOL_INDEX,
        DEFAULT_SUM_INDICES,
        BarConfig(kind="time", time_seconds=5.0),
    )
    assert bars.shape[0] >= 1
    assert bars.shape[1] == feats.shape[1]
    deltas = np.diff(bar_ts)
    assert np.all(deltas >= 5.0 - 1e-6)


def test_volume_bars_close_when_threshold_met():
    feats, ts = _toy_features(200)
    bars, _ = aggregate_array(
        feats,
        ts,
        DEFAULT_MID_INDEX,
        DEFAULT_BID_VOL_INDEX,
        DEFAULT_SUM_INDICES,
        BarConfig(kind="volume", volume_threshold=10.0),
    )
    assert bars.shape[0] >= 1


def test_sum_indices_actually_sum_within_a_bar():
    feats, ts = _toy_features(50)
    feats[:, DEFAULT_SUM_INDICES[0]] = 1.0
    bars, _ = aggregate_array(
        feats,
        ts,
        DEFAULT_MID_INDEX,
        DEFAULT_BID_VOL_INDEX,
        DEFAULT_SUM_INDICES,
        BarConfig(kind="time", time_seconds=5.0),
    )
    assert np.all(bars[:, DEFAULT_SUM_INDICES[0]] >= 1.0)


def test_unknown_bar_kind_raises():
    feats, ts = _toy_features(20)
    with pytest.raises(ValueError):
        aggregate_array(
            feats, ts, DEFAULT_MID_INDEX, DEFAULT_BID_VOL_INDEX,
            DEFAULT_SUM_INDICES, BarConfig(kind="not-a-real-bar"),
        )
