"""Aggregation of raw LOB tick streams into denoised bars.

Polymarket median half-spread is around 200 bps so per-tick mid moves are
dominated by spread-bouncing noise rather than signal. This module implements
the standard Lopez de Prado financial data structures so the world model can
be trained on aggregated bars instead of raw ticks.

Bar types implemented:
- Time bars: fixed wall-clock interval.
- Volume bars: aggregate until cumulative top-of-book size threshold met.
- Dollar bars: aggregate until cumulative notional threshold met.
- Tick-imbalance bars: aggregate until signed-tick imbalance crosses a threshold.
- CUSUM bars: aggregate until directional CUSUM filter triggers.

Each bar yields the same 94-dim flat feature vector as the raw-tick path so
the rest of the pipeline (encoder, replay buffer) is agnostic to the
aggregation choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np


@dataclass
class BarConfig:
    """Configuration for bar aggregation.

    The `kind` selects the sampling rule. Only the field matching `kind` is
    consulted; the others are ignored.
    """

    kind: str = "time"
    time_seconds: float = 5.0
    volume_threshold: float = 200.0
    dollar_threshold: float = 1000.0
    tick_imbalance_threshold: float = 8.0
    cusum_threshold: float = 0.005


@dataclass
class _BarAccumulator:
    """Running aggregator over a sequence of input feature vectors.

    The bar's output mid/spread/imbalance are taken from the LAST tick in the
    bar (close-of-bar semantics). Volume and OFI fields are summed across the
    bar. Other tick-level fields are averaged across the bar.
    """

    feature_dim: int
    sum_buf: np.ndarray
    last_buf: np.ndarray
    count: int = 0
    cum_volume: float = 0.0
    cum_dollar: float = 0.0
    cum_signed_tick: float = 0.0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    bar_start_ts: float | None = None

    @classmethod
    def empty(cls, feature_dim: int) -> "_BarAccumulator":
        return cls(
            feature_dim=feature_dim,
            sum_buf=np.zeros(feature_dim, dtype=np.float32),
            last_buf=np.zeros(feature_dim, dtype=np.float32),
        )

    def reset(self) -> None:
        self.sum_buf.fill(0.0)
        self.last_buf.fill(0.0)
        self.count = 0
        self.cum_volume = 0.0
        self.cum_dollar = 0.0
        self.cum_signed_tick = 0.0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.bar_start_ts = None


def _emit(acc: _BarAccumulator, sum_indices: tuple[int, ...]) -> np.ndarray:
    """Materialize a bar feature vector from an accumulator.

    Sum-mode features (sum_indices) are summed across the bar; everything
    else is taken from the last tick to preserve close-of-bar semantics.
    """
    out = acc.last_buf.copy()
    if acc.count > 0:
        for i in sum_indices:
            out[i] = float(acc.sum_buf[i])
    return out


def aggregate_to_bars(
    feature_stream: Iterable[np.ndarray],
    timestamps: Iterable[float],
    mid_index: int,
    volume_index: int | None,
    sum_indices: tuple[int, ...],
    config: BarConfig,
) -> Iterator[tuple[np.ndarray, float]]:
    """Stream aggregator that yields (bar_feature_vector, bar_close_ts) tuples.

    Parameters
    ----------
    feature_stream
        Iterable of (feature_dim,) float arrays, one per raw tick.
    timestamps
        Iterable of monotonically non-decreasing tick timestamps in seconds.
    mid_index
        Index into the feature vector for the (normalized or raw) midprice.
        Used for tick-imbalance and CUSUM rules.
    volume_index
        Index of the (log) top-of-book volume aggregate. Required for volume
        and dollar bars. Pass None if not used.
    sum_indices
        Indices of features that should be summed (rather than carried from
        the last tick) when emitting a bar. Typical: OFI top, dmid, dspread,
        dimbalance, trade_intensity.
    config
        BarConfig describing which sampling rule to apply.
    """

    acc: _BarAccumulator | None = None
    last_mid: float | None = None
    feature_dim: int | None = None

    for x, ts in zip(feature_stream, timestamps):
        if feature_dim is None:
            feature_dim = int(x.shape[0])
            acc = _BarAccumulator.empty(feature_dim)
        assert acc is not None
        acc.last_buf[:] = x
        acc.sum_buf += x
        acc.count += 1
        if acc.bar_start_ts is None:
            acc.bar_start_ts = float(ts)

        mid = float(x[mid_index])
        signed_tick = 0.0
        if last_mid is not None:
            dmid = mid - last_mid
            signed_tick = 1.0 if dmid > 0 else (-1.0 if dmid < 0 else 0.0)
            acc.cusum_pos = max(0.0, acc.cusum_pos + dmid)
            acc.cusum_neg = min(0.0, acc.cusum_neg + dmid)
        last_mid = mid
        acc.cum_signed_tick += signed_tick

        if volume_index is not None:
            vol_term = float(x[volume_index])
            acc.cum_volume += vol_term
            acc.cum_dollar += vol_term * mid

        triggered = _bar_triggered(acc, ts, config)
        if triggered:
            yield _emit(acc, sum_indices), float(ts)
            acc.reset()
            acc.bar_start_ts = float(ts)


def _bar_triggered(acc: _BarAccumulator, ts: float, config: BarConfig) -> bool:
    """Decide whether the current bar should close at this tick."""
    if config.kind == "time":
        if acc.bar_start_ts is None:
            return False
        return float(ts) - acc.bar_start_ts >= config.time_seconds
    if config.kind == "volume":
        return acc.cum_volume >= config.volume_threshold
    if config.kind == "dollar":
        return acc.cum_dollar >= config.dollar_threshold
    if config.kind == "tick_imbalance":
        return abs(acc.cum_signed_tick) >= config.tick_imbalance_threshold
    if config.kind == "cusum":
        return max(acc.cusum_pos, -acc.cusum_neg) >= config.cusum_threshold
    raise ValueError(f"Unknown BarConfig.kind: {config.kind!r}")


def aggregate_array(
    features: np.ndarray,
    timestamps: np.ndarray,
    mid_index: int,
    volume_index: int | None,
    sum_indices: tuple[int, ...],
    config: BarConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-array convenience wrapper around aggregate_to_bars.

    Returns (bar_features, bar_close_timestamps).
    """
    bars = list(
        aggregate_to_bars(
            features, timestamps, mid_index, volume_index, sum_indices, config
        )
    )
    if not bars:
        return (
            np.zeros((0, features.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    feats = np.stack([b[0] for b in bars], axis=0).astype(np.float32)
    ts = np.array([b[1] for b in bars], dtype=np.float64)
    return feats, ts


# Default sum-mode indices for the project's 94-dim feature layout.
# K=10 levels x F_LEVEL=8 = 80, then F_TICK=14. Tick layout (offset 80):
# 0 mid, 1 spread, 2 log_spread, 3 imbalance, 4 microprice, 5 weighted_mid_disp,
# 6 log_bid_vol, 7 log_ask_vol, 8 dmid, 9 dspread, 10 dimbalance,
# 11 ofi_top, 12 trade_intensity, 13 rolling_vol.
DEFAULT_TICK_BASE = 80
DEFAULT_MID_INDEX = DEFAULT_TICK_BASE + 0
DEFAULT_BID_VOL_INDEX = DEFAULT_TICK_BASE + 6
DEFAULT_SUM_INDICES: tuple[int, ...] = (
    DEFAULT_TICK_BASE + 8,   # dmid
    DEFAULT_TICK_BASE + 9,   # dspread
    DEFAULT_TICK_BASE + 10,  # dimbalance
    DEFAULT_TICK_BASE + 11,  # ofi_top
    DEFAULT_TICK_BASE + 12,  # trade_intensity
)
