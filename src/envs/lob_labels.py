"""Label generation for LOB direction prediction.

Default project target is the sign of the next-tick midprice change with a
fixed threshold for the flat bucket. That target is dominated by spread-
bouncing noise on Polymarket because the median half-spread is around 200 bps.

This module implements two cleaner alternatives:

1. Triple-barrier labels (Lopez de Prado 2018): for each tick, label by which
   of {profit barrier, stop-loss barrier, time barrier} hits first within a
   forward-looking horizon. This is the canonical denoised LOB direction
   target and pairs naturally with binary-contract markets because the
   contract itself has a terminal payoff barrier.
2. Multi-threshold direction sweep: a vector of {-1, 0, 1} labels at multiple
   threshold values, used for reporting accuracy as a curve over thresholds
   instead of a single 1% bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class TripleBarrierConfig:
    """Forward-looking barriers for direction labeling.

    profit_threshold and stop_threshold are absolute mid-return thresholds
    (e.g. 0.005 for 50 bps). horizon is the maximum number of forward ticks
    the label can look ahead before the time barrier fires.
    """

    profit_threshold: float = 0.005
    stop_threshold: float = 0.005
    horizon: int = 32


def triple_barrier_labels_numpy(
    mid: np.ndarray, config: TripleBarrierConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy reference implementation for triple-barrier labels.

    Parameters
    ----------
    mid
        1D array of mid prices in raw (un-normalized) units. Shape (T,).
    config
        TripleBarrierConfig with profit and stop thresholds plus a horizon.

    Returns
    -------
    labels
        Int64 array of shape (T,) with values in {0, 1, 2}: 0 = stop hit,
        1 = time barrier (flat), 2 = profit hit. The last `horizon` entries
        are flagged invalid via the mask.
    valid_mask
        Bool array of shape (T,). True where the forward window fits within
        the input.
    """
    T = int(mid.shape[0])
    labels = np.full((T,), 1, dtype=np.int64)
    mask = np.zeros((T,), dtype=bool)
    if T <= 1:
        return labels, mask
    h = int(config.horizon)
    pt = float(config.profit_threshold)
    st = float(config.stop_threshold)

    for t in range(T):
        end = min(T, t + 1 + h)
        if end <= t + 1:
            continue
        base = mid[t]
        if not np.isfinite(base) or base <= 0:
            continue
        future = mid[t + 1 : end]
        rets = (future - base) / base
        hit_profit = np.argmax(rets >= pt) if np.any(rets >= pt) else -1
        hit_stop = np.argmax(rets <= -st) if np.any(rets <= -st) else -1
        if hit_profit < 0 and hit_stop < 0:
            labels[t] = 1
        elif hit_profit < 0:
            labels[t] = 0
        elif hit_stop < 0:
            labels[t] = 2
        else:
            labels[t] = 2 if hit_profit < hit_stop else 0
        if end - (t + 1) >= 1:
            mask[t] = True
    return labels, mask


def triple_barrier_labels_torch(
    mid: torch.Tensor, config: TripleBarrierConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized torch implementation of triple-barrier labels.

    Parameters
    ----------
    mid
        Tensor of shape (B, L) with raw mid prices.
    config
        TripleBarrierConfig.

    Returns
    -------
    labels
        Long tensor of shape (B, L) with values in {0, 1, 2}.
    valid_mask
        Bool tensor of shape (B, L). True where the forward window fits
        within the sequence.
    """
    B, L = mid.shape
    h = int(config.horizon)
    pt = float(config.profit_threshold)
    st = float(config.stop_threshold)
    device = mid.device

    labels = torch.full((B, L), 1, dtype=torch.long, device=device)
    valid = torch.zeros((B, L), dtype=torch.bool, device=device)
    if L <= 1 or h <= 0:
        return labels, valid

    pad = mid[:, -1:].expand(B, h)
    extended = torch.cat([mid, pad], dim=1)
    base = mid.unsqueeze(-1)
    future = extended.unfold(1, h, 1)[:, 1 : L + 1]
    safe_base = torch.where(base.abs() < 1e-12, torch.ones_like(base), base)
    rets = (future - base) / safe_base

    hit_profit = (rets >= pt).any(dim=-1)
    hit_stop = (rets <= -st).any(dim=-1)
    profit_idx = torch.argmax((rets >= pt).int(), dim=-1)
    stop_idx = torch.argmax((rets <= -st).int(), dim=-1)

    profit_first = hit_profit & (~hit_stop | (profit_idx < stop_idx))
    stop_first = hit_stop & (~hit_profit | (stop_idx < profit_idx))
    labels = torch.where(profit_first, torch.full_like(labels, 2), labels)
    labels = torch.where(stop_first, torch.full_like(labels, 0), labels)

    horizon_index = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    valid = horizon_index < (L - 1)
    return labels, valid


def multi_threshold_direction_labels(
    mid_diff: torch.Tensor, thresholds: Sequence[float]
) -> torch.Tensor:
    """Three-class direction labels at multiple thresholds.

    Parameters
    ----------
    mid_diff
        Tensor of shape (B, L-1) containing successive mid-price differences
        in normalized space. Same units as the existing DirectionHead target.
    thresholds
        Sequence of positive floats. Each yields a separate (B, L-1) label
        tensor.

    Returns
    -------
    labels
        Long tensor of shape (B, L-1, len(thresholds)) with values in
        {0, 1, 2}.
    """
    B, Lm1 = mid_diff.shape
    out = torch.full((B, Lm1, len(thresholds)), 1, dtype=torch.long, device=mid_diff.device)
    for i, t in enumerate(thresholds):
        up = mid_diff > float(t)
        down = mid_diff < -float(t)
        out[..., i] = torch.where(up, torch.full_like(out[..., i], 2), out[..., i])
        out[..., i] = torch.where(down, torch.full_like(out[..., i], 0), out[..., i])
    return out
