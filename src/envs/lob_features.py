"""Feature engineering for Polymarket LOB snapshots.

Converts a BacktestData.timeline into per-tick feature tensors:

  per_level[t, k, :F_LEVEL]   K=10 ordered depth tokens (bids then asks)
  per_tick [t, :F_TICK]       14 aggregate + order-flow scalars

Features are aggregate statistics relative to the midprice, not raw prices:
raw top-K levels carry noise, and order flow (volume deltas, OFI) is a
stronger directional signal than absolute price magnitudes.
"""
# region imports
from __future__ import annotations
import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lob.backtester import BacktestData, OrderBookSnapshot, TickData, build_timeline
# endregion
logger = logging.getLogger(__name__)
K_LEVELS = 10
F_LEVEL = 8
F_TICK = 14
FEATURE_DIM_FLAT = K_LEVELS * F_LEVEL + F_TICK
LEVEL_FEATURE_NAMES = (
    "rel_price",
    "log_size",
    "log_cum_depth",
    "vol_share",
    "price_gap",
    "level_index",
    "side",
    "log_staleness",
)
TICK_FEATURE_NAMES = (
    "mid",
    "spread",
    "log_spread",
    "imbalance",
    "microprice",
    "weighted_mid_disp",
    "log_bid_vol",
    "log_ask_vol",
    "dmid",
    "dspread",
    "dimbalance",
    "ofi_top",
    "trade_intensity",
    "rolling_vol",
)
BINARY_TICK_FEATURE_NAMES = (
    "boundary_distance",
    "boundary_scaled_depth",
    "logit_mid_velocity",
    "logit_mid_acceleration",
    "amihud_illiquidity",
    "variance_ratio",
)
F_TICK_BINARY = F_TICK + len(BINARY_TICK_FEATURE_NAMES)
FLAT_FEATURE_NAMES = tuple(
    f"level{k}.{name}"
    for k in range(K_LEVELS)
    for name in LEVEL_FEATURE_NAMES
) + tuple(f"tick.{name}" for name in TICK_FEATURE_NAMES)
FLAT_FEATURE_NAMES_BINARY = FLAT_FEATURE_NAMES + tuple(
    f"tick.{name}" for name in BINARY_TICK_FEATURE_NAMES
)
LEVEL_DETERMINISTIC_INDICES = (5, 6)
DEFAULT_NORM_CLIP = 8.0
DEFAULT_LEVEL_STD_FLOOR = np.asarray(
    [1e-4, 5e-2, 5e-2, 2e-2, 1e-4, 1.0, 1.0, 1e-1],
    dtype=np.float32,
)
DEFAULT_TICK_STD_FLOOR = np.asarray(
    [
        1e-2, 5e-3, 5e-3, 2e-2, 1e-2, 1e-4, 5e-2,
        5e-2, 1e-3, 1e-3, 1e-2, 1.0, 1.0, 1e-3,
    ],
    dtype=np.float32,
)
DEFAULT_BINARY_TICK_STD_FLOOR = np.asarray(
    [1e-3, 1e-2, 1e-3, 1e-3, 1e-6, 1e-2],
    dtype=np.float32,
)


def _tick_std_floor_for_width(f_tick: int, eps: float) -> np.ndarray:
    # Polymarket base (14) and binary-extended (20) schemas get pinned floors; other
    # widths such as FI-2010 fall back to a flat eps floor.
    if f_tick == DEFAULT_TICK_STD_FLOOR.shape[0]:
        return DEFAULT_TICK_STD_FLOOR
    if f_tick == F_TICK_BINARY:
        return np.concatenate([DEFAULT_TICK_STD_FLOOR, DEFAULT_BINARY_TICK_STD_FLOOR])
    return np.full(f_tick, eps, dtype=np.float32)


@dataclass
class LOBSequence:
    market_slug: str
    per_level: np.ndarray  # Shape (T, K_LEVELS, F_LEVEL) float32.
    per_tick: np.ndarray   # Shape (T, F_TICK) float32.
    midprice: np.ndarray   # Shape (T,) float32, raw unnormalized mid.
    ts_sec: np.ndarray     # Shape (T,) int64.
    yes_outcome: np.ndarray | None = None  # Shape (T,) float32 in {0, 1}, or nan if unknown.

    def to_flat(self) -> np.ndarray:
        # Use the array's own shape so FI-2010 (K, 4) works alongside Polymarket (K, 8).
        T, k, f_level = self.per_level.shape
        return np.concatenate(
            [self.per_level.reshape(T, k * f_level), self.per_tick],
            axis=1,
        ).astype(np.float32)


def _encode_levels(book: OrderBookSnapshot, mid: float) -> np.ndarray:
    out = np.zeros((K_LEVELS, F_LEVEL), dtype=np.float32)
    if mid <= 0.0:
        return out
    entries: list[tuple[float, float, int]] = []
    for lvl in book.bids[:K_LEVELS]:
        entries.append((float(lvl.price), float(lvl.size), +1))
    for lvl in book.asks[:K_LEVELS - len(entries)]:
        entries.append((float(lvl.price), float(lvl.size), -1))
    bid_cum_size = 0.0
    ask_cum_size = 0.0
    prev_price = mid
    for k, (price, size, side) in enumerate(entries[:K_LEVELS]):
        if side == +1:
            bid_cum_size += size
            cum_depth = bid_cum_size
        else:
            ask_cum_size += size
            cum_depth = ask_cum_size
        total_cum = bid_cum_size + ask_cum_size
        vol_share = cum_depth / total_cum if total_cum > 0 else 0.0
        gap = price - prev_price if k > 0 else price - mid
        prev_price = price
        out[k, 0] = (price - mid) / max(mid, 1e-6)
        out[k, 1] = math.log1p(max(size, 0.0))
        out[k, 2] = math.log1p(max(cum_depth, 0.0))
        out[k, 3] = vol_share
        out[k, 4] = gap
        out[k, 5] = (2.0 * float(k) / max(K_LEVELS - 1, 1)) - 1.0
        out[k, 6] = float(side)
        # Index 7 is filled in by extract_features() with log book staleness.
    return out


def append_binary_market_features(
    per_tick: np.ndarray,
    midprice: np.ndarray,
    vol_window: int = 20,
) -> np.ndarray:
    """Append six Polymarket-specific binary-market tick features.

    Polymarket prices are bounded probabilities in [0, 1], so the informative
    coordinates differ from a generic LOB: distance to the 0/1 boundary, the
    log-odds (logit) velocity and acceleration of the implied probability, and
    structural-break statistics (Amihud illiquidity, variance ratio) that double
    as the regime-inference structural prior. Input per_tick is (T, F_TICK) raw
    (unnormalized) and midprice is the raw mid in (0, 1); returns (T, F_TICK_BINARY).
    """
    eps = 1.0e-6
    mid = np.clip(midprice.astype(np.float64), eps, 1.0 - eps)
    boundary_distance = np.minimum(mid, 1.0 - mid)
    log_depth_sum = per_tick[:, 6].astype(np.float64) + per_tick[:, 7].astype(np.float64)
    boundary_scaled_depth = log_depth_sum * (2.0 * boundary_distance)
    logit_mid = np.log(mid / (1.0 - mid))
    logit_velocity = np.zeros_like(logit_mid)
    logit_velocity[1:] = logit_mid[1:] - logit_mid[:-1]
    logit_acceleration = np.zeros_like(logit_mid)
    logit_acceleration[1:] = logit_velocity[1:] - logit_velocity[:-1]
    depth_proxy = log_depth_sum + eps
    amihud_instant = np.abs(logit_velocity) / depth_proxy
    amihud_cumsum = np.concatenate([[0.0], np.cumsum(amihud_instant)])
    amihud_rolling = np.zeros_like(logit_mid)
    variance_ratio = np.ones_like(logit_mid)
    num_ticks = logit_mid.shape[0]
    for t in range(num_ticks):
        window_start = max(0, t + 1 - vol_window)
        window_count = t + 1 - window_start
        amihud_rolling[t] = (amihud_cumsum[t + 1] - amihud_cumsum[window_start]) / window_count
        if window_count >= 4:
            window_returns = logit_velocity[window_start : t + 1]
            var_one_step = float(np.var(window_returns))
            two_step_returns = window_returns[:-1] + window_returns[1:]
            var_two_step = float(np.var(two_step_returns))
            variance_ratio[t] = var_two_step / (2.0 * var_one_step + eps) if var_one_step > 0.0 else 1.0
    binary_block = np.stack(
        [
            boundary_distance,
            boundary_scaled_depth,
            logit_velocity,
            logit_acceleration,
            amihud_rolling,
            variance_ratio,
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([per_tick.astype(np.float32), binary_block], axis=1)


def extract_features(
    timeline: list[TickData],
    market_slug: str,
    vol_window: int = 20,
    yes_outcome: float | None = None,
    include_binary_features: bool = False,
) -> LOBSequence:
    """Per-tick feature tensors for a single market. Skips ticks with no
    2-sided YES book.
    """
    per_level_rows: list[np.ndarray] = []
    per_tick_rows: list[np.ndarray] = []
    mids: list[float] = []
    ts_list: list[int] = []
    mid_window: list[float] = []
    prev_mid: float | None = None
    prev_spread: float | None = None
    prev_imbalance: float | None = None
    prev_top5_sizes: np.ndarray | None = None
    prev_top_bid_size: float | None = None
    prev_top_ask_size: float | None = None
    for tick in timeline:
        sb = tick.order_books.get(market_slug)
        if sb is None:
            continue
        book = sb.yes_book
        if len(book.bids) == 0 or len(book.asks) == 0:
            continue
        mid = book.mid
        if mid <= 0.0:
            continue
        best_bid = book.best_bid
        best_ask = book.best_ask
        spread = best_ask - best_bid
        top_bid_size = float(book.bids[0].size)
        top_ask_size = float(book.asks[0].size)
        top_sum = top_bid_size + top_ask_size
        imbalance = top_bid_size / top_sum if top_sum > 0 else 0.5
        total_bid_vol = book.total_bid_size
        total_ask_vol = book.total_ask_size
        if top_sum > 0:
            microprice = (best_ask * top_bid_size + best_bid * top_ask_size) / top_sum
        else:
            microprice = mid
        weighted_mid_disp = (microprice - mid) / max(mid, 1e-6)
        lvl_tokens = _encode_levels(book, mid)
        book_ts = int(tick.book_timestamps.get(market_slug, tick.ts_sec))
        staleness = max(float(tick.ts_sec - book_ts), 0.0)
        lvl_tokens[:, 7] = math.log1p(staleness)
        dmid = (mid - prev_mid) if prev_mid is not None else 0.0
        dspread = (spread - prev_spread) if prev_spread is not None else 0.0
        dimb = (imbalance - prev_imbalance) if prev_imbalance is not None else 0.0
        # Signed top-of-book order-flow imbalance (Cont-Kukanov style).
        if prev_top_bid_size is not None and prev_top_ask_size is not None:
            ofi_top = (top_bid_size - prev_top_bid_size) - (top_ask_size - prev_top_ask_size)
        else:
            ofi_top = 0.0
        top5 = np.zeros(10, dtype=np.float32)
        for k in range(min(5, len(book.bids))):
            top5[k] = float(book.bids[k].size)
        for k in range(min(5, len(book.asks))):
            top5[5 + k] = float(book.asks[k].size)
        if prev_top5_sizes is not None:
            trade_intensity = float(np.abs(top5 - prev_top5_sizes).sum())
        else:
            trade_intensity = 0.0
        prev_top5_sizes = top5
        mid_window.append(mid)
        if len(mid_window) > vol_window:
            mid_window.pop(0)
        rolling_vol = float(np.std(mid_window)) if len(mid_window) >= 2 else 0.0
        tick_vec = np.array([
            mid, spread, math.log1p(max(spread, 0.0)),
            imbalance, microprice, weighted_mid_disp,
            math.log1p(max(total_bid_vol, 0.0)),
            math.log1p(max(total_ask_vol, 0.0)),
            dmid, dspread, dimb,
            ofi_top, trade_intensity, rolling_vol,
        ], dtype=np.float32)
        assert tick_vec.shape[0] == F_TICK
        per_level_rows.append(lvl_tokens)
        per_tick_rows.append(tick_vec)
        mids.append(mid)
        ts_list.append(int(tick.ts_sec))
        prev_mid = mid
        prev_spread = spread
        prev_imbalance = imbalance
        prev_top_bid_size = top_bid_size
        prev_top_ask_size = top_ask_size
    if not per_tick_rows:
        raise RuntimeError(f"No usable ticks for market {market_slug}")
    per_level_array = np.stack(per_level_rows, axis=0)
    per_tick_array = np.stack(per_tick_rows, axis=0)
    midprice_array = np.asarray(mids, dtype=np.float32)
    if include_binary_features:
        per_tick_array = append_binary_market_features(per_tick_array, midprice_array, vol_window)
    return LOBSequence(
        market_slug=market_slug,
        per_level=per_level_array,
        per_tick=per_tick_array,
        midprice=midprice_array,
        ts_sec=np.asarray(ts_list, dtype=np.int64),
        yes_outcome=(
            np.full(len(ts_list), float(yes_outcome), dtype=np.float32)
            if yes_outcome is not None
            else None
        ),
    )
BASIC_TICK_FEATURE_NAMES = (
    "mid",
    "spread",
    "log_spread",
    "imbalance",
    "microprice",
    "log_total_vol",
)


def compute_basic_tick_features(per_level_fi2010: np.ndarray) -> np.ndarray:
    """Derive 6 stationary tick aggregates from FI-2010-style level data.

    Input shape is (T, K, 4) where the last axis is ordered
    (ask_price, ask_size, bid_price, bid_size). Output shape is (T, 6).
    Pure function: no cross-tick state, so it works on already-shuffled data.
    """
    if per_level_fi2010.ndim != 3 or per_level_fi2010.shape[-1] != 4:
        raise ValueError(
            "compute_basic_tick_features expects shape (T, K, 4); got "
            f"{tuple(per_level_fi2010.shape)}"
        )
    ask_prices = per_level_fi2010[:, :, 0]
    ask_sizes = per_level_fi2010[:, :, 1]
    bid_prices = per_level_fi2010[:, :, 2]
    bid_sizes = per_level_fi2010[:, :, 3]
    best_ask = ask_prices[:, 0]
    best_bid = bid_prices[:, 0]
    top_ask_size = ask_sizes[:, 0]
    top_bid_size = bid_sizes[:, 0]
    mid = 0.5 * (best_ask + best_bid)
    spread = best_ask - best_bid
    log_spread = np.log1p(np.maximum(spread, 0.0))
    top_sum = top_bid_size + top_ask_size
    imbalance = np.where(top_sum > 0.0, top_bid_size / np.maximum(top_sum, 1e-12), 0.5)
    microprice_num = best_ask * top_bid_size + best_bid * top_ask_size
    microprice = np.where(top_sum > 0.0, microprice_num / np.maximum(top_sum, 1e-12), mid)
    total_vol = ask_sizes.sum(axis=1) + bid_sizes.sum(axis=1)
    log_total_vol = np.log1p(np.maximum(total_vol, 0.0))
    return np.stack(
        [mid, spread, log_spread, imbalance, microprice, log_total_vol],
        axis=1,
    ).astype(np.float32)


def pick_longest_market(data: BacktestData) -> str:
    counts: dict[str, int] = {}
    for tick in data.timeline:
        for slug, sb in tick.order_books.items():
            if sb.yes_book.bids and sb.yes_book.asks:
                counts[slug] = counts.get(slug, 0) + 1
    if not counts:
        raise RuntimeError("No markets with 2-sided books in timeline")
    return max(counts.items(), key=lambda kv: kv[1])[0]
# Normalization.


@dataclass
class NormalizationStats:
    per_level_mean: np.ndarray
    per_level_std: np.ndarray
    per_tick_mean: np.ndarray
    per_tick_std: np.ndarray
    clip_value: float = DEFAULT_NORM_CLIP

    def to_json(self) -> dict:
        f_tick = int(self.per_tick_mean.shape[0])
        tick_feature_names = list(TICK_FEATURE_NAMES)
        if f_tick == F_TICK_BINARY:
            tick_feature_names = list(TICK_FEATURE_NAMES) + list(BINARY_TICK_FEATURE_NAMES)
        return {
            "per_level_mean": self.per_level_mean.tolist(),
            "per_level_std":  self.per_level_std.tolist(),
            "per_tick_mean":  self.per_tick_mean.tolist(),
            "per_tick_std":   self.per_tick_std.tolist(),
            "clip_value": self.clip_value,
            "level_feature_names": list(LEVEL_FEATURE_NAMES),
            "tick_feature_names": tick_feature_names,
            "level_std_floor": DEFAULT_LEVEL_STD_FLOOR.tolist(),
            "tick_std_floor": _tick_std_floor_for_width(f_tick, 1e-6).tolist(),
            "deterministic_level_indices": list(LEVEL_DETERMINISTIC_INDICES),
        }
    @classmethod
    def from_json(cls, d: dict) -> "NormalizationStats":
        per_level_mean = np.asarray(d["per_level_mean"], dtype=np.float32)
        per_level_std = np.asarray(d["per_level_std"], dtype=np.float32)
        per_tick_mean = np.asarray(d["per_tick_mean"], dtype=np.float32)
        per_tick_std = np.asarray(d["per_tick_std"], dtype=np.float32)
        # Apply the canonical Polymarket floors and deterministic-index pins only
        # when the saved stats match that schema's widths.
        if per_level_mean.shape[0] == DEFAULT_LEVEL_STD_FLOOR.shape[0]:
            per_level_std = np.maximum(per_level_std, DEFAULT_LEVEL_STD_FLOOR)
            for idx in LEVEL_DETERMINISTIC_INDICES:
                per_level_mean[idx] = 0.0
                per_level_std[idx] = 1.0
        if per_tick_std.shape[0] in (DEFAULT_TICK_STD_FLOOR.shape[0], F_TICK_BINARY):
            per_tick_std = np.maximum(per_tick_std, _tick_std_floor_for_width(per_tick_std.shape[0], 1e-6))
        return cls(
            per_level_mean=per_level_mean,
            per_level_std=per_level_std,
            per_tick_mean=per_tick_mean,
            per_tick_std=per_tick_std,
            clip_value=float(d.get("clip_value", DEFAULT_NORM_CLIP)),
        )


def fit_normalization(
    seq: LOBSequence,
    eps: float = 1e-6,
    clip_value: float = DEFAULT_NORM_CLIP,
) -> NormalizationStats:
    # Infer per-level feature width from the array. Polymarket uses 8 (with the
    # canonical std_floor and deterministic indices); FI-2010 and other schemas
    # fall back to a flat eps floor and no deterministic pinning.
    f_level = int(seq.per_level.shape[-1])
    f_tick = int(seq.per_tick.shape[-1])
    pl = seq.per_level.reshape(-1, f_level)
    pt = seq.per_tick
    pl_mean = pl.mean(axis=0).astype(np.float32)
    pt_mean = pt.mean(axis=0).astype(np.float32)
    level_floor = (
        DEFAULT_LEVEL_STD_FLOOR if f_level == DEFAULT_LEVEL_STD_FLOOR.shape[0]
        else np.full(f_level, eps, dtype=np.float32)
    )
    tick_floor = _tick_std_floor_for_width(f_tick, eps)
    pl_std = np.maximum(pl.std(axis=0), level_floor).astype(np.float32)
    pt_std = np.maximum(pt.std(axis=0), tick_floor).astype(np.float32)
    # Polymarket level index and side are deterministic token metadata; pin them
    # so tiny distribution shifts cannot blow up the z-score. Skip when the
    # active schema does not have those slots.
    if f_level == DEFAULT_LEVEL_STD_FLOOR.shape[0]:
        for idx in LEVEL_DETERMINISTIC_INDICES:
            pl_mean[idx] = 0.0
            pl_std[idx] = 1.0
    return NormalizationStats(
        per_level_mean=pl_mean,
        per_level_std=np.maximum(pl_std, eps).astype(np.float32),
        per_tick_mean=pt_mean,
        per_tick_std=np.maximum(pt_std, eps).astype(np.float32),
        clip_value=float(clip_value),
    )


def apply_normalization(seq: LOBSequence, stats: NormalizationStats) -> LOBSequence:
    pl = (seq.per_level - stats.per_level_mean) / stats.per_level_std
    pt = (seq.per_tick - stats.per_tick_mean) / stats.per_tick_std
    pl = np.clip(pl, -stats.clip_value, stats.clip_value)
    pt = np.clip(pt, -stats.clip_value, stats.clip_value)
    if not np.isfinite(pl).all() or not np.isfinite(pt).all():
        raise ValueError(f"Non-finite normalized features for market {seq.market_slug}")
    return LOBSequence(
        market_slug=seq.market_slug,
        per_level=pl.astype(np.float32),
        per_tick=pt.astype(np.float32),
        midprice=seq.midprice,
        ts_sec=seq.ts_sec,
        yes_outcome=seq.yes_outcome,
    )


def make_aggregate_only(seq: LOBSequence) -> LOBSequence:
    """Zero depth tokens while preserving tick-level aggregate features."""
    return LOBSequence(
        market_slug=seq.market_slug,
        per_level=np.zeros_like(seq.per_level, dtype=np.float32),
        per_tick=seq.per_tick.astype(np.float32),
        midprice=seq.midprice,
        ts_sec=seq.ts_sec,
        yes_outcome=seq.yes_outcome,
    )


def normalized_feature_diagnostics(seq: LOBSequence, clip_value: float | None = None) -> dict:
    flat = seq.to_flat()
    flat_dim = flat.shape[1] if flat.ndim == 2 else 0
    abs_flat = np.abs(flat)
    finite = bool(np.isfinite(flat).all())
    max_abs = float(abs_flat.max()) if flat.size else 0.0
    feature_max_abs = abs_flat.max(axis=0) if flat.size else np.zeros(flat_dim)
    top_idx = np.argsort(feature_max_abs)[-5:][::-1]
    cap = DEFAULT_NORM_CLIP if clip_value is None else float(clip_value)
    # Pick the schema-appropriate feature-name tuple, defaulting to indices for unknown widths.
    if flat_dim == len(FLAT_FEATURE_NAMES):
        name_table = FLAT_FEATURE_NAMES
    elif flat_dim == len(FLAT_FEATURE_NAMES_BINARY):
        name_table = FLAT_FEATURE_NAMES_BINARY
    else:
        name_table = tuple(f"feature_{i}" for i in range(flat_dim))
    return {
        "finite": finite,
        "max_abs": max_abs,
        "clip_value": cap,
        "within_clip": bool(finite and max_abs <= cap + 1e-5),
        "top_features": [
            (name_table[int(i)], float(feature_max_abs[int(i)]))
            for i in top_idx
        ],
    }


def denormalize_flat(flat: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    # Infer the schema (K, F_level) from the normalization stats so the same code path
    # handles Polymarket (8 per-level features) and FI-2010 (4 per-level features).
    arr = np.asarray(flat, dtype=np.float32).copy()
    f_level = int(stats.per_level_mean.shape[0])
    f_tick = int(stats.per_tick_mean.shape[0])
    total = arr.shape[-1]
    level_dim = total - f_tick
    if level_dim % f_level != 0:
        raise ValueError(
            f"denormalize_flat: feature width {total} not divisible into "
            f"(K * {f_level}) + {f_tick}"
        )
    k = level_dim // f_level
    levels = arr[..., :level_dim].reshape(*arr.shape[:-1], k, f_level)
    ticks = arr[..., level_dim:]
    levels *= stats.per_level_std
    levels += stats.per_level_mean
    ticks *= stats.per_tick_std
    ticks += stats.per_tick_mean
    return arr


def save_normalization(stats: NormalizationStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_json(), indent=2))


def load_normalization(path: Path) -> NormalizationStats:
    return NormalizationStats.from_json(json.loads(path.read_text()))
# CLI entry point.


def _cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--market", type=str, default=None)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--norm-out", type=Path, default=None)
    p.add_argument("--norm-clip", type=float, default=DEFAULT_NORM_CLIP)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    bt = build_timeline(data_dir=args.data_dir, hours=args.hours)
    slug = args.market or pick_longest_market(bt)
    print(f"market: {slug}")
    seq = extract_features(bt.timeline, slug)
    print(f"per_level shape: {seq.per_level.shape}")
    print(f"per_tick  shape: {seq.per_tick.shape}")
    print(f"midprice  range: [{seq.midprice.min():.4f}, {seq.midprice.max():.4f}], "
          f"mean={seq.midprice.mean():.4f}")
    print(f"per_tick  stats (mean, std):")
    for i in range(F_TICK):
        print(f"  f{i:02d}  mean={seq.per_tick[:, i].mean():+.4f}  "
              f"std={seq.per_tick[:, i].std():.4f}")
    if args.norm_out:
        stats = fit_normalization(seq, clip_value=args.norm_clip)
        save_normalization(stats, args.norm_out)
        print(f"normalization saved to {args.norm_out}")
        seq_norm = apply_normalization(seq, stats)
        diag = normalized_feature_diagnostics(seq_norm, stats.clip_value)
        print(f"normalized max_abs={diag['max_abs']:.4f}, finite={diag['finite']}")
        print("top normalized feature magnitudes:")
        for name, value in diag["top_features"]:
            print(f"  {name}: {value:.4f}")
if __name__ == "__main__":
    _cli()
