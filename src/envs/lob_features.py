"""Feature engineering for Polymarket LOB snapshots.

Converts a BacktestData.timeline into per-tick feature tensors:

  per_level[t, k, :F_LEVEL]   K=10 ordered depth tokens (bids then asks)
  per_tick [t, :F_TICK]       14 aggregate + order-flow scalars

Features are aggregate statistics relative to the midprice, not raw prices:
raw top-K levels carry noise, and order flow (volume deltas, OFI) is a
stronger directional signal than absolute price magnitudes.
"""

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

logger = logging.getLogger(__name__)

K_LEVELS = 10
F_LEVEL = 8
F_TICK = 14
FEATURE_DIM_FLAT = K_LEVELS * F_LEVEL + F_TICK


@dataclass
class LOBSequence:
    market_slug: str
    per_level: np.ndarray  # (T, K_LEVELS, F_LEVEL) float32
    per_tick: np.ndarray   # (T, F_TICK) float32
    midprice: np.ndarray   # (T,) float32, raw unnormalized mid
    ts_sec: np.ndarray     # (T,) int64

    def to_flat(self) -> np.ndarray:
        T = self.per_level.shape[0]
        return np.concatenate(
            [self.per_level.reshape(T, K_LEVELS * F_LEVEL), self.per_tick],
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
        out[k, 5] = float(k)
        out[k, 6] = float(side)
        # out[k, 7] filled in by extract_features with book staleness
    return out


def extract_features(
    timeline: list[TickData],
    market_slug: str,
    vol_window: int = 20,
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
        staleness = float(tick.ts_sec - book_ts)
        lvl_tokens[:, 7] = staleness

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

    return LOBSequence(
        market_slug=market_slug,
        per_level=np.stack(per_level_rows, axis=0),
        per_tick=np.stack(per_tick_rows, axis=0),
        midprice=np.asarray(mids, dtype=np.float32),
        ts_sec=np.asarray(ts_list, dtype=np.int64),
    )


def pick_longest_market(data: BacktestData) -> str:
    counts: dict[str, int] = {}
    for tick in data.timeline:
        for slug, sb in tick.order_books.items():
            if sb.yes_book.bids and sb.yes_book.asks:
                counts[slug] = counts.get(slug, 0) + 1
    if not counts:
        raise RuntimeError("No markets with 2-sided books in timeline")
    return max(counts.items(), key=lambda kv: kv[1])[0]


# --- Normalization ---------------------------------------------------------


@dataclass
class NormalizationStats:
    per_level_mean: np.ndarray
    per_level_std: np.ndarray
    per_tick_mean: np.ndarray
    per_tick_std: np.ndarray

    def to_json(self) -> dict:
        return {
            "per_level_mean": self.per_level_mean.tolist(),
            "per_level_std":  self.per_level_std.tolist(),
            "per_tick_mean":  self.per_tick_mean.tolist(),
            "per_tick_std":   self.per_tick_std.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "NormalizationStats":
        return cls(
            per_level_mean=np.asarray(d["per_level_mean"], dtype=np.float32),
            per_level_std=np.asarray(d["per_level_std"], dtype=np.float32),
            per_tick_mean=np.asarray(d["per_tick_mean"], dtype=np.float32),
            per_tick_std=np.asarray(d["per_tick_std"], dtype=np.float32),
        )


def fit_normalization(seq: LOBSequence, eps: float = 1e-6) -> NormalizationStats:
    pl = seq.per_level.reshape(-1, F_LEVEL)
    pt = seq.per_tick
    return NormalizationStats(
        per_level_mean=pl.mean(axis=0).astype(np.float32),
        per_level_std=(pl.std(axis=0) + eps).astype(np.float32),
        per_tick_mean=pt.mean(axis=0).astype(np.float32),
        per_tick_std=(pt.std(axis=0) + eps).astype(np.float32),
    )


def apply_normalization(seq: LOBSequence, stats: NormalizationStats) -> LOBSequence:
    pl = (seq.per_level - stats.per_level_mean) / stats.per_level_std
    pt = (seq.per_tick - stats.per_tick_mean) / stats.per_tick_std
    return LOBSequence(
        market_slug=seq.market_slug,
        per_level=pl.astype(np.float32),
        per_tick=pt.astype(np.float32),
        midprice=seq.midprice,
        ts_sec=seq.ts_sec,
    )


def save_normalization(stats: NormalizationStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_json(), indent=2))


def load_normalization(path: Path) -> NormalizationStats:
    return NormalizationStats.from_json(json.loads(path.read_text()))


# --- CLI entry point -------------------------------------------------------


def _cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--market", type=str, default=None)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--norm-out", type=Path, default=None)
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
        stats = fit_normalization(seq)
        save_normalization(stats, args.norm_out)
        print(f"normalization saved to {args.norm_out}")


if __name__ == "__main__":
    _cli()
