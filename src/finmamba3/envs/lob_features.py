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
from finmamba3.backtester import BacktestData, OrderBookSnapshot, TickData, build_timeline
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
# The spot/TTE/asset block (Phase 0.1/0.2/0.6) is the causal channels the prior pipeline
# discarded: the underlying Chainlink-spot path that defines settlement, a time-to-expiry
# clock, and a one-hot asset id. It is always appended last, so its flat offset is derivable
# as FeatureDimTick - SPOT_BLOCK_WIDTH no matter whether the binary block is present.
SPOT_TICK_FEATURE_NAMES = (
    "spot_logret_since_open",
    "spot_realized_vol",
    "spot_signed_distance",
    "tte_frac",
    "log_seconds_to_expiry",
    "asset_is_btc",
    "asset_is_eth",
    "asset_is_sol",
)
SPOT_BLOCK_WIDTH = len(SPOT_TICK_FEATURE_NAMES)
F_TICK_SPOT = F_TICK + SPOT_BLOCK_WIDTH
F_TICK_BINARY_SPOT = F_TICK_BINARY + SPOT_BLOCK_WIDTH
# Spot return/vol/distance are O(1e-3); the clock and one-hot are O(1). The floors keep a
# near-constant channel (e.g. a single-asset slice's one-hot) from exploding under z-score.
DEFAULT_SPOT_TICK_STD_FLOOR = np.asarray(
    [1e-4, 1e-4, 1e-4, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2],
    dtype=np.float32,
)


def _tick_std_floor_for_width(f_tick: int, eps: float) -> np.ndarray:
    # The floor is composed from the same blocks that build the tick vector, so every
    # Polymarket schema (base 14, +binary, +spot, +binary+spot) gets pinned floors; other
    # widths such as FI-2010 fall back to a flat eps floor.
    base = DEFAULT_TICK_STD_FLOOR
    binary = DEFAULT_BINARY_TICK_STD_FLOOR
    spot = DEFAULT_SPOT_TICK_STD_FLOOR
    if f_tick == base.shape[0]:
        return base
    if f_tick == base.shape[0] + binary.shape[0]:
        return np.concatenate([base, binary])
    if f_tick == base.shape[0] + spot.shape[0]:
        return np.concatenate([base, spot])
    if f_tick == base.shape[0] + binary.shape[0] + spot.shape[0]:
        return np.concatenate([base, binary, spot])
    return np.full(f_tick, eps, dtype=np.float32)


def tick_feature_names_for_width(f_tick: int) -> tuple:
    # Mirror _tick_std_floor_for_width so saved normalization and diagnostics label the
    # channels with the schema that produced them.
    base = TICK_FEATURE_NAMES
    binary = BINARY_TICK_FEATURE_NAMES
    spot = SPOT_TICK_FEATURE_NAMES
    if f_tick == len(base):
        return base
    if f_tick == len(base) + len(binary):
        return base + binary
    if f_tick == len(base) + len(spot):
        return base + spot
    if f_tick == len(base) + len(binary) + len(spot):
        return base + binary + spot
    return tuple(f"tick_{i}" for i in range(f_tick))


FLAT_FEATURE_NAMES_SPOT = FLAT_FEATURE_NAMES + tuple(
    f"tick.{name}" for name in SPOT_TICK_FEATURE_NAMES
)
FLAT_FEATURE_NAMES_BINARY_SPOT = FLAT_FEATURE_NAMES_BINARY + tuple(
    f"tick.{name}" for name in SPOT_TICK_FEATURE_NAMES
)


@dataclass
class LOBSequence:
    market_slug: str
    per_level: np.ndarray  # Shape (T, K_LEVELS, F_LEVEL) float32.
    per_tick: np.ndarray   # Shape (T, F_TICK) float32.
    midprice: np.ndarray   # Shape (T,) float32, raw unnormalized mid.
    ts_sec: np.ndarray     # Shape (T,) int64.
    yes_outcome: np.ndarray | None = None  # Shape (T,) float32 in {0, 1}, or nan if unknown.
    event_counts: np.ndarray | None = None  # Shape (T, 2) float32 buy/sell arrivals for Hawkes loss.
    # Raw (unnormalized) supervision channels for the settlement loss (Phase 0.3): tte_frac in
    # [0, 1] weights the outcome BCE toward expiry, and the signed spot distance supplies the
    # observable running spot-sign target. None unless include_spot_features built them.
    tte_frac: np.ndarray | None = None
    spot_signed_distance: np.ndarray | None = None
    # Raw (unnormalized) book-relative edge supervision channels (Phase 2): best ask for YES/NO
    # tokens, YES book mid, and total YES book depth. Populated by extract_features() for every
    # usable tick; NaN when the NO book has no ask side. Used by BookRelativeEdgeHead.
    yes_ask: np.ndarray | None = None
    no_ask: np.ndarray | None = None
    yes_mid: np.ndarray | None = None
    book_depth: np.ndarray | None = None

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


def _asset_chainlink(tick: TickData, asset: str) -> float:
    # The Chainlink oracle is the source of truth for settlement, so the spot path the model
    # conditions on must read the same series compute_settlements resolves the outcome from.
    if asset == "BTC":
        return float(tick.chainlink_btc)
    if asset == "ETH":
        return float(tick.chainlink_eth)
    if asset == "SOL":
        return float(tick.chainlink_sol)
    raise ValueError(f"Unknown asset {asset!r}; expected BTC, ETH or SOL.")


def _asset_onehot(asset: str) -> tuple:
    if asset == "BTC":
        return (1.0, 0.0, 0.0)
    if asset == "ETH":
        return (0.0, 1.0, 0.0)
    if asset == "SOL":
        return (0.0, 0.0, 1.0)
    raise ValueError(f"Unknown asset {asset!r}; expected BTC, ETH or SOL.")


def extract_features(
    timeline: list[TickData],
    market_slug: str,
    vol_window: int = 20,
    yes_outcome: float | None = None,
    include_binary_features: bool = False,
    asset: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    spot_open: float | None = None,
    include_spot_features: bool = False,
) -> LOBSequence:
    """Per-tick feature tensors for a single market. Skips ticks with no
    2-sided YES book.

    When include_spot_features is set, the eight-channel spot/TTE/asset block (Phase
    0.1/0.2/0.6) is appended after the base (and optional binary) tick channels: the
    Chainlink-spot path that defines settlement, a time-to-expiry clock, and a one-hot
    asset id. asset, start_ts and end_ts are then required because the block is anchored to
    the market's open reference and expiry. spot_open (the open Chainlink reference the
    label compares against) is taken from the caller when positive, else recovered from the
    timeline. The raw tte_frac and signed distance are returned on the sequence so the
    settlement loss can weight by expiry and supervise the observable spot sign without
    re-deriving them from z-scored obs.
    """
    if include_spot_features and (asset is None or start_ts is None or end_ts is None):
        raise ValueError(
            "extract_features(include_spot_features=True) requires asset, start_ts and end_ts."
        )
    spot_open_ref = float(spot_open) if spot_open is not None else 0.0
    if include_spot_features and spot_open_ref <= 0.0:
        # The Chainlink open is the value nearest start_ts; the series is per-second dense so a
        # forward-filled tick at the boundary carries the open the settlement is defined against.
        best_gap = None
        for tick in timeline:
            spot_at_tick = _asset_chainlink(tick, asset)
            if spot_at_tick <= 0.0:
                continue
            gap = abs(int(tick.ts_sec) - int(start_ts))
            if best_gap is None or gap < best_gap:
                best_gap = gap
                spot_open_ref = spot_at_tick
        if spot_open_ref <= 0.0:
            raise RuntimeError(f"No positive Chainlink open reference for market {market_slug}")
    per_level_rows: list[np.ndarray] = []
    per_tick_rows: list[np.ndarray] = []
    mids: list[float] = []
    ts_list: list[int] = []
    mid_window: list[float] = []
    spot_rows: list[np.ndarray] = []
    tte_raw_list: list[float] = []
    spot_dist_raw_list: list[float] = []
    event_count_rows: list[np.ndarray] = []
    spot_logret_window: list[float] = []
    yes_ask_raw_list: list[float] = []
    no_ask_raw_list: list[float] = []
    yes_mid_raw_list: list[float] = []
    book_depth_raw_list: list[float] = []
    prev_spot: float | None = None
    tte_span = max(int(end_ts) - int(start_ts), 1) if include_spot_features else 1
    asset_onehot = _asset_onehot(asset) if include_spot_features else (0.0, 0.0, 0.0)
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
            buy_events = 1.0 if top_bid_size > prev_top_bid_size else 0.0
            sell_events = 1.0 if top_ask_size > prev_top_ask_size else 0.0
        else:
            ofi_top = 0.0
            buy_events = 0.0
            sell_events = 0.0
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
        event_count_rows.append(np.array([buy_events, sell_events], dtype=np.float32))
        mids.append(mid)
        ts_list.append(int(tick.ts_sec))
        # Book-relative edge supervision channels: YES ask/mid/depth always available (YES book is
        # two-sided by the loop guard above); NO ask is NaN when the NO book has no ask levels.
        no_book = sb.no_book
        yes_ask_raw_list.append(float(book.best_ask))
        yes_mid_raw_list.append(float(book.mid))
        no_ask_raw_list.append(float(no_book.best_ask) if no_book.asks else float('nan'))
        book_depth_raw_list.append(float(book.total_bid_size + book.total_ask_size))
        if include_spot_features:
            spot_now = _asset_chainlink(tick, asset)
            if spot_now <= 0.0:
                spot_now = spot_open_ref
            spot_logret = math.log(spot_now / spot_open_ref)
            signed_distance = (spot_now - spot_open_ref) / spot_open_ref
            # Realized vol is the rolling std of per-tick spot log-returns over the same window
            # as the mid's rolling_vol, so the two volatilities share one cadence and definition.
            step_logret = math.log(spot_now / prev_spot) if prev_spot is not None and prev_spot > 0.0 else 0.0
            spot_logret_window.append(step_logret)
            if len(spot_logret_window) > vol_window:
                spot_logret_window.pop(0)
            spot_vol = float(np.std(spot_logret_window)) if len(spot_logret_window) >= 2 else 0.0
            seconds_to_expiry = max(int(end_ts) - int(tick.ts_sec), 0)
            tte_frac = min(max(seconds_to_expiry / tte_span, 0.0), 1.0)
            log_sec_to_expiry = math.log1p(float(seconds_to_expiry))
            spot_rows.append(np.array([
                spot_logret, spot_vol, signed_distance,
                tte_frac, log_sec_to_expiry,
                asset_onehot[0], asset_onehot[1], asset_onehot[2],
            ], dtype=np.float32))
            tte_raw_list.append(tte_frac)
            spot_dist_raw_list.append(signed_distance)
            prev_spot = spot_now
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
    tte_frac_array = None
    spot_signed_distance_array = None
    if include_spot_features:
        per_tick_array = np.concatenate([per_tick_array, np.stack(spot_rows, axis=0)], axis=1)
        tte_frac_array = np.asarray(tte_raw_list, dtype=np.float32)
        spot_signed_distance_array = np.asarray(spot_dist_raw_list, dtype=np.float32)
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
        event_counts=np.stack(event_count_rows, axis=0).astype(np.float32),
        tte_frac=tte_frac_array,
        spot_signed_distance=spot_signed_distance_array,
        yes_ask=np.asarray(yes_ask_raw_list, dtype=np.float32),
        no_ask=np.asarray(no_ask_raw_list, dtype=np.float32),
        yes_mid=np.asarray(yes_mid_raw_list, dtype=np.float32),
        book_depth=np.asarray(book_depth_raw_list, dtype=np.float32),
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


def _slug_asset(slug: str) -> str:
    """Asset symbol a market slug belongs to, for per-asset market selection (default BTC)."""
    s = slug.lower()
    if s.startswith("eth") or s.startswith("ethereum"):
        return "ETH"
    if s.startswith("sol") or s.startswith("solana"):
        return "SOL"
    return "BTC"


def pick_top_markets(data: BacktestData, n: int, min_ticks: int = 128, assets: list[str] | None = None) -> list[str]:
    """Return the slugs of the n longest 2-sided-book markets, longest first.

    Counts ticks whose YES book has both sides, then keeps only markets with at
    least min_ticks usable ticks so each yields at least one training window.
    When assets is given, restricts to those asset symbols (e.g. ['ETH']) so a
    per-asset model can be trained. Fewer than n markets may qualify; the caller
    gets whatever exists.
    """
    count_by_slug: dict[str, int] = {}
    for tick in data.timeline:
        for slug, sb in tick.order_books.items():
            if sb.yes_book.bids and sb.yes_book.asks:
                count_by_slug[slug] = count_by_slug.get(slug, 0) + 1
    qualified_by_count = [
        (slug, count) for slug, count in count_by_slug.items()
        if count >= min_ticks and (assets is None or _slug_asset(slug) in assets)
    ]
    if not qualified_by_count:
        raise RuntimeError(f"No markets with >= {min_ticks} two-sided-book ticks in timeline")
    qualified_by_count.sort(key=lambda kv: kv[1], reverse=True)
    top_slugs = [slug for slug, _ in qualified_by_count[:n]]
    return top_slugs


def pick_longest_market(data: BacktestData) -> str:
    # min_ticks=1 keeps the original contract: return the single longest market
    # regardless of length, raising only when no 2-sided book exists at all.
    return pick_top_markets(data, 1, min_ticks=1)[0]
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
        tick_feature_names = list(tick_feature_names_for_width(f_tick))
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
        event_counts=seq.event_counts,
        tte_frac=seq.tte_frac,
        spot_signed_distance=seq.spot_signed_distance,
        yes_ask=seq.yes_ask,
        no_ask=seq.no_ask,
        yes_mid=seq.yes_mid,
        book_depth=seq.book_depth,
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
        event_counts=seq.event_counts,
        tte_frac=seq.tte_frac,
        spot_signed_distance=seq.spot_signed_distance,
        yes_ask=seq.yes_ask,
        no_ask=seq.no_ask,
        yes_mid=seq.yes_mid,
        book_depth=seq.book_depth,
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
    elif flat_dim == len(FLAT_FEATURE_NAMES_SPOT):
        name_table = FLAT_FEATURE_NAMES_SPOT
    elif flat_dim == len(FLAT_FEATURE_NAMES_BINARY_SPOT):
        name_table = FLAT_FEATURE_NAMES_BINARY_SPOT
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
