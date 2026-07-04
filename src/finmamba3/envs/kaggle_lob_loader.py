"""Kaggle high-frequency crypto spot LOB loader (Binance-style books).

The Kaggle dataset ships one CSV per asset and resolution (ADA, BTC, ETH x
{1sec, 1min, 5min}). Each row is a book snapshot:

  system_time, midpoint, spread, buys, sells,
  bids_distance_0..14, asks_distance_0..14    (signed *relative* distance from mid)
  bids_notional_0..14, asks_notional_0..14    (USD depth per level)
  {bids,asks}_{cancel,limit,market}_notional_0..14   (order-flow decomposition)

This loader is the spot sibling of fi2010_loader.py: it emits the repo's LOBSequence
in the Polymarket base 8/14 schema (K=10, F_level=8, F_tick=14, FeatureDim=94) so the
encoder, normalization and decoder are unchanged. The K=10 rows are the five best bid
levels then the five best ask levels, so the encoder sees a symmetric two-sided book in
the existing LEVEL_FEATURE_NAMES layout. This is spot data with no settlement or expiry,
so no binary-market and no spot/TTE features are produced.

Public entry point: load_kaggle_lob(csv_path, asset, resolution).
"""
# region imports
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas
from finmamba3.envs.lob_features import F_LEVEL, F_TICK, K_LEVELS, LOBSequence
# endregion
logger = logging.getLogger(__name__)
# The encoder's K=10 tokens are split into five bid then five ask levels so both sides reach the encoder (module docstring).
KAGGLE_K_LEVELS = K_LEVELS
KAGGLE_LEVELS_PER_SIDE = K_LEVELS // 2
# The CSV ships 15 raw levels per side; the deepest five are dropped to match the encoder's K=10.
KAGGLE_BOOK_DEPTH = 15
KAGGLE_F_LEVEL = F_LEVEL
KAGGLE_F_TICK = F_TICK
KAGGLE_FEATURE_DIM = KAGGLE_K_LEVELS * KAGGLE_F_LEVEL + KAGGLE_F_TICK
# Rows per hour by resolution maps an --hours slice to a pandas nrows cap; the 1sec files are ~1.8 GB, so this is load-bearing.
ROWS_PER_HOUR_BY_RESOLUTION = {"1sec": 3600, "1min": 60, "5min": 12}
ASSET_NAMES = ("BTC", "ETH", "ADA")
# Pure-sign next-tick direction by default; training buckets direction itself, so this only labels the diagnostic array.
DEFAULT_FLAT_THRESHOLD = 0.0
DEFAULT_VOL_WINDOW = 20


@dataclass
class KaggleLOBSequence:
    sequence: LOBSequence
    # Shape (T,), int64 in {0=down, 1=flat, 2=up}; next-tick mid direction.
    direction_labels: np.ndarray
    asset: str
    resolution: str


def _read_kaggle_frame(csv_path: Path, max_rows: int | None) -> pandas.DataFrame:
    # Read only the columns the 8/14 schema consumes; usecols keeps the 1sec files from materializing unused columns.
    needed = ["midpoint", "spread", "buys", "sells"]
    for k in range(KAGGLE_LEVELS_PER_SIDE):
        needed.append(f"bids_distance_{k}")
        needed.append(f"asks_distance_{k}")
    for k in range(KAGGLE_BOOK_DEPTH):
        needed.append(f"bids_notional_{k}")
        needed.append(f"asks_notional_{k}")
        needed.append(f"bids_limit_notional_{k}")
        needed.append(f"asks_limit_notional_{k}")
        needed.append(f"bids_market_notional_{k}")
        needed.append(f"asks_market_notional_{k}")
    # `nrows` caps the read for an --hours slice; dropna removes gap snapshots so the finiteness check guards genuine bugs.
    frame = pandas.read_csv(csv_path, usecols=needed, nrows=max_rows)
    frame = frame[needed].dropna(axis=0, how="any").reset_index(drop=True)
    if frame.shape[0] < 2:
        raise RuntimeError(f"Kaggle CSV {csv_path} yields fewer than two usable rows after dropna.")
    return frame


def _encode_side(distance: np.ndarray, notional: np.ndarray, side_sign: float) -> np.ndarray:
    """Five-level x eight-feature per-side tokens in the LEVEL_FEATURE_NAMES layout.

    Bids (side +1) and asks (side -1) share one definition so the encoder sees a symmetric book.
    Cumulative depth and volume share are computed within the side, which is the clean per-side
    reading of the Polymarket vol_share feature for a dense two-sided book. log_staleness is zero
    because a Kaggle snapshot has no book-versus-tick timestamp lag.
    """
    num_ticks = distance.shape[0]
    tokens = np.zeros((num_ticks, KAGGLE_LEVELS_PER_SIDE, KAGGLE_F_LEVEL), dtype=np.float32)
    cumulative_depth = np.cumsum(np.clip(notional, 0.0, None), axis=1)
    side_total_depth = np.clip(cumulative_depth[:, -1:], 1e-9, None)
    volume_share = cumulative_depth / side_total_depth
    relative_gap = np.zeros_like(distance)
    relative_gap[:, 0] = distance[:, 0]
    relative_gap[:, 1:] = distance[:, 1:] - distance[:, :-1]
    tokens[:, :, 0] = distance
    tokens[:, :, 1] = np.log1p(np.clip(notional, 0.0, None))
    tokens[:, :, 2] = np.log1p(cumulative_depth)
    tokens[:, :, 3] = volume_share
    tokens[:, :, 4] = relative_gap
    tokens[:, :, 5] = np.linspace(-1.0, 1.0, KAGGLE_LEVELS_PER_SIDE, dtype=np.float32)
    tokens[:, :, 6] = side_sign
    return tokens


def _rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    """Trailing standard deviation over `window` samples, matching extract_features' rolling_vol.

    The first sample has no history and is zero; thereafter each value is the std of the trailing
    window, so the Kaggle tick block measures volatility on the same definition as the Polymarket
    pipeline rather than inventing a second one.
    """
    num_values = series.shape[0]
    rolling = np.zeros(num_values, dtype=np.float64)
    for end in range(1, num_values):
        start = max(0, end + 1 - window)
        rolling[end] = float(np.std(series[start : end + 1]))
    return rolling.astype(np.float32)


def _build_tick_features(
    mid: np.ndarray,
    spread: np.ndarray,
    buys: np.ndarray,
    sells: np.ndarray,
    bid_distance_top: np.ndarray,
    ask_distance_top: np.ndarray,
    bid_notional_all: np.ndarray,
    ask_notional_all: np.ndarray,
    vol_window: int,
) -> np.ndarray:
    """The 14 TICK_FEATURE_NAMES aggregates for the spot book (no binary-market block).

    Best bid/ask prices are recovered from the signed relative distances and the mid; microprice,
    imbalance and OFI use the top-of-book notionals as the size proxy. trade_intensity is the bar's
    traded notional (buys + sells), the spot analogue of the Polymarket top-5 size churn.
    """
    best_bid = mid * (1.0 + bid_distance_top)
    best_ask = mid * (1.0 + ask_distance_top)
    top_bid_size = bid_notional_all[:, 0]
    top_ask_size = ask_notional_all[:, 0]
    top_sum = top_bid_size + top_ask_size
    safe_top = np.clip(top_sum, 1e-9, None)
    imbalance = np.where(top_sum > 0.0, top_bid_size / safe_top, 0.5)
    microprice = np.where(top_sum > 0.0, (best_ask * top_bid_size + best_bid * top_ask_size) / safe_top, mid)
    weighted_mid_disp = (microprice - mid) / np.clip(mid, 1e-9, None)
    total_bid_vol = bid_notional_all.sum(axis=1)
    total_ask_vol = ask_notional_all.sum(axis=1)
    dmid = np.zeros_like(mid)
    dmid[1:] = mid[1:] - mid[:-1]
    dspread = np.zeros_like(spread)
    dspread[1:] = spread[1:] - spread[:-1]
    dimbalance = np.zeros_like(imbalance)
    dimbalance[1:] = imbalance[1:] - imbalance[:-1]
    bid_size_change = np.zeros_like(top_bid_size)
    bid_size_change[1:] = top_bid_size[1:] - top_bid_size[:-1]
    ask_size_change = np.zeros_like(top_ask_size)
    ask_size_change[1:] = top_ask_size[1:] - top_ask_size[:-1]
    ofi_top = bid_size_change - ask_size_change
    trade_intensity = buys + sells
    rolling_vol = _rolling_std(mid, vol_window)
    tick = np.stack(
        [
            mid, spread, np.log1p(np.clip(spread, 0.0, None)),
            imbalance, microprice, weighted_mid_disp,
            np.log1p(np.clip(total_bid_vol, 0.0, None)),
            np.log1p(np.clip(total_ask_vol, 0.0, None)),
            dmid, dspread, dimbalance,
            ofi_top, trade_intensity, rolling_vol,
        ],
        axis=1,
    ).astype(np.float32)
    return tick


def _direction_labels(mid: np.ndarray, flat_threshold: float) -> np.ndarray:
    """Next-tick direction from the fractional mid change in the head's {0=down, 1=flat, 2=up} encoding.

    The flat bucket is |return| <= flat_threshold; the final tick has no successor and is flat-filled.
    This array is the loader's diagnostic / benchmark label, analogous to FI2010Sequence.direction_labels;
    the world model derives its own next-tick target from the normalized mid channel during training.
    """
    num_values = mid.shape[0]
    labels = np.full(num_values, 1, dtype=np.int64)
    forward_return = np.zeros(num_values, dtype=np.float64)
    forward_return[:-1] = (mid[1:] - mid[:-1]) / np.clip(mid[:-1], 1e-9, None)
    labels[forward_return > flat_threshold] = 2
    labels[forward_return < -flat_threshold] = 0
    labels[-1] = 1
    return labels


def max_rows_for_hours(hours: float | None, resolution: str) -> int | None:
    """Translate an --hours slice into a pandas nrows cap for the given resolution.

    None hours means read the whole file; a positive value is rounded up to at least two rows so the
    smallest slice still yields a usable sequence.
    """
    if hours is None or hours <= 0.0:
        return None
    if resolution not in ROWS_PER_HOUR_BY_RESOLUTION:
        raise ValueError(f"resolution must be one of {sorted(ROWS_PER_HOUR_BY_RESOLUTION)}, got {resolution!r}.")
    return max(2, int(round(hours * ROWS_PER_HOUR_BY_RESOLUTION[resolution])))


def load_kaggle_lob(
    csv_path: Path,
    asset: str,
    resolution: str,
    max_rows: int | None = None,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> KaggleLOBSequence:
    """Load one Kaggle CSV as an LOBSequence in the 8/14 schema plus next-tick direction labels.

    Returns raw (un-normalized) per_level and per_tick so the trainer fits its own z-score stats,
    exactly like load_fi2010_split. asset and resolution are recorded for provenance and config
    wiring; they do not change the feature math (the regime axes are computed on the mid series).
    """
    if asset not in ASSET_NAMES:
        raise ValueError(f"asset must be one of {list(ASSET_NAMES)}, got {asset!r}.")
    if resolution not in ROWS_PER_HOUR_BY_RESOLUTION:
        raise ValueError(f"resolution must be one of {sorted(ROWS_PER_HOUR_BY_RESOLUTION)}, got {resolution!r}.")
    frame = _read_kaggle_frame(Path(csv_path), max_rows)
    mid = frame["midpoint"].to_numpy(dtype=np.float64)
    if not (mid > 0.0).all():
        raise ValueError(f"Kaggle CSV {csv_path} has non-positive midpoint values; the book is malformed.")
    spread = frame["spread"].to_numpy(dtype=np.float64)
    buys = frame["buys"].to_numpy(dtype=np.float64)
    sells = frame["sells"].to_numpy(dtype=np.float64)
    bid_distance = np.stack([frame[f"bids_distance_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_LEVELS_PER_SIDE)], axis=1)
    ask_distance = np.stack([frame[f"asks_distance_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_LEVELS_PER_SIDE)], axis=1)
    bid_notional_top = np.stack([frame[f"bids_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_LEVELS_PER_SIDE)], axis=1)
    ask_notional_top = np.stack([frame[f"asks_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_LEVELS_PER_SIDE)], axis=1)
    bid_notional_all = np.stack([frame[f"bids_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_BOOK_DEPTH)], axis=1)
    ask_notional_all = np.stack([frame[f"asks_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_BOOK_DEPTH)], axis=1)
    bid_limit_all = np.stack([frame[f"bids_limit_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_BOOK_DEPTH)], axis=1)
    ask_limit_all = np.stack([frame[f"asks_limit_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_BOOK_DEPTH)], axis=1)
    bid_market_all = np.stack([frame[f"bids_market_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_BOOK_DEPTH)], axis=1)
    ask_market_all = np.stack([frame[f"asks_market_notional_{k}"].to_numpy(dtype=np.float64) for k in range(KAGGLE_BOOK_DEPTH)], axis=1)
    bid_tokens = _encode_side(bid_distance, bid_notional_top, +1.0)
    ask_tokens = _encode_side(ask_distance, ask_notional_top, -1.0)
    per_level = np.concatenate([bid_tokens, ask_tokens], axis=1).astype(np.float32)
    per_tick = _build_tick_features(
        mid, spread, buys, sells, bid_distance[:, 0], ask_distance[:, 0],
        bid_notional_all, ask_notional_all, vol_window,
    )
    if not np.isfinite(per_level).all() or not np.isfinite(per_tick).all():
        raise ValueError(f"Non-finite features built from {csv_path}; check the source columns.")
    direction_labels = _direction_labels(mid, flat_threshold)
    num_ticks = mid.shape[0]
    buy_event_count = (bid_limit_all > 0.0).sum(axis=1) + (ask_market_all > 0.0).sum(axis=1)
    sell_event_count = (ask_limit_all > 0.0).sum(axis=1) + (bid_market_all > 0.0).sum(axis=1)
    event_counts = np.stack([buy_event_count, sell_event_count], axis=1).astype(np.float32)
    sequence = LOBSequence(
        market_slug=f"kaggle_{asset.lower()}_{resolution}",
        per_level=per_level,
        per_tick=per_tick,
        midprice=mid.astype(np.float32),
        ts_sec=np.arange(num_ticks, dtype=np.int64),
        yes_outcome=None,
        event_counts=event_counts,
    )
    logger.info(
        f"kaggle {asset} {resolution}: {num_ticks} ticks, per_level={per_level.shape}, "
        f"per_tick={per_tick.shape}, direction {{0,1,2}} -> "
        f"{np.bincount(direction_labels, minlength=3).tolist()}"
    )
    return KaggleLOBSequence(sequence=sequence, direction_labels=direction_labels, asset=asset, resolution=resolution)
