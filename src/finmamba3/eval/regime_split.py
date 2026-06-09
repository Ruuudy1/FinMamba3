"""Regime-shift train/test split for evaluating non-stationarity adaptation.

The episodic-memory ablation thesis is "memory helps under regime shift,"
which only makes sense if there is an explicit shift. This module produces
two complementary splits over a sequence of resolved markets:

1. Time split: train on markets resolved before a cutoff date; evaluate on
   markets resolved after. Tests pure temporal generalization.
2. Volatility split: train on markets whose realized mid-volatility is below
   the training-set median; evaluate on the high-vol tail. Tests whether
   the model survives a vol regime change without seeing it in training.

Both splits return the same MarketLifecycle objects already used by the
backtester, so no data-format change is required downstream.
"""
# region imports
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from finmamba3.backtester.data_loader import _asset_from_slug
from finmamba3.backtester.strategy import MarketLifecycle
# endregion
if TYPE_CHECKING:
    from finmamba3.backtester.data_loader import TickData


@dataclass
class RegimeSplitResult:
    train_markets: list[MarketLifecycle]
    test_markets: list[MarketLifecycle]
    description: str


@dataclass
class WindowSplitResult:
    low_vol_windows: list[tuple[int, int]]
    high_vol_windows: list[tuple[int, int]]
    description: str


def window_volatility_split(
    midprice: np.ndarray,
    window_len: int,
    quantile: float = 0.5,
) -> WindowSplitResult:
    """Split one concatenated LOB stream into low- and high-volatility windows.

    FI-2010 ships as a single event stream with no market or day boundaries, so the
    per-market volatility_split has no segment key to group by. The stream is chopped
    into non-overlapping windows of window_len events, realized volatility is measured
    per window, and the windows are split at the vol quantile: windows at or below the
    threshold form the low-vol reference regime and the rest form the high-vol shifted
    regime. Each returned (start, end) pair indexes the stream so the caller can slice
    its LOBSequence into one sub-sequence per window, the direct analog of the per-market
    list volatility_split returns. This keeps the FI-2010 difference-of-differences a fair
    comparison with the Polymarket vol-median split: only the dataset changes, not the
    split definition.

    Volatility is the standard deviation of the first difference of the mid, not of
    fractional returns: FI-2010 mids are already z-scored or decimal-scaled and can be
    zero or negative, which makes a fractional return ill-defined, while the split only
    needs a consistent volatility ordering across windows measured on the same scale.
    """
    num_events = int(midprice.shape[0])
    assert window_len >= 2, "window_len must be at least 2 to estimate volatility."
    mid = np.asarray(midprice, dtype=np.float64)
    bounds = [
        (start, start + window_len)
        for start in range(0, num_events - window_len + 1, window_len)
    ]
    assert len(bounds) >= 2, (
        f"stream of {num_events} events yields fewer than two full windows at "
        f"window_len={window_len}; lower window_len or pass a longer stream."
    )
    vols = np.fromiter(
        (float(np.std(np.diff(mid[start:end]))) for start, end in bounds),
        dtype=np.float64,
        count=len(bounds),
    )
    threshold = float(np.quantile(vols, quantile))
    low_vol_windows = [bounds[i] for i in range(len(bounds)) if vols[i] <= threshold]
    high_vol_windows = [bounds[i] for i in range(len(bounds)) if vols[i] > threshold]
    return WindowSplitResult(
        low_vol_windows=low_vol_windows,
        high_vol_windows=high_vol_windows,
        description=f"win{window_len}-vol>{threshold:.5f}",
    )


def realized_vol_from_timeline(timeline: list[TickData]) -> dict[str, float]:
    """Realized YES-mid return volatility per market over the loaded timeline.

    Samples each market's YES-book mid once per distinct book snapshot, skipping
    forward-filled repeats so idle ticks do not deflate the estimate, then takes
    the std of one-step fractional mid-returns. Markets with fewer than two
    usable mids are omitted; volatility_split reads a missing slug as undefined.
    """
    mids_by_slug: dict[str, list[float]] = {}
    last_book_ts: dict[str, int] = {}
    for tick in timeline:
        for slug, book in tick.order_books.items():
            if last_book_ts.get(slug) == book.book_ts:
                continue
            last_book_ts[slug] = book.book_ts
            mid = book.yes_book.mid
            if mid > 0:
                mids_by_slug.setdefault(slug, []).append(mid)
    vol_by_slug: dict[str, float] = {}
    for slug, mids in mids_by_slug.items():
        if len(mids) < 2:
            continue
        arr = np.asarray(mids, dtype=np.float64)
        vol_by_slug[slug] = float(np.std(np.diff(arr) / arr[:-1]))
    return vol_by_slug


def market_spot_series(
    timeline: list["TickData"],
    lifecycles: list[MarketLifecycle],
) -> dict[str, np.ndarray]:
    """Per-market underlying Chainlink-spot price series over its [start_ts, end_ts] window.

    Forward-filled repeats are dropped so idle seconds do not distort downstream estimators; a
    market with fewer than two distinct spot prices is omitted. Both the spot-vol axis (Phase 0.5)
    and the predictability axis (Phase 0.7) read these series, so the per-market extraction lives
    here once.
    """
    num_ticks = len(timeline)
    ts_array = np.fromiter((int(tick.ts_sec) for tick in timeline), dtype=np.int64, count=num_ticks)
    price_by_asset = {
        "BTC": np.fromiter((float(tick.chainlink_btc) for tick in timeline), dtype=np.float64, count=num_ticks),
        "ETH": np.fromiter((float(tick.chainlink_eth) for tick in timeline), dtype=np.float64, count=num_ticks),
        "SOL": np.fromiter((float(tick.chainlink_sol) for tick in timeline), dtype=np.float64, count=num_ticks),
    }
    series_by_slug: dict[str, np.ndarray] = {}
    for lifecycle in lifecycles:
        prices = price_by_asset[_asset_from_slug(lifecycle.market_slug)]
        lo = int(np.searchsorted(ts_array, lifecycle.start_ts, side="left"))
        hi = int(np.searchsorted(ts_array, lifecycle.end_ts, side="right"))
        window = prices[lo:hi]
        window = window[window > 0.0]
        if window.shape[0] < 2:
            continue
        moved = window[np.concatenate([[True], window[1:] != window[:-1]])]
        if moved.shape[0] >= 2:
            series_by_slug[lifecycle.market_slug] = moved
    return series_by_slug


def spot_realized_vol_from_timeline(
    timeline: list["TickData"],
    lifecycles: list[MarketLifecycle],
) -> dict[str, float]:
    """Realized underlying-spot return volatility per market (Phase 0.5, the honest regime axis).

    The prior split measured volatility on the YES-mid (the implied probability), which
    mechanically explodes toward 0/1 near settlement, so "high vol" was confounded with "near
    expiry". The std of one-step fractional Chainlink-spot returns inside each market's window
    removes that confound.
    """
    vol_by_slug: dict[str, float] = {}
    for slug, series in market_spot_series(timeline, lifecycles).items():
        vol_by_slug[slug] = float(np.std(np.diff(series) / series[:-1]))
    return vol_by_slug


def time_split(
    markets: list[MarketLifecycle],
    cutoff_ts: float,
) -> RegimeSplitResult:
    """Split markets by their end_ts against a Unix-second cutoff."""
    train = [m for m in markets if m.end_ts < cutoff_ts]
    test = [m for m in markets if m.end_ts >= cutoff_ts]
    return RegimeSplitResult(
        train_markets=train,
        test_markets=test,
        description=f"time<{cutoff_ts}",
    )


def volatility_split(
    markets: list[MarketLifecycle],
    realized_vol: dict[str, float],
    quantile: float = 0.5,
) -> RegimeSplitResult:
    """Split markets by realized mid-vol against a training-set quantile.

    realized_vol is a dict mapping market_slug -> realized rolling volatility
    measured on the training side, usually std of mid-returns over the market's
    lifetime. Markets with vol below the threshold go to train; the rest test.
    """
    if not markets:
        return RegimeSplitResult(train_markets=[], test_markets=[], description="empty")
    vol_by_market = realized_vol
    vols = np.fromiter(
        (vol_by_market.get(m.market_slug, np.nan) for m in markets),
        dtype=np.float64,
        count=len(markets),
    )
    finite = vols[np.isfinite(vols)]
    if finite.size == 0:
        return RegimeSplitResult(
            train_markets=list(markets),
            test_markets=[],
            description="vol-undefined",
        )
    threshold = float(np.quantile(finite, quantile))
    train = [m for m, v in zip(markets, vols) if np.isfinite(v) and v <= threshold]
    test = [m for m, v in zip(markets, vols) if np.isfinite(v) and v > threshold]
    return RegimeSplitResult(
        train_markets=train,
        test_markets=test,
        description=f"vol>{threshold:.5f}",
    )
