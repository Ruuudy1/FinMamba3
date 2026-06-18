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
from finmamba3.backtester.strategy import MarketLifecycle
# endregion
if TYPE_CHECKING:
    from finmamba3.backtester.data_loader import TickData


@dataclass
class RegimeSplitResult:
    train_markets: list[MarketLifecycle]
    test_markets: list[MarketLifecycle]
    description: str


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
