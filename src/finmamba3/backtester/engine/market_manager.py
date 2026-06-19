"""
Market Manager — Tracks active markets, lifecycle state, and settlements.

State machine per market: UPCOMING -> ACTIVE -> SETTLED
"""

# region imports
from __future__ import annotations
import logging
from dataclasses import dataclass
from finmamba3.backtester.data_loader import TickData
from finmamba3.backtester.strategy import (
    MarketLifecycle,
    MarketStatus,
    MarketView,
    OrderBookSnapshot,
    Settlement,
)
# endregion

logger = logging.getLogger(__name__)


class MarketManager:
    """
    Manages market lifecycles and builds MarketView objects each tick.

    Promotes markets to ACTIVE when current_time >= start_ts.
    Settles markets when current_time >= end_ts.
    """

    def __init__(
        self,
        lifecycles: list[MarketLifecycle],
        settlements: dict[str, Settlement],
    ):
        self._lifecycles = {lc.market_slug: lc for lc in lifecycles}
        self._settlements = settlements
        self._settled_this_tick: list[Settlement] = []

    @property
    def lifecycles(self) -> dict[str, MarketLifecycle]:
        return self._lifecycles

    @property
    def settlements(self) -> dict[str, Settlement]:
        return self._settlements

    def get_settled_this_tick(self) -> list[Settlement]:
        """Return markets that were settled on the most recent tick."""
        return list(self._settled_this_tick)

    def update(self, current_ts: int) -> dict[str, MarketView]:
        """
        Advance market states and return current active MarketViews.

        Call this once per tick. Returns dict of slug -> MarketView for
        all currently active markets.
        """
        self._settled_this_tick = []
        active_views: dict[str, MarketView] = {}

        for slug, lc in self._lifecycles.items():
            # Transition: UPCOMING -> ACTIVE
            if lc.status == MarketStatus.UPCOMING and current_ts >= lc.start_ts:
                if current_ts < lc.end_ts:
                    lc.status = MarketStatus.ACTIVE
                else:
                    # Already past end — settle immediately
                    lc.status = MarketStatus.SETTLED
                    if slug in self._settlements:
                        self._settled_this_tick.append(self._settlements[slug])
                    continue

            # Transition: ACTIVE -> SETTLED
            if lc.status == MarketStatus.ACTIVE and current_ts >= lc.end_ts:
                lc.status = MarketStatus.SETTLED
                if slug in self._settlements:
                    self._settled_this_tick.append(self._settlements[slug])
                continue

            # Build MarketView for active markets
            if lc.status == MarketStatus.ACTIVE:
                time_remaining_s = max(0.0, float(lc.end_ts - current_ts))
                duration = float(lc.end_ts - lc.start_ts)
                time_remaining_frac = time_remaining_s / duration if duration > 0 else 0.0

                active_views[slug] = MarketView(
                    market_slug=slug,
                    interval=lc.interval,
                    start_ts=lc.start_ts,
                    end_ts=lc.end_ts,
                    time_remaining_s=time_remaining_s,
                    time_remaining_frac=time_remaining_frac,
                )

        return active_views

    def enrich_views(
        self,
        views: dict[str, MarketView],
        tick: TickData,
    ) -> dict[str, MarketView]:
        """
        Enrich MarketViews with current tick's order book and price data.

        Returns new dict with updated frozen MarketView objects (since they're frozen,
        we create new instances with the additional data).
        """
        enriched: dict[str, MarketView] = {}

        for slug, view in views.items():
            # The repo timeline stores each market's book as a StoredBook (yes_book/no_book attrs).
            stored = tick.order_books.get(slug)
            yes_book = stored.yes_book if stored is not None else OrderBookSnapshot()
            no_book = stored.no_book if stored is not None else OrderBookSnapshot()

            # Get price data from market_prices
            price_row = tick.market_prices.get(slug, {})
            yes_price = float(price_row.get("yes_price", 0) or 0)
            no_price = float(price_row.get("no_price", 0) or 0)
            yes_bid = float(price_row.get("yes_bid", 0) or 0)
            yes_ask = float(price_row.get("yes_ask", 0) or 0)
            no_bid = float(price_row.get("no_bid", 0) or 0)
            no_ask = float(price_row.get("no_ask", 0) or 0)

            # If no price data from table, derive from books
            if yes_price == 0 and yes_book.mid > 0:
                yes_price = yes_book.mid
            if no_price == 0 and no_book.mid > 0:
                no_price = no_book.mid
            if yes_bid == 0 and yes_book.best_bid > 0:
                yes_bid = yes_book.best_bid
            if yes_ask == 0 and yes_book.best_ask > 0:
                yes_ask = yes_book.best_ask
            if no_bid == 0 and no_book.best_bid > 0:
                no_bid = no_book.best_bid
            if no_ask == 0 and no_book.best_ask > 0:
                no_ask = no_book.best_ask

            enriched[slug] = MarketView(
                market_slug=slug,
                interval=view.interval,
                start_ts=view.start_ts,
                end_ts=view.end_ts,
                time_remaining_s=view.time_remaining_s,
                time_remaining_frac=view.time_remaining_frac,
                yes_book=yes_book,
                no_book=no_book,
                yes_price=yes_price,
                no_price=no_price,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
            )

        return enriched

    def is_market_active(self, slug: str) -> bool:
        """Check if a market is currently active."""
        lc = self._lifecycles.get(slug)
        return lc is not None and lc.status == MarketStatus.ACTIVE

    def get_all_settled(self) -> list[str]:
        """Return slugs of all settled markets."""
        return [
            slug for slug, lc in self._lifecycles.items()
            if lc.status == MarketStatus.SETTLED
        ]
