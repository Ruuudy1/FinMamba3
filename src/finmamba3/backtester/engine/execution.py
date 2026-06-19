"""
Execution Engine — Order matching against recorded order books.

Handles validation, latency simulation, staleness checks, and walk-the-book
fill logic for realistic backtesting.
"""

# region imports
from __future__ import annotations
import logging
from dataclasses import dataclass
from finmamba3.backtester.strategy import (
    Fill,
    MarketView,
    Order,
    Side,
    Token,
)
# endregion

logger = logging.getLogger(__name__)

# Position limits
MAX_SHARES_PER_TOKEN = 500

# Staleness: reject fills if book is older than this many seconds
MAX_BOOK_STALENESS_S = 5

# Execution latency: orders at tick T fill at tick T+1
EXECUTION_LATENCY_TICKS = 1


@dataclass
class RejectedOrder:
    """An order that failed validation."""
    order: Order
    reason: str


@dataclass
class PendingOrder:
    """An order queued for execution at a future tick."""
    order: Order
    submit_tick: int  # tick when the order was submitted
    execute_tick: int  # tick when the order should be executed


class ExecutionEngine:
    """
    Handles order validation, queuing, and execution against recorded books.

    Pipeline:
    1. Validate — market is ACTIVE, sufficient cash, within position limits
    2. Latency — orders submitted at tick T fill at tick T+1
    3. Staleness — reject if last order book snapshot > 5s old
    4. Walk the book — consume recorded order book levels
    5. Book depletion — multiple orders on same tick see depleted book
    """

    def __init__(self):
        self._pending: list[PendingOrder] = []
        self._rejected: list[RejectedOrder] = []
        # Track consumed liquidity per tick per book for depletion
        self._depleted: dict[str, dict] = {}  # key = f"{slug}:{token}:{side}" -> consumed sizes per level

    def queue_orders(
        self,
        orders: list[Order],
        current_tick: int,
        cash: float,
        positions: dict,  # slug -> Position
        active_markets: dict[str, MarketView],
    ) -> tuple[list[PendingOrder], list[RejectedOrder]]:
        """
        Validate orders and queue valid ones for execution at next tick.

        Returns (queued, rejected) lists.
        """
        queued: list[PendingOrder] = []
        rejected: list[RejectedOrder] = []

        for order in orders:
            reason = self._validate_order(order, cash, positions, active_markets)
            if reason:
                rej = RejectedOrder(order=order, reason=reason)
                rejected.append(rej)
                self._rejected.append(rej)
                logger.debug(f"Order rejected: {reason} — {order}")
                continue

            pending = PendingOrder(
                order=order,
                submit_tick=current_tick,
                execute_tick=current_tick + EXECUTION_LATENCY_TICKS,
            )
            queued.append(pending)
            self._pending.append(pending)

            # Pre-reserve cash for BUY orders to prevent over-spending
            if order.side == Side.BUY:
                price = order.limit_price if order.limit_price else 1.0
                cash -= order.size * price

        return queued, rejected

    def execute_pending(
        self,
        current_tick: int,
        market_views: dict[str, MarketView],
        book_timestamps: dict[str, int],
    ) -> list[Fill]:
        """
        Execute all pending orders that are due at the current tick.

        Walks the recorded order book for realistic fills. Multiple orders
        on the same book see depleted liquidity.
        """
        # Reset depletion tracking for this tick
        self._depleted = {}

        due = [p for p in self._pending if p.execute_tick <= current_tick]
        remaining = [p for p in self._pending if p.execute_tick > current_tick]
        self._pending = remaining

        fills: list[Fill] = []

        for pending in due:
            order = pending.order
            view = market_views.get(order.market_slug)

            if not view:
                self._rejected.append(RejectedOrder(
                    order=order, reason="market no longer active at execution time"
                ))
                continue

            # Staleness check
            book_ts = book_timestamps.get(order.market_slug, 0)
            if current_tick - book_ts > MAX_BOOK_STALENESS_S:
                self._rejected.append(RejectedOrder(
                    order=order,
                    reason=f"book stale ({current_tick - book_ts}s > {MAX_BOOK_STALENESS_S}s)",
                ))
                continue

            # Walk the book
            fill = self._walk_the_book(order, view, current_tick)
            if fill:
                fills.append(fill)
            else:
                self._rejected.append(RejectedOrder(
                    order=order, reason="no liquidity available at limit price"
                ))

        return fills

    def _validate_order(
        self,
        order: Order,
        cash: float,
        positions: dict,
        active_markets: dict[str, MarketView],
    ) -> str | None:
        """Validate an order. Returns reason string if invalid, None if valid."""
        # Market must be active
        if order.market_slug not in active_markets:
            return f"market {order.market_slug} not active"

        # Size must be positive
        if order.size <= 0:
            return f"invalid size: {order.size}"

        # Limit price must be valid if set
        if order.limit_price is not None:
            if order.limit_price <= 0 or order.limit_price >= 1:
                return f"invalid limit price: {order.limit_price} (must be 0 < p < 1)"

        # Position limits
        pos = positions.get(order.market_slug)
        if order.side == Side.BUY:
            current = 0.0
            if pos:
                current = pos.yes_shares if order.token == Token.YES else pos.no_shares
            if current + order.size > MAX_SHARES_PER_TOKEN:
                return (
                    f"position limit exceeded: {current} + {order.size} > "
                    f"{MAX_SHARES_PER_TOKEN}"
                )

            # Cash check (conservative: use limit price or 1.0 for market orders)
            price = order.limit_price if order.limit_price else 1.0
            cost = order.size * price
            if cost > cash:
                return f"insufficient cash: need ${cost:.2f}, have ${cash:.2f}"

        elif order.side == Side.SELL:
            # Can only sell tokens you own (no short selling)
            current = 0.0
            if pos:
                current = pos.yes_shares if order.token == Token.YES else pos.no_shares
            if order.size > current:
                return (
                    f"cannot sell {order.size} {order.token.value} shares, "
                    f"only own {current}"
                )

        return None

    def _walk_the_book(
        self,
        order: Order,
        view: MarketView,
        current_tick: int,
    ) -> Fill | None:
        """
        Walk the recorded order book to fill an order.

        BUY orders consume ASK levels (buying from sellers).
        SELL orders consume BID levels (selling to buyers).
        Tracks depletion so multiple orders don't double-count liquidity.
        """
        # Select the right book side
        if order.token == Token.YES:
            book = view.yes_book
        else:
            book = view.no_book

        if order.side == Side.BUY:
            levels = list(book.asks)  # buy from asks (ascending price)
        else:
            levels = list(book.bids)  # sell to bids (descending price)

        if not levels:
            return None

        # Depletion key
        depletion_key = f"{order.market_slug}:{order.token.value}:{order.side.value}"
        consumed = self._depleted.get(depletion_key, {})

        remaining_size = order.size
        total_cost = 0.0
        total_filled = 0.0

        for i, level in enumerate(levels):
            # Check limit price
            if order.limit_price is not None:
                if order.side == Side.BUY and level.price > order.limit_price:
                    break
                if order.side == Side.SELL and level.price < order.limit_price:
                    break

            # Available size at this level (accounting for depletion)
            already_consumed = consumed.get(i, 0.0)
            available = max(0.0, level.size - already_consumed)

            if available <= 0:
                continue

            fill_at_level = min(remaining_size, available)
            total_cost += fill_at_level * level.price
            total_filled += fill_at_level
            remaining_size -= fill_at_level

            # Track depletion
            consumed[i] = already_consumed + fill_at_level

            if remaining_size <= 1e-9:
                break

        self._depleted[depletion_key] = consumed

        if total_filled <= 1e-9:
            return None

        avg_price = total_cost / total_filled

        return Fill(
            market_slug=order.market_slug,
            token=order.token,
            side=order.side,
            size=total_filled,
            avg_price=avg_price,
            cost=total_cost,
            timestamp=current_tick,
            order=order,
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def total_rejected(self) -> int:
        return len(self._rejected)

    @property
    def rejected_orders(self) -> list[RejectedOrder]:
        return list(self._rejected)
