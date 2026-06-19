"""
Portfolio — Cash, positions, P&L tracking.

Tracks cash balance, token positions per market, and handles settlement payouts.
"""

# region imports
from __future__ import annotations
import logging
from dataclasses import dataclass
from finmamba3.backtester.strategy import (
    Fill,
    MarketView,
    PositionView,
    Settlement,
    Side,
    Token,
)
# endregion

logger = logging.getLogger(__name__)

DEFAULT_STARTING_CASH = 10_000.0


@dataclass
class Position:
    """Mutable position in a single market."""
    market_slug: str
    yes_shares: float = 0.0
    no_shares: float = 0.0
    cost_basis: float = 0.0  # total cost paid for this position

    def to_view(self) -> PositionView:
        return PositionView(
            market_slug=self.market_slug,
            yes_shares=self.yes_shares,
            no_shares=self.no_shares,
            cost_basis=self.cost_basis,
        )


@dataclass
class PortfolioSnapshot:
    """Immutable snapshot of portfolio state at a point in time."""
    timestamp: int
    cash: float
    positions: dict[str, PositionView]
    total_value: float
    realized_pnl: float
    unrealized_pnl: float


class Portfolio:
    """
    Manages cash balance and token positions.

    Settlement: YES outcome -> yes_shares pay $1 each, no_shares pay $0
                NO outcome -> no_shares pay $1 each, yes_shares pay $0
    """

    def __init__(self, starting_cash: float = DEFAULT_STARTING_CASH):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self._fill_history: list[Fill] = []
        self._settlement_history: list[Settlement] = []

    def get_position(self, slug: str) -> Position:
        """Get or create position for a market."""
        if slug not in self.positions:
            self.positions[slug] = Position(market_slug=slug)
        return self.positions[slug]

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill to the portfolio."""
        pos = self.get_position(fill.market_slug)

        if fill.side == Side.BUY:
            self.cash -= fill.cost
            if fill.token == Token.YES:
                pos.yes_shares += fill.size
            else:
                pos.no_shares += fill.size
            pos.cost_basis += fill.cost
        else:  # SELL
            self.cash += fill.cost
            if fill.token == Token.YES:
                pos.yes_shares -= fill.size
            else:
                pos.no_shares -= fill.size
            pos.cost_basis -= fill.cost

        self._fill_history.append(fill)

    def apply_settlement(self, settlement: Settlement) -> float:
        """
        Settle a market and return the P&L from this settlement.

        YES outcome: yes_shares pay $1, no_shares pay $0
        NO outcome: no_shares pay $1, yes_shares pay $0
        """
        pos = self.get_position(settlement.market_slug)

        if settlement.outcome == Token.YES:
            payout = pos.yes_shares * 1.0  # YES shares pay $1
            # NO shares pay $0
        else:
            payout = pos.no_shares * 1.0  # NO shares pay $1
            # YES shares pay $0

        pnl = payout - pos.cost_basis
        self.cash += payout
        self.realized_pnl += pnl

        logger.debug(
            f"Settlement {settlement.market_slug}: {settlement.outcome.value} | "
            f"payout=${payout:.2f} pnl=${pnl:+.2f} "
            f"(yes={pos.yes_shares:.1f}, no={pos.no_shares:.1f})"
        )

        # Clear the position
        pos.yes_shares = 0.0
        pos.no_shares = 0.0
        pos.cost_basis = 0.0

        self._settlement_history.append(settlement)
        return pnl

    def mark_to_market(self, market_views: dict[str, MarketView]) -> float:
        """
        Calculate total portfolio value using current market mid-prices.

        Returns total value = cash + sum of all position mark-to-market values.
        """
        total = self.cash

        for slug, pos in self.positions.items():
            if pos.yes_shares <= 0 and pos.no_shares <= 0:
                continue

            view = market_views.get(slug)
            if view:
                # Value at current mid-prices
                yes_value = pos.yes_shares * view.yes_price
                no_value = pos.no_shares * view.no_price
                total += yes_value + no_value
            else:
                # No market data — value at cost basis
                total += pos.cost_basis

        return total

    def unrealized_pnl(self, market_views: dict[str, MarketView]) -> float:
        """Calculate unrealized P&L across all open positions."""
        total_unrealized = 0.0
        for slug, pos in self.positions.items():
            if pos.yes_shares <= 0 and pos.no_shares <= 0:
                continue

            view = market_views.get(slug)
            if view:
                current_value = pos.yes_shares * view.yes_price + pos.no_shares * view.no_price
                total_unrealized += current_value - pos.cost_basis

        return total_unrealized

    def snapshot(
        self, timestamp: int, market_views: dict[str, MarketView]
    ) -> PortfolioSnapshot:
        """Create an immutable snapshot of current portfolio state."""
        total_value = self.mark_to_market(market_views)
        unrealized = self.unrealized_pnl(market_views)

        positions = {
            slug: pos.to_view()
            for slug, pos in self.positions.items()
            if pos.yes_shares > 0 or pos.no_shares > 0
        }

        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=positions,
            total_value=total_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
        )

    def get_position_views(self) -> dict[str, PositionView]:
        """Get read-only views of all positions with non-zero holdings."""
        return {
            slug: pos.to_view()
            for slug, pos in self.positions.items()
            if pos.yes_shares > 0 or pos.no_shares > 0 or pos.cost_basis != 0
        }

    @property
    def total_fills(self) -> int:
        return len(self._fill_history)

    @property
    def fill_history(self) -> list[Fill]:
        return list(self._fill_history)

    @property
    def settlement_history(self) -> list[Settlement]:
        return list(self._settlement_history)
