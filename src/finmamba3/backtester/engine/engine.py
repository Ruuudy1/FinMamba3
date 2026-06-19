"""
Backtest Engine — Main tick loop orchestrator.

Coordinates data loading, market management, execution, and portfolio tracking
to run a strategy over historical data.
"""

# region imports
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from finmamba3.backtester.data_loader import BacktestData, TickData
from .execution import ExecutionEngine
from .market_manager import MarketManager
from .portfolio import Portfolio, PortfolioSnapshot
from finmamba3.backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketLifecycle,
    MarketState,
    Settlement,
)
# endregion

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    strategy_name: str
    start_ts: int
    end_ts: int
    starting_cash: float
    final_cash: float
    final_portfolio_value: float
    total_pnl: float
    total_trades: int
    total_settlements: int
    total_rejected: int
    portfolio_snapshots: list[PortfolioSnapshot]
    fills: list[Fill]
    settlements: list[Settlement]
    elapsed_seconds: float


class BacktestEngine:
    """
    Main backtest engine that drives the tick loop.

    Per tick:
    1. Settle expired markets -> update portfolio, notify strategy
    2. Update active markets from MarketManager
    3. Execute pending orders (queued from previous tick)
    4. Build MarketState snapshot
    5. Call strategy.on_tick(state) -> get new orders
    6. Validate and queue orders for next tick
    7. Record portfolio snapshot
    """

    def __init__(
        self,
        data: BacktestData,
        strategy: BaseStrategy,
        starting_cash: float = 10_000.0,
        snapshot_interval: int = 1,  # record portfolio every N ticks
    ):
        self.data = data
        self.strategy = strategy
        self.starting_cash = starting_cash
        self.snapshot_interval = snapshot_interval

        # Initialize components — deep-copy lifecycles so data can be reused
        self.portfolio = Portfolio(starting_cash=starting_cash)
        fresh_lifecycles = [
            MarketLifecycle(
                market_slug=lc.market_slug,
                interval=lc.interval,
                start_ts=lc.start_ts,
                end_ts=lc.end_ts,
            )
            for lc in data.lifecycles
        ]
        self.market_manager = MarketManager(
            lifecycles=fresh_lifecycles,
            settlements=data.settlements,
        )
        self.execution = ExecutionEngine()

        # Results
        self.snapshots: list[PortfolioSnapshot] = []
        self.all_fills: list[Fill] = []
        self.all_settlements: list[Settlement] = []

    def run(self) -> BacktestResult:
        """Run the complete backtest and return results."""
        if not self.data.timeline:
            logger.warning("No timeline data to backtest")
            return self._empty_result()

        strategy_name = type(self.strategy).__name__
        logger.info(
            f"Starting backtest: {strategy_name} | "
            f"{len(self.data.timeline)} ticks | "
            f"${self.starting_cash:,.0f} starting cash | "
            f"{len(self.data.lifecycles)} markets"
        )

        t0 = time.time()
        total_ticks = len(self.data.timeline)

        for i, tick in enumerate(self.data.timeline):
            self._process_tick(tick)

            # Progress logging
            if (i + 1) % 3600 == 0 or i == total_ticks - 1:
                pv = self.portfolio.mark_to_market(
                    self.market_manager.update(tick.ts_sec)
                )
                logger.info(
                    f"Tick {i + 1}/{total_ticks} | "
                    f"ts={tick.ts_sec} | "
                    f"portfolio=${pv:,.2f} | "
                    f"trades={self.portfolio.total_fills} | "
                    f"settled={len(self.all_settlements)}"
                )

        elapsed = time.time() - t0

        # Final valuation
        final_views = {}
        if self.data.timeline:
            last_tick = self.data.timeline[-1]
            final_views = self.market_manager.update(last_tick.ts_sec)
            final_views = self.market_manager.enrich_views(final_views, last_tick)

        final_value = self.portfolio.mark_to_market(final_views)

        result = BacktestResult(
            strategy_name=strategy_name,
            start_ts=self.data.start_ts,
            end_ts=self.data.end_ts,
            starting_cash=self.starting_cash,
            final_cash=self.portfolio.cash,
            final_portfolio_value=final_value,
            total_pnl=final_value - self.starting_cash,
            total_trades=self.portfolio.total_fills,
            total_settlements=len(self.all_settlements),
            total_rejected=self.execution.total_rejected,
            portfolio_snapshots=self.snapshots,
            fills=self.all_fills,
            settlements=self.all_settlements,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"Backtest complete: {strategy_name} | "
            f"{elapsed:.1f}s | "
            f"P&L=${result.total_pnl:+,.2f} | "
            f"trades={result.total_trades} | "
            f"rejected={result.total_rejected}"
        )

        return result

    def _process_tick(self, tick: TickData) -> None:
        """Process a single tick of the backtest."""
        ts = tick.ts_sec

        # 1. Update market states (settles expired markets)
        active_views = self.market_manager.update(ts)

        # 2. Handle settlements
        settled = self.market_manager.get_settled_this_tick()
        for settlement in settled:
            self.portfolio.apply_settlement(settlement)
            self.all_settlements.append(settlement)
            self.strategy.on_settlement(settlement)

        # 3. Enrich market views with current tick data
        enriched_views = self.market_manager.enrich_views(active_views, tick)

        # 4. Execute pending orders from previous tick
        fills = self.execution.execute_pending(
            current_tick=ts,
            market_views=enriched_views,
            book_timestamps=tick.book_timestamps,
        )

        for fill in fills:
            self.portfolio.apply_fill(fill)
            self.all_fills.append(fill)
            self.strategy.on_fill(fill)

        # 5. Build MarketState for strategy
        total_value = self.portfolio.mark_to_market(enriched_views)
        state = MarketState(
            timestamp=ts,
            timestamp_utc=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            markets=enriched_views,
            btc_mid=tick.btc_mid,
            btc_spread=tick.btc_spread,
            chainlink_btc=tick.chainlink_btc,
            cash=self.portfolio.cash,
            positions=self.portfolio.get_position_views(),
            total_portfolio_value=total_value,
        )

        # 6. Call strategy
        orders = self.strategy.on_tick(state)
        # 7. Validate and queue orders
        if orders:
            self.execution.queue_orders(
                orders=orders,
                current_tick=ts,
                cash=self.portfolio.cash,
                positions={
                    slug: self.portfolio.get_position(slug)
                    for slug in enriched_views
                },
                active_markets=enriched_views,
            )

        # 8. Record portfolio snapshot
        if ts % self.snapshot_interval == 0:
            snap = self.portfolio.snapshot(ts, enriched_views)
            self.snapshots.append(snap)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            strategy_name=type(self.strategy).__name__,
            start_ts=0,
            end_ts=0,
            starting_cash=self.starting_cash,
            final_cash=self.starting_cash,
            final_portfolio_value=self.starting_cash,
            total_pnl=0.0,
            total_trades=0,
            total_settlements=0,
            total_rejected=0,
            portfolio_snapshots=[],
            fills=[],
            settlements=[],
            elapsed_seconds=0.0,
        )
