"""Vendored data loader + dataclasses from the DATAHACKS2026 backtester.

Engine / execution / portfolio / scoring components are intentionally
omitted; only the pure-function timeline builder is needed here.
"""

from .data_loader import BacktestData, TickData, build_timeline
from .strategy import (
    MarketLifecycle,
    MarketStatus,
    OrderBookLevel,
    OrderBookSnapshot,
    Settlement,
    StoredBook,
    Token,
)

__all__ = [
    "BacktestData",
    "TickData",
    "build_timeline",
    "MarketLifecycle",
    "MarketStatus",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "Settlement",
    "StoredBook",
    "Token",
]
