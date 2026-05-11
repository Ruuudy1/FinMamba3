"""Vendored data loader + dataclasses from the DATAHACKS2026 backtester.

The Gymnasium environment in ``envs.polymarket_lob_env`` implements a small
execution and portfolio layer around these pure timeline structures.
"""
# region imports
from .data_loader import BacktestData, TickData, build_timeline
from .strategy import (
    MarketLifecycle,
    MarketStatus,
    OrderBookLevel,
    OrderBookSnapshot,
    Fill,
    Order,
    Side,
    Settlement,
    StoredBook,
    Token,
)
# endregion
__all__ = [
    "BacktestData",
    "TickData",
    "build_timeline",
    "MarketLifecycle",
    "MarketStatus",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "Fill",
    "Order",
    "Side",
    "Settlement",
    "StoredBook",
    "Token",
]
