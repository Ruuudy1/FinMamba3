"""Timeline data loader, dataclasses, and the vendored execution engine for the Polymarket backtester.

``data_loader`` builds the per-second timeline (``build_timeline``) and ``strategy`` holds the dataclass
contracts plus ``BaseStrategy``. The full execution engine (tick loop, walk-the-book execution,
portfolio) lives in the ``engine`` subpackage and consumes these same structures; it is imported
explicitly via ``finmamba3.backtester.engine`` rather than re-exported here, so the data-pipeline
modules that only need ``build_timeline`` do not pull the engine in.
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
