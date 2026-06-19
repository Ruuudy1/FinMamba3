"""Vendored DATAHACKS2026 execution engine, repointed onto this repo's backtester dataclasses.

Only the behavioral modules (tick loop, walk-the-book execution, market lifecycle, portfolio) are
vendored; the timeline structures and strategy contracts live in ``finmamba3.backtester.data_loader``
and ``finmamba3.backtester.strategy``. ``ExecutionEngine`` and ``Portfolio`` stay engine-private.
"""
# region imports
from .engine import BacktestEngine, BacktestResult
from .execution import MAX_SHARES_PER_TOKEN
# endregion
__all__ = ["BacktestEngine", "BacktestResult", "MAX_SHARES_PER_TOKEN"]
