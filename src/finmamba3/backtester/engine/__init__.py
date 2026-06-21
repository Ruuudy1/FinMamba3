"""Vendored DATAHACKS2026 execution engine, repointed onto this repo's backtester dataclasses.

Only the behavioral modules (tick loop, walk-the-book execution, market lifecycle, portfolio) are
vendored; the timeline structures and strategy contracts live in ``finmamba3.backtester.data_loader``
and ``finmamba3.backtester.strategy``. ``ExecutionEngine`` and ``Portfolio`` stay engine-private.

The house code style is deliberately not applied to this subpackage: it is vendored to keep a clean
upstream diff and to protect the parity oracle (``tests/test_engine_cpp_parity.py``), so its comments,
docstrings, and spacing follow the upstream convention rather than this repo's.
"""
# region imports
from .engine import BacktestEngine, BacktestResult
from .execution import MAX_SHARES_PER_TOKEN
# endregion
__all__ = ["BacktestEngine", "BacktestResult", "MAX_SHARES_PER_TOKEN"]
