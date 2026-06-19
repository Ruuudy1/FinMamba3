"""Opt-in native C++ backtester engine (Phase 2 accelerator), parity-checked against the Python engine.

The compiled extension is NOT built by ``pip install``; build it explicitly with
``python -m finmamba3.backtester.engine_cpp.build`` (needs a C++ toolchain: MSVC on Windows, gcc in the
CUDA image). The timeline is marshalled into dense arrays once (``marshal_dense``) and the whole tick
loop runs in C++ (``run_marshalled``), callback-free. The pure-Python ``finmamba3.backtester.engine``
stays the canonical reference and is the default; this path is opt-in.

The marshalling is timeline-only (independent of strategy/params), so a threshold sweep marshals once
and runs the native engine many times — that amortised re-run is where the speedup lands; a single run
is dominated by the Python-side marshalling.

v1 supports the default run_pnl_backtest path: fixed-size sizing, optional CUSUM gate, and the tte /
500-share position gates. Options not yet ported (Kelly sizing, calibration temperature, the
predictability gate, the book-depth band, edge mode) are rejected up front so the native path can never
silently diverge from the reference.
"""
# region imports
from dataclasses import dataclass
from pathlib import Path
import numpy as np
# endregion
# Import the compiled module only when it has been built, without a try/except: a glob of the package
# directory tells us whether `python -m finmamba3.backtester.engine_cpp.build` has produced the binary.
_BUILT_EXTENSIONS = list(Path(__file__).parent.glob("_engine*.pyd")) + list(Path(__file__).parent.glob("_engine*.so"))
if _BUILT_EXTENSIONS:
    from . import _engine
else:
    _engine = None
_BUILD_HINT = (
    "The native C++ backtester engine is not built. Build it with "
    "`python -m finmamba3.backtester.engine_cpp.build` (requires a C++ toolchain), or use the default "
    "Python engine."
)
_TOKEN_YES = 0
_TOKEN_NO = 1


@dataclass
class MarshalledTimeline:
    """Dense per-(tick, market) arrays + the slug->index map, built once and reused across runs."""
    arrays: tuple
    market_index_by_slug: dict


@dataclass
class CppBacktestResult:
    """Subset of the Python BacktestResult that the PnL adapter reads, returned by the native engine."""
    total_pnl: float
    total_trades: int
    fills: list  # Each entry is (market_slug, token_str, size, avg_price).
    snapshot_values: list  # Per-snapshot mark-to-market portfolio value, for the Sharpe computation.


def _assert_native_supported(params: dict, strategy_kind: int) -> None:
    # Fail-fast: the v1 native engine reproduces only the fixed-size settlement path, so reject any
    # option whose logic is not yet ported rather than running it and silently diverging.
    assert params.get("sizing", "fixed") == "fixed", "cpp engine v1 supports only sizing='fixed'."
    assert float(params.get("calibration_temperature", 1.0)) == 1.0, "cpp engine v1 does not support calibration_temperature != 1."
    assert not params.get("use_predictability_gate", False), "cpp engine v1 does not support the predictability gate."
    assert float(params.get("min_book_depth", 0.0)) == 0.0, "cpp engine v1 does not support min_book_depth."
    assert float(params.get("max_book_depth", 0.0)) == 0.0, "cpp engine v1 does not support max_book_depth."
    assert params.get("ev_by_slug_ts") is None, "cpp engine v1 does not support edge mode."


def marshal_dense(bt, slugs: list, prob_by_slug_ts: dict) -> MarshalledTimeline:
    """Flatten a scoped BacktestData + precomputed probabilities into dense arrays for the native engine.

    Timeline-only and strategy-independent, so a sweep marshals once and reuses the result for every
    strategy / parameter value. bt is pnl_backtest._data_for_slugs's per-regime scope; slugs gives the
    market order; prob_by_slug_ts supplies the model's per-tick settlement YES probability.
    """
    timeline = bt.timeline
    n_ticks = len(timeline)
    n_markets = len(slugs)
    market_index_by_slug = {slug: index for index, slug in enumerate(slugs)}
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    ts = np.fromiter((int(tick.ts_sec) for tick in timeline), dtype=np.int64, count=n_ticks)
    chainlink = np.fromiter((float(tick.chainlink_btc) for tick in timeline), dtype=np.float64, count=n_ticks)
    market_start = np.zeros(n_markets, dtype=np.int64)
    market_end = np.zeros(n_markets, dtype=np.int64)
    market_outcome = np.full(n_markets, -1, dtype=np.int32)
    for slug, index in market_index_by_slug.items():
        lifecycle = lifecycle_by_slug[slug]
        market_start[index] = int(lifecycle.start_ts)
        market_end[index] = int(lifecycle.end_ts)
        settlement = bt.settlements.get(slug)
        if settlement is not None:
            market_outcome[index] = 1 if settlement.outcome.value == "YES" else 0
    max_levels = 1
    for tick in timeline:
        for slug in slugs:
            stored = tick.order_books.get(slug)
            if stored is not None:
                max_levels = max(max_levels, len(stored.yes_book.asks), len(stored.no_book.asks))
    present = np.zeros((n_ticks, n_markets), dtype=np.int8)
    book_ts = np.zeros((n_ticks, n_markets), dtype=np.int64)
    yes_mid = np.zeros((n_ticks, n_markets), dtype=np.float64)
    yes_best_ask = np.zeros((n_ticks, n_markets), dtype=np.float64)
    no_best_ask = np.zeros((n_ticks, n_markets), dtype=np.float64)
    yes_price = np.zeros((n_ticks, n_markets), dtype=np.float64)
    no_price = np.zeros((n_ticks, n_markets), dtype=np.float64)
    model_prob = np.full((n_ticks, n_markets), np.nan, dtype=np.float64)
    yes_ask_px = np.zeros((n_ticks, n_markets, max_levels), dtype=np.float64)
    yes_ask_sz = np.zeros((n_ticks, n_markets, max_levels), dtype=np.float64)
    yes_ask_n = np.zeros((n_ticks, n_markets), dtype=np.int32)
    no_ask_px = np.zeros((n_ticks, n_markets, max_levels), dtype=np.float64)
    no_ask_sz = np.zeros((n_ticks, n_markets, max_levels), dtype=np.float64)
    no_ask_n = np.zeros((n_ticks, n_markets), dtype=np.int32)
    for tick_index, tick in enumerate(timeline):
        now = int(tick.ts_sec)
        price_by_slug = tick.market_prices
        for slug, market in market_index_by_slug.items():
            probability = prob_by_slug_ts.get((slug, now))
            if probability is not None:
                model_prob[tick_index, market] = float(probability)
            stored = tick.order_books.get(slug)
            if stored is None:
                continue
            yes_book = stored.yes_book
            no_book = stored.no_book
            present[tick_index, market] = 1
            book_ts[tick_index, market] = int(tick.book_timestamps.get(slug, now))
            yes_mid[tick_index, market] = yes_book.mid
            yes_best_ask[tick_index, market] = yes_book.best_ask
            no_best_ask[tick_index, market] = no_book.best_ask
            price_row = price_by_slug.get(slug, {})
            derived_yes_price = float(price_row.get("yes_price", 0) or 0)
            derived_no_price = float(price_row.get("no_price", 0) or 0)
            if derived_yes_price == 0 and yes_book.mid > 0:
                derived_yes_price = yes_book.mid
            if derived_no_price == 0 and no_book.mid > 0:
                derived_no_price = no_book.mid
            yes_price[tick_index, market] = derived_yes_price
            no_price[tick_index, market] = derived_no_price
            yes_level_count = min(len(yes_book.asks), max_levels)
            for level in range(yes_level_count):
                yes_ask_px[tick_index, market, level] = yes_book.asks[level].price
                yes_ask_sz[tick_index, market, level] = yes_book.asks[level].size
            yes_ask_n[tick_index, market] = yes_level_count
            no_level_count = min(len(no_book.asks), max_levels)
            for level in range(no_level_count):
                no_ask_px[tick_index, market, level] = no_book.asks[level].price
                no_ask_sz[tick_index, market, level] = no_book.asks[level].size
            no_ask_n[tick_index, market] = no_level_count
    arrays = (
        ts, chainlink, market_start, market_end, market_outcome,
        present, book_ts, yes_mid, yes_best_ask, no_best_ask, yes_price, no_price, model_prob,
        yes_ask_px, yes_ask_sz, yes_ask_n, no_ask_px, no_ask_sz, no_ask_n,
    )
    return MarshalledTimeline(arrays=arrays, market_index_by_slug=market_index_by_slug)


def run_marshalled(
    marshalled: MarshalledTimeline, strategy_kind: int, params: dict,
    starting_cash: float, snapshot_interval: int,
) -> CppBacktestResult:
    """Run the native engine over an already-marshalled timeline (the cheap, repeatable part of a sweep)."""
    if _engine is None:
        raise RuntimeError(_BUILD_HINT)
    _assert_native_supported(params, strategy_kind)
    raw = _engine.run_backtest(
        *marshalled.arrays,
        int(strategy_kind), float(params["edge_threshold"]), float(params["order_size"]),
        float(params["max_tte_frac"]), float(params["min_tte_frac"]), bool(params["use_cusum"]),
        float(params["cusum_threshold"]), float(starting_cash), int(snapshot_interval),
    )
    slug_by_index = {index: slug for slug, index in marshalled.market_index_by_slug.items()}
    fills = [
        (slug_by_index[market], "YES" if token == _TOKEN_YES else "NO", float(size), float(avg_price))
        for (market, token, size, avg_price) in raw["fills"]
    ]
    return CppBacktestResult(
        total_pnl=float(raw["total_pnl"]),
        total_trades=int(raw["total_trades"]),
        fills=fills,
        snapshot_values=[float(value) for value in raw["snapshot_values"]],
    )


def run_backtest_cpp(
    bt, slugs: list, prob_by_slug_ts: dict, strategy_kind: int, params: dict,
    starting_cash: float, snapshot_interval: int,
) -> CppBacktestResult:
    """Marshal a scoped BacktestData and run the native engine once (marshal + run; see run_marshalled)."""
    marshalled = marshal_dense(bt, slugs, prob_by_slug_ts)
    return run_marshalled(marshalled, strategy_kind, params, starting_cash, snapshot_interval)
