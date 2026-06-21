"""Opt-in native C++ backtester engine (Phase 2 accelerator), parity-checked against the Python engine.

The compiled extension is NOT built by ``pip install``; build it explicitly with
``python -m finmamba3.backtester.engine_cpp.build`` (needs a C++ toolchain: MSVC on Windows, gcc in the
CUDA image). The timeline is flattened once into a sparse CSR over the present (tick, market) cells
(``marshal_timeline``) and the whole tick loop runs in C++ (``run_marshalled``), callback-free. The
pure-Python ``finmamba3.backtester.engine`` stays the canonical reference and is the default; this is opt-in.

The dense (T, M) grid is ~99% empty (markets are short-lived) and its ask-ladder dimension reaches
hundreds of levels, which is infeasible at headline scale; the sparse CSR lists only the active cells
and truncates each ladder at the reachable depth, so marshalling is O(present cells) in both time and
memory. It is also timeline-only (independent of strategy/params), so a threshold sweep marshals once
and re-runs the native engine per value; that amortised re-run is the largest speedup.

The native engine covers the full WorldModelStrategy / NaiveLagStrategy surface: fixed or quarter-Kelly
sizing, the optional CUSUM gate, the tte / 500-share position gates, the predictability gate (live spot
window and per-market precomputed ER), the two-sided book-depth band, settlement-probability trading
with optional temperature calibration, and edge mode. Per-run (slug, ts) signals are aligned onto the
cell order before the run; the run/strategy configuration is one dict, so adding a future option does
not change the C++ signature. The Python engine remains the parity oracle.
"""
# region imports
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from finmamba3.backtester.engine import MAX_SHARES_PER_TOKEN
from finmamba3.backtester.strategy import Token
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
# Strategy-kind enum shared with the C++ run_backtest (strategy_kind argument).
STRATEGY_WORLD_MODEL = 0
STRATEGY_NAIVE_LAG = 1
# Passed for a cell-aligned signal that the current run does not use; the C++ engine never indexes it.
_EMPTY_SIGNAL = np.empty(0, dtype=np.float64)


@dataclass
class MarshalledTimeline:
    """Sparse CSR timeline arrays + the slug->index map, built once and reused across runs."""
    arrays: tuple
    market_index_by_slug: dict
    settlement_by_slug: dict  # Scoped slug -> Settlement, to rebuild the run's settled set for per_trade_pnls.
    cell_ts: np.ndarray  # Each cell's tick second, to align per-run (slug, ts) signals onto the cell order.


@dataclass(frozen=True)
class _CppFill:
    """The fill fields per_trade_pnls reads; a structural stand-in for the Python engine's Fill."""
    market_slug: str
    token: Token
    size: float
    avg_price: float


@dataclass(frozen=True)
class _CppSnapshot:
    """The one snapshot field _sharpe_from_snapshots reads; a stand-in for PortfolioSnapshot."""
    total_value: float


@dataclass
class CppBacktestResult:
    """Drop-in for the Python BacktestResult over the fields the PnL adapter reads (pnl, trades, fills,
    settlements, snapshots), so per_trade_pnls and _sharpe_from_snapshots consume it unchanged."""
    total_pnl: float
    total_trades: int
    fills: list  # list[_CppFill]
    settlements: list  # The scoped Settlements, for per_trade_pnls' outcome lookup.
    portfolio_snapshots: list  # list[_CppSnapshot], for _sharpe_from_snapshots.


@dataclass
class AlignedSignals:
    """Per-run (slug, ts) signals aligned onto the marshalled cell order, plus the flags they imply.

    gate_er / gate_dir depend only on the gate window and ev only on the model (none on the swept
    threshold), so a constant-window sweep aligns these once and reuses them across every value, which
    is what lifts the gated-sweep speedup (the per-value alignment was the residual O(present cells) cost).
    """
    gate_er_cells: np.ndarray
    gate_dir_cells: np.ndarray
    ev_yes_cells: np.ndarray
    ev_no_cells: np.ndarray
    gate_mode: int
    use_edge: bool


def _align_scalar(marshalled: MarshalledTimeline, by_slug_ts: dict, default: float) -> np.ndarray:
    # Map a per-(slug, ts) signal dict onto the marshalled cell order (one value per present cell), so a
    # gate the strategy reads by (slug, ts) becomes a flat per-cell array the native engine indexes.
    cell_market = marshalled.arrays[6]
    cell_ts = marshalled.cell_ts
    slug_by_index = {index: slug for slug, index in marshalled.market_index_by_slug.items()}
    aligned = np.full(cell_market.shape[0], default, dtype=np.float64)
    lookup = by_slug_ts.get
    for cell in range(cell_market.shape[0]):
        value = lookup((slug_by_index[int(cell_market[cell])], int(cell_ts[cell])))
        if value is not None:
            aligned[cell] = value
    return aligned


def _align_pair(marshalled: MarshalledTimeline, by_slug_ts: dict) -> tuple:
    # Map a per-(slug, ts) (yes, no) EV pair dict onto the cell order; NaN marks a cell with no EV, which
    # the native engine skips exactly as the Python strategy skips a (slug, ts) absent from ev_by_slug_ts.
    cell_market = marshalled.arrays[6]
    cell_ts = marshalled.cell_ts
    slug_by_index = {index: slug for slug, index in marshalled.market_index_by_slug.items()}
    yes_out = np.full(cell_market.shape[0], np.nan, dtype=np.float64)
    no_out = np.full(cell_market.shape[0], np.nan, dtype=np.float64)
    lookup = by_slug_ts.get
    for cell in range(cell_market.shape[0]):
        pair = lookup((slug_by_index[int(cell_market[cell])], int(cell_ts[cell])))
        if pair is not None:
            yes_out[cell] = pair[0]
            no_out[cell] = pair[1]
    return yes_out, no_out


def _append_truncated_asks(asks, price_out: list, size_out: list) -> None:
    # Copy ask levels into the flat ladders until cumulative size reaches the per-token share cap, then
    # stop: walk-the-book never consumes more than that on one book per tick (each strategy issues at most
    # one buy per market per tick, so there is no same-tick depletion), making the deeper tail unreachable.
    cumulative = 0.0
    for level in asks:
        price_out.append(level.price)
        size_out.append(level.size)
        cumulative += level.size
        if cumulative >= MAX_SHARES_PER_TOKEN:
            break


def marshal_timeline(bt, slugs: list, prob_by_slug_ts: dict, include_depth: bool = False) -> MarshalledTimeline:
    """Flatten a scoped BacktestData + precomputed probabilities into a sparse CSR for the native engine.

    Timeline-only and (apart from include_depth) strategy-independent, so a sweep marshals once and reuses
    the result for every strategy / parameter value. bt is pnl_backtest._data_for_slugs's per-regime scope;
    slugs gives the market order; prob_by_slug_ts supplies the model's per-tick settlement YES probability.
    Only the present (tick, market) cells are emitted, grouped per tick by tick_offset, with the ask ladders
    truncated at the reachable depth, so this is O(present cells), not O(ticks * markets). include_depth adds
    the full two-sided YES depth per cell (a full-ladder sum, so computed only when the depth band is used).
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
    tick_offset = np.zeros(n_ticks + 1, dtype=np.int64)
    cell_market: list = []
    cell_prob: list = []
    cell_book_ts: list = []
    cell_yes_mid: list = []
    cell_yes_price: list = []
    cell_no_price: list = []
    cell_yes_off: list = [0]
    cell_no_off: list = [0]
    yes_ask_px: list = []
    yes_ask_sz: list = []
    no_ask_px: list = []
    no_ask_sz: list = []
    cell_ts: list = []
    cell_yes_depth: list = []
    index_of = market_index_by_slug.get
    prob_of = prob_by_slug_ts.get
    for tick_index, tick in enumerate(timeline):
        now = int(tick.ts_sec)
        price_by_slug = tick.market_prices
        for slug, stored in tick.order_books.items():
            market = index_of(slug)
            if market is None:
                continue
            yes_book = stored.yes_book
            no_book = stored.no_book
            yes_mid = yes_book.mid
            no_mid = no_book.mid
            probability = prob_of((slug, now))
            cell_market.append(market)
            cell_ts.append(now)
            cell_prob.append(np.nan if probability is None else probability)
            cell_book_ts.append(int(stored.book_ts))
            cell_yes_mid.append(yes_mid)
            if include_depth:
                cell_yes_depth.append(yes_book.total_bid_size + yes_book.total_ask_size)
            price_row = price_by_slug.get(slug)
            derived_yes_price = float(price_row.get("yes_price", 0) or 0) if price_row else 0.0
            derived_no_price = float(price_row.get("no_price", 0) or 0) if price_row else 0.0
            if derived_yes_price == 0 and yes_mid > 0:
                derived_yes_price = yes_mid
            if derived_no_price == 0 and no_mid > 0:
                derived_no_price = no_mid
            cell_yes_price.append(derived_yes_price)
            cell_no_price.append(derived_no_price)
            _append_truncated_asks(yes_book.asks, yes_ask_px, yes_ask_sz)
            cell_yes_off.append(len(yes_ask_px))
            _append_truncated_asks(no_book.asks, no_ask_px, no_ask_sz)
            cell_no_off.append(len(no_ask_px))
        tick_offset[tick_index + 1] = len(cell_market)
    arrays = (
        ts, chainlink, market_start, market_end, market_outcome, tick_offset,
        np.asarray(cell_market, dtype=np.int32), np.asarray(cell_prob, dtype=np.float64),
        np.asarray(cell_book_ts, dtype=np.int64), np.asarray(cell_yes_mid, dtype=np.float64),
        np.asarray(cell_yes_price, dtype=np.float64), np.asarray(cell_no_price, dtype=np.float64),
        np.asarray(cell_yes_off, dtype=np.int64), np.asarray(yes_ask_px, dtype=np.float64),
        np.asarray(yes_ask_sz, dtype=np.float64), np.asarray(cell_no_off, dtype=np.int64),
        np.asarray(no_ask_px, dtype=np.float64), np.asarray(no_ask_sz, dtype=np.float64),
        np.asarray(cell_yes_depth, dtype=np.float64),
    )
    return MarshalledTimeline(
        arrays=arrays, market_index_by_slug=market_index_by_slug,
        settlement_by_slug=dict(bt.settlements), cell_ts=np.asarray(cell_ts, dtype=np.int64),
    )


def align_run_signals(marshalled: MarshalledTimeline, strategy_kind: int, params: dict) -> AlignedSignals:
    """Align this run's per-(slug, ts) signals onto the cell order, returning them with the gate flags.

    Threshold-independent (see AlignedSignals), so a constant-window sweep calls this once per arm and
    passes the result to run_marshalled for every value; a single run lets run_marshalled call it.
    """
    gate_er = params.get("gate_er_by_slug_ts")
    gate_dir = params.get("gate_dir_by_slug_ts")
    edge_signals = params.get("ev_by_slug_ts")
    gate_mode = 1 if (bool(params.get("use_predictability_gate", False)) and gate_er is not None) else 0
    use_edge = edge_signals is not None
    gate_er_cells = _align_scalar(marshalled, gate_er, 0.0) if gate_mode == 1 else _EMPTY_SIGNAL
    gate_dir_cells = (
        _align_scalar(marshalled, gate_dir, 1.0)
        if (gate_mode == 1 and strategy_kind == STRATEGY_NAIVE_LAG and gate_dir is not None) else _EMPTY_SIGNAL
    )
    ev_yes_cells, ev_no_cells = _align_pair(marshalled, edge_signals) if use_edge else (_EMPTY_SIGNAL, _EMPTY_SIGNAL)
    return AlignedSignals(gate_er_cells, gate_dir_cells, ev_yes_cells, ev_no_cells, gate_mode, use_edge)


def run_marshalled(
    marshalled: MarshalledTimeline, strategy_kind: int, params: dict,
    starting_cash: float, snapshot_interval: int, signals: "AlignedSignals | None" = None,
) -> CppBacktestResult:
    """Run the native engine over an already-marshalled timeline (the cheap, repeatable part of a sweep).

    The per-run signals (predictability ER, naive trend direction, edge EV) are aligned by signals, which
    a sweep hoists out of its loop and passes in; when omitted they are aligned here. The book-depth column
    is baked into the marshalling and the flat config dict carries the scalars, so adding a gate never
    changes the C++ signature.
    """
    if _engine is None:
        raise RuntimeError(_BUILD_HINT)
    if signals is None:
        signals = align_run_signals(marshalled, strategy_kind, params)
    config = {
        "strategy_kind": int(strategy_kind),
        "edge_threshold": float(params.get("edge_threshold", 0.05)),
        "order_size": float(params.get("order_size", 50.0)),
        "max_tte_frac": float(params.get("max_tte_frac", 1.0)),
        "min_tte_frac": float(params.get("min_tte_frac", 0.0)),
        "use_cusum": bool(params.get("use_cusum", False)),
        "cusum_threshold": float(params.get("cusum_threshold", 0.003)),
        "starting_cash": float(starting_cash),
        "snapshot_interval": int(snapshot_interval),
        "sizing_kind": 1 if params.get("sizing", "fixed") == "kelly" else 0,
        "kelly_fraction": float(params.get("kelly_fraction", 0.25)),
        "kelly_cap": float(params.get("kelly_cap", 0.05)),
        "bankroll": float(starting_cash),
        "calibration_temperature": float(params.get("calibration_temperature", 1.0)),
        "use_predictability_gate": bool(params.get("use_predictability_gate", False)),
        "gate_mode": signals.gate_mode,
        "predictability_threshold": float(params.get("predictability_threshold", 0.3)),
        "predictability_window": int(params.get("predictability_window", 120)),
        "min_book_depth": float(params.get("min_book_depth", 0.0)),
        "max_book_depth": float(params.get("max_book_depth", 0.0)),
        "use_edge": signals.use_edge,
    }
    raw = _engine.run_backtest(
        *marshalled.arrays, signals.gate_er_cells, signals.gate_dir_cells, signals.ev_yes_cells, signals.ev_no_cells, config,
    )
    slug_by_index = {index: slug for slug, index in marshalled.market_index_by_slug.items()}
    token_by_index = {_TOKEN_YES: Token.YES, _TOKEN_NO: Token.NO}
    fills = [
        _CppFill(slug_by_index[market], token_by_index[token], float(size), float(avg_price))
        for (market, token, size, avg_price) in raw["fills"]
    ]
    settlements = [marshalled.settlement_by_slug[slug_by_index[market]] for market in raw["settled_markets"]]
    return CppBacktestResult(
        total_pnl=float(raw["total_pnl"]),
        total_trades=int(raw["total_trades"]),
        fills=fills,
        settlements=settlements,
        portfolio_snapshots=[_CppSnapshot(float(value)) for value in raw["snapshot_values"]],
    )


def run_backtest_cpp(
    bt, slugs: list, prob_by_slug_ts: dict, strategy_kind: int, params: dict,
    starting_cash: float, snapshot_interval: int,
) -> CppBacktestResult:
    """Marshal a scoped BacktestData and run the native engine once (marshal + run; see run_marshalled)."""
    include_depth = float(params.get("min_book_depth", 0.0)) > 0.0 or float(params.get("max_book_depth", 0.0)) > 0.0
    marshalled = marshal_timeline(bt, slugs, prob_by_slug_ts, include_depth=include_depth)
    return run_marshalled(marshalled, strategy_kind, params, starting_cash, snapshot_interval)
