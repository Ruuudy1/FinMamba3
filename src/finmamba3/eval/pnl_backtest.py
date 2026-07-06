"""PnL backtest adapter: judge a frozen world model by simulated trading PnL, not a forecasting proxy.

This module is the one place the torch world model meets the execution engine, now vendored in this
repo as ``finmamba3.backtester.engine`` and consuming this repo's timeline dataclasses directly. The
frozen model's spot-conditioned settlement YES probability is precomputed with this repo's exact
training feature pipeline, so the probability the strategy trades on is the one the model was trained
to produce (no train/serve skew). A thin WorldModelStrategy turns a divergence from the book's implied
probability into an order, which the engine matches T+1 and settles for the headline PnL / Sharpe per
spot-volatility regime. This is the settlement-accurate, order-matching evaluation path that produces
the paper's economic tables; ``finmamba3.eval.backtest`` is the lighter gym-env sketch alternative.

The per-regime report labels a reference regime (where the model has the most reason to profit:
predictable / low-vol) and a shifted, harder one. degradation = pnl(reference) - pnl(shifted), so a
positive degradation means edge lost under the shift, and the FiLM gap subtracts the two arms'
degradations.
"""
# region imports
from __future__ import annotations
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import yaml
from finmamba3.backtester import build_timeline
from finmamba3.backtester.data_loader import BacktestData
from finmamba3.backtester.engine import BacktestEngine
from finmamba3.backtester.engine_cpp import (
    STRATEGY_NAIVE_LAG, STRATEGY_WORLD_MODEL, align_run_signals, marshal_timeline, run_marshalled,
)
from finmamba3.backtester.strategy import BaseStrategy, MarketLifecycle, Settlement
from finmamba3.config import DotDict, parse_args_and_update_config
from finmamba3.envs.lob_features import apply_normalization, extract_features, load_normalization
from finmamba3.eval.compare_direction import load_world_model
from finmamba3.eval.predictability import predictability_from_timeline
from finmamba3.eval.regime_split import spot_realized_vol_from_timeline, volatility_split
from finmamba3.sequence_builder import _append_cross_interval, _settlement_yes_outcome, _spot_feature_kwargs
# Split-out helpers are re-imported so this module stays the public surface its callers (tests, simple_baseline) import from.
from finmamba3.eval.survivability import (
    _sharpe_from_snapshots, bootstrap_survivability, bootstrap_survivability_by_market,
    per_trade_pnls, per_trade_pnls_by_market,
)
from finmamba3.eval.signals import _gate_signals_by_slug_ts, world_model_edge_ev_series, world_model_yes_prob_series
from finmamba3.eval.strategies import NaiveLagStrategy, WorldModelStrategy
# endregion
logger = logging.getLogger(__name__)


def _usable_tick_count_by_slug(bt) -> dict:
    """Count the ticks whose YES book has both sides, matching extract_features' usability test.

    Lets the caller drop markets that would raise "no usable ticks" or yield no scoreable window.
    """
    count_by_slug = {}
    for tick in bt.timeline:
        for slug, stored in tick.order_books.items():
            if stored.yes_book.bids and stored.yes_book.asks:
                count_by_slug[slug] = count_by_slug.get(slug, 0) + 1
    return count_by_slug


def _eval_sequence(bt, slug, stats, include_binary: bool, include_spot: bool, include_cross: bool):
    """Build one held-out market's normalized sequence through the exact training pipeline.

    Reuses the train stats so the model is served the scale it learned on.
    """
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    settlement = bt.settlements.get(slug)
    yes_outcome = _settlement_yes_outcome(settlement)
    spot_kwargs = _spot_feature_kwargs(slug, settlement, lifecycle_by_slug) if include_spot else {}
    raw = extract_features(
        bt.timeline, slug, yes_outcome=yes_outcome,
        include_binary_features=include_binary, include_spot_features=include_spot, **spot_kwargs,
    )
    if include_cross:
        raw = _append_cross_interval(bt, slug, raw, lifecycle_by_slug)
    return apply_normalization(raw, stats)


def _data_for_slugs(bt, slugs: list[str]) -> BacktestData:
    """Scope the engine data to one regime's slugs while the shared timeline carries every book.

    The engine only tracks, trades and settles the lifecycles it is handed, so restricting them to
    one regime's slugs scopes the run to that regime. The vendored engine consumes this repo's
    timeline dataclasses directly, so there is no per-tick translation: the shared bt.timeline is
    reused as-is and only the lifecycles/settlements are scoped.
    """
    wanted = {slug: True for slug in slugs}
    lifecycles = [
        MarketLifecycle(lc.market_slug, lc.interval, lc.start_ts, lc.end_ts)
        for lc in bt.lifecycles if lc.market_slug in wanted
    ]
    settlements = {}
    for slug in slugs:
        st = bt.settlements.get(slug)
        if st is not None:
            settlements[slug] = Settlement(
                st.market_slug, st.interval, st.outcome,
                st.start_ts, st.end_ts, st.chainlink_open, st.chainlink_close,
            )
    return BacktestData(
        timeline=bt.timeline, lifecycles=lifecycles, settlements=settlements,
        start_ts=bt.start_ts, end_ts=bt.end_ts,
    )


def _arch_overrides(regime_film_enabled: bool, is_mimo: bool, n_layer: int | None) -> list[str]:
    overrides = [
        f"Models.WorldModel.RegimeFiLM.Enabled={'true' if regime_film_enabled else 'false'}",
        f"Models.WorldModel.Mamba3.is_mimo={'true' if is_mimo else 'false'}",
    ]
    if n_layer is not None:
        overrides.append(f"Models.WorldModel.Mamba3.n_layer={int(n_layer)}")
    return overrides


def _load_config(config_path: Path, overrides: list[str]) -> DotDict:
    with open(config_path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    return DotDict(parse_args_and_update_config(cfg_raw, argv=[f"--{kv.split('=')[0]}={kv.split('=')[1]}" for kv in overrides]))


def _cpp_params(strategy_kwargs: dict, ev_by_slug_ts: dict | None) -> dict:
    # Flatten the strategy kwargs into the flat params dict the native engine and its fail-fast guard read; the guard rejects any key whose logic is not yet ported, so the cpp run cannot diverge.
    return {**strategy_kwargs, "ev_by_slug_ts": ev_by_slug_ts}


def _naive_cpp_params(order_size: float, max_tte_frac: float, cusum_threshold: float) -> dict:
    # The ungated naive baseline as the native engine reads it: fixed size on a CUSUM event, none of the not-yet-ported gates engaged, so it passes the guard on the v1 native path.
    return {
        "edge_threshold": 0.0, "order_size": order_size, "max_tte_frac": max_tte_frac, "min_tte_frac": 0.0,
        "use_cusum": True, "cusum_threshold": cusum_threshold, "sizing": "fixed",
        "calibration_temperature": 1.0, "use_predictability_gate": False,
        "min_book_depth": 0.0, "max_book_depth": 0.0, "ev_by_slug_ts": None,
    }


def _run_engine_arm(
    engine: str, engine_data: BacktestData, strategy: BaseStrategy,
    marshalled, strategy_kind: int, params: dict, cash: float, signals=None,
):
    """Run one strategy arm through the Python reference engine or the native C++ engine.

    The cpp path replays the timeline marshal_timeline flattened once per regime; signals are the per-run
    cell-aligned gate arrays, which a sweep hoists out of its loop (constant gate window) and passes in,
    else run_marshalled aligns them. Both paths return the same result surface (pnl, trades, fills, snapshots).
    """
    if engine == "python":
        return BacktestEngine(engine_data, strategy, starting_cash=cash, snapshot_interval=60).run()
    return run_marshalled(marshalled, strategy_kind, params, cash, 60, signals=signals)


def _sweep_aligned(engine: str, cache: dict, marshalled, strategy_kind: int, params: dict, window: int):
    # Memoise the cpp sweep's cell-aligned signals per (window, arm): the gate ER / direction depend only on the gate window, not the swept threshold, so a constant-window sweep aligns once and reuses across values.
    # The Python engine needs no alignment, so this is a no-op for it.
    if engine != "cpp":
        return None
    key = (window, strategy_kind)
    if key not in cache:
        cache[key] = align_run_signals(marshalled, strategy_kind, params)
    return cache[key]


def _engine_workers(engine: str, engine_threads: int) -> int:
    # The cpp engine releases the GIL inside its tick loop, so independent backtests run on real cores; the python reference engine holds the GIL throughout, so it always stays serial.
    # 0 = auto.
    if engine != "cpp":
        return 1
    if engine_threads > 0:
        return engine_threads
    return os.cpu_count() or 1


def _run_concurrent(tasks: list, max_workers: int) -> list:
    # Run independent zero-arg backtest tasks, preserving input order so the result equals the serial path exactly.
    # A single task or max_workers <= 1 stays on the calling thread (the python engine, and the --engine-threads 1 parity escape hatch).
    # The marshalled timeline and aligned signals each task reads are immutable, and each backtest allocates its own engine state, so the runs do not interfere.
    if max_workers <= 1 or len(tasks) <= 1:
        return [task() for task in tasks]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = [pool.submit(task) for task in tasks]
        return [future.result() for future in futures]


def run_pnl_backtest(
    wm, bt, stats, slugs: list[str], device: torch.device,
    include_binary: bool, include_spot: bool, include_cross: bool,
    strategy_kwargs: dict, cash: float, prob_source: str = "settlement",
    slippage_per_share: float = 0.0, engine: str = "python", engine_threads: int = 0,
) -> dict:
    """Backtest one regime's slugs with the world-model strategy and the naive-lag baseline.

    Returns each arm's PnL/Sharpe/trades, the world model's per-trade bootstrap survivability, and
    the model-minus-naive PnL so the report can apply the anti-artifact gate (the model must beat
    the mechanical oracle-lag baseline). The engine data is built once and replayed for both arms.
    prob_source='edge' routes through the BookRelativeEdgeHead instead of the settlement head.
    slippage_per_share subtracts a per-share execution cost from each fill in the bootstrap only.
    """
    prob_by_slug_ts = {}
    ev_by_slug_ts = {}
    for slug in slugs:
        seq = _eval_sequence(bt, slug, stats, include_binary, include_spot, include_cross)
        if prob_source == "edge":
            for ts, ev_pair in world_model_edge_ev_series(wm, seq, device).items():
                ev_by_slug_ts[(slug, ts)] = ev_pair
                # `prob_by_slug_ts` is not used in edge mode but must be non-empty for strategy init.
                prob_by_slug_ts[(slug, ts)] = 0.5
        else:
            for ts, prob in world_model_yes_prob_series(wm, seq, device).items():
                prob_by_slug_ts[(slug, ts)] = prob
    engine_data = _data_for_slugs(bt, slugs)
    include_depth = strategy_kwargs["min_book_depth"] > 0.0 or strategy_kwargs["max_book_depth"] > 0.0
    marshalled = marshal_timeline(engine_data, slugs, prob_by_slug_ts, include_depth=include_depth) if engine == "cpp" else None
    edge_signals = ev_by_slug_ts if prob_source == "edge" else None
    model_strategy = WorldModelStrategy(prob_by_slug_ts, bankroll=cash, ev_by_slug_ts=edge_signals, **strategy_kwargs)
    naive_strategy = NaiveLagStrategy(
        cusum_threshold=strategy_kwargs["cusum_threshold"],
        order_size=strategy_kwargs["order_size"],
        max_tte_frac=strategy_kwargs["max_tte_frac"],
    )
    model_params = _cpp_params(strategy_kwargs, edge_signals)
    naive_params = _naive_cpp_params(strategy_kwargs["order_size"], strategy_kwargs["max_tte_frac"], strategy_kwargs["cusum_threshold"])
    # Both arms replay the same immutable timeline independently, so run them concurrently on the cpp path.
    model_result, naive_result = _run_concurrent(
        [
            lambda: _run_engine_arm(engine, engine_data, model_strategy, marshalled, STRATEGY_WORLD_MODEL, model_params, cash),
            lambda: _run_engine_arm(engine, engine_data, naive_strategy, marshalled, STRATEGY_NAIVE_LAG, naive_params, cash),
        ],
        _engine_workers(engine, engine_threads),
    )
    bootstrap = bootstrap_survivability(per_trade_pnls(model_result, slippage_per_share), bankroll=cash)
    bootstrap_market = bootstrap_survivability_by_market(per_trade_pnls_by_market(model_result, slippage_per_share), bankroll=cash)
    return {
        "model": {
            "pnl": float(model_result.total_pnl),
            "sharpe": _sharpe_from_snapshots(model_result.portfolio_snapshots),
            "trades": int(model_result.total_trades),
            "bootstrap": bootstrap,
            "bootstrap_market": bootstrap_market,
        },
        "naive": {
            "pnl": float(naive_result.total_pnl),
            "trades": int(naive_result.total_trades),
        },
        "model_minus_naive": float(model_result.total_pnl - naive_result.total_pnl),
        "n_markets": len(slugs),
        "n_probs": len(prob_by_slug_ts),
    }


def settlement_calibration(prob_by_slug_ts: dict, bt, num_bins: int = 10) -> dict:
    """Reliability of the settlement YES probability over all precomputed (slug, ts): Brier + decile bins.

    The economic edge needs a calibrated p_model (newgoal-2 1c), so this checks whether the settlement
    head's probability matches the realized YES frequency. Brier is the mean squared error to the binary
    outcome; each bin compares the mean predicted probability to the realized YES rate, so a diagonal
    reliability curve means the head is calibrated and its divergence from the book is a real signal.
    """
    probs = []
    outcomes = []
    for (slug, _ts), prob in prob_by_slug_ts.items():
        outcome = _settlement_yes_outcome(bt.settlements.get(slug))
        if outcome is None:
            continue
        probs.append(prob)
        outcomes.append(outcome)
    if not probs:
        return {"brier": float("nan"), "n": 0, "bins": []}
    prob_array = np.asarray(probs, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    for i in range(num_bins):
        upper = prob_array <= edges[i + 1] if i == num_bins - 1 else prob_array < edges[i + 1]
        mask = (prob_array >= edges[i]) & upper
        if int(mask.sum()) > 0:
            bins.append({
                "bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_prob": float(prob_array[mask].mean()),
                "realized_yes": float(outcome_array[mask].mean()),
            })
    # Fit a temperature T minimizing the NLL of sigmoid(logit(p)/T) against the outcome, the standard post-hoc calibration; T > 1 softens an over-confident head.
    # A grid is enough at this resolution.
    clipped = np.clip(prob_array, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.5, 6.0, 56):
        calibrated = np.clip(1.0 / (1.0 + np.exp(-logit / temperature)), 1e-6, 1.0 - 1e-6)
        nll = float(-(outcome_array * np.log(calibrated) + (1.0 - outcome_array) * np.log(1.0 - calibrated)).mean())
        if nll < best_nll:
            best_nll, best_temperature = nll, float(temperature)
    return {
        "brier": float(np.mean((prob_array - outcome_array) ** 2)),
        "n": len(probs), "temperature": best_temperature, "bins": bins,
    }


def delay_prob_series(prob_by_slug_ts: dict, delay_secs: int) -> dict:
    """Shift each model probability later by delay_secs: the signal-delay latency stress test.

    The economic edge is a latency play (it harvests the book's lag behind spot), so it is maximally
    sensitive to how the model's signal is timed. Re-keying each (slug, ts) probability to (slug,
    ts + delay) makes the strategy at tick t trade on the model output from t - delay, leaving the
    causal predictability gate (computed on the original ticks) untouched and staling only the model
    signal. An edge that survives only because the signal is timestamp-aligned to the book vanishes at
    the first second of delay; a real edge degrades gracefully. delay_secs <= 0 is the identity.
    """
    if delay_secs <= 0:
        return prob_by_slug_ts
    return {(slug, ts + delay_secs): prob for (slug, ts), prob in prob_by_slug_ts.items()}


def run_threshold_sweep(
    wm, bt, stats, slugs: list[str], device: torch.device,
    include_binary: bool, include_spot: bool, include_cross: bool,
    base_kwargs: dict, values: list[float], cash: float, slippage_per_share: float = 0.0,
    sweep_param: str = "predictability", sample_mode: str = "random_sample", engine: str = "python",
    signal_delay_secs: int = 0, engine_threads: int = 0,
) -> dict:
    """Sweep one strategy hyperparameter on the full market set, precomputing the YES probs once.

    The deployable selective-participation strategy trades the whole market set (no regime split). With
    sweep_param='predictability' it fits the gate ER threshold; with 'edge' it fits the model-vs-book
    edge threshold at the gate threshold already in base_kwargs. Because the probs and the naive
    baseline are independent of these, they are computed once, so the sweep is cheap. Returns per-value
    total PnL, trade count, model-minus-naive, and bootstrap survivability for picking a train value.
    """
    prob_by_slug_ts = {}
    for slug in slugs:
        seq = _eval_sequence(bt, slug, stats, include_binary, include_spot, include_cross)
        for ts, prob in world_model_yes_prob_series(wm, seq, device, sample_mode=sample_mode).items():
            prob_by_slug_ts[(slug, ts)] = prob
    # The signal-delay stress test stales only the model probability; the gate is re-derived on the delayed trade ticks below so it still reads the genuine causal ER at trade time.
    # The native engine marshals probs ahead of the loop, so delay is restricted to the python reference engine.
    assert signal_delay_secs <= 0 or engine == "python", "signal_delay_secs requires --engine python"
    traded_probs = delay_prob_series(prob_by_slug_ts, signal_delay_secs)
    engine_data = _data_for_slugs(bt, slugs)
    include_depth = base_kwargs["min_book_depth"] > 0.0 or base_kwargs["max_book_depth"] > 0.0
    marshalled = marshal_timeline(engine_data, slugs, traded_probs, include_depth=include_depth) if engine == "cpp" else None
    gate_signals_by_window: dict = {}
    aligned_by_window_arm: dict = {}
    def _row_task(value, strategy, model_params, model_signals, naive_strategy, naive_sweep_params, naive_signals):
        # Bind one sweep value's prepared inputs into a zero-arg closure so _run_concurrent can fan the values across cores; the bootstrap is folded in so each task builds its whole row off-thread.
        def _task():
            model_result = _run_engine_arm(engine, engine_data, strategy, marshalled, STRATEGY_WORLD_MODEL, model_params, cash, model_signals)
            naive_result = _run_engine_arm(engine, engine_data, naive_strategy, marshalled, STRATEGY_NAIVE_LAG, naive_sweep_params, cash, naive_signals)
            trade_pnls = per_trade_pnls(model_result, slippage_per_share)
            boot = bootstrap_survivability(trade_pnls, bankroll=cash)
            boot_market = bootstrap_survivability_by_market(per_trade_pnls_by_market(model_result, slippage_per_share), bankroll=cash)
            return {
                "value": float(value),
                "sweep_param": sweep_param,
                "pnl": float(model_result.total_pnl),
                "pnl_after_slippage": float(sum(trade_pnls)),
                "trades": int(model_result.total_trades),
                "naive_pnl": float(naive_result.total_pnl),
                "naive_trades": int(naive_result.total_trades),
                "model_minus_naive": float(model_result.total_pnl - naive_result.total_pnl),
                "frac_profitable": boot["frac_profitable"],
                "drawdown_p95": boot["drawdown_p95"],
                "n_markets_traded": boot_market["n_markets"],
                "frac_profitable_market": boot_market["frac_profitable"],
                "drawdown_p95_market": boot_market["drawdown_p95"],
                "ci_low_market": boot_market["ci_low"],
            }
        return _task
    tasks = []
    for value in values:
        kwargs = dict(base_kwargs)
        kwargs["use_predictability_gate"] = True
        if sweep_param == "edge":
            kwargs["edge_threshold"] = value
        elif sweep_param == "window":
            kwargs["predictability_window"] = int(value)
        else:
            kwargs["predictability_threshold"] = value
        window = int(kwargs["predictability_window"])
        # The gate signals depend only on the window, so memoise them here (serially, before the parallel fan-out below): a threshold or edge sweep computes them once and the cpp arms align them once too.
        if window not in gate_signals_by_window:
            gate_signals_by_window[window] = _gate_signals_by_slug_ts(bt, traded_probs, window)
        gate_er, gate_dir = gate_signals_by_window[window]
        strategy = WorldModelStrategy(traded_probs, bankroll=cash, gate_er_by_slug_ts=gate_er, **kwargs)
        model_params = {**_cpp_params(kwargs, None), "gate_er_by_slug_ts": gate_er}
        # Fair baseline: the identical asset-correct predictability gate, buying the observable spot-trend direction with no model, so the model must beat trend-following over the same selective gate.
        naive_pred_threshold = value if sweep_param == "predictability" else base_kwargs["predictability_threshold"]
        naive_strategy = NaiveLagStrategy(
            cusum_threshold=base_kwargs["cusum_threshold"], order_size=base_kwargs["order_size"],
            max_tte_frac=base_kwargs["max_tte_frac"], use_predictability_gate=True,
            predictability_threshold=naive_pred_threshold, predictability_window=window,
            gate_er_by_slug_ts=gate_er, gate_dir_by_slug_ts=gate_dir,
            min_book_depth=base_kwargs["min_book_depth"], max_book_depth=base_kwargs["max_book_depth"],
        )
        naive_sweep_params = {
            **base_kwargs, "use_predictability_gate": True, "predictability_threshold": naive_pred_threshold,
            "predictability_window": window, "ev_by_slug_ts": None,
            "gate_er_by_slug_ts": gate_er, "gate_dir_by_slug_ts": gate_dir,
        }
        model_signals = _sweep_aligned(engine, aligned_by_window_arm, marshalled, STRATEGY_WORLD_MODEL, model_params, window)
        naive_signals = _sweep_aligned(engine, aligned_by_window_arm, marshalled, STRATEGY_NAIVE_LAG, naive_sweep_params, window)
        tasks.append(_row_task(value, strategy, model_params, model_signals, naive_strategy, naive_sweep_params, naive_signals))
    rows = _run_concurrent(tasks, _engine_workers(engine, engine_threads))
    return {
        "n_markets": len(slugs), "n_probs": len(prob_by_slug_ts),
        "calibration": settlement_calibration(prob_by_slug_ts, bt), "sweep": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PnL backtest of a frozen world model per spot-vol regime")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path)
    p.add_argument("--regime-film", action="store_true", help="Rebuild the model with RegimeFiLM on (treatment arm).")
    p.add_argument("--is-mimo", action="store_true")
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--hours-val", type=float, default=6.0)
    p.add_argument("--assets", default="BTC", help="Comma-separated asset filter; the engine references are BTC-only.")
    p.add_argument("--intervals", default=None, help="Comma-separated interval filter (e.g. '15m'); newgoal-2 validates 15m before 5m.")
    p.add_argument(
        "--regime-axis", choices=("predictability", "spot_vol"), default="predictability",
        help="predictability = Efficiency-Ratio split (newgoal-2 primary axis); spot_vol = Phase 0.5 split.",
    )
    p.add_argument("--volatility-quantile", type=float, default=0.5)
    p.add_argument("--edge-threshold", type=float, default=0.05)
    p.add_argument("--order-size", type=float, default=50.0)
    p.add_argument("--max-tte-frac", type=float, default=1.0)
    p.add_argument(
        "--min-tte-frac", type=float, default=0.0,
        help="Skip near-expiry trades below this time-remaining fraction (the edge is early-market; near-expiry reverses).",
    )
    p.add_argument("--use-cusum", action="store_true", help="Gate trades on a CUSUM spot-move event (newgoal-2 1b).")
    p.add_argument("--cusum-threshold", type=float, default=0.003, help="CUSUM spot-return threshold; tune on train.")
    p.add_argument("--sizing", choices=("fixed", "kelly"), default="fixed", help="kelly = quarter-Kelly convex sizing (1c).")
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--kelly-cap", type=float, default=0.05)
    p.add_argument(
        "--predictability-gate", action="store_true",
        help="Trade only when the causal rolling spot efficiency ratio is above threshold (selective participation).",
    )
    p.add_argument("--predictability-threshold", type=float, default=0.3, help="Causal-ER gate threshold; tune on train.")
    p.add_argument("--predictability-window", type=int, default=120, help="Ticks of recent spot for the causal ER gate.")
    p.add_argument(
        "--sweep-thresholds", default=None,
        help="Comma-separated predictability-gate thresholds to sweep on the full market set (train-tuning); "
        "precomputes probs once and reports per-threshold deployable PnL + survivability.",
    )
    p.add_argument(
        "--slippage-per-share", type=float, default=0.0,
        help="Per-share execution cost subtracted from each fill in the sweep's bootstrap (deployability stress test).",
    )
    p.add_argument(
        "--signal-delay-secs", type=int, default=0,
        help="Stale the model signal by N seconds before trading (latency stress test); the gate stays at trade time. python engine only.",
    )
    p.add_argument(
        "--calibration-temperature", type=float, default=1.0,
        help="Post-hoc temperature on the settlement prob (fit on train); >1 softens the over-confident head for Kelly.",
    )
    p.add_argument(
        "--sweep-param", choices=("predictability", "edge", "window"), default="predictability",
        help="Which hyperparameter --sweep-thresholds varies: the gate ER (default), the model-vs-book edge, or the gate window.",
    )
    p.add_argument(
        "--min-book-depth", type=float, default=0.0,
        help="Book-depth gate: trade only when two-sided depth clears this floor (tests asset-identity vs depth).",
    )
    p.add_argument(
        "--max-book-depth", type=float, default=0.0,
        help="Book-depth ceiling (0 = none): skip the most-liquid/efficient books, trading the liquidity band.",
    )
    p.add_argument(
        "--prob-source", choices=("settlement", "edge"), default="settlement",
        help="settlement: trade from settlement probability (existing path); edge: trade from predicted EV (BookRelativeEdgeHead).",
    )
    p.add_argument(
        "--deterministic-latent", action="store_true",
        help="Use the deterministic latent ('probs') in the prob precompute so the sweep PnL is reproducible.",
    )
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--out", type=Path, default=Path("reports/pnl_backtest.json"))
    p.add_argument("--device", default=None)
    p.add_argument(
        "--engine", choices=("python", "cpp"), default="python",
        help="python = pure-Python reference engine (default); cpp = native engine (build with "
        "`python -m finmamba3.backtester.engine_cpp.build`). Parity-checked; amortises across a sweep.",
    )
    p.add_argument(
        "--engine-threads", type=int, default=0,
        help="Threads for independent cpp backtests (sweep values, both arms); 0 = auto (os.cpu_count()), "
        "1 = serial. The cpp engine releases the GIL so these run on real cores; the python engine ignores it.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(args.config, _arch_overrides(args.regime_film, args.is_mimo, args.n_layer))
    include_binary = cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    include_spot = cfg.Models.WorldModel.Encoder.get("SpotFeatures", False)
    include_cross = cfg.Models.WorldModel.Encoder.get("CrossIntervalContext", False)
    wm = load_world_model(cfg, args.checkpoint, device)
    if args.prob_source == "edge":
        assert wm.use_edge_head, "checkpoint has no edge head; train with EdgeHead.Enabled=True or use --prob-source settlement."
    else:
        assert wm.use_settlement_head, "checkpoint has no settlement head; the PnL strategy trades its YES probability."
    assets = [a.strip().upper() for a in args.assets.split(",")]
    intervals = [s.strip() for s in args.intervals.split(",")] if args.intervals else None
    bt = build_timeline(data_dir=args.data_val, hours=args.hours_val, assets=assets, intervals=intervals)
    stats = load_normalization(args.norm_path)
    # Only markets with at least one scoreable window can be traded, so the split is taken over those; this also avoids extract_features raising on a market whose YES book never has both sides.
    count_by_slug = _usable_tick_count_by_slug(bt)
    tradeable = [lc for lc in bt.lifecycles if count_by_slug.get(lc.market_slug, 0) >= 64]
    # The reference regime is where the model has the most reason to profit (predictable / low-vol); the shifted regime is the harder one.
    # Degradation = pnl(reference) - pnl(shifted), so a positive degradation means edge lost under the shift, and the FiLM gap subtracts the two arms' degradations.
    if args.regime_axis == "predictability":
        score_by_slug = predictability_from_timeline(bt.timeline, tradeable, metric="efficiency_ratio")
        split = volatility_split(tradeable, realized_vol=score_by_slug, quantile=args.volatility_quantile)
        reference_label, reference_markets = "high_pred", split.test_markets
        shifted_label, shifted_markets = "low_pred", split.train_markets
    else:
        score_by_slug = spot_realized_vol_from_timeline(bt.timeline, tradeable)
        split = volatility_split(tradeable, realized_vol=score_by_slug, quantile=args.volatility_quantile)
        reference_label, reference_markets = "low_vol", split.train_markets
        shifted_label, shifted_markets = "high_vol", split.test_markets
    reference_slugs = [m.market_slug for m in reference_markets]
    shifted_slugs = [m.market_slug for m in shifted_markets]
    assert reference_slugs and shifted_slugs, (
        f"{args.regime_axis} split is degenerate: {reference_label}={len(reference_slugs)} "
        f"{shifted_label}={len(shifted_slugs)}; widen --hours-val."
    )
    logger.info(f"{args.regime_axis} split ({split.description}): {reference_label}={len(reference_slugs)} {shifted_label}={len(shifted_slugs)} markets")
    strategy_kwargs = {
        "edge_threshold": args.edge_threshold,
        "order_size": args.order_size,
        "max_tte_frac": args.max_tte_frac,
        "min_tte_frac": args.min_tte_frac,
        "use_cusum": bool(args.use_cusum),
        "cusum_threshold": args.cusum_threshold,
        "sizing": args.sizing,
        "kelly_fraction": args.kelly_fraction,
        "kelly_cap": args.kelly_cap,
        "use_predictability_gate": bool(args.predictability_gate),
        "predictability_threshold": args.predictability_threshold,
        "predictability_window": args.predictability_window,
        "calibration_temperature": args.calibration_temperature,
        "min_book_depth": args.min_book_depth,
        "max_book_depth": args.max_book_depth,
    }
    if args.sweep_thresholds:
        values = [float(t) for t in args.sweep_thresholds.split(",")]
        all_slugs = [lc.market_slug for lc in tradeable]
        sweep_report = run_threshold_sweep(
            wm, bt, stats, all_slugs, device,
            include_binary, include_spot, include_cross, strategy_kwargs, values, args.cash, args.slippage_per_share,
            args.sweep_param, "probs" if args.deterministic_latent else "random_sample", engine=args.engine,
            signal_delay_secs=args.signal_delay_secs, engine_threads=args.engine_threads,
        )
        sweep_report["signal_delay_secs"] = int(args.signal_delay_secs)
        sweep_report["checkpoint"] = str(args.checkpoint)
        sweep_report["data_val"] = str(args.data_val)
        sweep_report["slippage_per_share"] = float(args.slippage_per_share)
        for row in sweep_report["sweep"]:
            logger.info(
                f"[sweep] {row['sweep_param']}={row['value']:.3f} pnl={row['pnl']:+.1f} (after_slip={row['pnl_after_slippage']:+.1f}) "
                f"trades={row['trades']} naive={row['naive_pnl']:+.1f} ({row['naive_trades']} tr) "
                f"m-n={row['model_minus_naive']:+.1f} P(profit)={row['frac_profitable']:.3f} dd95={row['drawdown_p95']:.3f}"
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(sweep_report, indent=2))
        logger.info(f"[sweep] report written to {args.out}")
        return 0
    report = {
        "checkpoint": str(args.checkpoint), "regime_film": bool(args.regime_film),
        "regime_axis": args.regime_axis, "split": split.description,
        "reference_label": reference_label, "shifted_label": shifted_label,
        "slippage_per_share": float(args.slippage_per_share),
    }
    for label, slugs in ((reference_label, reference_slugs), (shifted_label, shifted_slugs)):
        report[label] = run_pnl_backtest(
            wm, bt, stats, slugs, device,
            include_binary, include_spot, include_cross, strategy_kwargs, args.cash,
            prob_source=args.prob_source,
            slippage_per_share=args.slippage_per_share, engine=args.engine, engine_threads=args.engine_threads,
        )
        cell = report[label]
        boot = cell["model"]["bootstrap"]
        logger.info(
            f"[pnl] {label}: model_pnl={cell['model']['pnl']:+.2f} naive_pnl={cell['naive']['pnl']:+.2f} "
            f"model-naive={cell['model_minus_naive']:+.2f} trades={cell['model']['trades']} "
            f"P(profit)={boot['frac_profitable']:.2f} dd95={boot['drawdown_p95']:.3f} markets={cell['n_markets']}"
        )
    report["degradation_model"] = report[reference_label]["model"]["pnl"] - report[shifted_label]["model"]["pnl"]
    report["degradation_naive"] = report[reference_label]["naive"]["pnl"] - report[shifted_label]["naive"]["pnl"]
    logger.info(f"[pnl] degradation_model ({reference_label}-{shifted_label}) = {report['degradation_model']:+.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info(f"[pnl] report written to {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
