"""Parity matrix: every ported strategy gate, native C++ vs Python reference, on the real timeline.

Each config builds the same strategy object (Python) and params dict (C++) and asserts identical
total_pnl / trades / sharpe / bootstrap. Needs no GPU or checkpoint. Usage: python gate_parity.py [hours].
"""
# region imports
import logging
import sys
from finmamba3.backtester.engine import BacktestEngine
from finmamba3.backtester.engine_cpp import STRATEGY_NAIVE_LAG, STRATEGY_WORLD_MODEL, marshal_timeline, run_marshalled
from finmamba3.eval.pnl_backtest import (
    NaiveLagStrategy, WorldModelStrategy, _data_for_slugs, _gate_signals_by_slug_ts,
    _sharpe_from_snapshots, bootstrap_survivability, per_trade_pnls,
)
from bench_common import BASE, CASH, build_validation_timeline, synth_probs
# endregion
logging.disable(logging.CRITICAL)
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0


def summarize(result):
    return (
        result.total_pnl, result.total_trades, _sharpe_from_snapshots(result.portfolio_snapshots),
        bootstrap_survivability(per_trade_pnls(result), bankroll=CASH)["frac_profitable"],
    )


def run_both(engine_data, slugs, prob_by_slug_ts, strategy, strategy_kind, params, include_depth):
    py = summarize(BacktestEngine(engine_data, strategy, starting_cash=CASH, snapshot_interval=60).run())
    marshalled = marshal_timeline(engine_data, slugs, prob_by_slug_ts, include_depth=include_depth)
    cpp = summarize(run_marshalled(marshalled, strategy_kind, params, CASH, 60))
    return py, cpp


def check(name, py, cpp):
    ok = (py[1] == cpp[1] and abs(py[0] - cpp[0]) < 1e-4 and abs(py[2] - cpp[2]) < 1e-6 and abs(py[3] - cpp[3]) < 1e-9)
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:34s} pnl py={py[0]:+10.3f} cpp={cpp[0]:+10.3f}  tr {py[1]}/{cpp[1]}  shpd={abs(py[2] - cpp[2]):.0e} bootd={abs(py[3] - cpp[3]):.0e}")
    return ok


def main():
    bt = build_validation_timeline(HOURS)
    slugs = [lifecycle.market_slug for lifecycle in bt.lifecycles]
    engine_data = _data_for_slugs(bt, slugs)
    prob_by_slug_ts = synth_probs(bt, slugs)
    ev_by_slug_ts = {key: (prob - 0.5, 0.5 - prob) for key, prob in prob_by_slug_ts.items()}
    gate_er, gate_dir = _gate_signals_by_slug_ts(bt, prob_by_slug_ts, 120)
    print(f"timeline T={len(bt.timeline)} M={len(slugs)} probs={len(prob_by_slug_ts)}")
    results = []
    def wm(extra_kwargs, extra_params, strategy_kwargs, include_depth=False):
        kwargs = {**BASE, **extra_kwargs}
        strat = WorldModelStrategy(prob_by_slug_ts, bankroll=CASH, **strategy_kwargs, **kwargs)
        params = {**kwargs, "ev_by_slug_ts": None, **extra_params}
        return run_both(engine_data, slugs, prob_by_slug_ts, strat, STRATEGY_WORLD_MODEL, params, include_depth)
    results.append(check("settlement fixed (baseline)", *wm({}, {}, {"ev_by_slug_ts": None})))
    results.append(check("kelly sizing", *wm({"sizing": "kelly"}, {}, {"ev_by_slug_ts": None})))
    results.append(check("calibration_temperature=2.5", *wm({"calibration_temperature": 2.5}, {}, {"ev_by_slug_ts": None})))
    results.append(check("use_cusum (model)", *wm({"use_cusum": True}, {}, {"ev_by_slug_ts": None})))
    results.append(check("depth band [100,5000]", *wm({"min_book_depth": 100.0, "max_book_depth": 5000.0}, {}, {"ev_by_slug_ts": None}, include_depth=True)))
    results.append(check("pred-gate live-window (model)", *wm({"use_predictability_gate": True}, {}, {"ev_by_slug_ts": None})))
    results.append(check("pred-gate per-market ER (model)", *wm({"use_predictability_gate": True}, {"gate_er_by_slug_ts": gate_er}, {"ev_by_slug_ts": None, "gate_er_by_slug_ts": gate_er})))
    results.append(check("edge mode (ev)", *wm({}, {"ev_by_slug_ts": ev_by_slug_ts}, {"ev_by_slug_ts": ev_by_slug_ts})))
    results.append(check("kelly + calib + per-market gate", *wm({"sizing": "kelly", "calibration_temperature": 1.8, "use_predictability_gate": True}, {"gate_er_by_slug_ts": gate_er}, {"ev_by_slug_ts": None, "gate_er_by_slug_ts": gate_er})))
    def naive(extra_kwargs, extra_params):
        strat = NaiveLagStrategy(cusum_threshold=0.003, order_size=50.0, max_tte_frac=1.0, **extra_kwargs)
        params = {"order_size": 50.0, "max_tte_frac": 1.0, "cusum_threshold": 0.003, **extra_params}
        return run_both(engine_data, slugs, prob_by_slug_ts, strat, STRATEGY_NAIVE_LAG, params, False)
    results.append(check("naive cusum", *naive({}, {"use_cusum": True})))
    results.append(check("naive pred-gate live-window", *naive({"use_predictability_gate": True}, {"use_predictability_gate": True})))
    results.append(check("naive pred-gate per-market ER", *naive(
        {"use_predictability_gate": True, "gate_er_by_slug_ts": gate_er, "gate_dir_by_slug_ts": gate_dir},
        {"use_predictability_gate": True, "gate_er_by_slug_ts": gate_er, "gate_dir_by_slug_ts": gate_dir},
    )))
    print(f"\n{sum(results)}/{len(results)} configs PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
