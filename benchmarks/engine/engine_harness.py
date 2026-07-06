"""Parity + profiling harness: native C++ backtester engine vs the Python reference, single run.

Drives the real validation timeline through both engines with deterministic synthetic probabilities,
exercising the exact pnl_backtest seam: marshal_timeline -> run_marshalled -> drop-in result ->
per_trade_pnls / sharpe. Needs no GPU or checkpoint. Usage: python engine_harness.py [hours].
"""
# region imports
import logging
import sys
import time
from finmamba3.backtester.engine import BacktestEngine
from finmamba3.backtester.engine_cpp import STRATEGY_NAIVE_LAG, STRATEGY_WORLD_MODEL, marshal_timeline, run_marshalled
from finmamba3.eval.pnl_backtest import (
    NaiveLagStrategy, WorldModelStrategy, _cpp_params, _data_for_slugs, _naive_cpp_params,
    _sharpe_from_snapshots, bootstrap_survivability, per_trade_pnls,
)
from bench_common import BASE, CASH, build_validation_timeline, synth_probs
# endregion
logging.disable(logging.CRITICAL)
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0


def timeit(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main():
    start = time.perf_counter()
    bt = build_validation_timeline(HOURS)
    build_secs = time.perf_counter() - start
    slugs = [lifecycle.market_slug for lifecycle in bt.lifecycles]
    engine_data = _data_for_slugs(bt, slugs)
    prob_by_slug_ts = synth_probs(bt, slugs)
    print(f"timeline: T={len(bt.timeline)} ticks  M={len(slugs)} markets  probs={len(prob_by_slug_ts)}  (build {build_secs:.1f}s)")
    model_strategy = WorldModelStrategy(prob_by_slug_ts, bankroll=CASH, ev_by_slug_ts=None, **BASE)
    naive_strategy = NaiveLagStrategy(cusum_threshold=0.003, order_size=50.0, max_tte_frac=1.0)
    py_model, py_model_secs = timeit(lambda: BacktestEngine(engine_data, model_strategy, starting_cash=CASH, snapshot_interval=60).run())
    py_naive, py_naive_secs = timeit(lambda: BacktestEngine(engine_data, naive_strategy, starting_cash=CASH, snapshot_interval=60).run())
    marshalled, marshal_secs = timeit(lambda: marshal_timeline(engine_data, slugs, prob_by_slug_ts))
    cpp_model, cpp_model_secs = timeit(lambda: run_marshalled(marshalled, STRATEGY_WORLD_MODEL, _cpp_params(BASE, None), CASH, 60))
    cpp_naive, cpp_naive_secs = timeit(lambda: run_marshalled(marshalled, STRATEGY_NAIVE_LAG, _naive_cpp_params(50.0, 1.0, 0.003), CASH, 60))
    py_sharpe = _sharpe_from_snapshots(py_model.portfolio_snapshots)
    cpp_sharpe = _sharpe_from_snapshots(cpp_model.portfolio_snapshots)
    py_boot = bootstrap_survivability(per_trade_pnls(py_model), bankroll=CASH)
    cpp_boot = bootstrap_survivability(per_trade_pnls(cpp_model), bankroll=CASH)
    print("\n--- PARITY (model arm) ---")
    print(f"  total_pnl    py={py_model.total_pnl:+.6f}  cpp={cpp_model.total_pnl:+.6f}  d={abs(py_model.total_pnl - cpp_model.total_pnl):.2e}")
    print(f"  total_trades py={py_model.total_trades}  cpp={cpp_model.total_trades}")
    print(f"  sharpe       py={py_sharpe:+.6f}  cpp={cpp_sharpe:+.6f}  d={abs(py_sharpe - cpp_sharpe):.2e}")
    print(f"  n_snapshots  py={len(py_model.portfolio_snapshots)}  cpp={len(cpp_model.portfolio_snapshots)}")
    print(f"  n_fills      py={len(py_model.fills)}  cpp={len(cpp_model.fills)}")
    print(f"  boot.frac_profitable py={py_boot['frac_profitable']:.6f}  cpp={cpp_boot['frac_profitable']:.6f}")
    print(f"  boot.dd95            py={py_boot['drawdown_p95']:.6f}  cpp={cpp_boot['drawdown_p95']:.6f}")
    print("--- PARITY (naive arm) ---")
    print(f"  total_pnl    py={py_naive.total_pnl:+.6f}  cpp={cpp_naive.total_pnl:+.6f}  d={abs(py_naive.total_pnl - cpp_naive.total_pnl):.2e}")
    print(f"  total_trades py={py_naive.total_trades}  cpp={cpp_naive.total_trades}")
    parity_ok = (
        py_model.total_trades == cpp_model.total_trades and
        abs(py_model.total_pnl - cpp_model.total_pnl) < 1e-4 and
        abs(py_sharpe - cpp_sharpe) < 1e-6 and
        abs(py_boot["frac_profitable"] - cpp_boot["frac_profitable"]) < 1e-9 and
        py_naive.total_trades == cpp_naive.total_trades and
        abs(py_naive.total_pnl - cpp_naive.total_pnl) < 1e-4
    )
    print(f"\nPARITY: {'PASS' if parity_ok else 'FAIL'}")
    print("\n--- PROFILE (seconds) ---")
    print(f"  python  model={py_model_secs:.4f}  naive={py_naive_secs:.4f}  total={py_model_secs + py_naive_secs:.4f}")
    print(f"  cpp     marshal={marshal_secs:.4f}  model={cpp_model_secs:.5f}  naive={cpp_naive_secs:.5f}")
    print(f"  speedup compute(model) = {py_model_secs / max(cpp_model_secs, 1e-9):.1f}x")
    single_py = py_model_secs + py_naive_secs
    single_cpp = marshal_secs + cpp_model_secs + cpp_naive_secs
    print(f"  speedup single-run (model+naive, marshal once) = {single_py / max(single_cpp, 1e-9):.2f}x")
    sweep_values = 20
    sweep_py = single_py * sweep_values
    sweep_cpp = marshal_secs + sweep_values * (cpp_model_secs + cpp_naive_secs)
    print(f"  speedup sweep x{sweep_values} (marshal once, rerun) = {sweep_py / max(sweep_cpp, 1e-9):.1f}x")
    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
