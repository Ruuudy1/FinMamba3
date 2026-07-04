"""Profile + parallel-parity of the cpp gated threshold sweep, GPU-free.

Builds a real timeline from the validation set and reuses the deterministic synthetic probabilities, then
drives the actual run_threshold_sweep at several --engine-threads counts. Reports the per-phase breakdown,
the wall-clock speedup vs threads=1, and asserts the produced rows are byte-identical across thread counts
(so threading changed only the schedule, not the math). Usage: python sweep_profile.py [hours] [n_values].
"""
# region imports
import cProfile
import logging
import os
import pstats
import sys
import time
import torch
from finmamba3.backtester.engine_cpp import _engine, align_run_signals, marshal_timeline, run_marshalled
from finmamba3.eval import pnl_backtest as pb
from finmamba3.eval.pnl_backtest import (
    STRATEGY_WORLD_MODEL, _cpp_params, _data_for_slugs, _gate_signals_by_slug_ts,
    bootstrap_survivability, per_trade_pnls, run_threshold_sweep,
)
from bench_common import BASE, CASH, build_validation_timeline, synth_probs
# endregion
logging.disable(logging.CRITICAL)
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
N_VALUES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
THREAD_COUNTS = [1, 2, 4, 8]


def main():
    if _engine is None:
        raise SystemExit("native engine not built: python -m finmamba3.backtester.engine_cpp.build")
    print(f"building timeline (hours={HOURS}) ...")
    bt = build_validation_timeline(HOURS)
    slugs = [lifecycle.market_slug for lifecycle in bt.lifecycles]
    prob_by_slug_ts = synth_probs(bt, slugs)
    series_by_slug = {}
    for (slug, ts), prob in prob_by_slug_ts.items():
        series_by_slug.setdefault(slug, {})[ts] = prob
    # Patch the two model touch-points so run_threshold_sweep needs no torch model: _eval_sequence
    # forwards the slug, and the prob series becomes that slug's precomputed synthetic series.
    pb._eval_sequence = lambda bt, slug, *a, **k: slug
    pb.world_model_yes_prob_series = lambda wm, seq, device, sample_mode=None, **k: series_by_slug.get(seq, {})
    print(f"timeline: T={len(bt.timeline)} markets={len(slugs)} probs={len(prob_by_slug_ts)} cpu={os.cpu_count()}")
    values = [round(0.2 + 0.08 * i, 4) for i in range(N_VALUES)]
    # Time the sweep's distinct phases once so the Amdahl floor (marshal-once + per-value bootstrap) is
    # visible next to the parallelizable per-run C++ loop.
    engine_data = _data_for_slugs(bt, slugs)
    start = time.perf_counter()
    marshalled = marshal_timeline(engine_data, slugs, prob_by_slug_ts, include_depth=False)
    marshal_secs = time.perf_counter() - start
    start = time.perf_counter()
    gate_er, _gate_dir = _gate_signals_by_slug_ts(bt, prob_by_slug_ts, 120)
    gate_secs = time.perf_counter() - start
    params = {**_cpp_params({**BASE, "use_predictability_gate": True, "predictability_threshold": 0.3}, None), "gate_er_by_slug_ts": gate_er}
    start = time.perf_counter()
    signals = align_run_signals(marshalled, STRATEGY_WORLD_MODEL, params)
    align_secs = time.perf_counter() - start
    start = time.perf_counter()
    result = run_marshalled(marshalled, STRATEGY_WORLD_MODEL, params, CASH, 60, signals=signals)
    run_secs = time.perf_counter() - start
    start = time.perf_counter()
    bootstrap_survivability(per_trade_pnls(result, 0.0), bankroll=CASH)
    boot_secs = time.perf_counter() - start
    print(f"\n--- phase breakdown ({len(slugs)} markets) ---")
    print(f"  marshal_timeline (once): {marshal_secs * 1000:8.1f} ms")
    print(f"  gate signals (once/win): {gate_secs * 1000:8.1f} ms")
    print(f"  align_run_signals      : {align_secs * 1000:8.1f} ms")
    print(f"  run_marshalled (1 C++) : {run_secs * 1000:8.1f} ms")
    print(f"  bootstrap (1 value)    : {boot_secs * 1000:8.1f} ms")
    print(f"  trades in 1 run        : {result.total_trades}")
    def run(threads):
        return run_threshold_sweep(
            None, bt, None, slugs, torch.device("cpu"), False, False, False, BASE, values, CASH,
            engine="cpp", engine_threads=threads,
        )["sweep"]
    print(f"\n--- sweep ({N_VALUES} values x 2 arms) wall-clock vs threads ---")
    base_rows = None
    serial_secs = None
    for threads in THREAD_COUNTS:
        start = time.perf_counter()
        rows = run(threads)
        elapsed = time.perf_counter() - start
        if base_rows is None:
            base_rows, serial_secs = rows, elapsed
        identical = rows == base_rows
        print(f"  threads={threads:>2}  {elapsed:7.3f}s  speedup={serial_secs / elapsed:5.2f}x  rows_identical={identical}")
        if not identical:
            raise SystemExit(f"PARITY FAILURE: threads={threads} produced different rows than threads=1")
    print("\n--- cProfile (threads=1) top cumulative ---")
    profiler = cProfile.Profile()
    profiler.enable()
    run(1)
    profiler.disable()
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
