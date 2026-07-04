"""Bitwise-identity check: vectorized bootstrap_survivability_by_market vs the original Python loop.

The vectorized survivability bootstrap must reproduce the original per-path double loop exactly, so the
threaded sweep that depends on it cannot move any reported number. Runs GPU-free: pure numpy, no model.
Usage: python verify_bootstrap_vec.py.
"""
# region imports
import numpy as np
from finmamba3.eval.pnl_backtest import bootstrap_survivability_by_market
# endregion


def reference_loop(pnls_by_market, bankroll=10_000.0, n_paths=10_000, seed=0):
    slugs = [slug for slug in pnls_by_market if pnls_by_market[slug]]
    if not slugs:
        return {"n_markets": 0, "n_trades": 0, "frac_profitable": 0.0, "drawdown_p95": 0.0, "ci_low": 0.0, "ci_high": 0.0, "mean_terminal": 0.0}
    cumulative_by_slug = [np.cumsum(np.asarray(pnls_by_market[slug], dtype=np.float64)) for slug in slugs]
    totals = np.array([float(cum[-1]) for cum in cumulative_by_slug])
    lows = np.array([float(cum.min()) for cum in cumulative_by_slug])
    highs = np.array([float(cum.max()) for cum in cumulative_by_slug])
    intrinsic_dd = np.array([float((np.maximum.accumulate(cum) - cum).max()) for cum in cumulative_by_slug])
    n_markets = len(slugs)
    n_trades = int(sum(cum.shape[0] for cum in cumulative_by_slug))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_markets, size=(n_paths, n_markets))
    terminal = totals[draws].sum(axis=1)
    drawdown = np.empty(n_paths, dtype=np.float64)
    for path in range(n_paths):
        peak = bankroll
        prefix = bankroll
        worst = 0.0
        for market_index in draws[path]:
            dip = max(intrinsic_dd[market_index], peak - (prefix + lows[market_index]))
            if dip > worst:
                worst = dip
            high = prefix + highs[market_index]
            if high > peak:
                peak = high
            prefix += totals[market_index]
        drawdown[path] = worst / bankroll
    return {
        "n_markets": n_markets, "n_trades": n_trades,
        "frac_profitable": float(np.mean(terminal > 0.0)),
        "drawdown_p95": float(np.percentile(drawdown, 95.0)),
        "ci_low": float(np.percentile(terminal, 2.5)),
        "ci_high": float(np.percentile(terminal, 97.5)),
        "mean_terminal": float(np.mean(terminal)),
    }


def random_case(rng, n_markets):
    pnls_by_market = {}
    for market_index in range(n_markets):
        n_trades = int(rng.integers(1, 12))
        pnls_by_market[f"slug-{market_index}"] = list((rng.standard_normal(n_trades) * rng.uniform(1, 80)).astype(float))
    return pnls_by_market


def main():
    rng = np.random.default_rng(123)
    worst_dd = 0.0
    for trial in range(40):
        n_markets = int(rng.integers(1, 250))
        case = random_case(rng, n_markets)
        reference = reference_loop(case)
        vectorized = bootstrap_survivability_by_market(case)
        if reference.keys() != vectorized.keys():
            raise SystemExit(f"KEY MISMATCH trial={trial}")
        for key in reference:
            if reference[key] != vectorized[key]:
                raise SystemExit(f"MISMATCH trial={trial} key={key} ref={reference[key]!r} new={vectorized[key]!r}")
        worst_dd = max(worst_dd, reference["drawdown_p95"])
    if reference_loop({}) != bootstrap_survivability_by_market({}):
        raise SystemExit("MISMATCH on the empty case")
    if reference_loop({"a": []}) != bootstrap_survivability_by_market({"a": []}):
        raise SystemExit("MISMATCH on the all-empty-market case")
    print(f"OK: 40 random cases + empties all BITWISE-IDENTICAL (sample dd95 max={worst_dd:.6f})")


if __name__ == "__main__":
    main()
