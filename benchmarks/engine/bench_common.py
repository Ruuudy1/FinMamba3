"""Shared fixtures for the native-engine benchmarks.

The validation timeline, the deterministic synthetic probabilities, and the common strategy kwargs live
here so the parity and profiling harnesses cannot drift apart. The engine logic is independent of where
the probabilities come from, so these synthetic probabilities exercise the exact pnl_backtest seam with
no GPU and no checkpoint.
"""
# region imports
import os
from pathlib import Path
from finmamba3.backtester import build_timeline
# endregion
CASH = 10_000.0
DATA_VAL = Path(os.environ.get("FINMAMBA_DATA_VAL", "data/validation"))
ASSETS = ["BTC", "ETH", "SOL"]
INTERVALS = ["5m", "15m"]
# Strategy kwargs shared by every benchmark arm. Individual harnesses override single keys on top.
BASE = {
    "edge_threshold": 0.05, "order_size": 50.0, "max_tte_frac": 1.0, "min_tte_frac": 0.0,
    "use_cusum": False, "cusum_threshold": 0.003, "sizing": "fixed", "kelly_fraction": 0.25,
    "kelly_cap": 0.05, "use_predictability_gate": False, "predictability_threshold": 0.3,
    "predictability_window": 120, "calibration_temperature": 1.0, "min_book_depth": 0.0, "max_book_depth": 0.0,
}


def build_validation_timeline(hours):
    return build_timeline(data_dir=DATA_VAL, hours=hours, assets=ASSETS, intervals=INTERVALS)


def synth_probs(bt, slugs):
    """Deterministic per-(slug, ts) probability that diverges from the book mid by up to +/-0.2, so the
    edge gate fires a realistic mix of YES, NO, and no-trade. Built only where the YES book has a mid."""
    wanted_by_slug = {slug: True for slug in slugs}
    prob_by_slug_ts = {}
    for tick in bt.timeline:
        ts = tick.ts_sec
        for slug in tick.order_books:
            if slug not in wanted_by_slug:
                continue
            mid = tick.order_books[slug].yes_book.mid
            if mid <= 0.0 or mid >= 1.0:
                continue
            noise = ((ts * 2654435761) ^ (len(slug) * 40503) ^ (ts % 7)) % 1000 / 1000.0
            prob_by_slug_ts[(slug, ts)] = min(max(mid + (noise - 0.5) * 0.4, 0.01), 0.99)
    return prob_by_slug_ts
