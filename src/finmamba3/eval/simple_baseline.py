"""Simple-model baseline for the PnL task: does the Mamba world model earn its keep?

The economic headline (pnl_backtest) compares the world model only to a no-model trend-following naive,
which answers "is there a latency edge" but not "do you need a world model to harvest it". This module
closes that gap: it fits a logistic regression and a gradient-boosted tree on the *same* spot-conditioned
per-tick features the world-model encoder consumes, on the same training split, then trades each model's
settlement probability through the identical strategy, engine, gate, and bootstrap. If a linear/GBT model
on the same features captures the same edge, the world-model machinery is not load-bearing for the trade;
if the world model wins, the architecture is justified. Either way it is the controlled baseline the
backtest was missing. No world-model forward pass is needed, so this runs without the mamba_ssm kernels.
"""
# region imports
from __future__ import annotations
import argparse
import gc
import json
import logging
from pathlib import Path
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from finmamba3.backtester import build_timeline
from finmamba3.backtester.engine import BacktestEngine
from finmamba3.config import DotDict, parse_args_and_update_config
from finmamba3.envs.lob_features import load_normalization
from finmamba3.eval.pnl_backtest import (
    NaiveLagStrategy, WorldModelStrategy, _eval_sequence, _gate_signals_by_slug_ts, _usable_tick_count_by_slug,
    bootstrap_survivability, bootstrap_survivability_by_market, per_trade_pnls, per_trade_pnls_by_market,
    settlement_calibration,
)
from finmamba3.eval.predictability import predictability_from_timeline
from finmamba3.eval.regime_split import volatility_split
from finmamba3.sequence_builder import _settlement_yes_outcome
# endregion
logger = logging.getLogger(__name__)


def _tte_weights(ts_sec: np.ndarray, start_ts: int, end_ts: int) -> np.ndarray:
    """Per-tick (1 - time-to-expiry) weight, the same ramp the settlement head trains under.

    Early ticks carry a noisy label (the outcome is barely decided), so the settlement loss down-weights
    them by 1 - tau; the baseline trains under the identical weighting so the only difference between it
    and the world model is the function class, not the supervision.
    """
    span = max(float(end_ts - start_ts), 1.0)
    tau = np.clip((float(end_ts) - ts_sec.astype(np.float64)) / span, 0.0, 1.0)
    return np.clip(1.0 - tau, 0.05, 1.0)


def _collect_supervised(bt, stats, slugs, include_binary, include_spot, include_cross, window: int, stride: int):
    """Stack per-tick features, settlement labels, and TTE weights across markets for model fitting.

    Reuses the exact training feature pipeline (_eval_sequence) so the baseline sees the same normalized
    inputs the world-model encoder does; the label is the market's realized YES outcome broadcast to every
    tick. Markets too short to score, or without a settled outcome, are skipped exactly as the world-model
    evaluator skips them. A stride above 1 keeps every stride-th tick of the concatenated time-ordered
    series, an unbiased subsample over the full training period that bounds the sklearn fit cost without
    truncating any part of the range.
    """
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    feature_rows = []
    label_rows = []
    weight_rows = []
    for slug in slugs:
        outcome = _settlement_yes_outcome(bt.settlements.get(slug))
        if outcome is None:
            continue
        seq = _eval_sequence(bt, slug, stats, include_binary, include_spot, include_cross)
        flat = seq.to_flat()
        if flat.shape[0] < window:
            continue
        lifecycle = lifecycle_by_slug[slug]
        feature_rows.append(flat)
        label_rows.append(np.full(flat.shape[0], float(outcome), dtype=np.float64))
        weight_rows.append(_tte_weights(seq.ts_sec, lifecycle.start_ts, lifecycle.end_ts))
    features = np.concatenate(feature_rows, axis=0)
    labels = np.concatenate(label_rows)
    weights = np.concatenate(weight_rows)
    if stride > 1:
        # Copy the strided views so the full-resolution base arrays are released before the fit.
        features = features[::stride].copy()
        labels = labels[::stride].copy()
        weights = weights[::stride].copy()
    return features, labels, weights


def _prob_series_from_model(model, bt, stats, slugs, include_binary, include_spot, include_cross, window: int) -> dict:
    """Per-tick YES probability from a fitted sklearn model, keyed (slug, ts) on the same ticks the model trades.

    Mirrors world_model_yes_prob_series' tradeable-tick set exactly: a probability is registered for every
    tick from window-1 onward (the positions the world model scores), so the strategy, gate, and engine see
    the baseline's probabilities at the identical (slug, ts) keys. Predictions are batched per market.
    """
    prob_by_slug_ts = {}
    for slug in slugs:
        seq = _eval_sequence(bt, slug, stats, include_binary, include_spot, include_cross)
        flat = seq.to_flat()
        if flat.shape[0] < window:
            continue
        scored = flat[window - 1 :]
        probs = model.predict_proba(scored)[:, 1]
        ts_scored = seq.ts_sec[window - 1 :]
        for offset in range(scored.shape[0]):
            prob_by_slug_ts[(slug, int(ts_scored[offset]))] = float(probs[offset])
    return prob_by_slug_ts


def _trade_arm(bt, prob_by_slug_ts, slugs, strategy_kwargs, cash: float) -> dict:
    """Run one probability source through the headline gated strategy and its fair naive, return PnL + survivability.

    Identical to pnl_backtest's gated single-value path: the asset-correct predictability gate, the same
    fixed sizing and edge threshold, the same NaiveLagStrategy control, and both the iid and the market-block
    bootstrap so the baseline is reported under the same survivability gate as the world model.
    """
    from finmamba3.eval.pnl_backtest import _data_for_slugs
    engine_data = _data_for_slugs(bt, slugs)
    window = int(strategy_kwargs["predictability_window"])
    gate_er, gate_dir = _gate_signals_by_slug_ts(bt, prob_by_slug_ts, window)
    model_strategy = WorldModelStrategy(prob_by_slug_ts, bankroll=cash, gate_er_by_slug_ts=gate_er, **strategy_kwargs)
    model_result = BacktestEngine(engine_data, model_strategy, starting_cash=cash, snapshot_interval=60).run()
    naive_strategy = NaiveLagStrategy(
        cusum_threshold=strategy_kwargs["cusum_threshold"], order_size=strategy_kwargs["order_size"],
        max_tte_frac=strategy_kwargs["max_tte_frac"], use_predictability_gate=True,
        predictability_threshold=strategy_kwargs["predictability_threshold"], predictability_window=window,
        gate_er_by_slug_ts=gate_er, gate_dir_by_slug_ts=gate_dir,
    )
    naive_result = BacktestEngine(engine_data, naive_strategy, starting_cash=cash, snapshot_interval=60).run()
    boot = bootstrap_survivability(per_trade_pnls(model_result), bankroll=cash)
    boot_market = bootstrap_survivability_by_market(per_trade_pnls_by_market(model_result), bankroll=cash)
    return {
        "pnl": float(model_result.total_pnl),
        "trades": int(model_result.total_trades),
        "naive_pnl": float(naive_result.total_pnl),
        "naive_trades": int(naive_result.total_trades),
        "model_minus_naive": float(model_result.total_pnl - naive_result.total_pnl),
        "bootstrap": boot,
        "bootstrap_market": boot_market,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logistic / GBT baseline on the same features as the world-model PnL trade")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-train", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path)
    p.add_argument("--assets", default="BTC")
    p.add_argument("--intervals", default=None)
    p.add_argument("--hours-train", type=float, default=1e9)
    p.add_argument("--hours-val", type=float, default=1e9)
    p.add_argument("--window", type=int, default=64, help="Context window; matches the world-model evaluator's tradeable-tick offset.")
    p.add_argument("--train-stride", type=int, default=1, help="Keep every K-th supervised training tick (unbiased over time) to bound the sklearn fit cost; 1 = full ~1.6M-tick fit.")
    p.add_argument("--predictability-threshold", type=float, default=0.60, help="Frozen ER gate (headline value).")
    p.add_argument("--predictability-window", type=int, default=120)
    p.add_argument("--edge-threshold", type=float, default=0.03)
    p.add_argument("--order-size", type=float, default=50.0)
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--out", type=Path, default=Path("reports/simple_baseline.json"))
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = DotDict(parse_args_and_update_config(yaml.safe_load(f), argv=[]))
    include_binary = cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    include_spot = cfg.Models.WorldModel.Encoder.get("SpotFeatures", False)
    include_cross = cfg.Models.WorldModel.Encoder.get("CrossIntervalContext", False)
    stats = load_normalization(args.norm_path)
    assets = [a.strip().upper() for a in args.assets.split(",")]
    intervals = [s.strip() for s in args.intervals.split(",")] if args.intervals else None

    bt_train = build_timeline(data_dir=args.data_train, hours=args.hours_train, assets=assets, intervals=intervals)
    train_count = _usable_tick_count_by_slug(bt_train)
    train_slugs = [lc.market_slug for lc in bt_train.lifecycles if train_count.get(lc.market_slug, 0) >= args.window]
    features, labels, weights = _collect_supervised(bt_train, stats, train_slugs, include_binary, include_spot, include_cross, args.window, args.train_stride)
    logger.info(f"[baseline] train: {len(train_slugs)} markets, {features.shape[0]} ticks (stride={args.train_stride}), {features.shape[1]} features, YES rate {labels.mean():.3f}")
    # The train timeline is unused past feature collection, so release it before the val timeline is built.
    # Holding both resident exceeds the 15 GiB WSL cap (the OOM that killed the first stride run mid val build).
    del bt_train, train_count, train_slugs
    gc.collect()
    logistic = LogisticRegression(max_iter=2000, C=1.0)
    logistic.fit(features, labels, sample_weight=weights)
    gbt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)
    gbt.fit(features, labels, sample_weight=weights)

    bt_val = build_timeline(data_dir=args.data_val, hours=args.hours_val, assets=assets, intervals=intervals)
    val_count = _usable_tick_count_by_slug(bt_val)
    tradeable_val = [lc for lc in bt_val.lifecycles if val_count.get(lc.market_slug, 0) >= args.window]
    val_slugs = [lc.market_slug for lc in tradeable_val]
    # The same predictability split the world-model evaluator uses, so the high/low-pred PnL is comparable
    # to the EV-head table: high_pred is the forecastable (reference) half, low_pred the random-walk half.
    score_by_slug = predictability_from_timeline(bt_val.timeline, tradeable_val, metric="efficiency_ratio")
    split = volatility_split(tradeable_val, realized_vol=score_by_slug, quantile=0.5)
    high_pred_slugs = [m.market_slug for m in split.test_markets]
    low_pred_slugs = [m.market_slug for m in split.train_markets]
    logger.info(f"[baseline] val: {len(val_slugs)} markets (high_pred={len(high_pred_slugs)} low_pred={len(low_pred_slugs)})")
    strategy_kwargs = {
        "edge_threshold": args.edge_threshold, "order_size": args.order_size, "max_tte_frac": 1.0, "min_tte_frac": 0.0,
        "use_cusum": False, "cusum_threshold": 0.003, "sizing": "fixed", "kelly_fraction": 0.25, "kelly_cap": 0.05,
        "use_predictability_gate": True, "predictability_threshold": args.predictability_threshold,
        "predictability_window": args.predictability_window, "calibration_temperature": 1.0,
        "min_book_depth": 0.0, "max_book_depth": 0.0,
    }
    report = {"config": str(args.config), "data_val": str(args.data_val), "n_val_markets": len(val_slugs),
              "n_high_pred": len(high_pred_slugs), "n_low_pred": len(low_pred_slugs), "arms": {}}
    for name, model in (("logistic", logistic), ("gbt", gbt)):
        prob_by_slug_ts = _prob_series_from_model(model, bt_val, stats, val_slugs, include_binary, include_spot, include_cross, args.window)
        arm = {
            "total": _trade_arm(bt_val, prob_by_slug_ts, val_slugs, strategy_kwargs, args.cash),
            "high_pred": _trade_arm(bt_val, prob_by_slug_ts, high_pred_slugs, strategy_kwargs, args.cash),
            "low_pred": _trade_arm(bt_val, prob_by_slug_ts, low_pred_slugs, strategy_kwargs, args.cash),
            "calibration": settlement_calibration(prob_by_slug_ts, bt_val),
            "n_probs": len(prob_by_slug_ts),
        }
        report["arms"][name] = arm
        total, hi, lo = arm["total"], arm["high_pred"], arm["low_pred"]
        logger.info(
            f"[baseline] {name}: total={total['pnl']:+.0f} (P={total['bootstrap_market']['frac_profitable']:.3f}, "
            f"{total['bootstrap_market']['n_markets']} mkts) hi={hi['pnl']:+.0f} (P={hi['bootstrap_market']['frac_profitable']:.3f}) "
            f"lo={lo['pnl']:+.0f} (P={lo['bootstrap_market']['frac_profitable']:.3f}) naive={total['naive_pnl']:+.0f} "
            f"Brier={arm['calibration']['brier']:.3f}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info(f"[baseline] report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
