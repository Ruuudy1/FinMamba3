"""Cross-regime generalization-gap evaluator for the regime-FiLM A/B.

The project thesis is generalization under regime shift: the regime-FiLM treatment
(RegimeFiLM.Enabled true) should lose less direction macro-F1 than the baseline
(RegimeFiLM.Enabled false) when the held-out markets move from a low-volatility to a
high-volatility regime. This script measures exactly that difference-of-differences,
which no single existing tool computed.

It splits the held-out markets at the realized-volatility median into a low-vol
reference regime and a high-vol shifted regime (reusing volatility_split), scores the
world model's direction head on each regime for both checkpoints (reusing the
compare_direction inference path), and reports:

    degradation_arm = macro_f1_low - macro_f1_high            (per arm)
    gap             = degradation_baseline - degradation_treatment

A positive gap means the treatment is more robust to the regime shift, supporting the
thesis. Macro-F1 is the reporting axis because it weights the three direction classes
equally, matching the LOBCAST benchmark and resisting the flat-class accuracy mirage.

The checkpoint stores only weights, so the baseline arm is rebuilt with FiLM off and
the treatment arm with FiLM on by definition of the A/B; both share --is-mimo.

Example
-------
    python -m finmamba3.eval.eval_regime_generalization \\
        --config configs/lob.yaml \\
        --baseline-checkpoint saved_models/lob/.../baseline/ckpt/world_model_final.pth \\
        --treatment-checkpoint saved_models/lob/.../treatment/ckpt/world_model_final.pth \\
        --data-val data/validation --norm-path saved_models/lob/normalization.json \\
        --label-mode triple_barrier --volatility-quantile 0.5
"""
# region imports
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from finmamba3.backtester import build_timeline
from finmamba3.config import DotDict, parse_args_and_update_config
from finmamba3.envs.lob_features import (
    F_LEVEL,
    K_LEVELS,
    apply_normalization,
    extract_features,
    load_normalization,
    make_aggregate_only,
    pick_top_markets,
)
from finmamba3.eval.compare_direction import (
    classification_metrics,
    load_world_model,
    world_model_direction_probs,
    world_model_prediction_mse,
    world_model_settlement_logloss,
)
from finmamba3.eval.regime_split import realized_vol_from_timeline, spot_realized_vol_from_timeline, volatility_split
from finmamba3.sequence_builder import _settlement_yes_outcome, _spot_feature_kwargs
# endregion
logger = logging.getLogger(__name__)
LEVEL_FLAT = K_LEVELS * F_LEVEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure the regime-FiLM generalization gap")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--baseline-checkpoint", required=True, type=Path, help="world_model_*.pth from the RegimeFiLM-off run")
    p.add_argument("--treatment-checkpoint", required=True, type=Path, help="world_model_*.pth from the RegimeFiLM-on run")
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", type=Path,
                   default=Path("saved_models") / "lob" / "normalization.json",
                   help="Train normalization stats; the held-out markets reuse them (no skew).")
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--volatility-quantile", type=float, default=0.5,
                   help="Markets above this realized-vol quantile form the high-vol regime.")
    p.add_argument("--vol-source", choices=("spot", "mid"), default="spot",
                   help="spot = underlying Chainlink realized vol over each market window (Phase 0.5, "
                        "the honest axis); mid = the legacy YES-mid vol, confounded with time-to-expiry.")
    p.add_argument("--max-markets", type=int, default=128,
                   help="Cap on the longest qualifying held-out markets to split and score; bounds cost.")
    p.add_argument("--min-ticks", type=int, default=128,
                   help="Drop markets with fewer two-sided ticks so every regime market yields windows.")
    p.add_argument("--metric", choices=("prediction_mse", "direction_macro_f1", "settlement_logloss"), default="prediction_mse",
                   help="prediction_mse = decoder NLL on the prior-predicted next tick (regime-sensitive, "
                        "non-degenerate); direction_macro_f1 = 3-class direction (collapses on Polymarket).")
    p.add_argument("--label-mode", choices=("next_tick", "triple_barrier"), default="triple_barrier",
                   help="Direction labelling for the direction_macro_f1 metric; triple_barrier is the denoised target.")
    p.add_argument("--threshold", type=float, default=0.01, help="Profit/stop barrier (triple_barrier) or next-tick threshold.")
    p.add_argument("--tb-horizon", type=int, default=32, help="Forward horizon in ticks for triple-barrier labelling.")
    p.add_argument("--windows-per-market", type=int, default=256,
                   help="Random windows scored per market; higher trims the per-regime macro-F1 noise.")
    p.add_argument("--is-mimo", action="store_true", help="Rebuild both arms as MIMO to match an H100 MIMO A/B; default SISO.")
    p.add_argument("--n-layer", type=int, default=None,
                   help="Override Mamba3.n_layer to match a non-default-depth training run.")
    p.add_argument("--out", type=Path, default=Path("reports/regime_generalization.md"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device_from_arg(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(config_path: Path, overrides: list[str]) -> DotDict:
    """Load the YAML config and apply dotted CLI-style overrides via the shared parser."""
    with open(config_path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    return DotDict(parse_args_and_update_config(cfg_raw, argv=overrides))


def _arch_overrides(regime_film_enabled: bool, is_mimo: bool, n_layer: int | None = None) -> list[str]:
    """Dotted overrides that pin the architecture flags a checkpoint depends on."""
    overrides = [
        "--Models.WorldModel.RegimeFiLM.Enabled", "true" if regime_film_enabled else "false",
        "--Models.WorldModel.Mamba3.is_mimo", "true" if is_mimo else "false",
    ]
    if n_layer is not None:
        overrides += ["--Models.WorldModel.Mamba3.n_layer", str(n_layer)]
    return overrides


def _split_slugs(bt, realized_vol: dict[str, float], max_markets: int, min_ticks: int,
                 quantile: float) -> tuple[list[str], list[str], str]:
    """Partition the qualifying held-out markets into (low_vol, high_vol) slug lists.

    Only the max_markets longest markets with at least min_ticks two-sided ticks qualify,
    which bounds the evaluation cost and uses the fullest (most window-rich) markets; the
    vol median is then taken over that pool so each regime is a clean low/high contrast.
    """
    qualified_slugs = pick_top_markets(bt, n=max_markets, min_ticks=min_ticks)
    keep_by_slug = {slug: True for slug in qualified_slugs}
    markets = [m for m in bt.lifecycles if m.market_slug in keep_by_slug]
    split = volatility_split(markets, realized_vol=realized_vol, quantile=quantile)
    low_slugs = [m.market_slug for m in split.train_markets]
    high_slugs = [m.market_slug for m in split.test_markets]
    return low_slugs, high_slugs, split.description


def generalization_gap(degradation_baseline: float, degradation_treatment: float) -> float:
    """The headline metric: how much less macro-F1 the treatment loses than the baseline.

    A positive value means the regime-FiLM treatment is more robust to the volatility
    regime shift than the baseline, which is the thesis's prediction.
    """
    return degradation_baseline - degradation_treatment


def _regime_sequences(bt, slugs: list[str], stats, aggregate_only: bool, include_binary_features: bool,
                      include_spot_features: bool = False) -> list:
    """Per-market normalized sequences for the given slugs, reusing the train stats.

    Mirrors build_market_sequences' inner loop but applies a pre-fit normalization
    instead of fitting one, so the held-out markets are scored on the training scale.
    """
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    seqs = []
    for slug in slugs:
        # Carry the market's realized settlement for settlement_logloss; the other metrics ignore yes_outcome.
        settlement = bt.settlements.get(slug)
        yes_outcome = _settlement_yes_outcome(settlement)
        spot_kwargs = _spot_feature_kwargs(slug, settlement, lifecycle_by_slug) if include_spot_features else {}
        raw = extract_features(
            bt.timeline, slug, yes_outcome=yes_outcome,
            include_binary_features=include_binary_features,
            include_spot_features=include_spot_features, **spot_kwargs,
        )
        seq = apply_normalization(raw, stats)
        if aggregate_only:
            seq = make_aggregate_only(seq)
        seqs.append(seq)
    return seqs


def _regime_macro_f1(wm, seqs: list, threshold: float, label_mode: str, tb_horizon: int,
                     windows_per_market: int, device: torch.device) -> float:
    """Pool the direction predictions across a regime's markets, then score one macro-F1.

    Pooling before classification_metrics keeps boundary-respecting per-market windows
    while still yielding a single macro-F1 for the regime.
    """
    probs_parts = []
    labels_parts = []
    for seq in seqs:
        probs, labels = world_model_direction_probs(
            wm, seq, threshold, label_mode, tb_horizon, LEVEL_FLAT, device,
            windows_per_market=windows_per_market,
        )
        probs_parts.append(probs)
        labels_parts.append(labels)
    pooled_probs = np.concatenate(probs_parts, axis=0)
    pooled_labels = np.concatenate(labels_parts, axis=0)
    return classification_metrics(pooled_probs, pooled_labels)["macro_f1"]


def _regime_prediction_mse(wm, seqs: list, windows_per_market: int, device: torch.device) -> float:
    """Pooled one-step-ahead decoder MSE across a regime's markets (lower is better).

    Summing squared error and element counts before dividing gives the exact element-weighted
    MSE over the regime, so longer markets are not over- or under-counted.
    """
    total_sse = 0.0
    total_count = 0
    for seq in seqs:
        sse, count = world_model_prediction_mse(wm, seq, device, windows_per_market=windows_per_market)
        total_sse += sse
        total_count += count
    assert total_count > 0, "no windows scored for this regime; markets are shorter than the window length."
    return total_sse / total_count


def _regime_settlement_logloss(wm, seqs: list, windows_per_market: int, device: torch.device) -> float:
    """Pooled settlement binary cross-entropy across a regime's markets (lower is better).

    Summing the BCE and the resolved-position count before dividing gives the exact position-weighted
    log-loss over the regime, so longer or better-resolved markets are not over- or under-counted.
    """
    total_bce = 0.0
    total_count = 0
    for seq in seqs:
        sum_bce, count = world_model_settlement_logloss(wm, seq, device, windows_per_market=windows_per_market)
        total_bce += sum_bce
        total_count += count
    assert total_count > 0, "no resolved settlement positions scored for this regime; markets lack yes_outcome."
    return total_bce / total_count


def _regime_score(wm, seqs: list, metric: str, threshold: float, label_mode: str, tb_horizon: int,
                  windows_per_market: int, device: torch.device) -> float:
    """Per-regime scalar for the chosen metric (macro-F1, one-step MSE, or settlement log-loss)."""
    if metric == "direction_macro_f1":
        return _regime_macro_f1(wm, seqs, threshold, label_mode, tb_horizon, windows_per_market, device)
    if metric == "settlement_logloss":
        return _regime_settlement_logloss(wm, seqs, windows_per_market, device)
    return _regime_prediction_mse(wm, seqs, windows_per_market, device)


def _degradation(low_value: float, high_value: float, higher_is_better: bool) -> float:
    """Quality lost moving from the low-vol to the high-vol regime (positive = worse in high-vol).

    macro-F1 is higher-is-better so the loss is low_value - high_value; prediction MSE is
    lower-is-better so the loss is high_value - low_value. Both make a positive number mean
    "the model got worse under the regime shift," so the gap sign is comparable across metrics.
    """
    if higher_is_better:
        return low_value - high_value
    return high_value - low_value


def _format_table(rows: list[dict], gap_value: float, regime_desc: str, metric: str,
                  regime_labels: tuple = ("low_vol", "high_vol")) -> str:
    """Render the per-arm degradation table and the headline generalization gap.

    gap = baseline_degradation - treatment_degradation, where degradation is the quality lost
    crossing into the shifted regime. A positive gap means the treatment degrades less than the
    baseline, the thesis's prediction, regardless of whether the metric is higher- or lower-better.
    regime_labels names the two regime columns so the predictability axis can read reference/shifted
    instead of the volatility axis's low_vol/high_vol.
    """
    reference_label, shifted_label = regime_labels
    head = f"| arm | {reference_label}_{metric} | {shifted_label}_{metric} | degradation |\n|---|---:|---:|---:|"
    lines = [head]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['low']:.5f} | {r['high']:.5f} | {r['degradation']:+.5f} |")
    verdict = "treatment more robust (thesis supported)" if gap_value > 0 else "treatment not more robust"
    lines.append("")
    lines.append(f"metric: {metric}   regime split: {regime_desc}")
    lines.append(f"**generalization gap (baseline_degradation - treatment_degradation) = {gap_value:+.5f}** -> {verdict}")
    return "\n".join(lines)


def main() -> int:
    """Run the two-arm A/B and write the per-regime degradation table.

    Both models are built before the held-out timeline is loaded. The mamba_ssm import pulls in a
    heavy transformers/dynamo chain whose transient allocation spike must land while host RAM
    is free; loading the large timeline first then importing was OSError(Errno 12) on the 4080.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = _device_from_arg(args.device)
    base_cfg = _load_config(args.config, [])
    aggregate_only = base_cfg.Models.WorldModel.Encoder.get("AggregateOnly", False)
    include_binary_features = base_cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    include_spot_features = base_cfg.Models.WorldModel.Encoder.get("SpotFeatures", False)
    # Build both arms before the timeline load; see the docstring's OOM ordering constraint.
    arms = []
    for arm_name, regime_film_enabled, checkpoint in (
        ("baseline", False, args.baseline_checkpoint),
        ("treatment", True, args.treatment_checkpoint),
    ):
        cfg = _load_config(args.config, _arch_overrides(regime_film_enabled, args.is_mimo, args.n_layer))
        wm = load_world_model(cfg, checkpoint, device)
        assert wm.use_direction_head, f"{arm_name} checkpoint has no direction head; cannot score direction macro-F1."
        arms.append((arm_name, wm))
    # One timeline load serves both the volatility split and the per-market feature source.
    bt = build_timeline(data_dir=args.data_val, hours=args.hours_val)
    if args.vol_source == "spot":
        realized_vol = spot_realized_vol_from_timeline(bt.timeline, bt.lifecycles)
    else:
        realized_vol = realized_vol_from_timeline(bt.timeline)
    low_slugs, high_slugs, split_desc = _split_slugs(
        bt, realized_vol, args.max_markets, args.min_ticks, args.volatility_quantile
    )
    assert low_slugs and high_slugs, (
        f"volatility split is degenerate: low={len(low_slugs)} high={len(high_slugs)}. "
        f"The held-out set needs at least two qualifying markets with defined volatility "
        f"to form both regimes; point --data-val at a multi-market split."
    )
    logger.info(f"volatility split ({split_desc}): low-vol={len(low_slugs)} high-vol={len(high_slugs)} markets")
    stats = load_normalization(args.norm_path)
    low_seqs = _regime_sequences(bt, low_slugs, stats, aggregate_only, include_binary_features, include_spot_features)
    high_seqs = _regime_sequences(bt, high_slugs, stats, aggregate_only, include_binary_features, include_spot_features)
    higher_is_better = args.metric == "direction_macro_f1"
    rows = []
    degradation_by_arm: dict[str, float] = {}
    for arm_name, wm in arms:
        low_value = _regime_score(wm, low_seqs, args.metric, args.threshold, args.label_mode,
                                  args.tb_horizon, args.windows_per_market, device)
        high_value = _regime_score(wm, high_seqs, args.metric, args.threshold, args.label_mode,
                                   args.tb_horizon, args.windows_per_market, device)
        degradation = _degradation(low_value, high_value, higher_is_better)
        degradation_by_arm[arm_name] = degradation
        rows.append({"arm": arm_name, "low": low_value, "high": high_value, "degradation": degradation})
        logger.info(
            f"{arm_name} [{args.metric}]: low-vol={low_value:.5f} high-vol={high_value:.5f} "
            f"degradation={degradation:+.5f}"
        )
    gap_value = generalization_gap(degradation_by_arm["baseline"], degradation_by_arm["treatment"])
    table = _format_table(rows, gap_value, split_desc, args.metric)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")
    logger.info(f"wrote {args.out}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
