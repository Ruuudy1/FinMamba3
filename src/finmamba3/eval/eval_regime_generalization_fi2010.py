"""FI-2010 cross-regime generalization-gap evaluator for the regime-FiLM A/B.

This is the FI-2010 sibling of eval_regime_generalization.py. The Polymarket gap came
back null, but Polymarket tick-direction is near-degenerate, so a null there cannot tell
"the mechanism does not help" apart from "the task had no signal." FI-2010 carries real
short-horizon LOB structure (LOBCAST reports 0.5-0.8 macro-F1), so it is the fair test of
the same thesis on data with signal. To keep the comparison honest the only thing that
changes across metrics is the scored quantity: the split definition stays a realized-volatility
median split over fixed-length windows of the one concatenated FI-2010 stream.

The --metric flag selects the regime-dependent quantity the gap is measured on:

  - prediction_mse:      one-step decoder MSE under a Gaussian decoder (the prior null's metric;
                         regime-*invariant* first moment, kept for reproduction).
  - studentt_nll:        one-step Student-t NLL on the price/scale channels (the regime's 2nd
                         moment; FiLM can express "this regime is more volatile" via the scale).
  - volume_nll:          one-step Student-t NLL on the size/volume channels (order-flow magnitude,
                         intrinsically regime-dependent; shares the studentt checkpoints).
  - direction_macro_f1:  3-class direction macro-F1 (LOBCAST-comparable; partly regime-invariant).

For every metric, per arm:

    degradation = (worse metric on high_vol) - (metric on low_vol)   normalized so positive = worse
    gap         = degradation(baseline, FiLM off) - degradation(treatment, FiLM on)

A positive gap means the treatment degrades less under the volatility shift, supporting the thesis.
The shared _degradation / generalization_gap / _format_table helpers handle both higher- and
lower-is-better metrics so the gap sign is comparable across all four.

Example
-------
    python -m finmamba3.eval.eval_regime_generalization_fi2010 \\
        --config configs/fi2010_studentt.yaml --metric studentt_nll \\
        --baseline-checkpoint saved_models/lob/LOB/<base>/ckpt/world_model_final.pth \\
        --treatment-checkpoint saved_models/lob/LOB/<treat>/ckpt/world_model_final.pth \\
        --data-val data/fi2010/validation --norm-path saved_models/lob/fi2010_norm_real.json \\
        --window-len 512 --out reports/volscale_gap_s0.md
"""
# region imports
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from finmamba3.envs.fi2010_loader import load_fi2010_split
from finmamba3.envs.lob_features import LOBSequence, apply_normalization, load_normalization
from finmamba3.eval.compare_direction import (
    classification_metrics,
    load_world_model,
    world_model_direction_probs,
    world_model_prediction_nll,
)
from finmamba3.eval.eval_regime_generalization import (
    _arch_overrides,
    _degradation,
    _format_table,
    _load_config,
    _device_from_arg,
    _regime_prediction_mse,
    generalization_gap,
)
from finmamba3.eval.predictability import window_predictability_split
from finmamba3.eval.regime_split import window_volatility_split
from finmamba3.sequence_builder import build_kaggle_sequences
# endregion
logger = logging.getLogger(__name__)
# A vol window shorter than the 64-event scoring windows yields nothing scorable; fail fast, not silently.
_MIN_WINDOW_LEN = 65
# One table of per-metric properties (Student-t need, degradation sign, NLL mask) keeps a new metric a one-line addition.
_METRIC_SPEC = {
    "prediction_mse": {"needs_studentt": False, "higher_is_better": False, "mask": None},
    "studentt_nll": {"needs_studentt": True, "higher_is_better": False, "mask": "price_scale"},
    "volume_nll": {"needs_studentt": True, "higher_is_better": False, "mask": "volume"},
    # recon_nll is the schema-agnostic G1 diagnostic; price_scale/volume are FI-2010 refinements kept for reproduction.
    "recon_nll": {"needs_studentt": True, "higher_is_better": False, "mask": "all"},
    "direction_macro_f1": {"needs_studentt": False, "higher_is_better": True, "mask": None},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure the regime-FiLM generalization gap on FI-2010")
    p.add_argument("--config", required=True, type=Path, help="FI-2010 config (e.g. configs/fi2010_studentt.yaml).")
    p.add_argument("--metric", choices=tuple(_METRIC_SPEC), default="prediction_mse",
                   help="Regime-dependent quantity the gap is measured on; see the module docstring.")
    p.add_argument("--dataset", choices=("fi2010", "kaggle"), default=None,
                   help="Which loader builds the val stream; default reads Dataset.Kind from the config.")
    p.add_argument("--regime-axis", choices=("predictability", "spot_vol"), default="predictability",
                   help="predictability = Efficiency-Ratio window split (G1 primary axis); "
                        "spot_vol = realized-vol window split (secondary axis).")
    p.add_argument("--baseline-checkpoint", required=True, type=Path,
                   help="world_model_*.pth from the RegimeFiLM-off FI-2010 run")
    p.add_argument("--treatment-checkpoint", required=True, type=Path,
                   help="world_model_*.pth from the RegimeFiLM-on FI-2010 run")
    p.add_argument("--data-val", required=True, type=Path, help="Directory holding the FI-2010 validation split text file.")
    p.add_argument("--norm-path", type=Path,
                   default=Path("saved_models") / "lob" / "fi2010_norm.json",
                   help="Training-time FI-2010 normalization stats so eval uses the train scale.")
    p.add_argument("--horizon", type=int, default=10, help="FI-2010 label horizon to load; one of 10, 20, 30, 50, 100.")
    p.add_argument("--max-events", type=int, default=None,
                   help="Cap events for a fast smoke run; default uses the full split.")
    p.add_argument("--window-len", type=int, default=512,
                   help="Events per vol window; the segment unit split at the vol median.")
    p.add_argument("--volatility-quantile", type=float, default=0.5,
                   help="Windows above this realized-vol quantile form the high-vol regime.")
    p.add_argument("--windows-per-segment", type=int, default=256,
                   help="Random 64-event windows scored per segment; higher trims the per-regime noise.")
    p.add_argument("--label-mode", choices=("next_tick", "triple_barrier"), default="next_tick",
                   help="Direction labelling for the direction_macro_f1 metric (ignored by the others).")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Next-tick or triple-barrier threshold for the direction_macro_f1 metric.")
    p.add_argument("--tb-horizon", type=int, default=32, help="Forward horizon in ticks for triple-barrier labelling.")
    p.add_argument("--is-mimo", action="store_true",
                   help="Rebuild both arms as MIMO to match an H100 MIMO A/B; default SISO.")
    p.add_argument("--n-layer", type=int, default=None,
                   help="Override Mamba3.n_layer to match a non-default-depth training run.")
    p.add_argument("--out", type=Path, default=Path("reports/regime_generalization_fi2010.md"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _feature_indices(k_levels: int, f_level: int, f_tick: int, mask_name: str) -> list[int]:
    """Flat-vector channel indices for the price/scale or volume regime, derived from the geometry.

    FI-2010's flat layout is K levels of (ask_price, ask_size, bid_price, bid_size) followed by the
    six tick aggregates (mid, spread, log_spread, imbalance, microprice, log_total_vol). The
    price/scale channels carry the regime's second moment (price-level dispersion); the volume
    channels carry the order-flow magnitude. Restricting the Student-t NLL to one or the other lets a
    single checkpoint answer the volatility question and the order-flow question separately. The
    per-level offsets come from f_level and the tick offsets from the fixed FI-2010 tick ordering, so
    a re-binned schema stays correct without hardcoding 46.
    """
    level_width = k_levels * f_level
    if mask_name == "all":
        # The full feature vector: the schema-agnostic reconstruction NLL over every channel.
        return list(range(level_width + f_tick))
    price_levels = [k * f_level + 0 for k in range(k_levels)] + [k * f_level + 2 for k in range(k_levels)]
    size_levels = [k * f_level + 1 for k in range(k_levels)] + [k * f_level + 3 for k in range(k_levels)]
    # Tick semantic offsets fixed by the FI-2010 schema: mid=0, microprice=4, log_total_vol=5.
    if mask_name == "price_scale":
        return price_levels + [level_width + 0, level_width + 4]
    assert mask_name == "volume", f"unknown feature mask {mask_name!r}; expected price_scale or volume."
    return size_levels + [level_width + 5]


def _slice_sequence(seq: LOBSequence, start: int, end: int) -> LOBSequence:
    """One sub-sequence covering the [start, end) slice of the concatenated stream.

    LOBSequence is a plain row-aligned dataclass, so slicing every field by the same range
    yields a valid shorter sequence the scoring path consumes exactly like a per-market one.
    """
    return LOBSequence(
        market_slug=f"{seq.market_slug}_w{start}",
        per_level=seq.per_level[start:end],
        per_tick=seq.per_tick[start:end],
        midprice=seq.midprice[start:end],
        ts_sec=seq.ts_sec[start:end],
        yes_outcome=None if seq.yes_outcome is None else seq.yes_outcome[start:end],
    )


def _regime_segments(seq: LOBSequence, windows: list[tuple[int, int]]) -> list[LOBSequence]:
    """Slice the normalized stream into one sub-sequence per window in the regime."""
    return [_slice_sequence(seq, start, end) for start, end in windows]


def _regime_prediction_nll(wm, seqs: list, feature_indices: torch.Tensor,
                           windows_per_segment: int, device: torch.device) -> float:
    """Pooled one-step Student-t NLL over a regime's window segments (lower is better).

    Summing NLL and element counts before dividing gives the exact element-weighted mean NLL over
    the regime, so longer segments are not over- or under-counted.
    """
    total_nll = 0.0
    total_count = 0
    for seq in seqs:
        sum_nll, count = world_model_prediction_nll(
            wm, seq, device, feature_indices, windows_per_market=windows_per_segment
        )
        total_nll += sum_nll
        total_count += count
    assert total_count > 0, "no windows scored for this regime; segments are shorter than the window length."
    return total_nll / total_count


def _regime_macro_f1(wm, seqs: list, threshold: float, label_mode: str, tb_horizon: int,
                     level_width: int, windows_per_segment: int, device: torch.device) -> float:
    """Pool direction-head predictions across a regime's segments, then score one macro-F1.

    Pooling before classification_metrics keeps boundary-respecting per-segment windows while still
    yielding a single macro-F1 for the regime, matching the Polymarket sibling's path.
    """
    probs_parts = []
    labels_parts = []
    for seq in seqs:
        probs, labels = world_model_direction_probs(
            wm, seq, threshold, label_mode, tb_horizon, level_width, device,
            windows_per_market=windows_per_segment,
        )
        probs_parts.append(probs)
        labels_parts.append(labels)
    pooled_probs = np.concatenate(probs_parts, axis=0)
    pooled_labels = np.concatenate(labels_parts, axis=0)
    return classification_metrics(pooled_probs, pooled_labels)["macro_f1"]


def _build_arms(config_path: Path, metric: str, baseline_ckpt: Path, treatment_ckpt: Path,
                is_mimo: bool, n_layer: int | None, device: torch.device) -> list[tuple[str, object]]:
    """Rebuild the baseline (FiLM off) and treatment (FiLM on) world models from their checkpoints.

    Both arms share the FI-2010 config and architecture flags; only the RegimeFiLM flag differs,
    which is the definition of the A/B. load_world_model fails fast if a rebuilt architecture does
    not match its checkpoint keys. The metric pins the decoder/head precondition: the Student-t
    metrics require a studentt decoder, prediction_mse forbids it, and direction_macro_f1 needs the
    direction head. direction_macro_f1 reads only the direction head, so it is decoder-agnostic and
    skips the decoder checks.
    """
    spec = _METRIC_SPEC[metric]
    arms = []
    for arm_name, regime_film_enabled, checkpoint in (
        ("baseline", False, baseline_ckpt),
        ("treatment", True, treatment_ckpt),
    ):
        cfg = _load_config(config_path, _arch_overrides(regime_film_enabled, is_mimo, n_layer))
        wm = load_world_model(cfg, checkpoint, device)
        if spec["needs_studentt"]:
            assert wm.decoder_kind == "studentt", (
                f"{arm_name} checkpoint uses a {wm.decoder_kind} decoder; metric {metric} needs a "
                f"studentt decoder. Train with configs/fi2010_studentt.yaml."
            )
        elif metric == "prediction_mse":
            # A studentt decoder returns (mean, log_scale) and would silently mis-decode under the Gaussian path.
            assert wm.decoder_kind != "studentt", (
                f"{arm_name} checkpoint uses a studentt decoder; metric {metric} assumes a Gaussian decoder."
            )
        if metric == "direction_macro_f1":
            assert wm.use_direction_head, f"{arm_name} checkpoint has no direction head; cannot score direction macro-F1."
        arms.append((arm_name, wm))
    return arms


def _load_val_sequence(dataset_kind: str, base_cfg, args) -> LOBSequence:
    """Normalized held-out val sequence for the chosen dataset, on the train normalization scale.

    FI-2010 loads its validation text file; Kaggle reproduces the exact chronological val carve the
    trainer used (the tail after HoursTrain) so the eval feature pipeline matches training with no skew.
    """
    if dataset_kind == "fi2010":
        bundle = load_fi2010_split(args.data_val, split="validation", horizon=args.horizon, max_events=args.max_events)
        stats = load_normalization(args.norm_path)
        return apply_normalization(bundle.sequence, stats)
    if dataset_kind == "kaggle":
        kaggle_cfg = base_cfg.Dataset.Kaggle
        val_norm, _, _ = build_kaggle_sequences(
            args.data_val, asset=str(kaggle_cfg.Asset), resolution=str(kaggle_cfg.Resolution),
            split="validation", hours_train=float(kaggle_cfg.HoursTrain),
            hours_val=float(kaggle_cfg.HoursVal), norm_path=args.norm_path, fit_stats=False,
            norm_clip=float(base_cfg.BasicSettings.get("NormClip", 8.0)),
            flat_threshold=float(kaggle_cfg.get("FlatThreshold", 0.0)),
        )
        return val_norm
    raise ValueError(f"--dataset must be fi2010 or kaggle, got {dataset_kind!r}.")


def _regime_windows(axis: str, midprice, window_len: int, quantile: float):
    """(reference, shifted) window bounds and a description for the chosen regime axis.

    predictability routes high-ER (forecastable) windows to the reference regime and low-ER (random-walk)
    windows to the shifted regime; spot_vol routes low realized-vol to the reference and high-vol to the
    shifted, the prior FI-2010 axis. The evaluator computes degradation = metric(reference) - metric(shifted).
    """
    if axis == "predictability":
        split = window_predictability_split(midprice, window_len, quantile)
        return split.reference_windows, split.shifted_windows, split.description
    if axis == "spot_vol":
        split = window_volatility_split(midprice, window_len, quantile)
        return split.low_vol_windows, split.high_vol_windows, split.description
    raise ValueError(f"--regime-axis must be predictability or spot_vol, got {axis!r}.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    assert args.window_len >= _MIN_WINDOW_LEN, (
        f"--window-len must be at least {_MIN_WINDOW_LEN} so each vol window yields scorable "
        f"64-event sub-windows; got {args.window_len}."
    )
    spec = _METRIC_SPEC[args.metric]
    device = _device_from_arg(args.device)
    # Build both models first; the mamba_ssm import RAM spike must land while host memory is free.
    arms = _build_arms(args.config, args.metric, args.baseline_checkpoint, args.treatment_checkpoint,
                       args.is_mimo, args.n_layer, device)
    base_cfg = _load_config(args.config, [])
    enc_cfg = base_cfg.Models.WorldModel.Encoder
    k_levels = int(enc_cfg.K)
    f_level = int(enc_cfg.FeatureDimLevel)
    f_tick = int(enc_cfg.FeatureDimTick)
    level_width = k_levels * f_level
    feature_indices = None
    if spec["mask"] is not None:
        index_list = _feature_indices(k_levels, f_level, f_tick, spec["mask"])
        feature_indices = torch.tensor(index_list, dtype=torch.long, device=device)
    dataset_kind = args.dataset or base_cfg.Dataset.get("Kind", "fi2010")
    val_norm = _load_val_sequence(dataset_kind, base_cfg, args)
    reference_windows, shifted_windows, split_desc = _regime_windows(
        args.regime_axis, val_norm.midprice, args.window_len, args.volatility_quantile
    )
    assert reference_windows and shifted_windows, (
        f"{args.regime_axis} window split is degenerate: reference={len(reference_windows)} "
        f"shifted={len(shifted_windows)}. Lower --window-len or pass a longer val split so both "
        f"regimes are non-empty."
    )
    logger.info(
        f"{dataset_kind} {args.regime_axis} split ({split_desc}): reference={len(reference_windows)} "
        f"shifted={len(shifted_windows)} windows of {args.window_len} events"
    )
    low_seqs = _regime_segments(val_norm, reference_windows)
    high_seqs = _regime_segments(val_norm, shifted_windows)
    rows = []
    degradation_by_arm: dict[str, float] = {}
    for arm_name, wm in arms:
        low_value = _score_regime(wm, low_seqs, args, spec, feature_indices, level_width, device)
        high_value = _score_regime(wm, high_seqs, args, spec, feature_indices, level_width, device)
        degradation = _degradation(low_value, high_value, spec["higher_is_better"])
        degradation_by_arm[arm_name] = degradation
        rows.append({"arm": arm_name, "low": low_value, "high": high_value, "degradation": degradation})
        logger.info(
            f"{arm_name} [{args.metric}]: reference={low_value:.5f} shifted={high_value:.5f} "
            f"degradation={degradation:+.5f}"
        )
    gap_value = generalization_gap(degradation_by_arm["baseline"], degradation_by_arm["treatment"])
    table = _format_table(rows, gap_value, split_desc, args.metric, regime_labels=("reference", "shifted"))
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")
    logger.info(f"wrote {args.out}")
    return 0


def _score_regime(wm, seqs: list, args, spec: dict, feature_indices, level_width: int, device: torch.device) -> float:
    """Per-regime scalar for the chosen metric, dispatched off the metric spec."""
    if args.metric == "direction_macro_f1":
        return _regime_macro_f1(wm, seqs, args.threshold, args.label_mode, args.tb_horizon,
                                level_width, args.windows_per_segment, device)
    if args.metric == "prediction_mse":
        return _regime_prediction_mse(wm, seqs, args.windows_per_segment, device)
    return _regime_prediction_nll(wm, seqs, feature_indices, args.windows_per_segment, device)
if __name__ == "__main__":
    sys.exit(main())
