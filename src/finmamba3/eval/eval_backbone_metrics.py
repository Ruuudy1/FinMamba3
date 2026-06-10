"""Single-model held-out metrics for the backbone ablation (G2): recon NLL + direction macro-F1.

Unlike the A/B evaluator this scores ONE checkpoint (no regime split, no baseline/treatment arms), so
it reports the two quantities the matched-parameter backbone comparison needs: full-channel Student-t
reconstruction NLL on the prior-predicted next tick (lower is better) and 3-class next-tick direction
macro-F1 (higher is better). It reuses the compare_direction scoring path so the numbers are
methodologically identical to the A/B, and the world-model param count is printed so the matched-budget
claim is empirical. MIMO is reached with --is-mimo (A100-only; the 4080 segfaults at build).

Example
-------
    python -m finmamba3.eval.eval_backbone_metrics --config configs/fi2010_studentt.yaml \\
        --dataset fi2010 --backbone Mamba2 --checkpoint <final.pth> \\
        --data-val data/fi2010/validation --norm-path saved_models/lob/fi2010_studentt_norm_s0.json
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
from finmamba3.envs.lob_features import apply_normalization, load_normalization
from finmamba3.eval.compare_direction import (
    classification_metrics,
    load_world_model,
    world_model_direction_probs,
    world_model_prediction_nll,
)
from finmamba3.eval.eval_regime_generalization import _device_from_arg, _load_config
from finmamba3.sequence_builder import build_kaggle_sequences
# endregion
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-model recon NLL + direction macro-F1 for the backbone ablation")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", choices=("fi2010", "kaggle"), required=True)
    p.add_argument("--backbone", default=None,
                   help="Override Models.WorldModel.Backbone (Mamba, Mamba2, Mamba3, Transformer).")
    p.add_argument("--checkpoint", required=True, type=Path, help="world_model_final.pth from the run.")
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path, help="Training normalization stats (no train/serve skew).")
    p.add_argument("--horizon", type=int, default=10, help="FI-2010 label horizon (ignored for Kaggle).")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.0, help="Next-tick direction flat-bucket threshold on normalized mid.")
    p.add_argument("--windows", type=int, default=512, help="Random windows scored over the val stream.")
    p.add_argument("--is-mimo", action="store_true", help="Rebuild as MIMO (A100-only; the 4080 segfaults at build).")
    p.add_argument("--out", type=Path, default=Path("reports/backbone_metrics.md"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _load_val_sequence(dataset: str, base_cfg, args):
    """Normalized held-out val sequence for the chosen dataset, on the train normalization scale."""
    if dataset == "fi2010":
        bundle = load_fi2010_split(args.data_val, split="validation", horizon=args.horizon, max_events=args.max_events)
        stats = load_normalization(args.norm_path)
        return apply_normalization(bundle.sequence, stats)
    kaggle_cfg = base_cfg.Dataset.Kaggle
    val_norm, _, _ = build_kaggle_sequences(
        args.data_val, asset=str(kaggle_cfg.Asset), resolution=str(kaggle_cfg.Resolution),
        split="validation", hours_train=float(kaggle_cfg.HoursTrain), hours_val=float(kaggle_cfg.HoursVal),
        norm_path=args.norm_path, fit_stats=False, norm_clip=float(base_cfg.BasicSettings.get("NormClip", 8.0)),
        flat_threshold=float(kaggle_cfg.get("FlatThreshold", 0.0)),
    )
    return val_norm


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = _device_from_arg(args.device)
    overrides = ["--Models.WorldModel.Mamba3.is_mimo", "true" if args.is_mimo else "false"]
    if args.backbone is not None:
        overrides += ["--Models.WorldModel.Backbone", args.backbone]
    cfg = _load_config(args.config, overrides)
    enc_cfg = cfg.Models.WorldModel.Encoder
    level_width = int(enc_cfg.K) * int(enc_cfg.FeatureDimLevel)
    feature_dim = level_width + int(enc_cfg.FeatureDimTick)
    wm = load_world_model(cfg, args.checkpoint, device)
    assert wm.decoder_kind == "studentt", (
        f"checkpoint uses a {wm.decoder_kind} decoder; recon NLL needs a Student-t decoder."
    )
    assert wm.use_direction_head, "checkpoint has no direction head; cannot score direction macro-F1."
    param_count = sum(parameter.numel() for parameter in wm.parameters())
    val_norm = _load_val_sequence(args.dataset, cfg, args)
    feature_indices = torch.arange(feature_dim, dtype=torch.long, device=device)
    sum_nll, count = world_model_prediction_nll(wm, val_norm, device, feature_indices, windows_per_market=args.windows)
    recon_nll = sum_nll / count
    probs, labels = world_model_direction_probs(
        wm, val_norm, args.threshold, "next_tick", 32, level_width, device, windows_per_market=args.windows
    )
    metrics = classification_metrics(probs, labels)
    backbone = args.backbone or str(cfg.Models.WorldModel.Backbone)
    mimo_tag = " (MIMO)" if args.is_mimo else ""
    table = (
        f"| backbone | params | recon_nll | direction_macro_f1 | accuracy |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| {backbone}{mimo_tag} | {param_count:,} | {recon_nll:.5f} | {metrics['macro_f1']:.5f} | {metrics['accuracy']:.5f} |"
    )
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")
    logger.info(
        f"{args.dataset} {backbone}{mimo_tag}: params={param_count:,} recon_nll={recon_nll:.5f} "
        f"macro_f1={metrics['macro_f1']:.5f} acc={metrics['accuracy']:.5f} -> wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
