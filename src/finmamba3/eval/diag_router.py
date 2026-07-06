"""Router-discrimination diagnostic for the regime-FiLM A/B.

The batch-mean router entropy (reg_H) reported during training stays at ln(R) whether the router is
genuinely vol-discriminative or degenerately uniform-per-step, because balanced bucket occupancy keeps
the batch mean flat either way. This script reads the two quantities that actually separate those cases
on held-out windows: the per-step router entropy (drops toward zero when each step commits to one
regime) and the agreement between the router's argmax regime and the realized-volatility bucket the
regime-supervision target uses. A supervised router that learned the vol axis shows low per-sample
entropy and high agreement; a uniform-collapsed router shows per-sample entropy near ln(R) and
chance-level agreement.

Example
-------
    python -m finmamba3.eval.diag_router --config configs/fi2010_regsup_studentt.yaml \\
        --checkpoint saved_models/lob/LOB/<id>/ckpt/world_model_final.pth \\
        --data-val data/fi2010/validation --norm-path saved_models/lob/fi2010_norm_real.json
"""
# region imports
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from finmamba3.eval.compare_direction import load_world_model
from finmamba3.eval.eval_regime_generalization import _arch_overrides, _device_from_arg, _load_config
from finmamba3.eval.eval_regime_generalization_fi2010 import _load_val_sequence
from finmamba3.models.regime_modulation import (
    efficiency_ratio_bucket_labels,
    efficiency_ratio_conditioning_feature,
    realized_vol_bucket_labels,
    realized_vol_conditioning_feature,
    regime_assignment_entropy,
    regime_per_sample_entropy,
)
# endregion
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose whether the regime-FiLM router discriminates volatility")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path, help="A RegimeFiLM-on checkpoint.")
    p.add_argument("--dataset", choices=("fi2010", "kaggle"), default=None,
                   help="Which loader builds the val stream; default reads Dataset.Kind from the config.")
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--supervise-axis", choices=("vol", "predictability"), default=None,
                   help="Rebuild the router on this axis to match the checkpoint's training (the config "
                        "default is not the CLI override the escalation trained with).")
    p.add_argument("--feed-obs-vol", action="store_true",
                   help="Rebuild with FeedObsVol on so the diagnostic feeds the router the same obs-derived "
                        "conditioning feature it trained on.")
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--num-windows", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = _device_from_arg(args.device)
    regime_overrides = []
    if args.supervise_axis is not None:
        regime_overrides += ["--Models.WorldModel.RegimeFiLM.SuperviseAxis", args.supervise_axis]
    if args.feed_obs_vol:
        regime_overrides += ["--Models.WorldModel.RegimeFiLM.FeedObsVol", "true"]
    cfg = _load_config(args.config, _arch_overrides(True, False, None) + regime_overrides)
    wm = load_world_model(cfg, args.checkpoint, device)
    assert wm.use_regime_film, "checkpoint has no regime-FiLM router to diagnose."
    # Read the supervised axis off the checkpoint so agreement is scored against the target it trained on.
    axis = wm.regime_film_supervise_axis
    bucket_labels_fn = efficiency_ratio_bucket_labels if axis == "predictability" else realized_vol_bucket_labels
    conditioning_fn = efficiency_ratio_conditioning_feature if axis == "predictability" else realized_vol_conditioning_feature
    dataset_kind = args.dataset or cfg.Dataset.get("Kind", "fi2010")
    flat = _load_val_sequence(dataset_kind, cfg, args).to_flat()
    total = flat.shape[0]
    length = args.window_len
    rng = np.random.default_rng(0)
    starts = rng.integers(0, total - length, size=min(args.num_windows, total - length))
    windows = np.stack([flat[s : s + length] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], length), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.straight_through_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        # Feed the obs-vol feature the checkpoint trained on; non-FeedObsVol checkpoints pass None and use their proxy.
        regime_vol = None
        if wm.regime_film_feed_obs_vol:
            regime_vol = conditioning_fn(obs[..., wm.midprice_index], wm.regime_film_vol_window)
        _, regime_aux = wm.sequence_model(flattened_sample, action, return_regime=True, regime_vol=regime_vol)
    regime_logits = regime_aux.regime_logits.float()
    mid = obs[:, :, wm.midprice_index]
    regime_labels = bucket_labels_fn(mid, wm.regime_film_vol_window, wm.regime_film_num_regimes)
    batch_mean_entropy = float(regime_assignment_entropy(regime_logits))
    per_sample_entropy = float(regime_per_sample_entropy(regime_logits))
    predicted = regime_logits.argmax(dim=-1)
    agreement = float((predicted == regime_labels).float().mean())
    max_entropy = float(np.log(wm.regime_film_num_regimes))
    chance = 1.0 / wm.regime_film_num_regimes
    logger.info(
        f"router diagnostic [{dataset_kind}, {axis} bucket] ({len(starts)} windows of {length}): "
        f"reg_H(batch-mean)={batch_mean_entropy:.4f}/{max_entropy:.4f}  "
        f"per_sample_H={per_sample_entropy:.4f}/{max_entropy:.4f}  "
        f"argmax-vs-{axis}-bucket agreement={agreement:.4f} (chance={chance:.4f})"
    )
    return 0
if __name__ == "__main__":
    sys.exit(main())
