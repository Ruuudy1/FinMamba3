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
from finmamba3.envs.fi2010_loader import load_fi2010_split
from finmamba3.envs.lob_features import apply_normalization, load_normalization
from finmamba3.eval.compare_direction import load_world_model
from finmamba3.eval.eval_regime_generalization import _arch_overrides, _device_from_arg, _load_config
from finmamba3.models.regime_modulation import (
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
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--num-windows", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = _device_from_arg(args.device)
    cfg = _load_config(args.config, _arch_overrides(True, False, None))
    wm = load_world_model(cfg, args.checkpoint, device)
    assert wm.use_regime_film, "checkpoint has no regime-FiLM router to diagnose."
    bundle = load_fi2010_split(args.data_val, split="validation", horizon=args.horizon, max_events=args.max_events)
    stats = load_normalization(args.norm_path)
    flat = apply_normalization(bundle.sequence, stats).to_flat()
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
        # Feed the same obs-vol conditioning feature the checkpoint trained on so the router state the
        # diagnostic reads matches training; a non-FeedObsVol checkpoint passes None and uses its proxy.
        regime_vol = None
        if wm.regime_film_feed_obs_vol:
            regime_vol = realized_vol_conditioning_feature(obs[..., wm.midprice_index], wm.regime_film_vol_window)
        _, regime_aux = wm.sequence_model(flattened_sample, action, return_regime=True, regime_vol=regime_vol)
    regime_logits = regime_aux.regime_logits.float()
    mid = obs[:, :, wm.midprice_index]
    vol_labels = realized_vol_bucket_labels(mid, wm.regime_film_vol_window, wm.regime_film_num_regimes)
    batch_mean_entropy = float(regime_assignment_entropy(regime_logits))
    per_sample_entropy = float(regime_per_sample_entropy(regime_logits))
    predicted = regime_logits.argmax(dim=-1)
    agreement = float((predicted == vol_labels).float().mean())
    max_entropy = float(np.log(wm.regime_film_num_regimes))
    chance = 1.0 / wm.regime_film_num_regimes
    logger.info(
        f"router diagnostic ({len(starts)} windows of {length}): "
        f"reg_H(batch-mean)={batch_mean_entropy:.4f}/{max_entropy:.4f}  "
        f"per_sample_H={per_sample_entropy:.4f}/{max_entropy:.4f}  "
        f"argmax-vs-vol-bucket agreement={agreement:.4f} (chance={chance:.4f})"
    )
    return 0
if __name__ == "__main__":
    sys.exit(main())
