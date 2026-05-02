"""Diagnose a trained world-model checkpoint for temporal prior collapse.

Loads a `world_model.pth` checkpoint, regenerates a 32-step autoregressive
imagine rollout, computes the categorical posterior/prior entropy on a
validation batch, and reports the top per-feature val MSE so the dominant
loss contributors are visible at a glance.

Use after a Phase A run that ended with `Imagine/mid_norm_std = 0` or with
an unexpectedly large `val_loss`. The script does not retrain; it inspects.

Example
-------
    python -m eval.diagnose_run \\
        --checkpoint saved_models/lob/LOB/1iemugot/ckpt/world_model.pth \\
        --config src/config_files/configure_lob.yaml \\
        --data-val data/validation \\
        --norm-path saved_models/lob/normalization.json \\
        --out-dir notes/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose a world-model checkpoint")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to world_model.pth")
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config used to build the model")
    p.add_argument("--data-val", required=True, type=Path,
                   help="Validation data directory")
    p.add_argument("--norm-path", type=Path,
                   default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    p.add_argument("--out-dir", type=Path, default=Path("notes"))
    p.add_argument("--market-slug", default=None)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--entropy-batch", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device_from_arg(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_world_model(args: argparse.Namespace, device: torch.device):
    """Build a WorldModel from config and populate weights from checkpoint."""
    import yaml
    from config_utils import DotDict, parse_args_and_update_config
    from sub_models.world_models import WorldModel

    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)

    # Action_dim is read by WorldModel from the agent action space; for diagnosis
    # we use a small default since action input is gated by Phase A defaults anyway.
    action_dim = 13
    wm = WorldModel(action_dim=action_dim, config=cfg, device=device).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = state.get("world_model", state.get("state_dict", state))
    missing, unexpected = wm.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"missing keys when loading checkpoint: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        logger.warning(f"unexpected keys when loading checkpoint: {len(unexpected)} (first 3: {unexpected[:3]})")
    wm.eval()
    return wm, cfg


def _load_val_sequence(args: argparse.Namespace, cfg):
    """Build the validation LOBSequence using the same path train_lob takes."""
    from envs.lob_features import (
        apply_normalization, extract_features, load_normalization, make_aggregate_only,
        pick_longest_market,
    )
    from lob.backtester import build_timeline

    bt = build_timeline(data_dir=args.data_val, hours=args.hours_val)
    slug = args.market_slug or pick_longest_market(bt)
    seq = extract_features(bt.timeline, slug)
    stats = load_normalization(args.norm_path)
    seq_norm = apply_normalization(seq, stats)
    if getattr(cfg.Models.WorldModel.Encoder, "AggregateOnly", False):
        seq_norm = make_aggregate_only(seq_norm)
    return seq_norm, stats, slug


def _imagine_rollout(wm, val_seq, context_len: int, horizon: int) -> np.ndarray:
    """Replicates train_lob._imagine_and_log decoding without wandb side-effects."""
    device = wm.device
    flat = val_seq.to_flat()
    T = flat.shape[0]
    if T < context_len + 1:
        raise RuntimeError(f"val sequence has only {T} ticks; need >= {context_len + 1}")
    ctx = flat[:context_len]
    obs = torch.from_numpy(ctx).float().to(device).unsqueeze(0)
    action = torch.zeros((1, context_len), dtype=torch.float32, device=device)
    decoded = []
    with torch.no_grad():
        ctx_latent = wm.encode_obs(obs)
        prefix_latent = ctx_latent
        prefix_action = action
        for step in range(horizon):
            if wm.model == "Transformer":
                from sub_models.attention_blocks import get_subsequent_mask_with_batch_length
                mask = get_subsequent_mask_with_batch_length(prefix_latent.shape[1], prefix_latent.device)
                feat = wm.sequence_model(prefix_latent, prefix_action, mask)
            else:
                feat = wm.sequence_model(prefix_latent, prefix_action)
            feat = wm.condition_dist_feat(feat[:, -1:])
            prior_logits = wm.dist_head.forward_prior(feat)
            prior_sample = wm.stright_throught_gradient(prior_logits)
            prior_flat = wm.flatten_sample(prior_sample)
            decoded.append(wm.obs_decoder(prior_flat).cpu().numpy()[0, 0])
            if step != horizon - 1:
                prefix_latent = torch.cat([prefix_latent, prior_flat], dim=1)
                prefix_action = torch.cat(
                    [prefix_action, torch.zeros((1, 1), dtype=torch.float32, device=device)], dim=1
                )
    return np.stack(decoded, axis=0)


def _categorical_entropy_stats(wm, val_seq, batch_size: int) -> dict[str, float]:
    """Return mean and std of per-step entropy for posterior and prior over a val batch."""
    device = wm.device
    flat = val_seq.to_flat()
    T = flat.shape[0]
    L = 64
    if T < L + 1:
        L = max(2, T - 1)
    rng = np.random.default_rng(0)
    starts = rng.integers(0, max(1, T - L), size=min(batch_size, max(1, T - L)))
    windows = np.stack([flat[s : s + L] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], L), dtype=torch.float32, device=device)
    with torch.no_grad():
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.stright_throught_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        if wm.model == "Transformer":
            from sub_models.attention_blocks import get_subsequent_mask_with_batch_length
            mask = get_subsequent_mask_with_batch_length(L, flattened_sample.device)
            dist_feat = wm.sequence_model(flattened_sample, action, mask)
        else:
            dist_feat = wm.sequence_model(flattened_sample, action)
        dist_feat = wm.condition_dist_feat(dist_feat)
        prior_logits = wm.dist_head.forward_prior(dist_feat)
    # Entropy is computed per categorical group (CategoricalDim groups of ClassDim classes).
    # Higher entropy means the latent is closer to uniform; full collapse is log(ClassDim).
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)
    post_h = _entropy(post_logits)
    prior_h = _entropy(prior_logits)
    log_classes = math.log(post_logits.shape[-1])
    return {
        "post_entropy_mean": float(post_h.mean().item()),
        "post_entropy_std": float(post_h.std().item()),
        "prior_entropy_mean": float(prior_h.mean().item()),
        "prior_entropy_std": float(prior_h.std().item()),
        "uniform_entropy": float(log_classes),
    }


def _per_feature_val_mse(wm, val_seq, batch_size: int = 64, batch_length: int = 64) -> list[tuple[str, float]]:
    """Top-by-MSE per-feature breakdown using the post-prior next-step prediction."""
    from envs.lob_features import FLAT_FEATURE_NAMES
    device = wm.device
    flat = val_seq.to_flat()
    T = flat.shape[0]
    if T < batch_length + 1:
        return []
    rng = np.random.default_rng(0)
    starts = rng.integers(0, T - batch_length, size=batch_size)
    windows = np.stack([flat[s : s + batch_length] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((batch_size, batch_length), dtype=torch.float32, device=device)
    with torch.no_grad():
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.stright_throught_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        if wm.model == "Transformer":
            from sub_models.attention_blocks import get_subsequent_mask_with_batch_length
            mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = wm.sequence_model(flattened_sample, action, mask)
        else:
            dist_feat = wm.sequence_model(flattened_sample, action)
        prior_logits = wm.dist_head.forward_prior(dist_feat[:, :-1])
        prior_sample = wm.stright_throught_gradient(prior_logits, sample_mode="probs")
        prior_flat = wm.flatten_sample(prior_sample)
        next_hat = wm.obs_decoder(prior_flat)
    target_next = obs[:, 1:].detach().float()
    pred_next = next_hat.detach().float()
    diff = pred_next - target_next
    feature_mse = (diff ** 2).mean(dim=(0, 1)).cpu().numpy()
    return sorted(
        [(FLAT_FEATURE_NAMES[i], float(v)) for i, v in enumerate(feature_mse)],
        key=lambda x: -x[1],
    )


def _maybe_plot(decoded: np.ndarray, out_path: Path) -> None:
    """Draw a 3-panel figure of mid, spread, imbalance over the rollout horizon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot.")
        return
    LEVEL_FLAT = 10 * 8
    mid = decoded[:, LEVEL_FLAT + 0]
    spread = decoded[:, LEVEL_FLAT + 1]
    imb = decoded[:, LEVEL_FLAT + 3]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(mid, marker="o", linewidth=1)
    axes[0].set_ylabel("mid_norm")
    axes[1].plot(spread, marker="o", linewidth=1, color="C1")
    axes[1].set_ylabel("spread_norm")
    axes[2].plot(imb, marker="o", linewidth=1, color="C2")
    axes[2].set_ylabel("imbalance_norm")
    axes[2].set_xlabel("rollout step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    logger.info(f"device: {device}")

    wm, cfg = _load_world_model(args, device)
    val_seq, _stats, slug = _load_val_sequence(args, cfg)
    logger.info(f"val market: {slug}, {val_seq.per_tick.shape[0]} ticks")

    decoded = _imagine_rollout(wm, val_seq, args.context_len, args.horizon)
    LEVEL_FLAT = 10 * 8
    mid = decoded[:, LEVEL_FLAT + 0]
    spread = decoded[:, LEVEL_FLAT + 1]
    imb = decoded[:, LEVEL_FLAT + 3]
    rollout_summary = {
        "horizon": int(args.horizon),
        "mid_norm_mean": float(mid.mean()),
        "mid_norm_std": float(mid.std()),
        "spread_norm_mean": float(spread.mean()),
        "spread_norm_std": float(spread.std()),
        "imbalance_norm_mean": float(imb.mean()),
        "imbalance_norm_std": float(imb.std()),
    }
    logger.info(f"rollout summary: {rollout_summary}")

    entropy_stats = _categorical_entropy_stats(wm, val_seq, args.entropy_batch)
    logger.info(f"entropy stats (uniform={entropy_stats['uniform_entropy']:.4f}):")
    logger.info(f"  posterior mean={entropy_stats['post_entropy_mean']:.4f} "
                f"std={entropy_stats['post_entropy_std']:.4f}")
    logger.info(f"  prior     mean={entropy_stats['prior_entropy_mean']:.4f} "
                f"std={entropy_stats['prior_entropy_std']:.4f}")

    feature_mse = _per_feature_val_mse(wm, val_seq)
    logger.info("top 10 per-feature val MSE:")
    for name, val in feature_mse[:10]:
        logger.info(f"  {name:32s} {val:>10.4f}")

    rollout_path = args.out_dir / f"diagnose_rollout_{slug}.npy"
    np.save(rollout_path, decoded)
    plot_path = args.out_dir / f"diagnose_rollout_{slug}.png"
    _maybe_plot(decoded, plot_path)
    summary = {
        "checkpoint": str(args.checkpoint),
        "slug": slug,
        "rollout": rollout_summary,
        "entropy": entropy_stats,
        "top_feature_mse": [{"feature": n, "mse": v} for n, v in feature_mse[:20]],
    }
    summary_path = args.out_dir / f"diagnose_summary_{slug}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"wrote {rollout_path}, {plot_path}, {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
