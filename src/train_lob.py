"""Offline world-model pretraining on Polymarket LOB features.

Trains the LOB encoder / Mamba / MLP decoder stack to reconstruct and
predict LOB feature sequences for a single market. No agent, no live env,
no reward signal. Produces a world-model checkpoint suitable for warm-
starting a downstream RL loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

from envs.lob_features import (
    FLAT_FEATURE_NAMES,
    FEATURE_DIM_FLAT,
    F_LEVEL,
    K_LEVELS,
    LOBSequence,
    apply_normalization,
    denormalize_flat,
    extract_features,
    fit_normalization,
    load_normalization,
    make_aggregate_only,
    normalized_feature_diagnostics,
    pick_longest_market,
    save_normalization,
)
from config_utils import DotDict, parse_args_and_update_config
from lob.backtester import build_timeline
from replay_buffer import ReplayBuffer
from training_steps import train_world_model_step

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parent


def _populate_buffer(buffer: ReplayBuffer, seq: LOBSequence) -> None:
    flat = seq.to_flat()
    T = flat.shape[0]
    for t in range(T):
        buffer.append(
            obs=flat[t],
            action=0,
            reward=0.0,
            termination=0.0,
        )
    logger.info(f"replay buffer: loaded {T} ticks for market {seq.market_slug}")


def _build_sequences(
    data_dir: Path,
    market_slug: str | None,
    hours: float,
    norm_path: Path,
    fit_stats: bool,
    norm_clip: float,
    aggregate_only: bool,
) -> tuple[LOBSequence, str, object]:
    bt = build_timeline(data_dir=data_dir, hours=hours)
    slug = market_slug or pick_longest_market(bt)
    settlement = bt.settlements.get(slug)
    yes_outcome = _settlement_yes_outcome(settlement)
    try:
        seq = extract_features(bt.timeline, slug, yes_outcome=yes_outcome)
    except RuntimeError:
        # Requested slug has no usable ticks in this split; fall back to the
        # longest market available in this split.
        slug = pick_longest_market(bt)
        settlement = bt.settlements.get(slug)
        yes_outcome = _settlement_yes_outcome(settlement)
        logger.warning(
            f"Market {market_slug!r} has no usable ticks in {data_dir}; "
            f"falling back to {slug!r}"
        )
        seq = extract_features(bt.timeline, slug, yes_outcome=yes_outcome)
    if fit_stats:
        stats = fit_normalization(seq, clip_value=norm_clip)
        save_normalization(stats, norm_path)
        logger.info(f"normalization fit on {slug}, saved to {norm_path}")
    else:
        stats = load_normalization(norm_path)
    seq_norm = apply_normalization(seq, stats)
    if aggregate_only:
        seq_norm = make_aggregate_only(seq_norm)
    return seq_norm, slug, stats


def _settlement_yes_outcome(settlement) -> float | None:
    if settlement is None:
        return None
    outcome = getattr(settlement.outcome, "value", settlement.outcome)
    if outcome == "YES":
        return 1.0
    if outcome == "NO":
        return 0.0
    return None


def _imagine_and_log(
    world_model: WorldModel,
    val_seq: LOBSequence,
    wlogger: WandbLogger,
    global_step: int,
    context_len: int = 16,
    horizon: int = 32,
) -> None:
    """Encode a short validation context, autoregressively roll forward,
    save decoded feature tensors for diagnostic plots.
    """
    world_model.eval()
    device = world_model.device
    T = val_seq.per_tick.shape[0]
    if T < context_len + 1:
        return
    start = 0
    ctx = val_seq.to_flat()[start : start + context_len]
    obs = torch.from_numpy(ctx).float().to(device, non_blocking=True).unsqueeze(0)
    action = torch.zeros((1, context_len), dtype=torch.float32, device=device)

    with torch.no_grad():
        ctx_latent = world_model.encode_obs(obs)
        prefix_latent = ctx_latent
        prefix_action = action
        decoded = []
        for step in range(horizon):
            if world_model.model == "Transformer":
                from sub_models.attention_blocks import get_subsequent_mask_with_batch_length

                temporal_mask = get_subsequent_mask_with_batch_length(
                    prefix_latent.shape[1], prefix_latent.device
                )
                feat = world_model.sequence_model(prefix_latent, prefix_action, temporal_mask)
            else:
                feat = world_model.sequence_model(prefix_latent, prefix_action)
            feat = world_model.condition_dist_feat(feat[:, -1:])
            prior_logits = world_model.dist_head.forward_prior(feat)
            prior_sample = world_model.stright_throught_gradient(prior_logits)
            prior_flat = world_model.flatten_sample(prior_sample)
            decoded.append(world_model.image_decoder(prior_flat).cpu().numpy()[0, 0])
            if step != horizon - 1:
                next_action = torch.zeros((1, 1), dtype=torch.float32, device=device)
                prefix_latent = torch.cat([prefix_latent, prior_flat], dim=1)
                prefix_action = torch.cat([prefix_action, next_action], dim=1)

    decoded = np.stack(decoded, axis=0)
    LEVEL_FLAT = 10 * 8
    mid_norm = decoded[:, LEVEL_FLAT + 0]
    spread_norm = decoded[:, LEVEL_FLAT + 1]
    imbalance_norm = decoded[:, LEVEL_FLAT + 3]
    wlogger.log("Imagine/mid_norm_mean", float(mid_norm.mean()), global_step=global_step)
    wlogger.log("Imagine/mid_norm_std", float(mid_norm.std()), global_step=global_step)
    wlogger.log("Imagine/spread_norm_mean", float(spread_norm.mean()), global_step=global_step)
    wlogger.log("Imagine/imbalance_norm_mean", float(imbalance_norm.mean()), global_step=global_step)

    ckpt_dir = Path(f"saved_models/lob/LOB/{wlogger.run.id}/imagine")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / f"rollout_step_{global_step}.npy", decoded)


def _validation_metrics(
    world_model: WorldModel,
    val_seq: LOBSequence,
    stats,
    batch_size: int,
    batch_length: int,
) -> tuple[dict[str, float], list[tuple[str, float]]]:
    world_model.eval()
    device = world_model.device
    flat = val_seq.to_flat()
    T = flat.shape[0]
    if T < batch_length + 1:
        return {"Val/reconstruction_loss": float("nan")}, []
    rng = np.random.default_rng(0)
    starts = rng.integers(0, T - batch_length, size=batch_size)
    windows = np.stack([flat[s : s + batch_length] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device, non_blocking=True)
    action = torch.zeros((batch_size, batch_length), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=world_model.use_amp
    ):
        embedding = world_model.encoder(obs)
        post_logits = world_model.dist_head.forward_post(embedding)
        sample = world_model.stright_throught_gradient(post_logits)
        flattened_sample = world_model.flatten_sample(sample)
        obs_hat = world_model.image_decoder(flattened_sample)
        reconstruction_loss = world_model.reconstruction_loss_func(obs_hat, obs)

        if world_model.model == "Transformer":
            from sub_models.attention_blocks import get_subsequent_mask_with_batch_length

            temporal_mask = get_subsequent_mask_with_batch_length(
                batch_length, flattened_sample.device
            )
            dist_feat = world_model.sequence_model(flattened_sample, action, temporal_mask)
        else:
            dist_feat = world_model.sequence_model(flattened_sample, action)
        prior_logits = world_model.dist_head.forward_prior(dist_feat[:, :-1])
        prior_sample = world_model.stright_throught_gradient(prior_logits, sample_mode="probs")
        prior_flat = world_model.flatten_sample(prior_sample)
        next_hat = world_model.image_decoder(prior_flat)

    target_next = obs[:, 1:].detach().float()
    pred_next = next_hat.detach().float()
    diff = pred_next - target_next
    feature_mse = (diff ** 2).mean(dim=(0, 1)).cpu().numpy()

    pred_np = pred_next.cpu().numpy()
    target_np = target_next.cpu().numpy()
    prev_np = obs[:, :-1].detach().float().cpu().numpy()
    pred_raw = denormalize_flat(pred_np, stats)
    target_raw = denormalize_flat(target_np, stats)
    prev_raw = denormalize_flat(prev_np, stats)

    level_flat = K_LEVELS * F_LEVEL
    mid_idx = level_flat + 0
    spread_idx = level_flat + 1
    imbalance_idx = level_flat + 3

    true_delta = target_raw[..., mid_idx] - prev_raw[..., mid_idx]
    pred_delta = pred_raw[..., mid_idx] - prev_raw[..., mid_idx]
    nonzero = np.abs(true_delta) > 1e-7
    if nonzero.any():
        mid_direction_accuracy = float(
            (np.sign(true_delta[nonzero]) == np.sign(pred_delta[nonzero])).mean()
        )
    else:
        mid_direction_accuracy = float("nan")

    spread_mae = float(np.abs(pred_raw[..., spread_idx] - target_raw[..., spread_idx]).mean())
    imbalance_mae = float(
        np.abs(pred_raw[..., imbalance_idx] - target_raw[..., imbalance_idx]).mean()
    )

    pred_yes = np.clip(pred_raw[..., mid_idx], 1e-6, 1.0 - 1e-6)
    outcome = val_seq.yes_outcome
    brier = float("nan")
    log_loss = float("nan")
    if outcome is not None and np.isfinite(outcome).any():
        labels = []
        for s in starts:
            labels.append(outcome[s + 1 : s + batch_length])
        y = np.stack(labels, axis=0).astype(np.float32)
        mask = np.isfinite(y)
        if mask.any():
            p = pred_yes[mask]
            yy = y[mask]
            brier = float(np.mean((p - yy) ** 2))
            log_loss = float(-np.mean(yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p)))

    max_abs_norm = float(np.abs(flat).max()) if flat.size else 0.0
    metrics = {
        "Val/reconstruction_loss": float(reconstruction_loss.item()),
        "Val/normalized_next_mse": float(feature_mse.mean()),
        "Val/max_abs_normalized_feature": max_abs_norm,
        "Val/mid_direction_accuracy": mid_direction_accuracy,
        "Val/yes_brier": brier,
        "Val/yes_log_loss": log_loss,
        "Val/spread_mae": spread_mae,
        "Val/imbalance_mae": imbalance_mae,
    }
    top_idx = np.argsort(feature_mse)[-5:][::-1]
    top_features = [(FLAT_FEATURE_NAMES[int(i)], float(feature_mse[int(i)])) for i in top_idx]
    return metrics, top_features


def _gpu_monitor(interval: int = 30) -> None:
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            util, mem_used, mem_total = result.stdout.strip().split(", ")
            logger.info(f"[GPU] util={util}%  mem={mem_used}/{mem_total} MiB")
        time.sleep(interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=SRC_DIR / "config_files" / "configure_lob.yaml")
    pre_parser.add_argument("--data-train", type=Path,
                            default=SRC_DIR.parent / "data" / "train")
    pre_parser.add_argument("--data-val", type=Path,
                            default=SRC_DIR.parent / "data" / "validation")
    pre_parser.add_argument("--market-slug", default=None)
    pre_parser.add_argument("--hours-train", type=float, default=6.0)
    pre_parser.add_argument("--hours-val", type=float, default=1.0)
    pre_parser.add_argument("--norm-path", type=Path,
                            default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    pre_args, remaining = pre_parser.parse_known_args()

    with open(pre_args.config, "r") as f:
        config = yaml.safe_load(f)
    config = parse_args_and_update_config(config, argv=remaining)
    config = DotDict(config)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(config.BasicSettings.Device)
    from utils import WandbLogger, seed_np_torch

    seed_np_torch(seed=config.BasicSettings.Seed)

    logger.info(f"building train features from {pre_args.data_train}")
    norm_clip = getattr(config.BasicSettings, "NormClip", 8.0)
    aggregate_only = getattr(config.Models.WorldModel.Encoder, "AggregateOnly", False)
    train_seq, slug, stats = _build_sequences(
        pre_args.data_train,
        pre_args.market_slug,
        pre_args.hours_train,
        pre_args.norm_path,
        fit_stats=True,
        norm_clip=norm_clip,
        aggregate_only=aggregate_only,
    )
    logger.info(f"building val features from {pre_args.data_val}")
    val_seq, _, _ = _build_sequences(
        pre_args.data_val,
        slug,
        pre_args.hours_val,
        pre_args.norm_path,
        fit_stats=False,
        norm_clip=norm_clip,
        aggregate_only=aggregate_only,
    )

    for split_name, seq in (("train", train_seq), ("val", val_seq)):
        diag = normalized_feature_diagnostics(seq, stats.clip_value)
        top = ", ".join(f"{name}={value:.3f}" for name, value in diag["top_features"])
        logger.info(
            f"{split_name} normalized max_abs={diag['max_abs']:.3f}, "
            f"finite={diag['finite']}, top=[{top}]"
        )
        if not diag["within_clip"]:
            raise RuntimeError(
                f"{split_name} normalized features exceed clip "
                f"{diag['clip_value']}: max_abs={diag['max_abs']}"
            )

    assert train_seq.to_flat().shape[1] == FEATURE_DIM_FLAT
    assert train_seq.to_flat().shape[1] == config.BasicSettings.FeatureDim, (
        f"Feature dim mismatch: computed {train_seq.to_flat().shape[1]} "
        f"vs config {config.BasicSettings.FeatureDim}"
    )

    action_dim = 1
    from sub_models.world_models import WorldModel

    world_model = WorldModel(action_dim=action_dim, config=config, device=device).cuda(device)
    n_params = sum(p.numel() for p in world_model.parameters())
    logger.info(f"world model: {n_params:,} params, encoder_type={world_model.encoder_type}")

    replay_buffer = ReplayBuffer(config, device=device)
    _populate_buffer(replay_buffer, train_seq)

    wlogger = WandbLogger(
        config=config,
        project=config.Wandb.Init.Project,
        mode=config.Wandb.Init.Mode,
    )
    logdir = f"./saved_models/{config.n}/{config.BasicSettings.Env_name}/{wlogger.run.id}"
    os.makedirs(f"{logdir}/ckpt", exist_ok=True)

    threading.Thread(target=_gpu_monitor, args=(30,), daemon=True).start()

    accum_steps = getattr(config.JointTrainAgent, 'AccumSteps', 1)
    max_steps = config.JointTrainAgent.SampleMaxSteps
    save_every = config.JointTrainAgent.SaveEverySteps
    val_every = max(save_every // 2, 500)
    imagine_every = save_every

    pbar = tqdm(range(max_steps), desc="pretrain")
    for step in pbar:
        train_world_model_step(
            replay_buffer=replay_buffer,
            world_model=world_model,
            batch_size=config.JointTrainAgent.BatchSize,
            batch_length=config.JointTrainAgent.BatchLength,
            logger=wlogger,
            epoch=config.JointTrainAgent.TrainDynamicsEpoch,
            global_step=step,
            accum_steps=accum_steps,
        )
        if step > 0 and step % val_every == 0:
            val_metrics, top_mse = _validation_metrics(
                world_model, val_seq, stats,
                config.JointTrainAgent.BatchSize,
                config.JointTrainAgent.BatchLength,
            )
            for key, value in val_metrics.items():
                if np.isfinite(value):
                    wlogger.log(key, value, global_step=step)
            for name, value in top_mse:
                safe_name = name.replace(".", "/")
                wlogger.log(f"Val/feature_mse/{safe_name}", value, global_step=step)
            if top_mse:
                logger.info(
                    "validation top feature MSE: "
                    + ", ".join(f"{name}={value:.4g}" for name, value in top_mse)
                )
            pbar.set_postfix(
                val_loss=f"{val_metrics.get('Val/reconstruction_loss', float('nan')):.4f}"
            )
        if step > 0 and step % imagine_every == 0:
            _imagine_and_log(world_model, val_seq, wlogger, step)
        if step > 0 and step % save_every == 0:
            torch.save(
                {"step": step, "world_model": world_model.state_dict(),
                 "optimizer": world_model.optimizer.state_dict()},
                f"{logdir}/ckpt/world_model.pth",
            )

    torch.save(
        {"step": max_steps, "world_model": world_model.state_dict(),
         "optimizer": world_model.optimizer.state_dict()},
        f"{logdir}/ckpt/world_model.pth",
    )
    logger.info(f"final checkpoint written to {logdir}/ckpt/world_model.pth")
    wlogger.close()


if __name__ == "__main__":
    main()
