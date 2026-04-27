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
    FEATURE_DIM_FLAT,
    LOBSequence,
    apply_normalization,
    extract_features,
    fit_normalization,
    load_normalization,
    pick_longest_market,
    save_normalization,
)
from lob.backtester import build_timeline
from replay_buffer import ReplayBuffer
from sub_models.world_models import WorldModel
from train import DotDict, parse_args_and_update_config
from utils import WandbLogger, seed_np_torch

logger = logging.getLogger(__name__)


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
) -> tuple[LOBSequence, str]:
    bt = build_timeline(data_dir=data_dir, hours=hours)
    slug = market_slug or pick_longest_market(bt)
    try:
        seq = extract_features(bt.timeline, slug)
    except RuntimeError:
        # Requested slug has no usable ticks in this split; fall back to the
        # longest market available in this split.
        slug = pick_longest_market(bt)
        logger.warning(
            f"Market {market_slug!r} has no usable ticks in {data_dir}; "
            f"falling back to {slug!r}"
        )
        seq = extract_features(bt.timeline, slug)
    if fit_stats:
        stats = fit_normalization(seq)
        save_normalization(stats, norm_path)
        logger.info(f"normalization fit on {slug}, saved to {norm_path}")
    else:
        stats = load_normalization(norm_path)
    seq_norm = apply_normalization(seq, stats)
    return seq_norm, slug


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
    from mamba_ssm import InferenceParams

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
        inference_params = InferenceParams(
            max_seqlen=context_len + horizon,
            max_batch_size=1,
        )
        ctx_feat = world_model.sequence_model(ctx_latent, action, inference_params)
        inference_params.seqlen_offset += ctx_feat.shape[1]

        prior_logits = world_model.dist_head.forward_prior(ctx_feat[:, -1:])
        prior_sample = world_model.stright_throught_gradient(prior_logits)
        prior_flat = world_model.flatten_sample(prior_sample)

        decoded = [world_model.image_decoder(prior_flat).cpu().numpy()[0, 0]]
        last_sample = prior_flat
        last_action = torch.zeros((1, 1), dtype=torch.float32, device=device)
        for _ in range(horizon - 1):
            feat = world_model.sequence_model(last_sample, last_action, inference_params)
            inference_params.seqlen_offset += 1
            prior_logits = world_model.dist_head.forward_prior(feat)
            prior_sample = world_model.stright_throught_gradient(prior_logits)
            last_sample = world_model.flatten_sample(prior_sample)
            decoded.append(world_model.image_decoder(last_sample).cpu().numpy()[0, 0])

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


def _validation_loss(
    world_model: WorldModel,
    val_seq: LOBSequence,
    batch_size: int,
    batch_length: int,
) -> float:
    world_model.eval()
    device = world_model.device
    flat = val_seq.to_flat()
    T = flat.shape[0]
    if T < batch_length + 1:
        return float("nan")
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
        loss = world_model.reconstruction_loss_func(obs_hat, obs)
    return float(loss.item())


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
    pre_parser.add_argument("--config", default="config_files/configure_lob.yaml")
    pre_parser.add_argument("--data-train", type=Path,
                            default=Path(r"C:\Users\ruuud\algoverse\Drama\data\train"))
    pre_parser.add_argument("--data-val", type=Path,
                            default=Path(r"C:\Users\ruuud\algoverse\Drama\data\validation"))
    pre_parser.add_argument("--market-slug", default=None)
    pre_parser.add_argument("--hours-train", type=float, default=6.0)
    pre_parser.add_argument("--hours-val", type=float, default=1.0)
    pre_parser.add_argument("--norm-path", type=Path,
                            default=Path(r"C:\Users\ruuud\algoverse\Drama\data\normalization.json"))
    pre_args, remaining = pre_parser.parse_known_args()

    # parse_args_and_update_config reads sys.argv; swap in the remaining args
    # so --Group.Key overrides still work.
    import sys as _sys
    _sys.argv = [_sys.argv[0]] + remaining

    with open(pre_args.config, "r") as f:
        config = yaml.safe_load(f)
    config = parse_args_and_update_config(config)
    config = DotDict(config)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(config.BasicSettings.Device)
    seed_np_torch(seed=config.BasicSettings.Seed)

    logger.info(f"building train features from {pre_args.data_train}")
    train_seq, slug = _build_sequences(
        pre_args.data_train,
        pre_args.market_slug,
        pre_args.hours_train,
        pre_args.norm_path,
        fit_stats=True,
    )
    logger.info(f"building val features from {pre_args.data_val}")
    val_seq, _ = _build_sequences(
        pre_args.data_val,
        slug,
        pre_args.hours_val,
        pre_args.norm_path,
        fit_stats=False,
    )

    assert train_seq.to_flat().shape[1] == FEATURE_DIM_FLAT
    assert train_seq.to_flat().shape[1] == config.BasicSettings.FeatureDim, (
        f"Feature dim mismatch: computed {train_seq.to_flat().shape[1]} "
        f"vs config {config.BasicSettings.FeatureDim}"
    )

    action_dim = 1
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

    from train import train_world_model_step

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
            vloss = _validation_loss(
                world_model, val_seq,
                config.JointTrainAgent.BatchSize,
                config.JointTrainAgent.BatchLength,
            )
            wlogger.log("Val/reconstruction_loss", vloss, global_step=step)
            pbar.set_postfix(val_loss=f"{vloss:.4f}")
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
