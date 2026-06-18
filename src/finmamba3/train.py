"""Offline world-model pretraining on Polymarket LOB features.

Trains the LOB encoder / Mamba / MLP decoder stack to reconstruct and
predict LOB feature sequences for a single market. No agent, no live env,
no reward signal. Produces a world-model checkpoint suitable for warm-
starting a downstream RL loop.
"""
# region imports
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
from finmamba3.envs.lob_features import (
    FLAT_FEATURE_NAMES,
    LOBSequence,
    denormalize_flat,
    make_aggregate_only,
    normalized_feature_diagnostics,
)
from finmamba3.envs.fi2010_loader import FLAT_FEATURE_NAMES_FI2010
from finmamba3.config import DotDict, parse_args_and_update_config
from finmamba3.replay_buffer import ReplayBuffer
from finmamba3.train_step import train_world_model_step
from finmamba3.sequence_builder import (
    build_fi2010_sequences,
    build_kaggle_sequences,
    build_market_sequences,
    build_sequences,
    populate_buffer,
    populate_buffer_multi,
)
# endregion
logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[1]


def imagine_rollout(
    world_model: "WorldModel",
    val_seq: LOBSequence,
    context_len: int = 16,
    horizon: int = 32,
) -> np.ndarray | None:
    """Encode a short validation context and autoregressively roll forward.

    Returns a (horizon, FEATURE_DIM_FLAT) decoded trajectory, or None if the
    val sequence is too short. Pure function with no wandb side-effects.
    """
    world_model.eval()
    device = world_model.device
    T = val_seq.per_tick.shape[0]
    if T < context_len + 1:
        return None
    ctx = val_seq.to_flat()[:context_len]
    obs = torch.from_numpy(ctx).float().to(device, non_blocking=True).unsqueeze(0)
    action = torch.zeros((1, context_len), dtype=torch.float32, device=device)
    decoded: list[np.ndarray] = []
    # Match training/validation autocast: the MIMO TileLang kernel only codegens a
    # bf16 MMA path, so an unwrapped FP32 sequence_model call fails nvcc compilation.
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=world_model.use_amp
    ):
        ctx_latent = world_model.encode_obs(obs)
        prefix_latent = ctx_latent
        prefix_action = action
        for step in range(horizon):
            if world_model.model == "Transformer":
                from finmamba3.models.attention import get_subsequent_mask_with_batch_length
                temporal_mask = get_subsequent_mask_with_batch_length(
                    prefix_latent.shape[1], prefix_latent.device
                )
                feat = world_model.sequence_model(prefix_latent, prefix_action, temporal_mask)
            else:
                feat = world_model.sequence_model(prefix_latent, prefix_action)
            feat = world_model.condition_dist_feat(feat[:, -1:])
            prior_logits = world_model.dist_head.forward_prior(feat)
            prior_sample = world_model.straight_through_gradient(prior_logits)
            prior_flat = world_model.flatten_sample(prior_sample)
            # The studentt decoder returns (mean, log_scale); the rollout trajectory is the predicted
            # mean, so take the first element. The Gaussian decoder returns the prediction directly.
            decoder_out = world_model.obs_decoder(prior_flat)
            mean_out = decoder_out[0] if world_model.decoder_kind == "studentt" else decoder_out
            decoded.append(mean_out.float().cpu().numpy()[0, 0])
            if step != horizon - 1:
                next_action = torch.zeros((1, 1), dtype=torch.float32, device=device)
                prefix_latent = torch.cat([prefix_latent, prior_flat], dim=1)
                prefix_action = torch.cat([prefix_action, next_action], dim=1)
    return np.stack(decoded, axis=0)


def _imagine_and_log(
    world_model: WorldModel,
    val_seq: LOBSequence,
    wlogger: WandbLogger,
    global_step: int,
    context_len: int = 16,
    horizon: int = 32,
) -> None:
    """Run an imagine rollout and emit wandb metrics plus the .npy snapshot."""
    decoded = imagine_rollout(world_model, val_seq, context_len, horizon)
    if decoded is None:
        return
    # midprice_index points at the start of the per-tick block; offsets 0/1/3 are
    # mid / spread / imbalance under both the Polymarket and FI-2010 schemas.
    level_flat = world_model.midprice_index
    mid_norm = decoded[:, level_flat + 0]
    spread_norm = decoded[:, level_flat + 1]
    imbalance_norm = decoded[:, level_flat + 3]
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
        sample = world_model.straight_through_gradient(post_logits)
        flattened_sample = world_model.flatten_sample(sample)
        # The studentt decoder returns (mean, log_scale) and its loss is the weighted NLL over both;
        # the Gaussian decoder returns a single tensor scored by weighted MSE. Mirror update() so the
        # studentt arm's Val/reconstruction_loss is the NLL that SaveBestOnly and early stopping read.
        decoder_out = world_model.obs_decoder(flattened_sample)
        if world_model.decoder_kind == "studentt":
            recon_mean, recon_log_scale = decoder_out
            reconstruction_loss = world_model.reconstruction_loss_func(
                world_model.obs_decoder, recon_mean, recon_log_scale, obs
            )
        else:
            obs_hat = decoder_out
            reconstruction_loss = world_model.reconstruction_loss_func(obs_hat, obs)
        if world_model.model == "Transformer":
            from finmamba3.models.attention import get_subsequent_mask_with_batch_length
            temporal_mask = get_subsequent_mask_with_batch_length(
                batch_length, flattened_sample.device
            )
            dist_feat = world_model.sequence_model(flattened_sample, action, temporal_mask)
        else:
            dist_feat = world_model.sequence_model(flattened_sample, action)
        prior_logits = world_model.dist_head.forward_prior(dist_feat[:, :-1])
        prior_sample = world_model.straight_through_gradient(prior_logits, sample_mode="probs")
        prior_flat = world_model.flatten_sample(prior_sample)
        # The one-step prediction MSE diagnostic compares the predicted mean to the next tick, so take
        # the studentt mean; the Gaussian decoder already returns that mean directly.
        prior_decode = world_model.obs_decoder(prior_flat)
        next_hat = prior_decode[0] if world_model.decoder_kind == "studentt" else prior_decode
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
    level_flat = world_model.midprice_index
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
    # Pick the feature-name tuple matching the active schema. Polymarket flat
    # feature dim is 94; FI-2010 flat is 46. Anything else falls back to indices.
    name_table = FLAT_FEATURE_NAMES
    if flat.shape[1] == len(FLAT_FEATURE_NAMES_FI2010):
        name_table = FLAT_FEATURE_NAMES_FI2010
    elif flat.shape[1] != len(FLAT_FEATURE_NAMES):
        name_table = tuple(f"feature_{i}" for i in range(flat.shape[1]))
    top_idx = np.argsort(feature_mse)[-5:][::-1]
    top_features = [(name_table[int(i)], float(feature_mse[int(i)])) for i in top_idx]
    return metrics, top_features


# Public aliases for use by external eval scripts. The underscored names remain
# the implementation; the public names mirror the API the eval CLIs import.
validate = _validation_metrics


def _gpu_monitor(interval: int = 30, sample_every: float = 2.0) -> None:
    # A single instantaneous nvidia-smi snapshot lands on an arbitrary instant of a
    # multi-second step and badly under-reads util. Sample across the whole window and
    # report mean + peak so the logged number reflects sustained occupancy.
    samples_per_window = max(1, int(interval / sample_every))
    while True:
        utils = []
        mem_used, mem_total = "?", "?"
        for _ in range(samples_per_window):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                util, mem_used, mem_total = result.stdout.strip().split(", ")
                utils.append(int(util))
            time.sleep(sample_every)
        if utils:
            mean_util = sum(utils) // len(utils)
            logger.info(f"[GPU] util mean={mean_util}% peak={max(utils)}% mem={mem_used}/{mem_total} MiB")


def _upload_checkpoints_async(folder: str, repo_id: str, token: str, lock: threading.Lock,
                              path_in_repo: str = "checkpoints/lob") -> None:
    # Fire-and-forget HF sync on a daemon thread so the upload never stalls the training
    # loop. The non-blocking lock drops the sync if a previous one is still in flight, so a
    # slow upload cannot pile up behind a faster save cadence.
    def _run():
        if not lock.acquire(blocking=False):
            return
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=folder,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message="Periodic checkpoint sync during training",
            )
            logger.info(f"[ckpt-sync] uploaded {folder} -> {repo_id}")
        except Exception as exc:
            # A transient network error must not kill a multi-hour run; log and retry next cadence.
            logger.warning(f"[ckpt-sync] upload failed: {exc}")
        finally:
            lock.release()
    threading.Thread(target=_run, daemon=True).start()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=SRC_DIR.parent / "configs" / "lob.yaml")
    pre_parser.add_argument("--data-train", type=Path,
                            default=SRC_DIR.parent / "data" / "train")
    pre_parser.add_argument("--data-val", type=Path,
                            default=SRC_DIR.parent / "data" / "validation")
    pre_parser.add_argument("--market-slug", default=None)
    pre_parser.add_argument(
        "--max-markets", type=int, default=None,
        help="Train on the top-N longest markets (multi-market substrate). Default: "
             "JointTrainAgent.MarketCount from the config, or 1 for single-market.",
    )
    pre_parser.add_argument("--hours-train", type=float, default=6.0)
    pre_parser.add_argument("--hours-val", type=float, default=1.0)
    pre_parser.add_argument(
        "--intervals", default=None,
        help="Comma-separated list of market intervals to use (e.g., '5m' or '5m,15m'). Default: all.",
    )
    pre_parser.add_argument("--norm-path", type=Path,
                            default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    pre_parser.add_argument(
        "--dataset", choices=("polymarket", "fi2010", "kaggle"), default=None,
        help="Override Dataset.Kind from the config file.",
    )
    pre_parser.add_argument(
        "--resume-checkpoint", type=Path, default=None,
        help="Warm-restart from this .pth (model + optimizer weights). The LR schedule "
             "and step counter reset; best_val_loss seeds from the checkpoint so the new "
             "run only saves a checkpoint when it beats the resumed one.",
    )
    pre_parser.add_argument(
        "--ckpt-repo", default=None,
        help="HF model repo id to sync checkpoints to during training (crash-safety).",
    )
    pre_parser.add_argument(
        "--ckpt-upload-every", type=int, default=0,
        help="Write the latest checkpoint and upload saved_models/<n> to --ckpt-repo every "
             "N steps. 0 disables in-training upload (the run still saves locally).",
    )
    pre_parser.add_argument(
        "--ckpt-path-in-repo", default="checkpoints/lob",
        help="Destination folder inside --ckpt-repo for the checkpoint sync. Set this per-dataset "
             "(e.g. mimo-ckpt/fi2010) so runs on different datasets do not collide in one folder, "
             "which lets a fresh session find and resume the right checkpoint.",
    )
    pre_parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    pre_args, remaining = pre_parser.parse_known_args()
    with open(pre_args.config, "r") as f:
        config = yaml.safe_load(f)
    config = parse_args_and_update_config(config, argv=remaining)
    config = DotDict(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(config.BasicSettings.Device)
    from finmamba3.training_utils import make_logger, seed_np_torch
    seed_np_torch(seed=config.BasicSettings.Seed)
    # Pick the data pipeline. Polymarket reads SQLite + per-tick CSV books;
    # FI-2010 reads a single space-separated text matrix per split.
    dataset_cfg = config.get("Dataset", None)
    config_dataset_kind = dataset_cfg.get("Kind", "polymarket") if dataset_cfg is not None else "polymarket"
    dataset_kind = pre_args.dataset or config_dataset_kind
    logger.info(f"dataset pipeline: {dataset_kind}")
    norm_clip = config.BasicSettings.get("NormClip", 8.0)
    aggregate_only = config.Models.WorldModel.Encoder.get("AggregateOnly", False)
    # The multi-market knob defaults to the config; 1 keeps the single-market path.
    max_markets = pre_args.max_markets if pre_args.max_markets is not None else int(config.JointTrainAgent.get("MarketCount", 1))
    # Parse intervals if provided
    intervals = None
    if pre_args.intervals:
        intervals = [i.strip() for i in pre_args.intervals.split(",")]
        logger.info(f"filtering markets to intervals: {intervals}")
    if dataset_kind == "polymarket":
        include_binary_features = config.Models.WorldModel.Encoder.BinaryMarketFeatures
        # Phase 0.1/0.2/0.6: SpotFeatures appends the causal spot path, the time-to-expiry clock
        # and the asset one-hot. Default off keeps the legacy 100-dim schema bit-identical.
        include_spot_features = config.Models.WorldModel.Encoder.get('SpotFeatures', False)
        # Phase 0.4: CrossIntervalContext appends the contemporaneous short-interval summary to hourly
        # markets (zeros elsewhere). Default off; enabling it widens the schema by three channels.
        include_cross_interval = config.Models.WorldModel.Encoder.get('CrossIntervalContext', False)
        # Per-asset training: restrict the market pool to these asset symbols (e.g. ['ETH']) so the model
        # learns one asset's dynamics; default None keeps the top markets across all assets (BTC-dominated).
        train_assets = config.Models.WorldModel.Encoder.get('Assets', None)
        logger.info(f"building train features from {pre_args.data_train} (max_markets={max_markets}, assets={train_assets})")
        if max_markets > 1:
            train_seqs, slugs, stats = build_market_sequences(
                pre_args.data_train,
                pre_args.market_slug,
                pre_args.hours_train,
                pre_args.norm_path,
                fit_stats=True,
                norm_clip=norm_clip,
                aggregate_only=aggregate_only,
                max_markets=max_markets,
                intervals=intervals,
                include_binary_features=include_binary_features,
                include_spot_features=include_spot_features,
                include_cross_interval=include_cross_interval,
                assets=train_assets,
            )
            # The training-loop val metric reads one sequence; the longest val market
            # is a stable, boundary-safe representative. Cross-market generalization is
            # measured post-hoc by eval/regime_split.py + eval/compare_direction.py.
            logger.info(f"building val features from {pre_args.data_val} (longest market)")
            val_seq, _, _ = build_sequences(
                pre_args.data_val,
                None,
                pre_args.hours_val,
                pre_args.norm_path,
                fit_stats=False,
                norm_clip=norm_clip,
                intervals=intervals,
                aggregate_only=aggregate_only,
                include_binary_features=include_binary_features,
                include_spot_features=include_spot_features,
                include_cross_interval=include_cross_interval,
            )
        else:
            train_seq, slug, stats = build_sequences(
                pre_args.data_train,
                pre_args.market_slug,
                pre_args.hours_train,
                pre_args.norm_path,
                fit_stats=True,
                norm_clip=norm_clip,
                aggregate_only=aggregate_only,
                intervals=intervals,
                include_binary_features=include_binary_features,
                include_spot_features=include_spot_features,
                include_cross_interval=include_cross_interval,
            )
            logger.info(f"building val features from {pre_args.data_val}")
            val_seq, _, _ = build_sequences(
                pre_args.data_val,
                slug,
                pre_args.hours_val,
                pre_args.norm_path,
                fit_stats=False,
                norm_clip=norm_clip,
                intervals=intervals,
                aggregate_only=aggregate_only,
                include_binary_features=include_binary_features,
                include_spot_features=include_spot_features,
                include_cross_interval=include_cross_interval,
            )
            train_seqs = [train_seq]
    elif dataset_kind == "fi2010":
        fi2010_cfg = dataset_cfg.get("FI2010", None) if dataset_cfg is not None else None
        horizon = int(fi2010_cfg.get("Horizon", 10)) if fi2010_cfg is not None else 10
        max_events_cfg = fi2010_cfg.get("MaxEvents", None) if fi2010_cfg is not None else None
        max_events = int(max_events_cfg) if max_events_cfg else None
        logger.info(f"loading FI-2010 train split from {pre_args.data_train}")
        train_seq, slug, stats = build_fi2010_sequences(
            pre_args.data_train, split="train", horizon=horizon,
            norm_path=pre_args.norm_path, fit_stats=True,
            norm_clip=norm_clip, max_events=max_events,
        )
        logger.info(f"loading FI-2010 validation split from {pre_args.data_val}")
        val_seq, _, _ = build_fi2010_sequences(
            pre_args.data_val, split="validation", horizon=horizon,
            norm_path=pre_args.norm_path, fit_stats=False,
            norm_clip=norm_clip, max_events=max_events,
        )
        if aggregate_only:
            train_seq = make_aggregate_only(train_seq)
            val_seq = make_aggregate_only(val_seq)
        # FI-2010 is a single sequence; the buffer's segment flags stay all-False.
        train_seqs = [train_seq]
    elif dataset_kind == "kaggle":
        kaggle_cfg = dataset_cfg.get("Kaggle", None) if dataset_cfg is not None else None
        if kaggle_cfg is None:
            raise ValueError("Dataset.Kind=kaggle requires a Dataset.Kaggle config block (Asset/Resolution/HoursTrain/HoursVal).")
        asset = str(kaggle_cfg.get("Asset", "BTC"))
        resolution = str(kaggle_cfg.get("Resolution", "1min"))
        hours_train = float(kaggle_cfg.get("HoursTrain", 168.0))
        hours_val = float(kaggle_cfg.get("HoursVal", 72.0))
        flat_threshold = float(kaggle_cfg.get("FlatThreshold", 0.0))
        logger.info(f"loading Kaggle {asset} {resolution} train slice from {pre_args.data_train}")
        train_seq, slug, stats = build_kaggle_sequences(
            pre_args.data_train, asset=asset, resolution=resolution, split="train",
            hours_train=hours_train, hours_val=hours_val, norm_path=pre_args.norm_path,
            fit_stats=True, norm_clip=norm_clip, flat_threshold=flat_threshold,
        )
        logger.info(f"loading Kaggle {asset} {resolution} validation slice from {pre_args.data_val}")
        val_seq, _, _ = build_kaggle_sequences(
            pre_args.data_val, asset=asset, resolution=resolution, split="validation",
            hours_train=hours_train, hours_val=hours_val, norm_path=pre_args.norm_path,
            fit_stats=False, norm_clip=norm_clip, flat_threshold=flat_threshold,
        )
        if aggregate_only:
            train_seq = make_aggregate_only(train_seq)
            val_seq = make_aggregate_only(val_seq)
        # Kaggle is a single continuous stream; the buffer's segment flags stay all-False.
        train_seqs = [train_seq]
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind!r}")
    diag_targets = [(f"train[{i}]", seq) for i, seq in enumerate(train_seqs)]
    diag_targets.append(("val", val_seq))
    for split_name, seq in diag_targets:
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
    # The config-driven assertion handles both Polymarket (94-dim) and FI-2010 (46-dim).
    # The legacy FEATURE_DIM_FLAT constant only applies to the Polymarket schema.
    assert train_seqs[0].to_flat().shape[1] == config.BasicSettings.FeatureDim, (
        f"Feature dim mismatch: computed {train_seqs[0].to_flat().shape[1]} "
        f"vs config {config.BasicSettings.FeatureDim}"
    )
    action_dim = 1
    from finmamba3.models.world_model import WorldModel
    world_model = WorldModel(action_dim=action_dim, config=config, device=device).cuda(device)
    n_params = sum(p.numel() for p in world_model.parameters())
    logger.info(f"world model: {n_params:,} params, encoder_type={world_model.encoder_type}")
    resume_best_val_loss = float('inf')
    if pre_args.resume_checkpoint is not None:
        ckpt = torch.load(pre_args.resume_checkpoint, map_location=device)
        world_model.load_state_dict(ckpt["world_model"])
        world_model.optimizer.load_state_dict(ckpt["optimizer"])
        resume_best_val_loss = float(ckpt.get("best_val_loss", float('inf')))
        logger.info(
            f"warm-restart from {pre_args.resume_checkpoint}: trained-to step "
            f"{ckpt.get('step', '?')}, best_val_loss={resume_best_val_loss:.4f}. "
            f"LR schedule and step counter reset for the new run."
        )
    # Probe the autocast/dtype path on a tiny batch so a misconfiguration shows
    # up before a 6-hour training run instead of after.
    try:
        with torch.no_grad():
            probe_obs = torch.zeros(
                (1, 4, config.BasicSettings.FeatureDim), device=device, dtype=torch.float32
            )
            probe_action = torch.zeros((1, 4), device=device, dtype=torch.float32)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=world_model.use_amp
            ):
                emb = world_model.encoder(probe_obs)
                seq = world_model.sequence_model(
                    torch.zeros(
                        (1, 4, world_model.stoch_flattened_dim),
                        device=device,
                        dtype=emb.dtype,
                    ),
                    probe_action,
                )
            logger.info(
                f"dtype probe: encoder->{emb.dtype}, sequence_model->{seq.dtype} "
                f"(use_amp={world_model.use_amp})"
            )
    except Exception as exc:
        logger.warning(f"dtype probe failed: {exc}")
    # Compile the encoder and decoder only. Mamba3 has its own Triton/TileLang
    # kernels; torch.compile would fight them and risks recompilations every
    # call. The transformer encoder and MLP decoder are pure PyTorch and benefit.
    if config.BasicSettings.get("Compile", False):
        try:
            world_model.encoder = torch.compile(
                world_model.encoder, mode="reduce-overhead", dynamic=False
            )
            world_model.obs_decoder = torch.compile(
                world_model.obs_decoder, mode="reduce-overhead", dynamic=False
            )
            logger.info("torch.compile enabled for encoder + obs_decoder")
        except Exception as exc:
            logger.warning(f"torch.compile unavailable, running eager: {exc}")
    replay_buffer = ReplayBuffer(config, device=device)
    if len(train_seqs) > 1:
        populate_buffer_multi(replay_buffer, train_seqs)
    else:
        populate_buffer(replay_buffer, train_seqs[0])
    wlogger = make_logger(
        config=config,
        project=config.Wandb.Init.Project,
        mode=config.Wandb.Init.Mode,
    )
    logdir = f"./saved_models/{config.n}/{config.BasicSettings.Env_name}/{wlogger.run.id}"
    os.makedirs(f"{logdir}/ckpt", exist_ok=True)
    threading.Thread(target=_gpu_monitor, args=(30,), daemon=True).start()
    ckpt_root = f"./saved_models/{config.n}"
    upload_every = pre_args.ckpt_upload_every if (pre_args.ckpt_repo and pre_args.hf_token) else 0
    upload_lock = threading.Lock()
    if upload_every > 0:
        logger.info(f"[ckpt-sync] syncing {ckpt_root} -> {pre_args.ckpt_repo} every {upload_every} steps")
    accum_steps = config.JointTrainAgent.get('AccumSteps', 1)
    log_every = config.JointTrainAgent.get('LogEvery', 50)
    max_steps = config.JointTrainAgent.SampleMaxSteps
    save_every = config.JointTrainAgent.SaveEverySteps
    val_every = max(save_every // 2, 500)
    imagine_every = save_every
    # Early stopping config
    early_stopping_patience = config.JointTrainAgent.get('EarlyStoppingPatience', 0)
    early_stop_metric = config.JointTrainAgent.get('EarlyStopMetric', 'Val/reconstruction_loss')
    save_best_only = config.JointTrainAgent.get('SaveBestOnly', False)
    # Tracking variables
    best_val_loss = resume_best_val_loss
    patience_counter = 0
    best_step = 0
    pbar = tqdm(range(max_steps), desc="pretrain", miniters=50, mininterval=0)
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
            log_every=log_every,
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
            # Echo the decision metrics to stdout so the offline log captures the
            # thesis-relevant signal, not just the reconstruction ELBO the run
            # early-stops on. NaN settlement metrics print as nan without breaking.
            logger.info(
                f"[val] step={step} "
                f"recon={val_metrics.get('Val/reconstruction_loss', float('nan')):.4f} "
                f"dir_acc={val_metrics.get('Val/mid_direction_accuracy', float('nan')):.4f} "
                f"yes_log_loss={val_metrics.get('Val/yes_log_loss', float('nan')):.4f} "
                f"yes_brier={val_metrics.get('Val/yes_brier', float('nan')):.4f}"
            )

            # --- Early stopping & best checkpoint logic ---
            current_val_loss = val_metrics.get(early_stop_metric, float('inf'))
            if not np.isfinite(current_val_loss):
                # When the selected metric is NaN this validation, fall through without comparing.
                current_val_loss = float('inf')

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_step = step
                patience_counter = 0
                if save_best_only:
                    torch.save(
                        {"step": step, "world_model": world_model.state_dict(),
                         "optimizer": world_model.optimizer.state_dict(),
                         "best_val_loss": best_val_loss},
                        f"{logdir}/ckpt/world_model_best.pth",
                    )
                    logger.info(f"new best val_loss={best_val_loss:.4f} at step {step}, checkpoint saved")
            else:
                patience_counter += 1
                if early_stopping_patience > 0:
                    logger.info(f"no improvement for {patience_counter}/{early_stopping_patience} validations")

            dir_acc = val_metrics.get("Val/mid_direction_accuracy", float("nan"))
            pbar.set_postfix(val_loss=f"{current_val_loss:.4f}", best=f"{best_val_loss:.4f}", dir_acc=f"{dir_acc:.3f}")

            # Early stopping check
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                logger.info(f"early stopping triggered at step {step}, best was {best_val_loss:.4f} at step {best_step}")
                break
        if step > 0 and step % imagine_every == 0:
            _imagine_and_log(world_model, val_seq, wlogger, step)
        if step > 0 and step % save_every == 0:
            torch.save(
                {"step": step, "world_model": world_model.state_dict(),
                 "optimizer": world_model.optimizer.state_dict()},
                f"{logdir}/ckpt/world_model.pth",
            )
        if upload_every > 0 and step > 0 and step % upload_every == 0:
            # Snapshot the latest weights, then push the whole checkpoint dir to HF so an
            # interrupted run still leaves a recent model on the Hub. The save runs inline
            # (it must finish before the async upload reads the file); the upload is detached.
            torch.save(
                {"step": step, "world_model": world_model.state_dict(),
                 "optimizer": world_model.optimizer.state_dict()},
                f"{logdir}/ckpt/world_model.pth",
            )
            _upload_checkpoints_async(ckpt_root, pre_args.ckpt_repo, pre_args.hf_token, upload_lock,
                                      pre_args.ckpt_path_in_repo)
    # Save final checkpoint
    torch.save(
        {"step": step, "world_model": world_model.state_dict(),
         "optimizer": world_model.optimizer.state_dict()},
        f"{logdir}/ckpt/world_model_final.pth",
    )
    logger.info(f"final checkpoint written to {logdir}/ckpt/world_model_final.pth")
    logger.info(f"best val_loss={best_val_loss:.4f} at step {best_step}")
    if save_best_only:
        logger.info(f"best checkpoint: {logdir}/ckpt/world_model_best.pth")
    if upload_every > 0:
        # Final synchronous flush so the very last weights land on HF before the process exits.
        from huggingface_hub import HfApi
        HfApi().upload_folder(
            folder_path=ckpt_root, path_in_repo=pre_args.ckpt_path_in_repo,
            repo_id=pre_args.ckpt_repo, repo_type="model", token=pre_args.hf_token,
            commit_message="Final checkpoint sync",
        )
        logger.info(f"[ckpt-sync] final upload {ckpt_root} -> {pre_args.ckpt_repo}")
    wlogger.close()
if __name__ == "__main__":
    main()
