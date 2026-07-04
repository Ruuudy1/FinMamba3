"""Compare direction prediction across the world model and baselines.

Loads a pretrained world-model checkpoint, fits the LinearAR baseline, trains
a fresh DeepLOB on the train split, and evaluates all three on the val split
across one or more direction thresholds. Emits a markdown comparison table whose
macro_f1 column is directly comparable to the LOBCAST benchmark (Prata et al.
2023): macro-F1 weights the three classes equally, unlike accuracy.

The DL arms (world_model, deeplob) honor --label-mode: next_tick (legacy sign of
the next-tick mid change) or triple_barrier (de Prado profit/stop/time barrier on
the raw mid, the denoised target for Polymarket's wide spread). LinearAR keeps its
own internal next-tick labelling.

The world-model arm is optional (skip with --baselines linear_ar,deeplob)
which lets you run the script CPU-only when no checkpoint is available.

Example
-------
    python -m eval.compare_direction \\
        --world-checkpoint saved_models/lob/LOB/1iemugot/ckpt/world_model.pth \\
        --config configs/lob.yaml \\
        --data-train data/train --data-val data/validation \\
        --thresholds 0.001,0.005,0.01 \\
        --baselines world_model,deeplob,linear_ar
"""
# region imports
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
# endregion
logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LOB direction prediction across methods")
    p.add_argument("--world-checkpoint", type=Path, default=None,
                   help="Path to world_model.pth; required if 'world_model' is in --baselines")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-train", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", type=Path, default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    p.add_argument("--market-slug", default=None)
    p.add_argument("--hours-train", type=float, default=6.0)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--thresholds", default="0.001,0.005,0.01", help="Comma-separated direction thresholds")
    p.add_argument("--label-mode", choices=("next_tick", "triple_barrier"), default="next_tick",
                   help="Direction labelling for the DL arms. next_tick = sign of the next-tick "
                        "mid change (legacy). triple_barrier = de Prado profit/stop/time barrier "
                        "on raw mid (denoised; recommended for Polymarket's wide spread).")
    p.add_argument("--tb-horizon", type=int, default=32, help="Forward horizon in ticks for triple-barrier labelling.")
    p.add_argument("--baselines", default="world_model,deeplob,linear_ar",
                   help="Comma-separated subset of {world_model,deeplob,linear_ar}")
    p.add_argument("--epochs-deeplob", type=int, default=3)
    p.add_argument("--batch-deeplob", type=int, default=64)
    p.add_argument("--lr-deeplob", type=float, default=1.0e-3)
    p.add_argument("--out", type=Path, default=Path("reports/direction_comparison.md"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device_from_arg(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _label_directions(mid_norm: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a 1-D midprice trajectory to (T-1,) class labels in {0, 1, 2}."""
    dmid = np.diff(mid_norm)
    labels = np.full_like(dmid, fill_value=1, dtype=np.int64)
    labels[dmid > threshold] = 2
    labels[dmid < -threshold] = 0
    return labels


def _nan_metrics() -> dict[str, float]:
    return {"accuracy": float("nan"), "macro_f1": float("nan"), "brier": float("nan")}


def _global_labels(val_seq, threshold: float, label_mode: str, tb_horizon: int, level_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-tick direction labels for the whole sequence, plus a validity mask.

    next_tick labels the sign of the next normalized-mid change (legacy). triple_barrier
    labels which of the profit/stop/time barriers a forward window hits first on the raw
    mid, the denoised target preferred for Polymarket's wide spread. Both are indexed per
    window as labels[s : s + batch_length - 1].
    """
    if label_mode == "triple_barrier":
        from finmamba3.envs.lob_labels import TripleBarrierConfig, triple_barrier_labels_numpy
        config = TripleBarrierConfig(profit_threshold=threshold, stop_threshold=threshold, horizon=tb_horizon)
        labels, valid = triple_barrier_labels_numpy(val_seq.midprice.astype(np.float64), config)
        return labels.astype(np.int64), valid
    mid_norm = val_seq.to_flat()[:, level_width + 0]
    labels = _label_directions(mid_norm, threshold)
    valid = np.ones_like(labels, dtype=bool)
    return labels, valid


def classification_metrics(pred_probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Top-1 accuracy, macro-averaged F1, and the three-class Brier score.

    Macro-F1 is the benchmark paper's reporting axis (Prata et al. 2023): it weights
    all three classes equally, so it is not inflated by the dominant flat class the way
    raw accuracy is.
    """
    if pred_probs.ndim != 2 or pred_probs.shape[1] != 3:
        raise ValueError(f"pred_probs must be (N, 3); got {pred_probs.shape}")
    if labels.size == 0:
        return _nan_metrics()
    pred_labels = pred_probs.argmax(axis=-1)
    accuracy = float((pred_labels == labels).mean())
    one_hot = np.eye(3)[labels]
    brier = float(((pred_probs - one_hot) ** 2).sum(axis=-1).mean())
    f1_sum = 0.0
    for class_index in range(3):
        true_positive = float(((pred_labels == class_index) & (labels == class_index)).sum())
        predicted_positive = float((pred_labels == class_index).sum())
        actual_positive = float((labels == class_index).sum())
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
        recall = true_positive / actual_positive if actual_positive > 0 else 0.0
        f1_sum += 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"accuracy": accuracy, "macro_f1": f1_sum / 3.0, "brier": brier}


def load_world_model(config, checkpoint_path, device: torch.device, action_dim: int = 1):
    """Build a WorldModel from a resolved config and load a checkpoint into it.

    The checkpoint stores only weights, never the config, so the caller must pass a
    config whose architecture flags (Mamba3.is_mimo, RegimeFiLM.Enabled) match the run
    that produced the checkpoint. A strict key check fails fast on any mismatch instead
    of silently dropping FiLM or MIMO weights the way a bare strict=False load would.

    action_dim defaults to 1 to match Phase-A pretraining (train.py builds the world model
    with action_dim=1); it feeds the sequence-model stem width, so a wrong value is a hard
    size-mismatch at load time.
    """
    from finmamba3.models.world_model import WorldModel
    wm = WorldModel(action_dim=action_dim, config=config, device=device).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get("world_model", state.get("state_dict", state))
    incompatible = wm.load_state_dict(sd, strict=False)
    assert not incompatible.missing_keys, (
        f"checkpoint is missing model keys, so the rebuilt architecture does not match it "
        f"(check is_mimo and the RegimeFiLM flag): {incompatible.missing_keys[:8]}"
    )
    assert not incompatible.unexpected_keys, (
        f"checkpoint has unexpected keys for the rebuilt architecture "
        f"(check is_mimo and the RegimeFiLM flag): {incompatible.unexpected_keys[:8]}"
    )
    wm.eval()
    return wm


def world_model_direction_probs(
    wm, val_seq, threshold: float, label_mode: str, tb_horizon: int,
    level_width: int, device: torch.device,
    windows_per_market: int = 64, window_len: int = 64, rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Direction-head class probabilities and matching labels for one market sequence.

    Samples fixed random windows and reads the per-position direction head, so the
    compare_direction world-model arm and the cross-regime gap evaluator score on
    identical methodology. Returns the valid-masked (probs, labels) so a caller can
    pool several markets before computing one metric.
    """
    flat = val_seq.to_flat()
    T = flat.shape[0]
    L = window_len
    if T < L + 1:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(rng_seed)
    starts = rng.integers(0, T - L, size=min(windows_per_market, T - L))
    windows = np.stack([flat[s : s + L] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], L), dtype=torch.float32, device=device)
    # Match training autocast so the MIMO TileLang kernel uses its bf16 (not FP32) MMA path.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.straight_through_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        if wm.model in ("Transformer", "TransformerModern"):
            from finmamba3.models.attention import get_subsequent_mask_with_batch_length
            mask = get_subsequent_mask_with_batch_length(L, flattened_sample.device)
            dist_feat = wm.sequence_model(flattened_sample, action, mask)
        else:
            dist_feat = wm.sequence_model(flattened_sample, action)
        dist_feat = wm.condition_dist_feat(dist_feat)
        direction_logits = wm.direction_head(dist_feat[:, :-1])
        direction_probs = torch.softmax(direction_logits.float(), dim=-1)
    labels_global, valid_global = _global_labels(val_seq, threshold, label_mode, tb_horizon, level_width)
    label_windows = [labels_global[s : s + L - 1] for s in starts]
    valid_windows = [valid_global[s : s + L - 1] for s in starts]
    labels_flat = np.stack(label_windows, axis=0).reshape(-1)
    valid_flat = np.stack(valid_windows, axis=0).reshape(-1)
    probs_np = direction_probs.cpu().numpy().reshape(-1, 3)
    return probs_np[valid_flat], labels_flat[valid_flat]


def world_model_settlement_logloss(
    wm, val_seq, device: torch.device,
    windows_per_market: int = 256, window_len: int = 64, rng_seed: int = 0,
) -> tuple[float, int]:
    """Summed binary cross-entropy and count of the settlement head over one market (lower is better).

    Mirrors the direction-probs forward but reads the settlement head, whose per-position logit predicts
    the market's terminal YES/NO outcome. Pooling (sum_bce, count) before dividing gives the exact
    position-weighted log-loss over a regime so longer markets are not over-weighted. Positions whose
    yes_outcome is unresolved (nan) are masked out. Requires the settlement head and a resolved market.
    """
    assert wm.use_settlement_head, "checkpoint has no settlement head; train with Settlement.Enabled=True."
    assert val_seq.yes_outcome is not None, "market has no yes_outcome; settlement log-loss is undefined."
    flat = val_seq.to_flat()
    T = flat.shape[0]
    L = window_len
    if T < L + 1:
        return 0.0, 0
    rng = np.random.default_rng(rng_seed)
    starts = rng.integers(0, T - L, size=min(windows_per_market, T - L))
    windows = np.stack([flat[s : s + L] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], L), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.straight_through_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        if wm.model in ("Transformer", "TransformerModern"):
            from finmamba3.models.attention import get_subsequent_mask_with_batch_length
            mask = get_subsequent_mask_with_batch_length(L, flattened_sample.device)
            dist_feat = wm.sequence_model(flattened_sample, action, mask)
        else:
            dist_feat = wm.sequence_model(flattened_sample, action)
        dist_feat = wm.condition_dist_feat(dist_feat)
        settle_logits = wm.settlement_head(dist_feat).float()
    outcome_windows = np.stack([val_seq.yes_outcome[s : s + L] for s in starts], axis=0)
    outcome = torch.from_numpy(outcome_windows).float().to(device)
    finite = torch.isfinite(outcome)
    safe_outcome = torch.where(finite, outcome, torch.zeros_like(outcome))
    per = F.binary_cross_entropy_with_logits(settle_logits, safe_outcome, reduction="none")
    masked = per * finite.float()
    return float(masked.sum().item()), int(finite.sum().item())


def world_model_prediction_mse(
    wm, val_seq, device: torch.device,
    windows_per_market: int = 256, window_len: int = 64, rng_seed: int = 0,
) -> tuple[float, int]:
    """Summed squared error and element count of the one-step-ahead decoder prediction.

    Encodes windows, runs the sequence model (which applies the regime-FiLM modulation for the
    treatment arm), decodes the prior-predicted next latent, and compares it to the actual next
    observation. This is the decoder NLL on the predicted next tick under a Gaussian (mse)
    decoder, i.e. the regime-sensitive reconstruction term that actually exercises FiLM (the
    posterior reconstruction barely does). Returning (sse, count) lets a caller pool markets exactly.
    """
    assert wm.decoder_kind != "studentt", (
        "prediction_mse assumes a Gaussian (mse) decoder; a studentt decoder returns (mean, log_scale)."
    )
    flat = val_seq.to_flat()
    T = flat.shape[0]
    L = window_len
    if T < L + 1:
        return 0.0, 0
    rng = np.random.default_rng(rng_seed)
    starts = rng.integers(0, T - L, size=min(windows_per_market, T - L))
    windows = np.stack([flat[s : s + L] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], L), dtype=torch.float32, device=device)
    # Match training autocast so the MIMO TileLang kernel uses its bf16 (not FP32) MMA path.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.straight_through_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        if wm.model in ("Transformer", "TransformerModern"):
            from finmamba3.models.attention import get_subsequent_mask_with_batch_length
            mask = get_subsequent_mask_with_batch_length(L, flattened_sample.device)
            dist_feat = wm.sequence_model(flattened_sample, action, mask)
        else:
            dist_feat = wm.sequence_model(flattened_sample, action)
        prior_logits = wm.dist_head.forward_prior(dist_feat[:, :-1])
        prior_sample = wm.straight_through_gradient(prior_logits, sample_mode="probs")
        prior_flat = wm.flatten_sample(prior_sample)
        next_hat = wm.obs_decoder(prior_flat).float()
    target_next = obs[:, 1:].float()
    sse = float(((next_hat - target_next) ** 2).sum().item())
    count = int(target_next.numel())
    return sse, count


def world_model_prediction_nll(
    wm, val_seq, device: torch.device, feature_indices: torch.Tensor,
    windows_per_market: int = 256, window_len: int = 64, rng_seed: int = 0,
) -> tuple[float, int]:
    """Summed Student-t NLL and element count of the one-step-ahead decoder prediction.

    The Gaussian prediction_mse cannot express "this regime is more volatile" because an MSE
    decoder has no scale parameter, so the regime-FiLM modulation had nothing to earn under it.
    A Student-t decoder emits a per-feature (mean, log_scale); its NLL on the prior-predicted next
    tick rewards a correctly widened scale in the high-vol regime, which is the regime-dependent
    quantity this campaign tests FiLM on. The forward is identical to world_model_prediction_mse up
    to prior_flat; only the decode and the likelihood differ. feature_indices restricts the NLL to a
    channel subset (price/scale or volume) so the same checkpoints answer both the volatility and the
    order-flow-magnitude question. Returning (sum_nll, count) lets a caller pool windows exactly.

    A FeedObsVol checkpoint conditioned its router on the obs-midprice realized volatility during
    training; the eval forward must hand the router the identical feature or the FiLM modulation
    diverges from training and the gap is meaningless. The likelihood is computed in float32 outside
    autocast so the Student-t lgamma/log terms keep full precision; bf16 would quantize the scale and
    bias the per-regime NLL comparison.
    """
    assert wm.decoder_kind == "studentt", (
        "prediction_nll assumes a Student-t decoder returning (mean, log_scale); a Gaussian decoder "
        "returns a single tensor and has no scale to score."
    )
    flat = val_seq.to_flat()
    T = flat.shape[0]
    L = window_len
    if T < L + 1:
        return 0.0, 0
    rng = np.random.default_rng(rng_seed)
    starts = rng.integers(0, T - L, size=min(windows_per_market, T - L))
    windows = np.stack([flat[s : s + L] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], L), dtype=torch.float32, device=device)
    # Match training autocast so the MIMO TileLang kernel uses its bf16 (not FP32) MMA path.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        sample = wm.straight_through_gradient(post_logits)
        flattened_sample = wm.flatten_sample(sample)
        if wm.model in ("Transformer", "TransformerModern"):
            from finmamba3.models.attention import get_subsequent_mask_with_batch_length
            mask = get_subsequent_mask_with_batch_length(L, flattened_sample.device)
            dist_feat = wm.sequence_model(flattened_sample, action, mask)
        else:
            # Hand a FeedObsVol router the identical realized-vol feature it was trained on (see docstring).
            regime_vol = None
            if wm.use_regime_film and wm.regime_film_feed_obs_vol:
                from finmamba3.models.regime_modulation import realized_vol_conditioning_feature
                regime_vol = realized_vol_conditioning_feature(obs[..., wm.midprice_index], wm.regime_film_vol_window)
            dist_feat = wm.sequence_model(flattened_sample, action, regime_vol=regime_vol)
        prior_logits = wm.dist_head.forward_prior(dist_feat[:, :-1])
        prior_sample = wm.straight_through_gradient(prior_logits, sample_mode="probs")
        prior_flat = wm.flatten_sample(prior_sample)
        mean, log_scale = wm.obs_decoder(prior_flat)
    target_next = obs[:, 1:].float()
    # Float32 outside autocast keeps the Student-t lgamma/log terms at full precision (see docstring).
    log_prob = wm.obs_decoder.log_prob(mean.float(), log_scale.float(), target_next)
    selected = log_prob.index_select(-1, feature_indices.to(log_prob.device))
    sum_nll = float((-selected).sum().item())
    count = int(selected.numel())
    return sum_nll, count


def _evaluate_world_model(args, threshold: float, val_seq, device: torch.device) -> dict[str, float]:
    """Run the world-model encoder + direction head on the val sequence."""
    import yaml
    from finmamba3.config import DotDict, parse_args_and_update_config
    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg = DotDict(parse_args_and_update_config(cfg_raw, argv=[]))
    wm = load_world_model(cfg, args.world_checkpoint, device)
    if not wm.use_direction_head:
        logger.warning("World model has no direction head; world_model arm will be skipped.")
        return _nan_metrics()
    probs, labels = world_model_direction_probs(wm, val_seq, threshold, args.label_mode, args.tb_horizon, 10 * 8, device)
    return classification_metrics(probs, labels)


def _train_eval_deeplob(args, threshold: float, train_seq, val_seq, device: torch.device) -> dict[str, float]:
    """Train DeepLOB on the train flat features and evaluate on val."""
    from finmamba3.baselines.deeplob import DeepLOB
    from finmamba3.envs.lob_features import F_LEVEL, K_LEVELS
    flat_train = train_seq.to_flat()
    flat_val = val_seq.to_flat()
    LEVEL_WIDTH = K_LEVELS * F_LEVEL
    train_per_level = flat_train[:, :LEVEL_WIDTH]
    val_per_level = flat_val[:, :LEVEL_WIDTH]
    L = 32
    def _windows(arr, L):
        starts = np.arange(0, arr.shape[0] - L)
        return np.stack([arr[s : s + L] for s in starts], axis=0)
    Xtr = _windows(train_per_level, L)
    Xva = _windows(val_per_level, L)
    if Xtr.shape[0] == 0 or Xva.shape[0] == 0:
        return _nan_metrics()
    train_labels, _train_valid = _global_labels(train_seq, threshold, args.label_mode, args.tb_horizon, LEVEL_WIDTH)
    val_labels, val_valid = _global_labels(val_seq, threshold, args.label_mode, args.tb_horizon, LEVEL_WIDTH)
    ytr = np.stack([train_labels[s : s + L - 1] for s in range(Xtr.shape[0])], axis=0)
    yva = np.stack([val_labels[s : s + L - 1] for s in range(Xva.shape[0])], axis=0)
    yva_valid = np.stack([val_valid[s : s + L - 1] for s in range(Xva.shape[0])], axis=0)
    model = DeepLOB(k_levels=K_LEVELS, f_level=F_LEVEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_deeplob)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    model.train()
    for _ in range(args.epochs_deeplob):
        idx = torch.randperm(Xtr_t.shape[0], device=device)
        for start in range(0, Xtr_t.shape[0], args.batch_deeplob):
            b = idx[start : start + args.batch_deeplob]
            logits = model(Xtr_t[b])
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 3), ytr_t[b, : L - 1].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(Xva).float().to(device)
        logits = model(Xva_t)
        probs = torch.softmax(logits[:, :-1].float(), dim=-1).cpu().numpy()
    valid_flat = yva_valid.reshape(-1)
    return classification_metrics(probs.reshape(-1, 3)[valid_flat], yva.reshape(-1)[valid_flat])


def _fit_eval_linear_ar(threshold: float, train_seq, val_seq) -> dict[str, float]:
    """Fit LinearAR on train and evaluate direction labels on val.

    LinearAR is a closed-form floor with its own next-tick labelling, so it ignores
    --label-mode. It emits hard labels, not probabilities; a deterministic one-hot is a
    fair Brier proxy and lets accuracy / macro-F1 share the common metric path.
    """
    from finmamba3.baselines.linear_ar import LinearAR, LinearARConfig
    flat_train = train_seq.to_flat().astype(np.float32)
    flat_val = val_seq.to_flat().astype(np.float32)
    cfg = LinearARConfig(lookback=16, threshold=threshold, midprice_index=80)
    ar = LinearAR(cfg)
    ar.fit(flat_train)
    pred, actual = ar.direction_labels(flat_val)
    if pred.size == 0:
        return _nan_metrics()
    return classification_metrics(np.eye(3)[pred], actual)


def _format_table(rows: list[dict]) -> str:
    """Render the rows as a markdown table. macro_f1 is the paper-comparable column."""
    head = "| method | threshold | accuracy | macro_f1 | brier |\n|---|---:|---:|---:|---:|"
    lines = [head]
    for r in rows:
        lines.append(f"| {r['method']} | {r['threshold']:.4f} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['brier']:.4f} |")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    methods = [m.strip() for m in args.baselines.split(",") if m.strip()]
    import yaml
    from finmamba3.config import DotDict, parse_args_and_update_config
    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)
    from finmamba3.sequence_builder import build_sequences
    norm_clip = cfg.BasicSettings.get("NormClip", 8.0)
    aggregate_only = cfg.Models.WorldModel.Encoder.get("AggregateOnly", False)
    # Eval sequences must reuse the training BinaryMarketFeatures flag or the encoder linear width mismatches.
    include_binary_features = cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    train_seq, slug, _stats = build_sequences(
        args.data_train, args.market_slug, args.hours_train, args.norm_path,
        fit_stats=True, norm_clip=norm_clip, aggregate_only=aggregate_only,
        include_binary_features=include_binary_features,
    )
    val_seq, _, _ = build_sequences(
        args.data_val, slug, args.hours_val, args.norm_path,
        fit_stats=False, norm_clip=norm_clip, aggregate_only=aggregate_only,
        include_binary_features=include_binary_features,
    )
    logger.info(f"train ticks: {train_seq.per_tick.shape[0]}, val ticks: {val_seq.per_tick.shape[0]}")
    rows = []
    for thr in thresholds:
        if "linear_ar" in methods:
            metrics = _fit_eval_linear_ar(thr, train_seq, val_seq)
            rows.append({"method": "linear_ar", "threshold": thr, **metrics})
            logger.info(f"linear_ar threshold={thr}: {metrics}")
        if "deeplob" in methods:
            metrics = _train_eval_deeplob(args, thr, train_seq, val_seq, device)
            rows.append({"method": "deeplob", "threshold": thr, **metrics})
            logger.info(f"deeplob   threshold={thr}: {metrics}")
        if "world_model" in methods:
            if args.world_checkpoint is None:
                logger.warning("world_model in --baselines but --world-checkpoint missing; skipping")
            else:
                metrics = _evaluate_world_model(args, thr, val_seq, device)
                rows.append({"method": "world_model", "threshold": thr, **metrics})
                logger.info(f"world_model threshold={thr}: {metrics}")
    table = _format_table(rows)
    print(table)
    with open(args.out, "w") as f:
        f.write(table + "\n")
    logger.info(f"wrote {args.out}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
