"""Compare next-tick direction prediction across the world model and baselines.

Loads a pretrained world-model checkpoint, fits the LinearAR baseline, trains
a fresh DeepLOB on the train split, and evaluates all three on the val split
across one or more direction thresholds. Emits a markdown comparison table.

The world-model arm is optional (skip with --baselines linear_ar,deeplob)
which lets you run the script CPU-only when no checkpoint is available.

Example
-------
    python -m eval.compare_direction \\
        --world-checkpoint saved_models/lob/LOB/1iemugot/ckpt/world_model.pth \\
        --config src/config_files/configure_lob.yaml \\
        --data-train data/train --data-val data/validation \\
        --thresholds 0.001,0.005,0.01 \\
        --baselines world_model,deeplob,linear_ar
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare LOB direction prediction across methods")
    p.add_argument("--world-checkpoint", type=Path, default=None,
                   help="Path to world_model.pth; required if 'world_model' is in --baselines")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-train", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", type=Path,
                   default=SRC_DIR.parent / "saved_models" / "lob" / "normalization.json")
    p.add_argument("--market-slug", default=None)
    p.add_argument("--hours-train", type=float, default=6.0)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--thresholds", default="0.001,0.005,0.01",
                   help="Comma-separated direction thresholds")
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


def _accuracy_brier(pred_probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute top-1 accuracy and three-class Brier score."""
    if pred_probs.ndim != 2 or pred_probs.shape[1] != 3:
        raise ValueError(f"pred_probs must be (N, 3); got {pred_probs.shape}")
    pred_labels = pred_probs.argmax(axis=-1)
    accuracy = float((pred_labels == labels).mean())
    one_hot = np.eye(3)[labels]
    brier = float(((pred_probs - one_hot) ** 2).sum(axis=-1).mean())
    return accuracy, brier


def _evaluate_world_model(args, threshold: float, val_seq, device: torch.device) -> dict[str, float]:
    """Run the world-model encoder + direction head on the val sequence."""
    import yaml
    from config_utils import DotDict, parse_args_and_update_config
    from sub_models.world_models import WorldModel

    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)
    wm = WorldModel(action_dim=13, config=cfg, device=device).to(device)
    state = torch.load(args.world_checkpoint, map_location=device, weights_only=False)
    sd = state.get("world_model", state.get("state_dict", state))
    wm.load_state_dict(sd, strict=False)
    wm.eval()
    if not getattr(wm, "use_direction_head", False):
        logger.warning("World model has no direction head; world_model arm will be skipped.")
        return {"accuracy": float("nan"), "brier": float("nan")}

    flat = val_seq.to_flat()
    T = flat.shape[0]
    L = 64
    if T < L + 1:
        return {"accuracy": float("nan"), "brier": float("nan")}
    rng = np.random.default_rng(0)
    starts = rng.integers(0, T - L, size=min(64, T - L))
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
        direction_logits = wm.direction_head(dist_feat[:, :-1])
        direction_probs = torch.softmax(direction_logits.float(), dim=-1)

    LEVEL_FLAT = 10 * 8
    mid_norm = obs[..., LEVEL_FLAT + 0]
    labels = []
    for s in starts:
        labels.append(_label_directions(flat[s : s + L, LEVEL_FLAT + 0], threshold))
    labels_np = np.stack(labels, axis=0)
    probs_np = direction_probs.cpu().numpy().reshape(-1, 3)
    labels_flat = labels_np.reshape(-1)
    accuracy, brier = _accuracy_brier(probs_np, labels_flat)
    return {"accuracy": accuracy, "brier": brier}


def _train_eval_deeplob(args, threshold: float, train_seq, val_seq, device: torch.device) -> dict[str, float]:
    """Train DeepLOB on the train flat features and evaluate on val."""
    from baselines.deeplob import DeepLOB
    from envs.lob_features import F_LEVEL, K_LEVELS

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
        return {"accuracy": float("nan"), "brier": float("nan")}
    train_mid = flat_train[:, LEVEL_WIDTH + 0]
    val_mid = flat_val[:, LEVEL_WIDTH + 0]
    ytr = np.stack(
        [_label_directions(train_mid[s : s + L], threshold) for s in range(Xtr.shape[0])], axis=0
    )
    yva = np.stack(
        [_label_directions(val_mid[s : s + L], threshold) for s in range(Xva.shape[0])], axis=0
    )

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
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 3), ytr_t[b, : L - 1].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(Xva).float().to(device)
        logits = model(Xva_t)
        probs = torch.softmax(logits[:, :-1].float(), dim=-1).cpu().numpy()
    accuracy, brier = _accuracy_brier(probs.reshape(-1, 3), yva[:, : L - 1].reshape(-1))
    return {"accuracy": accuracy, "brier": brier}


def _fit_eval_linear_ar(threshold: float, train_seq, val_seq) -> dict[str, float]:
    """Fit LinearAR on train and evaluate direction labels on val."""
    from baselines.linear_ar import LinearAR, LinearARConfig

    flat_train = train_seq.to_flat().astype(np.float32)
    flat_val = val_seq.to_flat().astype(np.float32)
    cfg = LinearARConfig(lookback=16, threshold=threshold, midprice_index=80)
    ar = LinearAR(cfg)
    ar.fit(flat_train)
    pred, actual = ar.direction_labels(flat_val)
    if pred.size == 0:
        return {"accuracy": float("nan"), "brier": float("nan")}
    accuracy = float((pred == actual).mean())
    # LinearAR does not emit class probabilities; fall back to a deterministic one-hot
    # for the Brier score, which is a fair proxy for a deterministic predictor.
    one_hot_pred = np.eye(3)[pred]
    one_hot_actual = np.eye(3)[actual]
    brier = float(((one_hot_pred - one_hot_actual) ** 2).sum(axis=-1).mean())
    return {"accuracy": accuracy, "brier": brier}


def _format_table(rows: list[dict]) -> str:
    """Render the rows as a markdown table sorted by method then threshold."""
    head = "| method | threshold | accuracy | brier |\n|---|---:|---:|---:|"
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['threshold']:.4f} | {r['accuracy']:.4f} | {r['brier']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    methods = [m.strip() for m in args.baselines.split(",") if m.strip()]

    import yaml
    from config_utils import DotDict, parse_args_and_update_config
    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)

    from train_lob import build_sequences
    norm_clip = getattr(cfg.BasicSettings, "NormClip", 8.0)
    aggregate_only = getattr(cfg.Models.WorldModel.Encoder, "AggregateOnly", False)
    train_seq, slug, _stats = build_sequences(
        args.data_train, args.market_slug, args.hours_train, args.norm_path,
        fit_stats=True, norm_clip=norm_clip, aggregate_only=aggregate_only,
    )
    val_seq, _, _ = build_sequences(
        args.data_val, slug, args.hours_val, args.norm_path,
        fit_stats=False, norm_clip=norm_clip, aggregate_only=aggregate_only,
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
