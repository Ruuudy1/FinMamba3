"""FI-2010 macro-F1 benchmark against the LOBCAST paper (Prata et al. 2023).

Trains DeepLOB on the FI-2010 train split using the dataset's published k-horizon
trend labels (the paper's labelling) and reports macro-F1 on the validation split,
directly comparable to Table 2 of arXiv:2308.01915. A majority-class floor
contextualizes the score, and an optional frozen-world-model linear probe places
the FinMamba3 representation on the same axis without retraining the backbone.

FI-2010 attaches one trend label per market observation (the trend over the next k
events), so each window of L observations yields a single prediction at the window's
last tick, unlike the per-step Polymarket comparison in eval/compare_direction.

Example
-------
    python -m finmamba3.eval.benchmark_fi2010 \\
        --config configs/fi2010.yaml \\
        --data-train data/train --data-val data/validation \\
        --horizon 10 --baselines deeplob,majority --epochs-deeplob 20
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
from finmamba3.eval.compare_direction import classification_metrics
# endregion
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FI-2010 macro-F1 benchmark vs the LOBCAST paper")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-train", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=10, help="FI-2010 label horizon: one of 10, 20, 30, 50, 100.")
    p.add_argument("--window", type=int, default=100, help="Observations per input window; the paper uses 100.")
    p.add_argument("--baselines", default="deeplob,majority,linear_ar",
                   help="Comma-separated subset of {deeplob,majority,linear_ar,world_model}.")
    p.add_argument("--world-checkpoint", type=Path, default=None,
                   help="FI-2010-trained world_model.pth; required if 'world_model' is in --baselines.")
    p.add_argument("--epochs-deeplob", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--max-events", type=int, default=None, help="Cap events per split for a fast smoke run.")
    p.add_argument("--norm-path", type=Path, default=Path("saved_models/lob/fi2010_norm.json"))
    p.add_argument("--out", type=Path, default=Path("reports/fi2010_benchmark.md"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _windows_and_labels(per_level_norm: np.ndarray, labels: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    # Each window predicts the published trend label at its last observation, matching
    # the FI-2010 single-label-per-observation convention the paper evaluates.
    T = per_level_norm.shape[0]
    flat = per_level_norm.reshape(T, -1)
    starts = np.arange(0, T - window)
    windows = np.stack([flat[s : s + window] for s in starts], axis=0)
    window_labels = labels[starts + window - 1]
    return windows, window_labels


def _eval_majority(ytr: np.ndarray, yva: np.ndarray) -> dict[str, float]:
    # The trivial floor predicts the train-majority class everywhere. Its macro-F1 is the
    # bar a real model must clear; the gap to it is what the paper finds collapses on unseen data.
    majority_class = int(np.bincount(ytr, minlength=3).argmax())
    probs = np.zeros((yva.shape[0], 3), dtype=np.float64)
    probs[:, majority_class] = 1.0
    return classification_metrics(probs, yva)


def _train_eval_deeplob(args, Xtr, ytr, Xva, yva, k_levels, f_level, device) -> dict[str, float]:
    from finmamba3.baselines.deeplob import DeepLOB
    model = DeepLOB(k_levels=k_levels, f_level=f_level).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    model.train()
    for _ in range(args.epochs_deeplob):
        idx = torch.randperm(Xtr_t.shape[0], device=device)
        for start in range(0, Xtr_t.shape[0], args.batch):
            b = idx[start : start + args.batch]
            logits = model(Xtr_t[b])[:, -1]
            loss = F.cross_entropy(logits, ytr_t[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    probs_chunks = []
    with torch.no_grad():
        Xva_t = torch.from_numpy(Xva).float().to(device)
        for start in range(0, Xva_t.shape[0], args.batch):
            logits = model(Xva_t[start : start + args.batch])[:, -1]
            probs_chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return classification_metrics(np.concatenate(probs_chunks, axis=0), yva)


def _world_model_features(wm, windows: np.ndarray, batch: int, device) -> np.ndarray:
    # Frozen-representation probe: take the sequence-model output at each window's last
    # tick as the learned feature, so the benchmark scores representation quality rather
    # than the pretraining direction head (which targets next-tick, not horizon-k, trends).
    feature_chunks = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        for start in range(0, windows.shape[0], batch):
            obs = torch.from_numpy(windows[start : start + batch]).float().to(device)
            action = torch.zeros((obs.shape[0], obs.shape[1]), dtype=torch.float32, device=device)
            embedding = wm.encoder(obs)
            post_logits = wm.dist_head.forward_post(embedding)
            flattened_sample = wm.flatten_sample(wm.straight_through_gradient(post_logits))
            dist_feat = wm.sequence_model(flattened_sample, action)
            dist_feat = wm.condition_dist_feat(dist_feat)
            feature_chunks.append(dist_feat[:, -1].float().cpu().numpy())
    return np.concatenate(feature_chunks, axis=0)


def _train_eval_world_model_probe(args, Xtr, ytr, Xva, yva, device) -> dict[str, float]:
    import yaml
    from finmamba3.config import DotDict, parse_args_and_update_config
    from finmamba3.models.world_model import WorldModel
    with open(args.config, "r") as f:
        cfg = DotDict(parse_args_and_update_config(yaml.safe_load(f), argv=[]))
    wm = WorldModel(action_dim=1, config=cfg, device=device).to(device)
    state = torch.load(args.world_checkpoint, map_location=device, weights_only=False)
    wm.load_state_dict(state.get("world_model", state.get("state_dict", state)), strict=False)
    wm.eval()
    train_features = _world_model_features(wm, Xtr, args.batch, device)
    val_features = _world_model_features(wm, Xva, args.batch, device)
    # A single linear layer on frozen features is the standard representation probe.
    probe = torch.nn.Linear(train_features.shape[1], 3).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1.0e-3)
    Xtr_t = torch.from_numpy(train_features).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    probe.train()
    for _ in range(200):
        logits = probe(Xtr_t)
        loss = F.cross_entropy(logits, ytr_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        probs = torch.softmax(probe(torch.from_numpy(val_features).float().to(device)).float(), dim=-1).cpu().numpy()
    return classification_metrics(probs, yva)


def _eval_linear_ar(train_norm, val_norm, k_levels: int, f_level: int, threshold: float) -> dict[str, float]:
    """Closed-form linear-AR direction floor on the normalized flat features (next-tick target).

    Regresses the next-tick mid return on a flat lookback window and reads the sign, the simplest
    linear predictor a real sequence model must beat. It labels the next-tick direction by design (not
    the published horizon-k trend the DeepLOB rows use), so it is reported as a next-tick floor, not a
    same-target competitor; the midprice index is derived from the schema geometry.
    """
    from finmamba3.baselines.linear_ar import LinearAR, LinearARConfig
    midprice_index = k_levels * f_level
    ar = LinearAR(LinearARConfig(lookback=16, threshold=threshold, midprice_index=midprice_index))
    ar.fit(train_norm.to_flat().astype(np.float32))
    predicted, actual = ar.direction_labels(val_norm.to_flat().astype(np.float32))
    if predicted.size == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "brier": float("nan")}
    return classification_metrics(np.eye(3)[predicted], actual)


def _format_fi2010_table(rows: list[dict], horizon: int) -> str:
    head = (
        f"FI-2010 benchmark (horizon k={horizon}), macro-F1 comparable to LOBCAST Table 2.\n\n"
        "| method | accuracy | macro_f1 | brier |\n|---|---:|---:|---:|"
    )
    lines = [head]
    for r in rows:
        lines.append(f"| {r['method']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['brier']:.4f} |")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.baselines.split(",") if m.strip()]
    from finmamba3.envs.fi2010_loader import FI2010_F_LEVEL, FI2010_K_LEVELS, load_fi2010_split
    from finmamba3.envs.lob_features import apply_normalization, fit_normalization, save_normalization
    bundle_train = load_fi2010_split(args.data_train, split="train", horizon=args.horizon, max_events=args.max_events)
    bundle_val = load_fi2010_split(args.data_val, split="validation", horizon=args.horizon, max_events=args.max_events)
    stats = fit_normalization(bundle_train.sequence, clip_value=8.0)
    save_normalization(stats, args.norm_path)
    train_norm = apply_normalization(bundle_train.sequence, stats)
    val_norm = apply_normalization(bundle_val.sequence, stats)
    Xtr, ytr = _windows_and_labels(train_norm.per_level, bundle_train.direction_labels, args.window)
    Xva, yva = _windows_and_labels(val_norm.per_level, bundle_val.direction_labels, args.window)
    logger.info(
        f"fi2010 windows: train={Xtr.shape}, val={Xva.shape}, "
        f"val_label_hist={np.bincount(yva, minlength=3).tolist()}"
    )
    rows = []
    if "majority" in methods:
        rows.append({"method": "majority", **_eval_majority(ytr, yva)})
        logger.info(f"majority: {rows[-1]}")
    if "linear_ar" in methods:
        rows.append({"method": "linear_ar (next-tick floor)",
                     **_eval_linear_ar(train_norm, val_norm, FI2010_K_LEVELS, FI2010_F_LEVEL, 0.0)})
        logger.info(f"linear_ar: {rows[-1]}")
    if "deeplob" in methods:
        rows.append({"method": "deeplob", **_train_eval_deeplob(args, Xtr, ytr, Xva, yva, FI2010_K_LEVELS, FI2010_F_LEVEL, device)})
        logger.info(f"deeplob: {rows[-1]}")
    if "world_model" in methods:
        if args.world_checkpoint is None:
            logger.warning("world_model in --baselines but --world-checkpoint missing; skipping")
        else:
            rows.append({"method": "world_model_probe", **_train_eval_world_model_probe(args, Xtr, ytr, Xva, yva, device)})
            logger.info(f"world_model_probe: {rows[-1]}")
    table = _format_fi2010_table(rows, args.horizon)
    print(table)
    args.out.write_text(table + "\n")
    logger.info(f"wrote {args.out}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
