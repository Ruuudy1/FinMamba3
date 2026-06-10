"""Frozen-feature capacity probe: is the predictability (ER) regime linearly decodable from the
world-model representation? Disambiguates the near-chance joint-router result. A linear probe is trained
on the FROZEN baseline (FiLM-off) conditioned dist_feat to predict the per-step ER bucket, on a held-out
window split. High accuracy => the regime IS in the representation (so joint FiLM 'won't use it'); near
chance => the per-step ER regime is not linearly decodable from the latent.
"""
# region imports
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from finmamba3.eval.compare_direction import load_world_model
from finmamba3.eval.eval_regime_generalization import _arch_overrides, _device_from_arg, _load_config
from finmamba3.eval.eval_regime_generalization_fi2010 import _load_val_sequence
from finmamba3.models.regime_modulation import efficiency_ratio_bucket_labels, realized_vol_bucket_labels
# endregion


def parse_args():
    p = argparse.ArgumentParser(description="Linear-probe capacity test for the ER regime on frozen features")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", choices=("fi2010", "kaggle"), required=True)
    p.add_argument("--checkpoint", required=True, type=Path, help="A FiLM-off baseline checkpoint.")
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--num-windows", type=int, default=400)
    p.add_argument("--vol-window", type=int, default=16)
    p.add_argument("--num-buckets", type=int, default=4)
    p.add_argument("--bucket", choices=("predictability", "vol"), default="predictability",
                   help="Which regime to decode: the ER (predictability) bucket or the realized-vol bucket.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = _device_from_arg(args.device)
    cfg = _load_config(args.config, _arch_overrides(False, False, None))
    wm = load_world_model(cfg, args.checkpoint, device)
    flat = _load_val_sequence(args.dataset, cfg, args).to_flat()
    total, length = flat.shape[0], args.window_len
    rng = np.random.default_rng(0)
    starts = rng.integers(0, total - length, size=min(args.num_windows, total - length))
    windows = np.stack([flat[s : s + length] for s in starts], axis=0)
    obs = torch.from_numpy(windows).float().to(device)
    action = torch.zeros((obs.shape[0], length), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
        embedding = wm.encoder(obs)
        post_logits = wm.dist_head.forward_post(embedding)
        flattened_sample = wm.flatten_sample(wm.straight_through_gradient(post_logits))
        dist_feat = wm.condition_dist_feat(wm.sequence_model(flattened_sample, action))
    features = dist_feat.float().reshape(-1, dist_feat.shape[-1])
    bucket_fn = efficiency_ratio_bucket_labels if args.bucket == "predictability" else realized_vol_bucket_labels
    labels = bucket_fn(obs[..., wm.midprice_index], args.vol_window, args.num_buckets).reshape(-1)
    num = features.shape[0]
    perm = torch.randperm(num, generator=torch.Generator().manual_seed(0))
    cut = int(0.7 * num)
    train_idx, val_idx = perm[:cut].to(device), perm[cut:].to(device)
    probe = torch.nn.Linear(features.shape[1], args.num_buckets).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    xtr, ytr = features[train_idx], labels[train_idx]
    for _ in range(300):
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(xtr), ytr)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        val_pred = probe(features[val_idx]).argmax(dim=-1)
        val_acc = float((val_pred == labels[val_idx]).float().mean())
    chance = 1.0 / args.num_buckets
    bucket_hist = torch.bincount(labels, minlength=args.num_buckets).tolist()
    print(f"[{args.dataset}] frozen-feature {args.bucket}-bucket linear probe: val_acc={val_acc:.4f} "
          f"(chance={chance:.4f}) n={num} bucket_hist={bucket_hist}")


if __name__ == "__main__":
    main()
