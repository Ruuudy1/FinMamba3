# region imports
from pathlib import Path

import numpy as np
import torch

from finmamba3.envs.fi2010_loader import load_fi2010_split
from finmamba3.envs.lob_features import apply_normalization, load_normalization
from finmamba3.eval.compare_direction import load_world_model, world_model_direction_probs
from finmamba3.eval.eval_regime_generalization import _arch_overrides, _load_config
# endregion


# The macro-F1 came back bit-identical across four independent checkpoints, which can only happen if
# every direction head emits a near-constant class. This dumps the predicted-class histogram so the
# collapse is confirmed directly rather than inferred from the identical scores.
def main() -> int:
    config_path = Path("configs/fi2010.yaml")
    checkpoint = Path("saved_models/lob/LOB/z7wi0tu2/ckpt/world_model_final.pth")
    device = torch.device("cuda")
    cfg = _load_config(config_path, _arch_overrides(False, False, None))
    world_model = load_world_model(cfg, checkpoint, device)
    enc_cfg = cfg.Models.WorldModel.Encoder
    level_width = int(enc_cfg.K) * int(enc_cfg.FeatureDimLevel)
    bundle = load_fi2010_split(Path("data/fi2010/validation"), split="validation", horizon=10)
    stats = load_normalization(Path("saved_models/lob/fi2010_norm_real.json"))
    val_norm = apply_normalization(bundle.sequence, stats)
    probs, labels = world_model_direction_probs(
        world_model, val_norm, threshold=0.0, label_mode="next_tick", tb_horizon=10,
        level_width=level_width, device=device, windows_per_market=256, window_len=512,
    )
    pred = probs.argmax(axis=-1)
    print(f"n_scored={pred.shape[0]}")
    print(f"pred class hist  (down/flat/up) = {np.bincount(pred, minlength=3).tolist()}")
    print(f"label class hist (down/flat/up) = {np.bincount(labels, minlength=3).tolist()}")
    print(f"unique predicted classes = {np.unique(pred).tolist()}")
    print(f"max-softmax-prob mean = {probs.max(axis=-1).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
