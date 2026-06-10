"""Tests for the single-model backbone-metrics evaluator (G2).

The recon-NLL and direction-F1 scoring helpers it calls are covered in compare_direction / fi2010
tests, so here we lock the module imports and the dataset val-loading glue (the new surface): the
fi2010 branch must return a normalized 46-dim sequence built from the train normalization scale.
"""
# region imports
from __future__ import annotations
import types
from pathlib import Path
import numpy as np
from finmamba3.envs.fi2010_loader import FI2010_FEATURE_DIM
from finmamba3.envs.lob_features import fit_normalization, save_normalization
from finmamba3.eval.eval_backbone_metrics import _load_val_sequence
# endregion


def _write_fi2010_split(path: Path, n_events: int = 80, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.1, 10.0, size=(n_events, 149)).astype(np.float32)
    for col in range(144, 149):
        matrix[:, col] = rng.choice([1.0, 2.0, 3.0], size=n_events)
    np.savetxt(path, matrix, fmt="%.6f")


def test_load_val_sequence_fi2010_returns_normalized_46dim(tmp_path):
    from finmamba3.envs.fi2010_loader import load_fi2010_split
    _write_fi2010_split(tmp_path / "Train_Dst_NoAuction_DecPre_CF_7.txt", seed=1)
    _write_fi2010_split(tmp_path / "Val_Dst_NoAuction_DecPre_CF_7.txt", seed=2)
    train_bundle = load_fi2010_split(tmp_path, split="train", horizon=10)
    stats = fit_normalization(train_bundle.sequence, clip_value=8.0)
    norm_path = tmp_path / "norm.json"
    save_normalization(stats, norm_path)
    args = types.SimpleNamespace(data_val=tmp_path, horizon=10, max_events=None, norm_path=norm_path)
    val = _load_val_sequence("fi2010", None, args)
    flat = val.to_flat()
    assert flat.shape[1] == FI2010_FEATURE_DIM, "fi2010 val must be the 46-dim schema."
    assert np.isfinite(flat).all()
    assert float(np.abs(flat).max()) <= 8.0 + 1e-4, "val must be on the train normalization scale (within clip)."
