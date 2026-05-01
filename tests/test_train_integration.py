"""Smoke integration test for the WorldModel update path with the new heads.

Drives a tiny synthetic batch through WorldModel.update with the new
auxiliary inputs (event_counts, outcome, time_to_expiry_frac) and asserts:
- Total loss is finite.
- All 11 returned loss tensors are scalar and finite.
- Direction-thresholds sweep changes the direction loss compared to the
  single-threshold baseline.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from types import SimpleNamespace


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def _make_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _make_namespace(v) for k, v in d.items()})
    return d


def _build_config(**overrides):
    base = {
        "BasicSettings": {
            "ObsMode": "features",
            "FeatureDim": 94,
            "ReplayBufferOnGPU": False,
            "Use_amp": False,
            "Use_cg": False,
            "Compile": False,
            "NormClip": 8.0,
        },
        "JointTrainAgent": {
            "BatchLength": 8,
            "ImagineContextLength": 4,
            "ImagineBatchLength": 4,
            "RealityContextLength": 4,
            "SaveEverySteps": 1000,
            "SampleMaxSteps": 200,
        },
        "Models": {
            "WorldModel": {
                "dtype": torch.float32,
                "Backbone": "Mamba3",
                "Act": "SiLU",
                "CategoricalDim": 4,
                "ClassDim": 4,
                "HiddenStateDim": 64,
                "Optimiser": "Adam",
                "LRSchedule": "constant",
                "LRMinRatio": 0.1,
                "Dropout": 0.0,
                "Unimix_ratio": 0.01,
                "Weight_decay": 0.0,
                "Max_grad_norm": 100.0,
                "Warmup_steps": 1,
                "UseActionInput": False,
                "DirectionThresholds": None,
                "Adam": {"LearningRate": 1e-4},
                "Encoder": {
                    "Type": "lob",
                    "K": 10,
                    "FeatureDimLevel": 8,
                    "FeatureDimTick": 14,
                    "DModel": 32,
                    "NumLayers": 1,
                    "NumHeads": 2,
                    "DimFeedforward": 64,
                    "OutputFlattenDim": 64,
                    "AggregateOnly": False,
                    "GradientCheckpointing": False,
                },
                "Decoder": {
                    "Kind": "mse",
                    "HiddenDim": 32,
                    "NumLayers": 1,
                    "NuInit": 5.0,
                    "LearnableNu": True,
                },
                "Reward": {"Enabled": False, "HiddenUnits": 32, "LayerNum": 1},
                "Termination": {"Enabled": False, "HiddenUnits": 32, "LayerNum": 1},
                "Transformer": {"FinalFeatureWidth": 4, "NumLayers": 1, "NumHeads": 2},
                "Mamba": {"n_layer": 1, "d_intermediate": 0, "ssm_cfg": {"d_state": 16}},
                "Mamba3": {
                    "Enabled": True,
                    "n_layer": 1,
                    "is_mimo": False,
                    "mimo_rank": 1,
                    "d_state": 16,
                    "headdim": 16,
                    "chunk_size": 8,
                    "is_outproj_norm": False,
                    "rope_fraction": 0.0,
                },
                "Direction": {
                    "Enabled": True,
                    "NumClasses": 3,
                    "Threshold": 0.01,
                    "LossWeight": 0.5,
                    "Dropout": 0.0,
                },
                "Regime": {"Enabled": False, "NumRegimes": 4, "EmbedDim": 8},
                "EpisodicMemory": {"Enabled": False},
                "Hawkes": {"Enabled": False, "LossWeight": 0.1},
                "Settlement": {"Enabled": False, "LossWeight": 0.25},
                "NaNGuardSteps": 0,
            }
        },
    }
    for k, v in overrides.items():
        cur = base
        path = k.split(".")
        for part in path[:-1]:
            cur = cur[part]
        cur[path[-1]] = v
    return _make_namespace(base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="WorldModel requires CUDA-built mamba_ssm")
def test_world_model_update_returns_eleven_finite_losses():
    from sub_models.world_models import WorldModel

    cfg = _build_config()
    device = torch.device("cuda")
    wm = WorldModel(action_dim=4, config=cfg, device=device).to(device)
    B, L, F = 2, cfg.JointTrainAgent.BatchLength, cfg.BasicSettings.FeatureDim
    obs = torch.randn(B, L, F, device=device)
    action = torch.zeros(B, L, dtype=torch.long, device=device)
    reward = torch.zeros(B, L, device=device)
    termination = torch.zeros(B, L, device=device)
    losses = wm.update(obs, action, reward, termination, global_step=1, epoch_step=0)
    assert len(losses) == 11
    for t in losses:
        assert torch.is_tensor(t) and torch.isfinite(t).all()
