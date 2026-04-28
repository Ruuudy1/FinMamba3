import math
import sys
import types
import unittest
from pathlib import Path

import torch
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _install_fake_mamba_modules():
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]

    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = []
    modules = types.ModuleType("mamba_ssm.modules")
    modules.__path__ = []

    class FakeMambaBlock(torch.nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.proj = torch.nn.Linear(d_model, d_model)

        def forward(self, x, **kwargs):
            return self.proj(x)

    mamba3 = types.ModuleType("mamba_ssm.modules.mamba3")
    mamba3.Mamba3 = FakeMambaBlock
    mamba2 = types.ModuleType("mamba_ssm.modules.mamba2")
    mamba2.Mamba2 = FakeMambaBlock

    sys.modules["mamba_ssm"] = pkg
    sys.modules["mamba_ssm.modules"] = modules
    sys.modules["mamba_ssm.modules.mamba3"] = mamba3
    sys.modules["mamba_ssm.modules.mamba2"] = mamba2


def _small_config(backbone="Mamba3"):
    with open(SRC / "config_files" / "configure_lob.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["BasicSettings"]["Device"] = "cpu"
    config["BasicSettings"]["Use_amp"] = False
    config["BasicSettings"]["Use_cg"] = False
    config["JointTrainAgent"]["BatchLength"] = 4
    config["JointTrainAgent"]["ImagineContextLength"] = 2
    config["JointTrainAgent"]["ImagineBatchLength"] = 2
    config["JointTrainAgent"]["RealityContextLength"] = 2
    config["JointTrainAgent"]["SaveEverySteps"] = 10

    wm = config["Models"]["WorldModel"]
    wm["dtype"] = torch.float32
    wm["Backbone"] = backbone
    wm["HiddenStateDim"] = 32
    wm["CategoricalDim"] = 4
    wm["ClassDim"] = 4
    wm["Optimiser"] = "Adam"
    wm["Dropout"] = 0.0
    wm["Warmup_steps"] = 1
    wm["Max_grad_norm"] = 10
    wm["Encoder"]["DModel"] = 32
    wm["Encoder"]["NumLayers"] = 1
    wm["Encoder"]["NumHeads"] = 4
    wm["Encoder"]["DimFeedforward"] = 64
    wm["Encoder"]["OutputFlattenDim"] = 64
    wm["Decoder"]["HiddenDim"] = 32
    wm["Decoder"]["NumLayers"] = 1
    wm["Reward"]["HiddenUnits"] = 32
    wm["Termination"]["HiddenUnits"] = 32
    wm["Mamba"]["n_layer"] = 1
    wm["Mamba"]["ssm_cfg"]["d_state"] = 4
    wm["Mamba3"]["n_layer"] = 1
    wm["Mamba3"]["d_state"] = 8
    wm["Mamba3"]["headdim"] = 16
    wm["Mamba3"]["chunk_size"] = 4
    # Disable optional auxiliary heads for backbone unit tests; they are
    # exercised end-to-end in the integration smoke test.
    wm.setdefault("Direction", {})["Enabled"] = False

    from config_utils import DotDict

    return DotDict(config)


class WorldModelMambaBackboneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pytorch_warmup  # noqa: F401
        except ModuleNotFoundError:
            warmup = types.ModuleType("pytorch_warmup")

            class LinearWarmup:
                def __init__(self, optimizer, warmup_period):
                    self.optimizer = optimizer
                    self.warmup_period = warmup_period

                def dampen(self):
                    return None

            warmup.LinearWarmup = LinearWarmup
            sys.modules["pytorch_warmup"] = warmup
        _install_fake_mamba_modules()

    def test_mamba3_mimo_sequence_shape_and_update(self):
        from sub_models.world_models import WorldModel

        config = _small_config("Mamba3")
        model = WorldModel(action_dim=1, config=config, device=torch.device("cpu"))
        self.assertEqual(model.model, "Mamba3")
        self.assertTrue(model.sequence_model.layers[0].kwargs["is_mimo"])

        latent = torch.randn(2, 4, model.stoch_flattened_dim)
        action = torch.zeros(2, 4)
        out = model.sequence_model(latent, action)
        self.assertEqual(tuple(out.shape), (2, 4, config.Models.WorldModel.HiddenStateDim))

        obs = torch.randn(2, 4, config.BasicSettings.FeatureDim)
        reward = torch.zeros(2, 4)
        termination = torch.zeros(2, 4)
        losses = model.update(obs, action, reward, termination, 0, 0)
        # update() now returns detached tensors so callers can defer GPU-CPU sync.
        self.assertTrue(all(torch.isfinite(v).item() for v in losses))

    def test_mamba2_fallback_constructs(self):
        from sub_models.world_models import WorldModel

        config = _small_config("Mamba2")
        model = WorldModel(action_dim=1, config=config, device=torch.device("cpu"))
        self.assertEqual(model.model, "Mamba2")
        latent = torch.randn(1, 3, model.stoch_flattened_dim)
        action = torch.zeros(1, 3)
        out = model.sequence_model(latent, action)
        self.assertEqual(tuple(out.shape), (1, 3, config.Models.WorldModel.HiddenStateDim))

    def test_episodic_memory_path_runs_finite(self):
        from sub_models.world_models import WorldModel

        config = _small_config("Mamba3")
        em = config.Models.WorldModel.EpisodicMemory
        em.Enabled = True
        em.Capacity = 64
        em.TopK = 2
        em.MinFillBeforeRetrieve = 0
        em.RetrieveEvery = 1
        model = WorldModel(action_dim=1, config=config, device=torch.device("cpu"))
        self.assertTrue(model.use_episodic_memory)
        obs = torch.randn(2, 4, config.BasicSettings.FeatureDim)
        action = torch.zeros(2, 4)
        reward = torch.zeros(2, 4)
        termination = torch.zeros(2, 4)
        # First call populates memory; second call should retrieve and fuse.
        model.update(obs, action, reward, termination, 0, 0)
        self.assertGreater(len(model.episodic_memory), 0)
        losses = model.update(obs, action, reward, termination, 1, 0)
        self.assertTrue(all(torch.isfinite(v).item() for v in losses))

    def test_direction_head_path_runs_finite(self):
        from sub_models.world_models import WorldModel

        config = _small_config("Mamba3")
        config.Models.WorldModel.Direction.Enabled = True
        config.Models.WorldModel.Direction.NumClasses = 3
        config.Models.WorldModel.Direction.LossWeight = 0.5
        config.Models.WorldModel.Direction.Threshold = 0.01
        config.Models.WorldModel.Direction.Dropout = 0.0
        model = WorldModel(action_dim=1, config=config, device=torch.device("cpu"))
        self.assertTrue(model.use_direction_head)
        obs = torch.randn(2, 4, config.BasicSettings.FeatureDim)
        action = torch.zeros(2, 4)
        reward = torch.zeros(2, 4)
        termination = torch.zeros(2, 4)
        losses = model.update(obs, action, reward, termination, 0, 0)
        self.assertEqual(len(losses), 9)  # 8 base losses + direction_loss
        self.assertTrue(all(torch.isfinite(v).item() for v in losses))


if __name__ == "__main__":
    unittest.main()
