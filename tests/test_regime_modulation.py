# region imports
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sub_models.regime_modulation import (  # noqa: E402
    RegimeFiLMModulator,
    regime_load_balance_loss,
)
# endregion


class RegimeFiLMModulatorTest(unittest.TestCase):
    def test_identity_at_init(self):
        torch.manual_seed(0)
        hidden_dim = 16
        n_layer = 3
        modulator = RegimeFiLMModulator(hidden_dim, n_layer, num_regimes=4, embed_dim=8)
        hidden = torch.randn(2, 5, hidden_dim)
        gammas, betas, regime_logits = modulator(hidden)
        self.assertEqual(gammas.shape, (2, 5, n_layer, hidden_dim))
        self.assertEqual(betas.shape, (2, 5, n_layer, hidden_dim))
        self.assertEqual(regime_logits.shape, (2, 5, 4))
        torch.testing.assert_close(gammas, torch.ones_like(gammas))
        torch.testing.assert_close(betas, torch.zeros_like(betas))
        block_input = torch.randn(2, 5, hidden_dim)
        for layer_index in range(n_layer):
            filmed = gammas[:, :, layer_index, :] * block_input + betas[:, :, layer_index, :]
            torch.testing.assert_close(filmed, block_input)
    def test_load_balance_loss_minimized_by_uniform(self):
        uniform_logits = torch.zeros(4, 6, 8)
        peaked_logits = torch.zeros(4, 6, 8)
        peaked_logits[..., 0] = 12.0
        self.assertLess(
            float(regime_load_balance_loss(uniform_logits)),
            float(regime_load_balance_loss(peaked_logits)),
        )
        self.assertGreaterEqual(float(regime_load_balance_loss(uniform_logits)), 0.0)
    def test_gradient_flows_through_zero_init_hyper(self):
        torch.manual_seed(0)
        modulator = RegimeFiLMModulator(16, 2, num_regimes=4, embed_dim=8)
        hidden = torch.randn(2, 5, 16)
        gammas, betas, _ = modulator(hidden)
        loss = ((gammas - 2.0) ** 2).mean() + (betas ** 2).mean()
        loss.backward()
        self.assertIsNotNone(modulator.hyper.weight.grad)
        self.assertGreater(float(modulator.hyper.weight.grad.abs().sum()), 0.0)
if __name__ == "__main__":
    unittest.main()
