# region imports
import sys
import unittest
from pathlib import Path
import torch
from finmamba3.models.regime_modulation import (  # noqa: E402
    RegimeFiLMModulator,
    causal_efficiency_ratio,
    causal_realized_vol,
    efficiency_ratio_bucket_labels,
    realized_vol_bucket_labels,
    realized_vol_conditioning_feature,
    regime_assignment_entropy,
    regime_load_balance_loss,
    regime_per_sample_entropy,
    regime_supervision_loss,
)
# endregion


class EfficiencyRatioSupervisionTest(unittest.TestCase):

    def test_causal_efficiency_ratio_trend_versus_zigzag(self):
        # A full-window trend has net == path so ER == 1; a zigzag cancels so ER ~ 0. Causal trailing.
        trend = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        zigzag = torch.tensor([[float(i % 2) for i in range(16)]])
        self.assertGreater(float(causal_efficiency_ratio(trend, window=8)[0, -1]), 0.99)
        self.assertLess(float(causal_efficiency_ratio(zigzag, window=8)[0, -1]), 0.2)

    def test_efficiency_ratio_buckets_rank_trend_above_random_walk(self):
        generator = torch.Generator().manual_seed(0)
        trend = torch.cumsum(torch.ones(1, 64), dim=1)
        random_walk = torch.cumsum(torch.randn(1, 64, generator=generator), dim=1)
        batch = torch.cat([trend, random_walk], dim=0)
        labels = efficiency_ratio_bucket_labels(batch, window=16, num_buckets=4)
        self.assertEqual(labels.shape, (2, 64))
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLess(int(labels.max()), 4)
        # The trending row averages a higher predictability bucket than the random-walk row.
        self.assertGreater(float(labels[0].float().mean()), float(labels[1].float().mean()))


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
        self.assertLess(float(regime_load_balance_loss(uniform_logits)), float(regime_load_balance_loss(peaked_logits)))
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

    def test_volatility_conditioning_widens_the_router_input_and_runs_self_contained(self):
        torch.manual_seed(0)
        modulator = RegimeFiLMModulator(16, 2, num_regimes=4, embed_dim=8, condition_on_vol=True, vol_window=3)
        # Conditioning appends a one-wide volatility proxy, so the router head reads hidden_dim + 1.
        self.assertEqual(modulator.regime_head.logits.in_features, 17)
        hidden = torch.randn(2, 5, 16)
        # The modulator derives the proxy from the summary itself, so the forward needs no extra input.
        gammas, betas, regime_logits = modulator(hidden)
        self.assertEqual(gammas.shape, (2, 5, 2, 16))
        self.assertEqual(regime_logits.shape, (2, 5, 4))
        vol = modulator._causal_latent_volatility(hidden)
        self.assertEqual(vol.shape, (2, 5, 1))
        # The first step has no prior, so its causal speed and thus its volatility proxy are zero.
        torch.testing.assert_close(vol[:, 0], torch.zeros_like(vol[:, 0]))

    def test_positive_init_scale_breaks_the_identity(self):
        torch.manual_seed(0)
        modulator = RegimeFiLMModulator(16, 2, num_regimes=4, embed_dim=8, init_scale=0.05)
        hidden = torch.randn(2, 5, 16)
        gammas, betas, _ = modulator(hidden)
        # A non-zero hypernetwork init means FiLM no longer reproduces gamma=1, beta=0 at step zero.
        self.assertFalse(torch.allclose(gammas, torch.ones_like(gammas)))
        self.assertFalse(torch.allclose(betas, torch.zeros_like(betas)))

    def test_assignment_entropy_peaks_for_uniform_and_collapses_for_peaked(self):
        uniform_logits = torch.zeros(4, 6, 8)
        peaked_logits = torch.zeros(4, 6, 8)
        peaked_logits[..., 0] = 12.0
        uniform_entropy = float(regime_assignment_entropy(uniform_logits))
        peaked_entropy = float(regime_assignment_entropy(peaked_logits))
        self.assertGreater(uniform_entropy, peaked_entropy)
        # A flat assignment over eight regimes sits at the log(8) entropy ceiling.
        self.assertAlmostEqual(uniform_entropy, float(torch.log(torch.tensor(8.0))), places=4)


class RegimeSupervisionTest(unittest.TestCase):

    def test_vol_bucket_labels_shape_and_range(self):
        torch.manual_seed(0)
        mid = torch.randn(3, 40)
        labels = realized_vol_bucket_labels(mid, window=4, num_buckets=4)
        self.assertEqual(labels.shape, (3, 40))
        self.assertEqual(labels.dtype, torch.long)
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLessEqual(int(labels.max()), 3)

    def test_vol_bucket_labels_rank_high_vol_above_low_vol(self):
        # A flat-then-choppy midprice must bucket the choppy tail above the flat head.
        flat = torch.zeros(1, 32)
        choppy = torch.zeros(1, 32)
        # Alternating jumps give the tail a large rolling std.
        choppy[0, ::2] = 5.0
        mid = torch.cat([flat, choppy], dim=1)
        labels = realized_vol_bucket_labels(mid, window=4, num_buckets=4)
        # The choppy second half sits in a strictly higher mean bucket than the flat first half.
        self.assertGreater(float(labels[0, 40:].float().mean()), float(labels[0, :28].float().mean()))

    def test_supervision_loss_lower_when_logits_match_labels(self):
        labels = torch.tensor([[0, 1, 2, 3]])
        matched = torch.zeros(1, 4, 4)
        for i in range(4):
            matched[0, i, i] = 12.0
        mismatched = torch.zeros(1, 4, 4)
        for i in range(4):
            mismatched[0, i, (i + 1) % 4] = 12.0
        self.assertLess(float(regime_supervision_loss(matched, labels)), float(regime_supervision_loss(mismatched, labels)))

    def test_conditioning_feature_is_monotonic_in_volatility(self):
        # Within one batch the feature must increase with realized volatility so a linear router head can bucket it into the supervised vol regimes, since the choppy tail reads higher than the flat head and the two regions share a batch so the batch standardization has variance to normalize.
        mid = torch.zeros(1, 64)
        # Alternating jumps make the tail high-vol, the head low-vol.
        mid[0, 32:][::2] = 5.0
        feat = realized_vol_conditioning_feature(mid, window=4)
        self.assertEqual(feat.shape, (1, 64, 1))
        self.assertGreater(float(feat[0, 36:].mean()), float(feat[0, :28].mean()))

    def test_conditioning_feature_is_rank_preserving_in_volatility(self):
        # The standardized feature stays an affine, rank-preserving transform of the causal realized vol the supervision label buckets, so feeding it gives the router exactly that axis.
        torch.manual_seed(0)
        mid = torch.randn(2, 40)
        vol = causal_realized_vol(mid, window=4)
        feat = realized_vol_conditioning_feature(mid, window=4).squeeze(-1)
        self.assertTrue(torch.equal(feat.reshape(-1).argsort(), vol.reshape(-1).argsort()))

    def test_external_vol_lets_supervision_discriminate_better_than_the_latent_proxy(self):
        # Head-to-head: a supervised router reading the external obs-vol feature must reach a strictly lower per-sample entropy than the same router restricted to its latent-speed proxy, since the external feature is the very axis the supervision label buckets; the proxy router stays near the uniform ceiling (the live failure mode) while the fed router commits to regimes.
        def _train(use_external):
            torch.manual_seed(0)
            mod = RegimeFiLMModulator(
                16, 2, num_regimes=4, embed_dim=8, condition_on_vol=True, vol_window=4, init_scale=0.03,
            )
            opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
            gen = torch.Generator().manual_seed(1)
            for _ in range(300):
                hidden = torch.randn(8, 16, 16, generator=gen)
                mid = torch.randn(8, 16, generator=gen)
                vol = causal_realized_vol(mid, 4)
                q = torch.quantile(vol.reshape(-1), torch.linspace(0, 1, 5)[1:-1])
                labels = torch.bucketize(vol, q)
                external = torch.log1p(vol).unsqueeze(-1) if use_external else None
                _, _, regime_logits = mod(hidden, external_vol=external)
                loss = regime_supervision_loss(regime_logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                _, _, regime_logits = mod(hidden, external_vol=external)
                return float(regime_per_sample_entropy(regime_logits))
        fed_entropy = _train(use_external=True)
        proxy_entropy = _train(use_external=False)
        self.assertLess(fed_entropy, proxy_entropy)
        # Below the ln(4)=1.386 uniform ceiling (weakly, in this noisy synthetic; the real latent discriminates far more).
        self.assertLess(fed_entropy, 1.34)

    def test_decouple_router_isolates_router_head_from_film_gradient(self):
        # Decoupling detaches the router logits on the path into the hypernetwork, so a FiLM-only loss leaves the router head untouched (no recon gradient can drag it to uniform) while the hypernetwork that emits FiLM still trains.
        torch.manual_seed(0)
        mod = RegimeFiLMModulator(16, 2, num_regimes=4, embed_dim=8, init_scale=0.03, decouple_router=True)
        hidden = torch.randn(2, 5, 16)
        gammas, betas, _ = mod(hidden)
        (gammas.pow(2).mean() + betas.pow(2).mean()).backward()
        head_grad = mod.regime_head.logits.weight.grad
        self.assertTrue(head_grad is None or float(head_grad.abs().sum()) == 0.0)
        self.assertGreater(float(mod.hyper.weight.grad.abs().sum()), 0.0)

    def test_coupled_router_passes_film_gradient_into_the_router_head(self):
        # Control: without decoupling, the FiLM loss does reach the router head, since that is the very coupling that lets reconstruction pin the router uniform.
        torch.manual_seed(0)
        mod = RegimeFiLMModulator(16, 2, num_regimes=4, embed_dim=8, init_scale=0.03, decouple_router=False)
        hidden = torch.randn(2, 5, 16)
        gammas, betas, _ = mod(hidden)
        (gammas.pow(2).mean() + betas.pow(2).mean()).backward()
        self.assertGreater(float(mod.regime_head.logits.weight.grad.abs().sum()), 0.0)

    def test_per_sample_entropy_separates_balanced_confident_from_degenerate_uniform(self):
        # Each step peaked at a different regime: batch-mean is ~uniform (assignment entropy high) but every step is confident, so per-sample entropy is ~0; this is exactly the distinction regime supervision targets, which the batch-mean diagnostic alone cannot see.
        confident = torch.full((1, 4, 4), -12.0)
        for i in range(4):
            confident[0, i, i] = 12.0
        uniform = torch.zeros(1, 4, 4)
        self.assertAlmostEqual(
            float(regime_assignment_entropy(confident)), float(regime_assignment_entropy(uniform)), places=3
        )
        self.assertLess(float(regime_per_sample_entropy(confident)), 0.1)
        self.assertGreater(float(regime_per_sample_entropy(uniform)), 1.3)
if __name__ == "__main__":
    unittest.main()
