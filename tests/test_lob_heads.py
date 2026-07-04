# region imports
import unittest
import torch
from finmamba3.models.lob_heads import BookRelativeEdgeHead, HawkesIntensityHead, SettlementHead
# endregion


class SettlementLossTest(unittest.TestCase):

    def test_bce_masks_nan_outcomes(self):
        # Three finite labels at logit 0 give per-element bce = log(2); the NaN tick drops out.
        logits = torch.zeros(4)
        outcome = torch.tensor([1.0, 0.0, float("nan"), 1.0])
        loss = SettlementHead.bce(logits, outcome)
        self.assertAlmostEqual(float(loss), 0.6931, places=3)
        self.assertTrue(torch.isfinite(loss))

    def test_tte_weighting_concentrates_on_late_ticks(self):
        # Phase 0.3: an early tick (frac=1) gets weight 0 and a late tick (frac=0) gets weight 1, so the realized-outcome supervision concentrates near expiry where the outcome is set.
        per = float(torch.log1p(torch.exp(torch.tensor(5.0))))
        logits = torch.tensor([5.0, 5.0])
        outcome = torch.tensor([0.0, 0.0])
        early_late = SettlementHead.bce(logits, outcome, torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(float(early_late), per / 2.0, places=3)
        all_late = SettlementHead.bce(logits, outcome, torch.tensor([0.0, 0.0]))
        self.assertAlmostEqual(float(all_late), per, places=3)

    def test_spot_sign_bce_targets_running_sign(self):
        # Positive distance targets 1 and negative targets 0; matching confident logits score near zero while reversed logits score large, so the aux genuinely supervises the observable sign.
        spot = torch.tensor([0.01, -0.01])
        aligned = SettlementHead.spot_sign_bce(torch.tensor([10.0, -10.0]), spot)
        reversed_loss = SettlementHead.spot_sign_bce(torch.tensor([-10.0, 10.0]), spot)
        self.assertLess(float(aligned), 0.01)
        self.assertGreater(float(reversed_loss), 5.0)

    def test_spot_sign_bce_masks_nan_and_weights_early(self):
        logits = torch.zeros(3)
        spot = torch.tensor([0.01, float("nan"), -0.01])
        loss = SettlementHead.spot_sign_bce(logits, spot)
        self.assertAlmostEqual(float(loss), 0.6931, places=3)
        # `tte_frac` weights the aux toward the start: a frac-0 (late) tick is fully down-weighted.
        early_only = SettlementHead.spot_sign_bce(torch.tensor([0.0, 0.0]), torch.tensor([0.01, -0.01]), torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(float(early_only), 0.6931 / 2.0, places=3)


class HawkesLossTest(unittest.TestCase):

    def test_poisson_nll_masks_nan_counts(self):
        log_intensity = torch.zeros(2, 2)
        counts = torch.tensor([[1.0, float("nan")], [0.0, 2.0]])
        loss = HawkesIntensityHead.poisson_nll(log_intensity, counts)
        self.assertAlmostEqual(float(loss), 1.0, places=3)
        self.assertTrue(torch.isfinite(loss))

    def test_poisson_nll_all_nan_returns_zero(self):
        log_intensity = torch.zeros(2, 2)
        counts = torch.full((2, 2), float("nan"))
        loss = HawkesIntensityHead.poisson_nll(log_intensity, counts)
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))


class BookRelativeEdgeHeadTest(unittest.TestCase):

    def setUp(self):
        self.head = BookRelativeEdgeHead(hidden_dim=32)

    def test_forward_output_shape(self):
        h = torch.randn(2, 10, 32)
        out = self.head(h)
        self.assertEqual(out.shape, (2, 10, 2))

    def test_ev_targets_computation(self):
        outcome = torch.tensor([[1.0, 0.0]])
        yes_ask = torch.tensor([[0.7, 0.3]])
        no_ask = torch.tensor([[0.3, 0.7]])
        ev_yes, ev_no = BookRelativeEdgeHead.ev_targets(outcome, yes_ask, no_ask)
        # Expected 1.0 - 0.7.
        self.assertAlmostEqual(float(ev_yes[0, 0]), 0.3, places=5)
        # Expected (1 - 1.0) - 0.3.
        self.assertAlmostEqual(float(ev_no[0, 0]), -0.3, places=5)
        # Expected 0.0 - 0.3.
        self.assertAlmostEqual(float(ev_yes[0, 1]), -0.3, places=5)
        # Expected (1 - 0.0) - 0.7.
        self.assertAlmostEqual(float(ev_no[0, 1]), 0.3, places=5)

    def test_finite_mask_excludes_nan_and_zero_depth(self):
        outcome = torch.tensor([[1.0, 1.0, float("nan"), 1.0]])
        yes_ask = torch.tensor([[0.7, float("nan"), 0.7, 0.7]])
        no_ask = torch.tensor([[0.3, 0.3, 0.3, 0.3]])
        book_depth = torch.tensor([[100.0, 100.0, 100.0, 0.0]])
        mask = BookRelativeEdgeHead.finite_mask(outcome, yes_ask, no_ask, book_depth)
        self.assertTrue(bool(mask[0, 0]))
        # NaN yes_ask must be excluded.
        self.assertFalse(bool(mask[0, 1]))
        # NaN outcome must be excluded.
        self.assertFalse(bool(mask[0, 2]))
        # Zero depth must be excluded.
        self.assertFalse(bool(mask[0, 3]))

    def test_action_labels_sit_buy_yes_buy_no(self):
        # `ev_yes`=0.10, ev_no=-0.10 gives BUY_YES; ev_yes=-0.10, ev_no=0.10 gives BUY_NO; both below threshold gives SIT.
        ev_yes = torch.tensor([[0.10, -0.10, 0.02]])
        ev_no = torch.tensor([[-0.10, 0.10, 0.01]])
        labels = BookRelativeEdgeHead.action_labels(ev_yes, ev_no, threshold=0.03)
        # Label 1 is BUY_YES.
        self.assertEqual(int(labels[0, 0]), 1)
        # Label 2 is BUY_NO.
        self.assertEqual(int(labels[0, 1]), 2)
        # Label 0 is SIT.
        self.assertEqual(int(labels[0, 2]), 0)

    def test_action_labels_tie_is_sit(self):
        # Equal EVs both above threshold: neither dominates, so SIT.
        ev_yes = torch.tensor([[0.10]])
        ev_no = torch.tensor([[0.10]])
        labels = BookRelativeEdgeHead.action_labels(ev_yes, ev_no, threshold=0.03)
        self.assertEqual(int(labels[0, 0]), 0)

    def test_huber_ev_loss_masked_and_finite(self):
        ev_hat = torch.zeros(1, 4, 2)
        ev_yes = torch.tensor([[0.1, 0.2, float("nan"), 0.1]])
        ev_no = torch.tensor([[-0.1, -0.2, -0.1, -0.1]])
        book_depth = torch.tensor([[100.0, 100.0, 100.0, 0.0]])
        outcome = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        yes_ask = torch.tensor([[0.9, 0.8, float("nan"), 0.9]])
        no_ask = torch.tensor([[1.1, 1.2, 1.1, 1.1]])
        mask = BookRelativeEdgeHead.finite_mask(outcome, yes_ask, no_ask, book_depth)
        loss = BookRelativeEdgeHead.huber_ev_loss(ev_hat, ev_yes, ev_no, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_action_ce_loss_all_masked_returns_zero(self):
        ev_hat = torch.zeros(1, 2, 2)
        ev_yes = torch.full((1, 2), float("nan"))
        ev_no = torch.full((1, 2), float("nan"))
        mask = torch.zeros(1, 2, dtype=torch.bool)
        loss = BookRelativeEdgeHead.action_ce_loss(ev_hat, ev_yes, ev_no, mask)
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_settlement_path_unaffected(self):
        # Existing settlement BCE must still work alongside the edge head (backward compat).
        logits = torch.zeros(4)
        outcome = torch.tensor([1.0, 0.0, float("nan"), 1.0])
        loss = SettlementHead.bce(logits, outcome)
        self.assertAlmostEqual(float(loss), 0.6931, places=3)


if __name__ == "__main__":
    unittest.main()
