# region imports
import unittest
import torch
from finmamba3.models.lob_heads import SettlementHead
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
        # Phase 0.3: an early tick (frac=1) gets weight 0 and a late tick (frac=0) gets weight 1,
        # so the realized-outcome supervision concentrates near expiry where the outcome is set.
        per = float(torch.log1p(torch.exp(torch.tensor(5.0))))
        logits = torch.tensor([5.0, 5.0])
        outcome = torch.tensor([0.0, 0.0])
        early_late = SettlementHead.bce(logits, outcome, torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(float(early_late), per / 2.0, places=3)
        all_late = SettlementHead.bce(logits, outcome, torch.tensor([0.0, 0.0]))
        self.assertAlmostEqual(float(all_late), per, places=3)
    def test_spot_sign_bce_targets_running_sign(self):
        # Positive distance targets 1 and negative targets 0; matching confident logits score near
        # zero while reversed logits score large, so the aux genuinely supervises the observable sign.
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
        # tte_frac weights the aux toward the start: a frac-0 (late) tick is fully down-weighted.
        early_only = SettlementHead.spot_sign_bce(
            torch.tensor([0.0, 0.0]), torch.tensor([0.01, -0.01]), torch.tensor([1.0, 0.0])
        )
        self.assertAlmostEqual(float(early_only), 0.6931 / 2.0, places=3)
if __name__ == "__main__":
    unittest.main()
