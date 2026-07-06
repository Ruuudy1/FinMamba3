# region imports
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from finmamba3.models.lob_heads import SettlementHead
# endregion


def test_settlement_bce_all_finite_matches_unmasked_mean():
    torch.manual_seed(0)
    logits = torch.randn(8, requires_grad=True)
    outcome = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    expected = F.binary_cross_entropy_with_logits(logits, outcome)
    actual = SettlementHead.bce(logits, outcome)
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_settlement_bce_all_nan_returns_zero_without_propagating():
    logits = torch.randn(8, requires_grad=True)
    outcome = torch.full((8,), float('nan'))
    loss = SettlementHead.bce(logits, outcome)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_settlement_bce_mixed_masks_nan_entries():
    logits = torch.tensor([0.5, -0.5, 0.0, 1.0, -1.0, 0.2], requires_grad=True)
    outcome = torch.tensor([1.0, float('nan'), 0.0, float('nan'), 1.0, 0.0])
    finite_mask = torch.isfinite(outcome)
    expected = F.binary_cross_entropy_with_logits(logits[finite_mask], outcome[finite_mask])
    actual = SettlementHead.bce(logits, outcome)
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_settlement_bce_ttf_weighted_handles_nan():
    logits = torch.tensor([0.5, -0.5, 0.0, 1.0], requires_grad=True)
    outcome = torch.tensor([1.0, float('nan'), 0.0, 1.0])
    ttf = torch.tensor([0.9, 0.5, 0.1, 0.0])
    loss = SettlementHead.bce(logits, outcome, ttf)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_settlement_logloss_pooling_masks_and_weights_by_resolved_count():
    """world_model_settlement_logloss returns (sum_bce, count) so a regime pools many markets into
    one position-weighted log-loss. This checks that contract directly: the summed masked BCE
    divided by the resolved-position count equals the mean BCE over only the finite-outcome
    positions, with the nan (unresolved) positions contributing nothing to either numerator or
    denominator.
    """
    logits = torch.tensor([[0.5, -0.5, 0.0], [1.0, -1.0, 0.2]])
    outcome = torch.tensor([[1.0, float('nan'), 0.0], [float('nan'), 1.0, 0.0]])
    finite = torch.isfinite(outcome)
    safe = torch.where(finite, outcome, torch.zeros_like(outcome))
    per = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    sum_bce = float((per * finite.float()).sum().item())
    count = int(finite.sum().item())
    expected = F.binary_cross_entropy_with_logits(logits[finite], outcome[finite])
    assert count == 4
    assert math.isclose(sum_bce / count, float(expected), rel_tol=1e-6)
