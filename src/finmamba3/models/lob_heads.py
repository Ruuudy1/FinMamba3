"""Auxiliary LOB heads for regime-aware world-model experiments."""
# region imports
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
# endregion


class DirectionHead(nn.Module):
    """Three-class head over next-tick midprice direction (down / flat / up).

    Forces the Mamba hidden state to encode predictive information about price
    movement, not just reconstructive information about the current tick. The
    target sign is derived inline from the normalized midprice channel of the
    obs vector, so no replay-buffer change is required.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 3,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2, **factory)
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim // 2, num_classes, **factory)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(self.dropout(hidden)))
        return self.head(h)
    @staticmethod
    def make_targets(
        mid_norm: torch.Tensor, threshold: float = 1.0e-2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Three-class targets from the normalized midprice tensor.

        mid_norm shape: (B, L). Returns (targets, mask) both (B, L-1). The
        target at position t corresponds to the change from tick t to t+1, so
        the head must use hidden_state[:, :-1] to predict it.
        Class 0 = down, 1 = flat, 2 = up. Mask is always True; threshold
        controls how aggressively small moves get bucketed as 'flat'.
        """
        dmid = mid_norm[:, 1:] - mid_norm[:, :-1]
        targets = torch.full_like(dmid, fill_value=1, dtype=torch.long)
        targets = torch.where(dmid > threshold, torch.full_like(targets, 2), targets)
        targets = torch.where(dmid < -threshold, torch.full_like(targets, 0), targets)
        mask = torch.ones_like(targets, dtype=torch.bool)
        return targets, mask


class RegimeHead(nn.Module):
    """Categorical regime head with a soft regime embedding."""

    def __init__(
        self,
        hidden_dim: int,
        num_regimes: int = 8,
        embed_dim: int = 32,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.logits = nn.Linear(hidden_dim, num_regimes, **factory)
        self.embedding = nn.Embedding(num_regimes, embed_dim, **factory)
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(hidden)
        probs = torch.softmax(logits, dim=-1)
        emb = probs @ self.embedding.weight
        return logits, emb


class RegimeConditioner(nn.Module):
    """Fuse Mamba hidden state with a learned regime embedding."""

    def __init__(
        self,
        hidden_dim: int,
        regime_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.proj = nn.Linear(hidden_dim + regime_dim, hidden_dim, **factory)
        self.gate = nn.Linear(hidden_dim + regime_dim, hidden_dim, **factory)
    def forward(self, hidden: torch.Tensor, regime_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, regime_emb], dim=-1)
        gate = torch.sigmoid(self.gate(x))
        return hidden + gate * torch.tanh(self.proj(x))


@dataclass
class MemoryBatch:
    values: torch.Tensor
    weights: torch.Tensor


class EpisodicMemory:
    """Small CPU-side top-k memory for hidden-state context retrieval.

    By default writes are FIFO (every observation is appended, oldest evicted).
    The optional `novelty_threshold` argument turns the write policy into a
    novelty filter: only entries whose novelty score exceeds the threshold
    are written. This is the learned-write-policy variant that differentiates
    the regime catalog from a sliding window of recent states.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        capacity: int = 50_000,
        novelty_threshold: float = 0.0,
    ) -> None:
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.capacity = int(capacity)
        self.novelty_threshold = float(novelty_threshold)
        self.keys = torch.empty((0, self.key_dim), dtype=torch.float32)
        self.values = torch.empty((0, self.value_dim), dtype=torch.float32)
    def __len__(self) -> int:
        return int(self.keys.shape[0])
    def add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        novelty: torch.Tensor | None = None,
    ) -> int:
        """Append entries, optionally filtered by per-entry novelty score.

        Returns the number of entries actually written.
        """
        keys_cpu = keys.detach().float().cpu().reshape(-1, self.key_dim)
        values_cpu = values.detach().float().cpu().reshape(-1, self.value_dim)
        if novelty is not None and self.novelty_threshold > 0.0:
            mask = (novelty.detach().float().cpu().reshape(-1) >= self.novelty_threshold)
            keys_cpu = keys_cpu[mask]
            values_cpu = values_cpu[mask]
        if keys_cpu.shape[0] == 0:
            return 0
        self.keys = torch.cat([self.keys, keys_cpu], dim=0)[-self.capacity :]
        self.values = torch.cat([self.values, values_cpu], dim=0)[-self.capacity :]
        return int(keys_cpu.shape[0])
    def retrieve(self, query: torch.Tensor, k: int = 4) -> MemoryBatch | None:
        if self.keys.numel() == 0:
            return None
        flat = query.detach().float().cpu().reshape(-1, self.key_dim)
        q = torch.nn.functional.normalize(flat, dim=-1)
        keys = torch.nn.functional.normalize(self.keys, dim=-1)
        scores = q @ keys.T
        top_scores, top_idx = torch.topk(scores, k=min(k, self.keys.shape[0]), dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        values = self.values[top_idx]
        fused = (values * weights.unsqueeze(-1)).sum(dim=1)
        return MemoryBatch(
            values=fused.reshape(*query.shape[:-1], self.value_dim).to(query.device),
            weights=weights.reshape(*query.shape[:-1], -1).to(query.device),
        )


class HawkesIntensityHead(nn.Module):
    """Predicts log-intensity for buy and sell event arrivals.

    Maps the conditioned Mamba hidden state to log-intensities lambda_buy and
    lambda_sell. Trained via a Poisson negative log-likelihood on observed
    event counts in a forward window. Adds order-arrival microstructure as
    an auxiliary self-supervised signal alongside the existing direction head.
    """

    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2, **factory)
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim // 2, 2, **factory)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(hidden))
        return self.head(h)
    @staticmethod
    def poisson_nll(
        log_intensity: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Per-step Poisson negative log-likelihood, masked and mean-reduced."""
        mask = torch.isfinite(counts).to(log_intensity.dtype)
        safe_counts = torch.where(torch.isfinite(counts), counts, torch.zeros_like(counts))
        intensity = log_intensity.exp()
        nll = intensity - safe_counts * log_intensity
        return (nll * mask).sum() / mask.sum().clamp(min=1.0)


class SettlementHead(nn.Module):
    """Predicts the binary settlement outcome of a Polymarket contract.

    Polymarket markets resolve to YES (1) or NO (0) at expiry. Conditioning
    the world model on the settlement outcome via an auxiliary head forces
    the latent to encode information that survives to terminal payoff,
    rather than just locally reconstructive features.
    """

    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2, **factory)
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim // 2, 1, **factory)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(hidden))
        return self.head(h).squeeze(-1)
    @staticmethod
    def bce(
        logits: torch.Tensor,
        outcome: torch.Tensor,
        time_to_expiry_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Binary cross-entropy, optionally weighted by closeness to expiry.

        time_to_expiry_frac in [0, 1]: 1 at start, 0 at expiry. Weighting by
        (1 - frac) puts more pressure near expiry where the binary outcome
        becomes most predictable.

        NaN entries in `outcome` (unresolved markets) are masked out: they
        contribute zero to the loss and zero to the denominator, so a batch
        with no resolved markets returns 0 without producing NaN gradients.
        When all outcomes are finite the result equals the unmasked mean.
        """
        outcome = outcome.to(logits.dtype)
        # mask.sum() drives a single device-side reduction per step; the cost is amortized over the rest of the loss compute.
        mask = torch.isfinite(outcome).to(logits.dtype)
        safe_outcome = torch.where(torch.isnan(outcome), torch.zeros_like(outcome), outcome)
        per = F.binary_cross_entropy_with_logits(logits, safe_outcome, reduction="none")
        if time_to_expiry_frac is None:
            return (per * mask).sum() / mask.sum().clamp(min=1.0)
        weight = (1.0 - time_to_expiry_frac.clamp(min=0.0, max=1.0)).to(logits.dtype)
        return (per * weight * mask).sum() / mask.sum().clamp(min=1.0)
    @staticmethod
    def spot_sign_bce(
        logits: torch.Tensor,
        spot_signed_distance: torch.Tensor,
        time_to_expiry_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Auxiliary BCE toward the observable running spot sign 1[spot >= open] (Phase 0.3).

        The contract settles on exactly this rule at expiry, so supervising the running sign
        early gives the head a dense, learnable target and keeps the causal spot sign salient
        in the latent. Weighting by tte_frac makes it dominate near the open, complementary to
        the realized-outcome term that bce() concentrates near expiry. NaN distances (a tick
        with no spot path) are masked out so they contribute neither loss nor denominator.
        """
        spot = spot_signed_distance.to(logits.dtype)
        target = (spot >= 0.0).to(logits.dtype)
        mask = torch.isfinite(spot).to(logits.dtype)
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if time_to_expiry_frac is None:
            weight = torch.ones_like(spot)
        else:
            weight = time_to_expiry_frac.clamp(min=0.0, max=1.0).to(logits.dtype)
        return (per * weight * mask).sum() / mask.sum().clamp(min=1.0)


class BookRelativeEdgeHead(nn.Module):
    """Predicts expected value of buying YES and NO at the current ask prices.

    Targets:
      ev_yes = outcome_yes - yes_ask    (positive means BUY_YES is profitable)
      ev_no  = (1 - outcome_yes) - no_ask  (positive means BUY_NO is profitable)

    Rows where yes_ask, no_ask, or outcome is NaN, or where book_depth == 0, are masked
    from all EV losses. The action CE derives three-class labels from ground-truth EVs:
    BUY_YES (1) or BUY_NO (2) when the respective EV exceeds the threshold and dominates
    the other side; SIT (0) otherwise.
    """

    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2, **factory)
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim // 2, 2, **factory)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(hidden))
        return self.head(h)

    @staticmethod
    def ev_targets(
        outcome: torch.Tensor,
        yes_ask: torch.Tensor,
        no_ask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EV targets from ground-truth outcome and ask prices. All (B, L)."""
        ev_yes = outcome - yes_ask
        ev_no = (1.0 - outcome) - no_ask
        return ev_yes, ev_no

    @staticmethod
    def finite_mask(
        outcome: torch.Tensor,
        yes_ask: torch.Tensor,
        no_ask: torch.Tensor,
        book_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Boolean (B, L) mask: True where all supervision inputs are finite and depth > 0."""
        return (
            torch.isfinite(outcome)
            & torch.isfinite(yes_ask)
            & torch.isfinite(no_ask)
            & torch.isfinite(book_depth)
            & (book_depth > 0.0)
        )

    @staticmethod
    def action_labels(
        ev_yes: torch.Tensor,
        ev_no: torch.Tensor,
        threshold: float = 0.03,
    ) -> torch.Tensor:
        """Action class labels: 0=SIT, 1=BUY_YES, 2=BUY_NO."""
        buy_yes = (ev_yes >= threshold) & (ev_yes > ev_no)
        buy_no = (ev_no >= threshold) & (ev_no > ev_yes)
        labels = torch.zeros_like(ev_yes, dtype=torch.long)
        labels = torch.where(buy_yes, torch.ones_like(labels), labels)
        labels = torch.where(buy_no, torch.full_like(labels, 2), labels)
        return labels

    @staticmethod
    def huber_ev_loss(
        ev_hat: torch.Tensor,
        ev_yes: torch.Tensor,
        ev_no: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss on [ev_yes, ev_no] targets, masked. ev_hat shape (B, L, 2).

        NaN targets at masked positions are replaced with 0 before the loss call so they
        cannot produce NaN gradients via 0 * NaN, matching the SettlementHead.bce pattern.
        """
        safe_yes = torch.where(torch.isfinite(ev_yes), ev_yes, torch.zeros_like(ev_yes))
        safe_no = torch.where(torch.isfinite(ev_no), ev_no, torch.zeros_like(ev_no))
        targets = torch.stack([safe_yes, safe_no], dim=-1)
        per = F.huber_loss(ev_hat, targets, reduction="none")
        mask2 = mask.unsqueeze(-1).expand_as(per).to(per.dtype)
        return (per * mask2).sum() / mask2.sum().clamp(min=1.0)

    @staticmethod
    def action_ce_loss(
        ev_hat: torch.Tensor,
        ev_yes: torch.Tensor,
        ev_no: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.03,
    ) -> torch.Tensor:
        """CE on {SIT, BUY_YES, BUY_NO} from EV predictions and ground-truth labels.

        SIT logit is fixed at `threshold` so the model prefers YES/NO exactly when
        the predicted EV exceeds the threshold that defines the action boundary.
        """
        labels = BookRelativeEdgeHead.action_labels(ev_yes, ev_no, threshold)
        sit_logit = torch.full_like(ev_hat[..., :1], threshold)
        logits = torch.cat([sit_logit, ev_hat], dim=-1)
        flat_logits = logits[mask].reshape(-1, 3)
        flat_labels = labels[mask].reshape(-1)
        if flat_labels.numel() == 0:
            return torch.zeros((), device=ev_hat.device, dtype=ev_hat.dtype)
        return F.cross_entropy(flat_logits, flat_labels)


class EpisodicMemoryFuser(nn.Module):
    """Gated residual fusion of a retrieved memory value into the Mamba hidden state.

    Mirrors the RegimeConditioner pattern: the gate decides per-feature how much
    of the retrieved context to inject. The retrieved value has no gradients, but
    the gate and projection are learned.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.proj = nn.Linear(hidden_dim + memory_dim, hidden_dim, **factory)
        self.gate = nn.Linear(hidden_dim + memory_dim, hidden_dim, **factory)
    def forward(self, hidden: torch.Tensor, memory_value: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, memory_value], dim=-1)
        gate = torch.sigmoid(self.gate(x))
        return hidden + gate * torch.tanh(self.proj(x))
