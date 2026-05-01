"""Auxiliary LOB heads for regime-aware world-model experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        mid_norm shape: (B, L). Returns (targets, mask) both (B, L-1) — the
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
        """Per-step Poisson negative log-likelihood, mean-reduced."""
        intensity = log_intensity.exp()
        nll = intensity - counts * log_intensity
        return nll.mean()


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
        """
        if time_to_expiry_frac is None:
            return F.binary_cross_entropy_with_logits(logits, outcome.to(logits.dtype))
        weight = (1.0 - time_to_expiry_frac.clamp(min=0.0, max=1.0)).to(logits.dtype)
        per = F.binary_cross_entropy_with_logits(
            logits, outcome.to(logits.dtype), reduction="none"
        )
        return (per * weight).mean()


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
