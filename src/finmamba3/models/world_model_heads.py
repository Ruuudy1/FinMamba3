# region imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from finmamba3.models.activations import ACTIVATION_BY_NAME
# endregion
RMSNorm = nn.RMSNorm


class DistHead(nn.Module):
    """Distribution head for posterior and prior categorical logits."""
    def __init__(self, image_feat_dim, hidden_state_dim, categorical_dim, class_dim, unimix_ratio=0.01, dtype=None, device=None) -> None:
        super().__init__()
        self.stoch_dim = categorical_dim
        self.post_head = nn.Linear(image_feat_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.prior_head = nn.Linear(hidden_state_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.unimix_ratio = unimix_ratio
        self.dtype=dtype
        self.device=device
    def unimix(self, logits, mixing_ratio=0.01):
        # Mix logits with uniform noise for uniform-prior regularization.
        if mixing_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            mixed_probs = mixing_ratio * torch.ones_like(probs, dtype=self.dtype, device=self.device) / self.stoch_dim + (1-mixing_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits
    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits
    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits


class RewardHead(nn.Module):
    def __init__(self, num_classes, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = ACTIVATION_BY_NAME[act]
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, num_classes, dtype=dtype, device=device)
    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationHead(nn.Module):
    def __init__(self, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = ACTIVATION_BY_NAME[act]
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)
    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # Remove the trailing singleton dimension.
        return termination
