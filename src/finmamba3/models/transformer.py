# region imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from finmamba3.models.attention import get_vector_mask
from finmamba3.models.attention import (
    PositionalEncoding1D, 
    AttentionBlock, 
    AttentionBlockKVCache,
    AttentionBlockPreNorm,
    AttentionBlockKVCachePreNorm
)
# endregion


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim
        # Combine the stochastic sample with the one-hot action embedding.
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.head = nn.Linear(feat_dim, stoch_dim)
    def forward(self, samples, action, mask):
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)
        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)
        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        # Combine the stochastic sample with the one-hot action embedding.
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)
    def forward(self, samples, action, mask):
        """Standard forward pass through the full sequence."""
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)
        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)
        return feats
    def reset_kv_cache_list(self, batch_size, dtype):
        """Clear and reinitialize the KV cache for a new episode."""
        param_example = next(iter(self.stem.parameters()))
        device = param_example.device
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))
    def forward_with_kv_cache(self, samples, action):
        """Autoregressive step using the KV cache stored in self.kv_cache_list."""
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+1, samples.device)
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        feats = self.layer_norm(feats)
        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)
        return feats


class StochasticTransformerModern(nn.Module):
    """Modern transformer with pre-norm architecture matching Mamba3's design.
    
    This class uses:
    - Pre-norm architecture (RMSNorm before each layer)
    - SiLU activation in FFN
    - Simpler stem (matching Mamba3)
    - Final RMSNorm (matching Mamba3)
    - 4x FFN expansion ratio (modern standard)
    
    This provides a fair comparison to Mamba3 by matching all architectural
    choices except the sequence model (attention vs SSM).
    """
    
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout, 
                 use_action_input=True, device=None, dtype=None):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.use_action_input = bool(use_action_input)
        
        factory = {"device": device, "dtype": dtype}
        
        # Simple stem matching Mamba3: Linear + RMSNorm + SiLU
        stem_in = stoch_dim + action_dim if self.use_action_input else stoch_dim
        self.stem = nn.Sequential(
            nn.Linear(stem_in, feat_dim, bias=True, **factory),
            nn.RMSNorm(feat_dim, **factory),
            nn.SiLU(),
        )
        
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        
        # Pre-norm transformer layers with 4x FFN expansion (modern standard)
        self.layer_stack = nn.ModuleList([
            AttentionBlockPreNorm(
                feat_dim=feat_dim, 
                hidden_dim=feat_dim * 4,  # 4x expansion like modern transformers
                num_heads=num_heads, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final norm matching Mamba3
        self.norm_f = nn.RMSNorm(feat_dim, **factory)
    
    def forward(self, samples, action, mask):
        """Standard forward pass through the full sequence."""
        if self.use_action_input:
            action = F.one_hot(action.long(), self.action_dim).to(dtype=samples.dtype)
            feats = self.stem(torch.cat([samples, action], dim=-1))
        else:
            feats = self.stem(samples)
        
        feats = self.position_encoding(feats)
        
        for layer in self.layer_stack:
            feats, attn = layer(feats, mask)
        
        output = self.norm_f(feats)
        return output


class StochasticTransformerModernKVCache(nn.Module):
    """Modern transformer with KV cache support for autoregressive generation.
    
    Matches StochasticTransformerModern but adds KV cache for efficient inference.
    """
    
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout,
                 use_action_input=True, device=None, dtype=None):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.use_action_input = bool(use_action_input)
        
        factory = {"device": device, "dtype": dtype}
        
        # Simple stem matching Mamba3: Linear + RMSNorm + SiLU
        stem_in = stoch_dim + action_dim if self.use_action_input else stoch_dim
        self.stem = nn.Sequential(
            nn.Linear(stem_in, feat_dim, bias=True, **factory),
            nn.RMSNorm(feat_dim, **factory),
            nn.SiLU(),
        )
        
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        
        # Pre-norm transformer layers with KV cache support
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCachePreNorm(
                feat_dim=feat_dim,
                hidden_dim=feat_dim * 4,  # 4x expansion like modern transformers
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final norm matching Mamba3
        self.norm_f = nn.RMSNorm(feat_dim, **factory)
    
    def forward(self, samples, action, mask):
        """Standard forward pass through the full sequence."""
        if self.use_action_input:
            action = F.one_hot(action.long(), self.action_dim).to(dtype=samples.dtype)
            feats = self.stem(torch.cat([samples, action], dim=-1))
        else:
            feats = self.stem(samples)
        
        feats = self.position_encoding(feats)
        
        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)
        
        output = self.norm_f(feats)
        return output
    
    def reset_kv_cache_list(self, batch_size, dtype):
        """Clear and reinitialize the KV cache for a new episode."""
        param_example = next(iter(self.stem.parameters()))
        device = param_example.device
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=device))
    
    def forward_with_kv_cache(self, samples, action):
        """Autoregressive step using the KV cache stored in self.kv_cache_list."""
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1]+1, samples.device)
        
        if self.use_action_input:
            action = F.one_hot(action.long(), self.action_dim).to(dtype=samples.dtype)
            feats = self.stem(torch.cat([samples, action], dim=-1))
        else:
            feats = self.stem(samples)
        
        feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
        
        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)
        
        output = self.norm_f(feats)
        return output
