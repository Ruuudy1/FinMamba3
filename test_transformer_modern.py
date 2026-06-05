"""Test script to verify the modern transformer implementation."""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from finmamba3.models.transformer import StochasticTransformerModernKVCache
from finmamba3.models.attention import get_subsequent_mask

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_modern_transformer():
    """Test the modern transformer forward pass."""
    print("Testing StochasticTransformerModern...")
    
    # Model config matching FI-2010 setup
    stoch_dim = 16 * 16  # 256 (categorical latent)
    action_dim = 1
    feat_dim = 512
    num_layers = 4
    num_heads = 8
    max_length = 64
    dropout = 0.1
    batch_size = 4
    seq_len = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = StochasticTransformerModernKVCache(
        stoch_dim=stoch_dim,
        action_dim=action_dim,
        feat_dim=feat_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=max_length,
        dropout=dropout,
        use_action_input=True,
        device=device,
        dtype=torch.float32
    ).to(device)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    
    # Create dummy inputs
    samples = torch.randn(batch_size, seq_len, stoch_dim, device=device)
    actions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = get_subsequent_mask(samples)
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            output = model(samples, actions, mask)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {samples.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, {feat_dim})")
        assert output.shape == (batch_size, seq_len, feat_dim), "Output shape mismatch!"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise
    
    # Test KV cache
    print("\nTesting KV cache...")
    try:
        model.reset_kv_cache_list(batch_size=batch_size, dtype=torch.float32)
        single_sample = torch.randn(batch_size, 1, stoch_dim, device=device)
        single_action = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        with torch.no_grad():
            output_cache = model.forward_with_kv_cache(single_sample, single_action)
        print(f"✓ KV cache forward pass successful!")
        print(f"  Input shape: {single_sample.shape}")
        print(f"  Output shape: {output_cache.shape}")
        print(f"  Expected shape: ({batch_size}, 1, {feat_dim})")
        assert output_cache.shape == (batch_size, 1, feat_dim), "KV cache output shape mismatch!"
    except Exception as e:
        print(f"✗ KV cache forward pass failed: {e}")
        raise
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    try:
        model.train()
        samples_grad = torch.randn(batch_size, seq_len, stoch_dim, device=device, requires_grad=True)
        actions_grad = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        mask_grad = get_subsequent_mask(samples_grad)
        
        output_grad = model(samples_grad, actions_grad, mask_grad)
        loss = output_grad.mean()
        loss.backward()
        
        print(f"✓ Gradient flow successful!")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Input grad shape: {samples_grad.grad.shape}")
        assert samples_grad.grad is not None, "No gradient for input!"
    except Exception as e:
        print(f"✗ Gradient flow failed: {e}")
        raise
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    
    # Print architecture summary
    print("\nArchitecture Summary:")
    print(f"  Backbone: TransformerModern (Pre-norm)")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {feat_dim}")
    print(f"  Heads: {num_heads}")
    print(f"  Head dim: {feat_dim // num_heads}")
    print(f"  FFN expansion: 4x ({feat_dim * 4})")
    print(f"  Total parameters: {total_params:,}")
    
    return model

if __name__ == "__main__":
    test_modern_transformer()
