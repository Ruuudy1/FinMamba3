# region imports
import torch
from finmamba3.models.transformer import StochasticTransformerModernKVCache
from finmamba3.models.attention import get_subsequent_mask_with_batch_length
# endregion


STOCH_DIM = 256
ACTION_DIM = 1
FEAT_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_LENGTH = 16
BATCH_SIZE = 3
SEQ_LEN = 8


def build_modern_transformer():
    return StochasticTransformerModernKVCache(
        stoch_dim=STOCH_DIM, action_dim=ACTION_DIM, feat_dim=FEAT_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, max_length=MAX_LENGTH,
        dropout=0.0, use_action_input=True, device=torch.device("cpu"), dtype=torch.float32,
    )


def test_modern_transformer_forward_returns_feat_dim_sequence():
    torch.manual_seed(0)
    model = build_modern_transformer()
    samples = torch.randn(BATCH_SIZE, SEQ_LEN, STOCH_DIM)
    action = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    causal_mask = get_subsequent_mask_with_batch_length(SEQ_LEN, samples.device)
    output = model(samples, action, causal_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, FEAT_DIM)
    assert torch.isfinite(output).all()


def test_modern_transformer_backpropagates_to_input():
    torch.manual_seed(0)
    model = build_modern_transformer()
    samples = torch.randn(BATCH_SIZE, SEQ_LEN, STOCH_DIM, requires_grad=True)
    action = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    causal_mask = get_subsequent_mask_with_batch_length(SEQ_LEN, samples.device)
    output = model(samples, action, causal_mask)
    output.mean().backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_modern_transformer_runs_without_action_input():
    torch.manual_seed(0)
    model = StochasticTransformerModernKVCache(
        stoch_dim=STOCH_DIM, action_dim=ACTION_DIM, feat_dim=FEAT_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, max_length=MAX_LENGTH,
        dropout=0.0, use_action_input=False, device=torch.device("cpu"), dtype=torch.float32,
    )
    samples = torch.randn(BATCH_SIZE, SEQ_LEN, STOCH_DIM)
    action = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    causal_mask = get_subsequent_mask_with_batch_length(SEQ_LEN, samples.device)
    output = model(samples, action, causal_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, FEAT_DIM)


def test_modern_transformer_kv_cache_single_step_matches_feat_dim():
    torch.manual_seed(0)
    model = build_modern_transformer()
    model.reset_kv_cache_list(batch_size=BATCH_SIZE, dtype=torch.float32)
    samples = torch.randn(BATCH_SIZE, 1, STOCH_DIM)
    action = torch.zeros(BATCH_SIZE, 1, dtype=torch.long)
    output = model.forward_with_kv_cache(samples, action)
    assert output.shape == (BATCH_SIZE, 1, FEAT_DIM)
    assert torch.isfinite(output).all()
