# Quick Start: Transformer Baseline

## Overview

A modern transformer baseline has been implemented for fair comparison against your Mamba3 model. The implementation is on branch `transformer-baseline`.

## What Was Done

✅ **Modern Pre-Norm Transformer Architecture**
- Matches Mamba3's architectural choices (RMSNorm, SiLU, pre-norm)
- 4 transformer layers to match Mamba3's 4 layers
- Same encoder, decoder, heads, and training setup as Mamba3
- Only difference: Self-Attention vs SSM

✅ **Configuration Files**
- `configs/fi2010_transformer.yaml` - For FI-2010 dataset
- `configs/lob_transformer_modern.yaml` - For Polymarket dataset

✅ **Code Implementation**
- Pre-norm attention blocks with RMSNorm
- Modern transformer classes with KV cache support
- Integration with world_model.py

✅ **Documentation & Testing**
- Full implementation documentation
- Test script to verify correctness
- Parameter count estimates

---

## How to Run

### Option 1: FI-2010 Baseline (Recommended First)

Compare against your run#4 (Mamba3 achieved val_loss=4.2546 at step 7000):

```bash
python -m finmamba3.train \
  --config configs/fi2010_transformer.yaml \
  --dataset fi2010
```

Expected runtime: ~2-4 hours for 8000 steps

### Option 2: Polymarket Baseline

Compare against your Polymarket Mamba3 runs:

```bash
python -m finmamba3.train \
  --config configs/lob_transformer_modern.yaml \
  --dataset polymarket \
  --hours-train 6 \
  --hours-val 1
```

Expected runtime: ~4-6 hours for 20000 steps

---

## Key Comparison Metrics

Track these to compare Mamba3 vs Transformer:

### Primary
1. **Validation Loss** - Direct performance comparison
2. **Direction Accuracy** - 3-class prediction accuracy
3. **Training Speed** - Iterations/sec, wall-clock time
4. **GPU Memory** - Peak usage during training

### Secondary
5. **Convergence Speed** - Steps to reach target loss
6. **Training Stability** - Loss variance over time
7. **Parameter Count** - Transformer will have ~50% more params
8. **Best Checkpoint** - Final best validation loss

---

## Expected Results

### Mamba3 (Your Baseline - run#4)
- FI-2010 val_loss: **4.2546** at step 7000
- Parameters: ~10M
- Training speed: ~3.0 it/s
- GPU memory: ~3.8 GB

### Transformer (To Be Determined)
- FI-2010 val_loss: **?** (compare to 4.2546)
- Parameters: ~15M (~50% more)
- Training speed: **?** (likely slower due to O(L²) attention)
- GPU memory: **?** (likely higher)

The goal is to see if the extra parameters and compute of the transformer can match or beat Mamba3's performance.

---

## What to Look For

### If Transformer Performs Better:
- Could indicate the task benefits from explicit attention mechanisms
- Short-range dependencies might be more important than long-range
- Worth exploring hybrid architectures

### If Mamba3 Performs Better:
- Validates SSM's efficiency for this task
- Linear complexity advantage is significant
- Continue optimizing Mamba3 architecture

### If Performance is Similar:
- Suggests both architectures are suitable
- Choose based on efficiency (Mamba3 is faster)
- Consider task-specific factors (deployment, inference speed)

---

## Files Created/Modified

### New Files
```
configs/fi2010_transformer.yaml          # FI-2010 config
configs/lob_transformer_modern.yaml      # Polymarket config
test_transformer_modern.py               # Test script
docs/transformer_baseline_implementation.md  # Full docs
docs/QUICKSTART_TRANSFORMER_BASELINE.md  # This file
```

### Modified Files
```
src/finmamba3/models/attention.py     # Pre-norm attention blocks
src/finmamba3/models/transformer.py   # Modern transformer classes
src/finmamba3/models/world_model.py   # Integration
```

---

## Validation Checklist

Before running experiments:

- [x] Syntax check passed (all files compile)
- [x] Architecture matches Mamba3 (pre-norm, RMSNorm, SiLU)
- [x] Config files created for both datasets
- [x] Documentation complete
- [ ] Run test on actual environment with PyTorch
- [ ] Verify training starts without errors
- [ ] Check parameter count matches estimates (~15M)
- [ ] Monitor first 100 steps for stability

---

## Troubleshooting

### If training fails to start:
1. Check that PyTorch and dependencies are installed
2. Verify config file path is correct
3. Check GPU availability with `nvidia-smi`
4. Look at error message - it should point to the issue

### If loss is NaN or unstable:
1. Reduce learning rate (try 2e-5 instead of 4e-5)
2. Increase warmup steps (try 1000 instead of 500)
3. Check gradient clipping is enabled (max_grad_norm: 1000)

### If OOM (Out of Memory):
1. Reduce batch size (try 32 instead of 64)
2. Reduce sequence length (try 32 instead of 64)
3. Enable gradient checkpointing in encoder

---

## Next Steps

1. **Run FI-2010 experiment first**
   - Fastest to run (~2-4 hours)
   - Clear baseline to compare against (run#4)
   - Will reveal any implementation issues quickly

2. **Analyze results**
   - Compare validation loss curves
   - Check training speed and memory
   - Look at direction accuracy

3. **Run Polymarket experiment**
   - Longer training (20K steps)
   - More complex dataset
   - Compare against your Polymarket Mamba3 runs

4. **Document findings**
   - Create comparison plots
   - Write up performance analysis
   - Share insights

---

## Quick Command Reference

```bash
# Switch to transformer baseline branch
git checkout transformer-baseline

# Run FI-2010 baseline
python -m finmamba3.train \
  --config configs/fi2010_transformer.yaml \
  --dataset fi2010

# Run Polymarket baseline
python -m finmamba3.train \
  --config configs/lob_transformer_modern.yaml \
  --dataset polymarket \
  --hours-train 6 \
  --hours-val 1

# Monitor training with wandb
wandb login  # if not already logged in
# Training logs will be saved offline by default

# Check GPU usage
nvidia-smi -l 1  # Update every second
```

---

## Questions?

Refer to:
1. **Full documentation**: `docs/transformer_baseline_implementation.md`
2. **Test script**: `test_transformer_modern.py`
3. **Config files**: `configs/fi2010_transformer.yaml`, `configs/lob_transformer_modern.yaml`

Good luck with your experiments! 🚀
