# Transformer Baseline Notebook

## Overview

`colab_lob_pretrain_transformer.ipynb` is a dedicated notebook for training the **Modern Transformer baseline** to compare against Mamba3.

## What's Different from the Mamba3 Notebook?

This notebook is identical to `colab_lob_pretrain.ipynb` except:

1. **Uses transformer configs instead of Mamba3 configs**
   - FI-2010: `configs/fi2010_transformer.yaml`
   - Polymarket: `configs/lob_transformer_modern.yaml`

2. **Updated documentation**
   - Title reflects "Transformer Baseline"
   - Comments explain the comparison purpose

3. **Same infrastructure**
   - Same data loading
   - Same dependency installation
   - Same logging and monitoring
   - Same output structure

## Usage

### Open in Google Colab

**Direct link:**
```
https://colab.research.google.com/github/YOUR_USERNAME/FinMamba3/blob/transformer-baseline/notebooks/colab_lob_pretrain_transformer.ipynb
```

*(Update YOUR_USERNAME with your GitHub username after pushing)*

### Configuration

In the first code cell, set:

```python
DATASET = "fi2010"  # or "polymarket"
```

The notebook will automatically use the corresponding transformer config:
- `"fi2010"` → `configs/fi2010_transformer.yaml`
- `"polymarket"` → `configs/lob_transformer_modern.yaml`

### Runtime Settings

```python
SMOKE_TEST = False      # Set True for quick 20-step test
RUN_PROBES = False      # Not used for transformer baseline
HOURS_TRAIN = 6         # Hours of training data
HOURS_VAL = 1           # Hours of validation data
MAX_STEPS = 8000        # Training steps (8000 for FI-2010, 20000 for Polymarket)
```

## What Gets Trained

### Modern Transformer Architecture

- **Pre-norm** with RMSNorm (matches Mamba3)
- **SiLU** activation (matches Mamba3)
- **4 transformer layers** (matches Mamba3's 4 layers)
- **Same encoder, decoder, heads** as Mamba3

### Only Difference: Sequence Model

- **Mamba3**: State Space Model (SSM) - Linear O(L) complexity
- **Transformer**: Self-Attention - Quadratic O(L²) complexity

This isolation enables fair comparison of the two architectures.

## Expected Results

### FI-2010 Dataset

**Mamba3 Baseline (run#4):**
- Validation loss: 4.2546 at step 7000
- Parameters: ~10M
- Training speed: ~3.0 it/s
- GPU memory: ~3.8 GB

**Transformer (To Measure):**
- Validation loss: ?
- Parameters: ~15M (~50% more due to explicit QKV projections)
- Training speed: ? (likely slower due to O(L²) attention)
- GPU memory: ? (likely higher)

### Polymarket Dataset

Compare against your Polymarket Mamba3 runs in the logs.

## Output Structure

Same as Mamba3 notebook:

```
saved_models/lob/LOB/<run_id>/
├── ckpt/
│   ├── world_model_best.pth      # Best validation checkpoint
│   └── world_model_final.pth     # Final checkpoint
└── normalization.json             # Feature normalization stats

logs/
└── stdout.txt                     # Training logs
```

## Comparing Results

After running both notebooks:

### 1. Validation Loss
Compare the best validation loss achieved:
```
Mamba3:     X.XXXX (from your logs)
Transformer: Y.YYYY (from this notebook)
```

### 2. Training Speed
Compare iterations/sec:
```
Mamba3:     ~3.0 it/s
Transformer: ? it/s
```

### 3. GPU Memory
Compare peak GPU usage:
```
Mamba3:     ~3.8 GB
Transformer: ? GB
```

### 4. Convergence
Compare steps to reach target loss:
```
Mamba3:     7000 steps to val_loss=4.25
Transformer: ? steps to val_loss=?
```

## Key Files

**Notebook:**
- `notebooks/colab_lob_pretrain_transformer.ipynb`

**Configs:**
- `configs/fi2010_transformer.yaml` - FI-2010 baseline
- `configs/lob_transformer_modern.yaml` - Polymarket baseline

**Implementation:**
- `src/finmamba3/models/transformer.py` - StochasticTransformerModern classes
- `src/finmamba3/models/attention.py` - Pre-norm attention blocks
- `src/finmamba3/models/world_model.py` - Integration

**Documentation:**
- `docs/transformer_baseline_implementation.md` - Full technical details
- `docs/QUICKSTART_TRANSFORMER_BASELINE.md` - Quick start guide

## Troubleshooting

### If training fails to start
1. Check HF_TOKEN is set in Colab Secrets
2. Verify GPU runtime is selected
3. Check error message in logs

### If loss is NaN or unstable
1. Reduce learning rate in config
2. Check gradient clipping is enabled
3. Verify data normalization stats

### If OOM (Out of Memory)
1. Reduce batch size in config
2. Reduce sequence length
3. Use a higher-memory GPU runtime

## Questions?

Refer to:
1. **Full documentation**: `docs/transformer_baseline_implementation.md`
2. **Quickstart guide**: `docs/QUICKSTART_TRANSFORMER_BASELINE.md`
3. **Original notebook**: `notebooks/colab_lob_pretrain.ipynb` (for comparison)

## Notes

- The transformer has ~50% more parameters than Mamba3, but this is acceptable for comparison
- Both models use identical encoder, decoder, and training setup
- The comparison isolates the SSM vs Attention architectural difference
- Results will show whether the extra compute/parameters of transformers are beneficial for this task

Good luck with your experiments! 🚀
