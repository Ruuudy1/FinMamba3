# Transformer Baseline - Complete Setup Guide

## 🎯 Quick Start

You now have **two separate notebooks** for running experiments:

### 1. Mamba3 Experiments (Original)
**Notebook**: `notebooks/colab_lob_pretrain.ipynb`

**Configs**:
- FI-2010: `configs/fi2010.yaml`
- Polymarket: `configs/lob.yaml`

### 2. Transformer Baseline Experiments (New)
**Notebook**: `notebooks/colab_lob_pretrain_transformer.ipynb`

**Configs**:
- FI-2010: `configs/fi2010_transformer.yaml`
- Polymarket: `configs/lob_transformer_modern.yaml`

---

## 📁 What Was Created

### Implementation Files
```
src/finmamba3/models/
├── attention.py              [MODIFIED] + Pre-norm attention blocks
├── transformer.py            [MODIFIED] + StochasticTransformerModern classes
└── world_model.py            [MODIFIED] + TransformerModern integration

configs/
├── fi2010_transformer.yaml          [NEW] FI-2010 transformer config
└── lob_transformer_modern.yaml      [NEW] Polymarket transformer config

notebooks/
├── colab_lob_pretrain_transformer.ipynb   [NEW] Transformer notebook
└── README_TRANSFORMER.md                   [NEW] Notebook documentation
```

### Documentation Files
```
docs/
├── transformer_baseline_implementation.md   [NEW] Technical details
├── QUICKSTART_TRANSFORMER_BASELINE.md      [NEW] Quick guide
└── [other files]

test_transformer_modern.py          [NEW] Test script
TRANSFORMER_BASELINE_SETUP.md       [NEW] This file
```

---

## 🚀 How to Use

### Option A: Google Colab (Recommended)

#### Step 1: Push to GitHub
```bash
# Make sure you're on transformer-baseline branch
git branch  # Should show: * transformer-baseline

# Push the branch to GitHub
git push origin transformer-baseline
```

#### Step 2: Open the Transformer Notebook in Colab

**URL format**:
```
https://colab.research.google.com/github/YOUR_USERNAME/FinMamba3/blob/transformer-baseline/notebooks/colab_lob_pretrain_transformer.ipynb
```

Replace `YOUR_USERNAME` with your GitHub username.

#### Step 3: Configure and Run

In the notebook's first cell:
```python
DATASET = "fi2010"  # or "polymarket"
SMOKE_TEST = False
MAX_STEPS = 8000    # 8000 for FI-2010, 20000 for Polymarket
```

Then **Run All Cells** ▶️

---

### Option B: Local Training (If You Have GPU)

```bash
# FI-2010 transformer baseline
python -m finmamba3.train \
  --config configs/fi2010_transformer.yaml \
  --dataset fi2010

# Polymarket transformer baseline
python -m finmamba3.train \
  --config configs/lob_transformer_modern.yaml \
  --dataset polymarket \
  --hours-train 6 \
  --hours-val 1
```

---

## 📊 Side-by-Side Comparison Plan

### Phase 1: FI-2010 (Faster, Recommended First)

| Experiment | Notebook | Config | Expected Time |
|------------|----------|--------|---------------|
| **Mamba3** | `colab_lob_pretrain.ipynb` | `fi2010.yaml` | 2-4 hours |
| **Transformer** | `colab_lob_pretrain_transformer.ipynb` | `fi2010_transformer.yaml` | 2-4 hours |

**Baseline to beat**: Mamba3 val_loss=4.2546 at step 7000

### Phase 2: Polymarket (Longer, More Complex)

| Experiment | Notebook | Config | Expected Time |
|------------|----------|--------|---------------|
| **Mamba3** | `colab_lob_pretrain.ipynb` | `lob.yaml` | 4-6 hours |
| **Transformer** | `colab_lob_pretrain_transformer.ipynb` | `lob_transformer_modern.yaml` | 4-6 hours |

---

## 📈 What to Compare

After running both experiments, compare:

### Primary Metrics
1. ✅ **Best Validation Loss** - Which model achieves lower loss?
2. ✅ **Training Speed** - Iterations/sec, wall-clock time
3. ✅ **Direction Accuracy** - 3-class midprice prediction
4. ✅ **GPU Memory Usage** - Peak memory during training

### Secondary Metrics
5. ✅ **Convergence Speed** - Steps to reach target loss
6. ✅ **Training Stability** - Loss variance, no NaN/Inf
7. ✅ **Parameter Count** - Transformer will have ~50% more
8. ✅ **Inference Speed** - Autoregressive rollout speed

### Questions to Answer
- Does the transformer's extra parameters (~50% more) improve performance?
- Is the transformer's O(L²) attention worth the extra compute?
- Does Mamba3's linear complexity translate to practical advantages?
- Which architecture learns better representations for LOB data?

---

## 🔍 Architectural Differences

### What's Identical (Fair Comparison)
✅ Encoder (LOB transformer encoder)
✅ Decoder (MSE/Student-t)
✅ Latent space (16×16 categorical)
✅ Auxiliary heads (direction, etc.)
✅ Stem (Linear + RMSNorm + SiLU)
✅ Normalization (Pre-norm RMSNorm)
✅ Activation (SiLU)
✅ Optimizer (Laprop)
✅ Learning rate schedule (Cosine)
✅ Batch size (64)
✅ Dropout rate
✅ Training steps

### What's Different (The Test)
❌ **Sequence Model**:
- **Mamba3**: State Space Model (SSM) - O(L) linear
- **Transformer**: Self-Attention - O(L²) quadratic

This isolation lets you measure the pure impact of the sequence modeling approach.

---

## 📝 Expected Parameter Counts

### FI-2010 Models

| Component | Mamba3 | Transformer |
|-----------|--------|-------------|
| Encoder | ~1.5M | ~1.5M (same) |
| Sequence Model | ~6-8M | ~12-13M |
| Decoder + Heads | ~1-2M | ~1-2M (same) |
| **Total** | **~10-12M** | **~15-17M** |

Transformer has ~50% more parameters due to explicit QKV projections and larger FFN.

---

## 🎬 Sample Workflow

### Day 1: FI-2010 Experiments

**Morning**: Run Mamba3
```
1. Open colab_lob_pretrain.ipynb in Colab
2. Set DATASET = "fi2010"
3. Run All Cells
4. Wait ~3 hours
5. Download logs and checkpoints
```

**Afternoon**: Run Transformer
```
1. Open colab_lob_pretrain_transformer.ipynb in Colab
2. Set DATASET = "fi2010"
3. Run All Cells
4. Wait ~3 hours
5. Download logs and checkpoints
```

**Evening**: Compare Results
```
1. Compare validation loss curves
2. Compare training speed (it/s)
3. Compare GPU memory usage
4. Analyze direction accuracy
5. Document findings
```

### Day 2: Analysis & Polymarket (Optional)

**Morning**: Analyze FI-2010 results
- Create comparison plots
- Write up initial findings
- Decide if Polymarket experiments are needed

**Afternoon**: Run Polymarket experiments (if needed)
- Similar process, but longer (20K steps)
- More complex dataset
- Additional metrics (Brier score, etc.)

---

## 🔧 Troubleshooting

### Notebook won't start
- ✅ Check HF_TOKEN is in Colab Secrets
- ✅ Verify GPU runtime is selected (not CPU)
- ✅ Check notebook branch is `transformer-baseline`

### Training fails
- ✅ Check config file path exists
- ✅ Verify data downloaded correctly
- ✅ Look at error message in logs

### Loss is NaN
- ✅ Reduce learning rate (try 2e-5 instead of 4e-5)
- ✅ Check gradient clipping is enabled
- ✅ Verify data normalization

### Out of Memory
- ✅ Reduce batch size (try 32 instead of 64)
- ✅ Reduce sequence length
- ✅ Use high-memory Colab runtime

---

## 📚 Documentation Reference

### For Quick Start
- `docs/QUICKSTART_TRANSFORMER_BASELINE.md`
- `notebooks/README_TRANSFORMER.md`

### For Technical Details
- `docs/transformer_baseline_implementation.md`
- `src/finmamba3/models/transformer.py` (code comments)
- `src/finmamba3/models/attention.py` (code comments)

### For Testing
- `test_transformer_modern.py` (unit tests)

---

## 🎯 Success Criteria

Your transformer baseline is successful if:

1. ✅ **Notebook runs without errors** on Colab
2. ✅ **Training converges** (loss decreases)
3. ✅ **No NaN/Inf** in loss or gradients
4. ✅ **Checkpoints save** correctly
5. ✅ **Results are comparable** to Mamba3 (within reasonable range)

---

## 🔄 Git Workflow

```bash
# Current branch
git branch  # Should show: * transformer-baseline

# View changes
git log --oneline -5

# Push to GitHub (to use in Colab)
git push origin transformer-baseline

# Create PR (when ready to merge)
gh pr create --base main --head transformer-baseline
```

---

## 📞 Need Help?

Check these in order:

1. **Quick issues**: `notebooks/README_TRANSFORMER.md`
2. **Config issues**: `docs/QUICKSTART_TRANSFORMER_BASELINE.md`
3. **Technical issues**: `docs/transformer_baseline_implementation.md`
4. **Code issues**: Read comments in modified files
5. **Test issues**: Run `test_transformer_modern.py`

---

## ✨ Summary

You now have:
- ✅ Modern transformer implementation (pre-norm, RMSNorm, SiLU)
- ✅ Dedicated Colab notebook for transformer experiments
- ✅ Configs for both FI-2010 and Polymarket
- ✅ Full documentation and testing
- ✅ Fair comparison setup against Mamba3

**Next step**: Open `colab_lob_pretrain_transformer.ipynb` in Colab and run your first transformer experiment! 🚀

---

## 🎊 Branch Status

**Current Branch**: `transformer-baseline`

**Commits**:
1. Implement modern transformer baseline
2. Add quickstart guide  
3. Add dedicated Colab notebook

**Ready to push to GitHub**: ✅ Yes

**Ready to run in Colab**: ✅ Yes

**Ready to compare against Mamba3**: ✅ Yes
