# FinDrama

FinDrama is the Polymarket limit-order-book version of Drama. The old Atari and
MemoryMaze paths have been removed; this repository now supports one workflow:
offline world-model pretraining on Polymarket LOB data, plus the first
Gymnasium trading environment for Phase B.

The key novelty axis vs. the upstream Drama paper (Wang et al., ICLR 2025) is
the Mamba-3 MIMO sequence backbone (Lahoti et al., ICLR 2026) applied to LOB
tick streams, plus a microstructure-aware feature encoder, an episodic-memory
ablation switch with optional learned write policy, and Lopez de Prado financial
data structures for tick-stream denoising.

## Run On Colab

Use exactly one notebook:

`notebooks/colab_lob_pretrain.ipynb`

The notebook works on any CUDA GPU (H100, L4, A100, T4). H100 is recommended.

In Colab:

1. Set the runtime to a GPU instance (Runtime, Change runtime type).
2. Open `notebooks/colab_lob_pretrain.ipynb`.
3. Confirm `REPO_URL = "https://github.com/Ruuudy1/FinDrama.git"` and
   `BRANCH = "improvements-research-review"` (or `master`) in the first code cell.
4. Add your `HF_TOKEN` to Colab Secrets (key icon, left sidebar). The token
   needs write access to upload compiled wheels and (optionally) checkpoints.
5. Run cells top to bottom.
6. Start with `SMOKE_TEST = True`. After one short update and validation print
   complete, set it to `False` for the full 20,000-step run.

The notebook installs PyTorch 2.6 CUDA 12.4, builds `causal-conv1d` and
`mamba-ssm` from source, downloads the data bundle automatically, and runs:

```bash
python -B src/train_lob.py --hours-train 6 --hours-val 1 --JointTrainAgent.SampleMaxSteps 20000
```

Checkpoints are written under:

```text
saved_models/lob/LOB/<run_id>/ckpt/world_model.pth
```

The final notebook cell uploads `saved_models/lob` to the HuggingFace dataset
repo at `ruuudy/FinDrama` under `checkpoints/lob/`.

## Wheel Cache

Compiled CUDA wheels (`causal-conv1d`, `mamba-ssm`) are cached on HuggingFace
so later runtimes skip the source build:

https://huggingface.co/datasets/ruuudy/FinDrama/tree/main

Wheels are keyed by Python version, PyTorch version, CUDA version, and GPU
architecture (for example `wheels-py312-torch260-cu124-sm90`). The first run
on a new GPU type builds and uploads; subsequent runs pull and install in
seconds.

Set `FORCE_REBUILD_WHEELS = True` in the first notebook cell to force a fresh
build after updating the dependency stack or if a cached wheel becomes stale.

## Data

The pretraining dataset is hosted on the HuggingFace dataset repo that also
hosts the wheel cache. Both splits are stored under `data/`:

```text
ruuudy/FinDrama
  data/
    train.tar.zip
    validation.tar.zip
  wheels-py.../
  checkpoints/lob/
```

Download both splits with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ruuudy/FinDrama",
    repo_type="dataset",
    allow_patterns=["data/train.tar.zip", "data/validation.tar.zip"],
    local_dir="./",
)
```

Or use the helper in `src/utils_hf.py`:

```python
from utils_hf import download_data
train_zip, val_zip = download_data(local_dir="./", revision=None)
```

Pin a `revision` in calling code for reproducibility. Authentication: the
HF token is read from the `HF_TOKEN` environment variable or the standard
HuggingFace cache. The notebook reads the same token from Colab Secrets.

The notebook extracts both archives into `data/train` and `data/validation`.

## Local Smoke Test

Install a CUDA PyTorch build first, then install the project dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy>=2,<3" "causal-conv1d>=1.4.0" --no-build-isolation
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation
pip install -r requirements.txt
```

Then run:

```bash
python -B src/train_lob.py --hours-train 1 --hours-val 0.25 --JointTrainAgent.SampleMaxSteps 20
```

Expected data layout:

```text
data/
  train/
    polymarket.db
    polymarket_books/
    binance_lob/
  validation/
    polymarket.db
    polymarket_books/
    binance_lob/
```

## Repository Layout

```text
notebooks/
  colab_lob_pretrain.ipynb
src/
  train_lob.py                    LOB pretraining entrypoint
  config_utils.py                 dotted CLI config overrides
  training_steps.py               world-model update loop
  replay_buffer.py
  agents.py                       actor-critic/PPO code for Phase B
  baselines/
    deeplob.py                    DeepLOB CNN+LSTM reference baseline
    linear_ar.py                  Linear vector autoregression floor baseline
  eval/
    backtest.py                   PnL/Sharpe/MaxDD harness for a frozen world model
    regime_split.py               Time and volatility splits for non-stationarity tests
  config_files/
    configure_lob.yaml            Default Mamba-3 MIMO baseline.
    configure_lob_em.yaml         Episodic memory enabled (FIFO writes).
    configure_lob_full_ablation.yaml  Student-t + Hawkes + Settlement + EM-novelty + multi-threshold.
    configure_lob_aggregate_only.yaml Tick-aggregate features only; per-level depth masked.
    configure_lob_studentt.yaml   Student-t reconstruction likelihood instead of MSE.
    configure_lob_mamba1.yaml     Mamba-1 backbone for the architecture sweep.
    configure_lob_mamba2.yaml     Mamba-2 backbone for the architecture sweep.
    configure_lob_transformer.yaml  Transformer backbone for the architecture sweep.
  envs/
    lob_features.py               94-dim microstructure-aware feature engineering
    lob_aggregation.py            Time/volume/dollar/tick-imbalance/CUSUM bars
    lob_labels.py                 Triple-barrier and multi-threshold direction targets
    polymarket_lob_env.py         Gymnasium trading environment with reward variants
  lob/backtester/                 Vendored DATAHACKS data structures + loader
  sub_models/
    lob_encoder.py                Transformer-over-depth-tokens encoder + Student-t decoder
    lob_auxiliary.py              Direction, regime, episodic memory, Hawkes, settlement heads
    fin_mamba.py                  FinDrama sequence wrapper for upstream Mamba
    world_models.py               Mamba3 MIMO world model
tests/
  test_lob_features.py
  test_lob_aggregation.py
  test_lob_labels.py
  test_baselines.py
  test_polymarket_lob_env.py
  test_world_model_mamba_backbones.py
  test_train_integration.py
```

## Literature Alignment

The project sits at the intersection of three lines of work. This section
records where we agree with the literature, where we deviate, and what to
cite. It exists so the team and reviewers can see at a glance whether each
design choice is defensible.

### Drama (Wang et al., arxiv 2410.08893, ICLR 2025)

Forked. Drama uses a 7M-parameter Mamba-2 world model on Atari100k and reports
a 105% normalized score with linear-time complexity. Key claim: parameter
efficiency vs. Transformer / RSSM / DreamerV3 baselines on a single laptop.

Where we deviate:
- We swap Mamba-2 for Mamba-3 MIMO (see below).
- We dropped Drama's "Dynamic Frequency-based Sampling" replay scheme in
  favour of an imagine-counter penalty in `replay_buffer.py:47`. This is an
  undocumented deviation; either bring it back as a baseline or write a
  paragraph defending why tick-frequency, not state-visit-frequency, is the
  bottleneck for LOB sequences.

### Mamba-3 MIMO (Lahoti et al., arxiv 2603.15569, ICLR 2026)

The strongest single novelty axis. Headline numbers: at 1.5B scale, Mamba-3 +
MIMO improves average downstream accuracy by 1.8 points over Gated DeltaNet
(0.6 from M3, 1.2 from MIMO), and Mamba-3 matches Mamba-2 perplexity at half
the state size. No published LOB / market-microstructure paper uses Mamba-3,
and only a handful even use Mamba-2 (`MambaTS`, `MambaStock`, `CryptoMamba`).
The framing for the paper should be "first Mamba-3 MIMO world model on LOB."

To validate the architectural claim we ship five matched-config yamls so the
ablation table can be produced with one command per backbone:
`configure_lob.yaml` (Mamba-3 MIMO), `configure_lob_mamba1.yaml`,
`configure_lob_mamba2.yaml`, `configure_lob_transformer.yaml`, and any
`is_mimo: false` variant of the default for the SISO column.

### Episodic and retrieval-augmented memory (arxiv 2506.06326, 2602.16192, 2202.08417)

The repo's `EpisodicMemory` (`src/sub_models/lob_auxiliary.py`) is a CPU-side
top-k cosine retriever with FIFO eviction. New: `UseNovelty` flag turns the
write policy into a KL-novelty filter so the buffer becomes a regime catalog
rather than a sliding window of recent states. The `configure_lob_em.yaml`
ablation enables the FIFO variant; `configure_lob_full_ablation.yaml`
enables the novelty-filtered variant. Both compare against the default (off).

### Modern world-model baselines

- DreamerV3 (Hafner et al., Nature 2025): RSSM with 32x32 categorical latents,
  symlog two-hot reward decoding, single hyperparameter set across domains.
  We borrow the symlog two-hot decoder (`functions_losses.py`) and shrink the
  categorical latent to 16x16 because the LOB-aggregate input is only 94 dims.
- TD-MPC2 (Hansen et al., ICLR 2024): decoder-free trajectory optimization at
  317M params across 80 continuous tasks. Discrete-action LOB does not need
  this directly, but the decoder-free idea motivates the optional Student-t
  decoder added in this branch (`lob_encoder.StudentTLOBDecoder`) - more
  honest than MSE on cents-discretized prices, easier to ablate against a
  decoder-free run later.
- R2-Dreamer (ICLR 2026): redundancy-reduced world model. Worth citing for
  positioning when the paper discusses why we keep a decoder.

### LOB-specific deep-learning baselines

We ship a port of DeepLOB (Zhang et al. 2018) and a closed-form linear AR
baseline at `src/baselines/`. Recent transformer-based competitors worth
adding next: TLOB (Bertini et al., arxiv 2502.15757) with dual spatial/
temporal attention; LiT (Frontiers AI 2025) with structured patches; HLOB
(ScienceDirect 2024) with persistence-aware blocks. The shared LOBFrame
codebase (arxiv 2403.09267) gives the canonical NASDAQ benchmark; we should
report numbers on Polymarket so the reviewer can compare regimes.

### Polymarket microstructure

[Sotskov et al. (arxiv 2604.24366)](https://arxiv.org/abs/2604.24366) measure
median half-spread on Polymarket near 200 bps - one to two orders of
magnitude wider than equity LOBs. This is why per-tick mid changes are
dominated by spread-bouncing noise rather than signal, and why the new
`src/envs/lob_aggregation.py` module is essential for an honest training
target. The `SoK: Decentralized Prediction Markets` paper (arxiv 2510.15612)
is the right taxonomy citation for positioning the dataset.

### State-space and SSM time-series work

For broader context: `MambaTS` (arxiv 2405.16440) is the SOTA SSM time-series
forecaster. `From S4 to Mamba` (arxiv 2503.18970) surveys the family.
`Mamba time series forecasting with uncertainty quantification` (PMC 2025)
is the right cite when discussing why a heavy-tailed decoder pairs naturally
with Mamba.

## Data Engineering

Polymarket median half-spread is roughly 200 bps. Raw per-tick mid changes
on Polymarket are mostly spread-bouncing noise, not signal. Three layered
denoising tools live in `src/envs/`:

1. **Bar aggregation** (`lob_aggregation.py`). Replace the raw tick stream
   with one of: time bars (5s/30s default), volume bars, dollar bars,
   tick-imbalance bars, or CUSUM bars. Lopez de Prado financial data
   structures applied directly to the 94-dim flat features.
2. **Triple-barrier labels** (`lob_labels.py`). Replace "sign of next-tick
   mid change" with "which of {profit, stop, time} barrier hits first."
   Available in numpy and torch. Multi-threshold sweep helper exists for
   the threshold-curve reporting.
3. **Multi-resolution encoder** (`MultiScaleEncoder` in `lob_encoder.py`).
   Wraps multiple `LOBEncoder` instances at different resolutions (raw
   ticks, 5s, 30s, 5min) and fuses them via a learned MLP. Designed for
   the 5min-to-15min/1hr binary-contract transfer experiment: train a
   shared encoder on resolved 5min markets, fine-tune the head on longer
   horizons.

Both bar aggregation and triple-barrier labeling are off by default to
preserve baseline reproducibility; opt in via the appropriate config
variant or by calling them from a custom data-loading script.

## Reward Function Variants

The advisor flagged that "a novel reward function is novel enough" and the
literature confirms it: most LOB RL papers copy a Cartea/Jaimungal market-
making reward or retrofit Sharpe-on-PnL. The env exposes three reward kinds
selectable via `PolymarketLOBEnv(reward_kind=...)`:

- `default` - Atari-style `tanh(delta_log/vol_scale)` minus turnover,
  inventory, and drawdown costs. Baseline.
- `settlement_calibrated` - default plus an extra reward proportional to
  `tanh((payoff - position_value) / vol_scale)` at every settlement event.
  Captures the binary-contract structure absent from a pure PnL reward.
- `risk_budgeted` - `tanh(delta_log / realized_rolling_std)` minus a
  variance penalty. Sharpe-style reward (Cartea/Jaimungal).

In Phase A pretraining the reward function is unused (rewards are zeroed in
the buffer). The variants matter once Phase B is wired up.

## Auxiliary Heads

Three optional heads are available alongside the existing reconstruction +
KL + DirectionHead stack. Each is gated by a config flag and contributes
zero loss when its required labels are absent.

- `HawkesIntensityHead` (`lob_auxiliary.py`). Predicts log-intensity for
  buy and sell event arrivals. Trained with Poisson NLL on observed event
  counts in a forward window. Requires `event_counts` to be threaded into
  `WorldModel.update()` via the data pipeline.
- `SettlementHead`. Predicts the binary contract outcome from the latent.
  Trained with BCE, optionally weighted by closeness-to-expiry so the
  pressure ramps up near resolution. Requires per-sequence `outcome` and
  optional `time_to_expiry_frac`.
- DirectionHead with `DirectionThresholds` list. Trains the same head with
  cross-entropy averaged across multiple direction-bucket thresholds, so
  accuracy can be reported as a curve over thresholds rather than pinned
  at one value.

Enable all three at once via `configure_lob_full_ablation.yaml`.

## Notes

- `train_lob.py` no longer imports `gym` or the removed Atari path.
- Normalized LOB features are clipped and checked before training.
- `Backbone: Mamba3` is the default. Full-sequence Phase A pretraining is the
  supported path; Phase B imagination uses full-prefix recomputation rather
  than Mamba3 step/inference-cache kernels.
- If a local run fails while importing `mamba_ssm.modules.mamba3`, the
  upstream source install is missing or was built against a different
  PyTorch CUDA wheel.
- Action input is enabled by default for backwards compatibility, but
  `Models.WorldModel.UseActionInput: False` removes the dead one-hot
  pathway during Phase A pretraining.
