# FinMamba3
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#the-team)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repository contains the research and code behind FinMamba3, a Mamba-3
world model for limit-order-book (LOB) microstructure. We built it to ask
whether conditioning a structured state-space backbone on an *inferred* market
regime improves behaviour under distribution shift. The campaign produced three
outcomes (full write-up in `finmamba3-paper.tex` and `RESULTS.md`):

1. **Regime conditioning is a decisive null.** Our central mechanism,
   RegimeFiLM, collapses to identity under joint training across every objective
   we test, on FI-2010, Polymarket, and the Kaggle crypto spot book, and we
   explain it mechanistically: an input-side affine is a *gauge* direction that
   folds losslessly into the block's own projections, so the optimizer returns
   it to identity.
2. **A spot-conditioned pipeline plus a selective-participation strategy is a
   genuine, survivable economic edge** on held-out Polymarket BTC, but it is
   *architecture-independent*: a logistic regression on the same features
   reproduces and exceeds it, so the pipeline and framing, not the sequence
   model, carry the result.
3. **A book-relative EV head** makes the settlement head's marginal
   low-predictability regime robust (P(profit) 0.63 -> 0.98 across three seeds)
   at comparable total held-out PnL, collapsing the high-vs-low regime spread
   6.2x.

It began as a fork of the Drama world-model framework (Wang et al., ICLR 2025)
but has since diverged completely: the Atari and MemoryMaze paths are gone, the
sequence backbone is a Mamba-3 stack (Lahoti et al., ICLR 2026) with matched
Mamba-1/2 and pre-norm Transformer baselines, and the data, features, rewards,
and evaluation are all rebuilt for financial microstructure.

## The Team

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/Ruuudy1">
          <img src="https://avatars.githubusercontent.com/u/130013367?s=100&v=4" width="100px;" alt="Rudy Osuna"/>
          <br /><sub><b>Rudy Osuna</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/rudy-osuna/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-Ruuudy1" title="Research">🔬</a>
        <a href="https://github.com/Ruuudy1/FinMamba3/commits?author=Ruuudy1" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/Hamzaq96">
          <img src="https://avatars.githubusercontent.com/u/30836331?s=100&v=4" width="100px;" alt="Hamza Qureshi"/>
          <br /><sub><b>Hamza Qureshi</b></sub></a>
        <br \><sub><a href="https://www.linkedin.com/in/hamza-qureshi-98373115a/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-Hamzaq96" title="Research">🔬</a>
        <a href="https://github.com/Ruuudy1/FinMamba3/commits?author=Hamzaq96" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/iradosla0">
          <img src="https://avatars.githubusercontent.com/u/229367996?s=100&v=4" width="100px;" alt="Ivan Radoslavov"/>
          <br /><sub><b>Ivan Radoslavov</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/ivan-asen-radoslavov-375b832a8/" title="LinkedIn">🔗 LinkedIn</a></sub>
        <br /><a href="#research-iradosla0" title="Research">🔬</a>
        <a href="https://github.com/Ruuudy1/FinMamba3/commits?author=iradosla0" title="Code">💻</a>
      </td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Overview

FinMamba3 supports two interchangeable workflows for offline world-model
pretraining:

1. Polymarket LOB tick data (the original pipeline, 94-dim features).
2. FI-2010 Nasdaq Helsinki LOB data (Ntakaris et al. 2018, 46-dim features),
   shipped with a loader, config, and HuggingFace mirror so the model can be
   benchmarked on a public LOB dataset whenever Polymarket data runs short.

The system adapts Drama to LOB microstructure with a microstructure-aware
feature encoder, an episodic-memory ablation switch with optional learned write
policy, and Lopez de Prado financial data structures for tick-stream denoising.
Under matched parameters no backbone (Mamba-1, Mamba-2, Mamba-3 SISO, or a
Transformer) dominates across datasets, so the backbone is reported as a
substrate, not a claimed advantage. We run Mamba-3 in its SISO configuration
throughout; the MIMO kernel exceeds the RTX 4080 shared-memory cap, so MIMO is
outside this project's hardware envelope and we make no claim about it.

## Run On Colab

Use exactly one notebook:

`notebooks/colab_lob_pretrain.ipynb`

The notebook works on any CUDA GPU for the SISO backbone the paper uses. The
codebase retains optional Mamba-3 MIMO support, but it is not part of the
paper's results and we make no claim about it: it requires the TileLang kernel
(no Python fallback) and an H100/H200 (sm_90) or B200 (sm_100) for the full
`d_state=128` path. To experiment with it anyway, A100 can fit the compatibility
`chunk_size=8` path, but the observed Polymarket MIMO run was ~9.49 s/it
(~39.5 hours for 15k steps); L4/T4 and workstation Blackwell cards generally need
`USE_MIMO = False`.

Colab Pro+ improves access but does not guarantee H100; Google notes that
runtime GPU resources vary and are subject to availability:
https://research.google.com/colaboratory/intl/en-GB/faq.html

Open in Colab in one click:

https://colab.research.google.com/github/Ruuudy1/FinMamba3/blob/main/notebooks/colab_lob_pretrain.ipynb

Then:

1. Set the runtime to a GPU instance (Runtime, Change runtime type). (The
   optional MIMO path needs H100/H200/B200; it is not part of the paper's
   results.)
2. Add your `HF_TOKEN` to Colab Secrets (key icon, left sidebar). The token
   needs write access for compiled wheels, checkpoints, and run logs.
3. Hit Run all.

By default the first cell sets:

```python
DATASET = "polymarket"  # reproduces the original Polymarket workflow
MAX_STEPS = 15000       # full Mamba-3 pretrain budget
RUN_PROBES = False
SMOKE_TEST = False
```

To run the public FI-2010 benchmark instead, switch `DATASET = "fi2010"`.
To run the three collapse-diagnosis probes instead of a full pretrain, set
`RUN_PROBES = True`. To verify plumbing in roughly 20 seconds before a real
run, set `SMOKE_TEST = True` and re-run all cells.

The notebook installs a CUDA PyTorch build for the detected GPU, builds
`causal-conv1d` and `mamba-ssm` from source (or pulls from the HF wheel cache
when keys match), downloads the chosen dataset, and runs:

```bash
python -m finmamba3.train \
    --config configs/lob.yaml \
    --dataset polymarket \
    --JointTrainAgent.SampleMaxSteps 15000
```

Checkpoints land under:

```text
saved_models/lob/LOB/<run_id>/ckpt/world_model.pth
```

The final cells upload checkpoints to the model repo
`sj-hryi/FinMamba3-checkpoints` under `checkpoints/lob/`, and stdout logs plus
wandb summaries to the dataset repo `sj-hryi/FinMamba3` under `logs/<run_date>/`.

## Wheel Cache

Compiled CUDA wheels (`causal-conv1d`, `mamba-ssm`) are cached in the dedicated
wheels dataset repo so later runtimes skip the source build:

https://huggingface.co/datasets/sj-hryi/FinMamba3-wheels/tree/main

Wheels are keyed by Python version, PyTorch version, CUDA version, and GPU
architecture (for example `wheels-py312-torch260-cu124-sm90`). The first run
on a new GPU type builds and uploads; subsequent runs pull and install in
seconds.

Set `FORCE_REBUILD_WHEELS = True` in the first notebook cell to force a fresh
build after updating the dependency stack or if a cached wheel becomes stale.

## Data

Both supported datasets live in the data-only HuggingFace dataset repo
`sj-hryi/FinMamba3`; checkpoints and prebuilt wheels live in separate repos:

```text
sj-hryi/FinMamba3                                             (dataset: data + logs)
  data/
    polymarket/
      train.tar.gz                                          Polymarket train bundle.
      validation.tar.gz                                     Polymarket val bundle.
    fi2010/
      train/Train_Dst_NoAuction_DecPre_CF_7.txt             FI-2010 train.
      validation/Val_Dst_NoAuction_DecPre_CF_7.txt          FI-2010 val.
  logs/<run_date>/
sj-hryi/FinMamba3-checkpoints                                 (model: checkpoints/lob + world_model.pth)
sj-hryi/FinMamba3-wheels                                      (dataset: prebuilt CUDA wheels)
```

### Polymarket

Download both splits with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sj-hryi/FinMamba3",
    repo_type="dataset",
    allow_patterns=["data/polymarket/train.tar.gz", "data/polymarket/validation.tar.gz"],
    local_dir="./",
)
```

Or use the helper in `src/finmamba3/hf_hub.py`:

```python
from finmamba3.hf_hub import download_data
train_zip, val_zip = download_data(local_dir="./", revision=None)
```

The notebook extracts both archives into `data/train` and `data/validation`.

### FI-2010

The FI-2010 NoAuction DecPre CF files are mirrored alongside the Polymarket
bundles. Pull them directly:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sj-hryi/FinMamba3",
    repo_type="dataset",
    allow_patterns=[
        "data/fi2010/train/Train_Dst_NoAuction_DecPre_CF_7.txt",
        "data/fi2010/validation/Val_Dst_NoAuction_DecPre_CF_7.txt",
    ],
    local_dir="./",
)
```

The trainer copies these into `data/train/` and `data/validation/`, then
`src/finmamba3/envs/fi2010_loader.py` parses the (149, N_events) matrix into a 46-dim
flat feature vector: 10 levels of (ask_price, ask_size, bid_price, bid_size)
plus 6 derived tick aggregates (mid, spread, log_spread, imbalance,
microprice, log_total_vol). The published 5-horizon direction labels are
loaded alongside and remapped to the world model's {0=down, 1=flat, 2=up}
convention.

Pin a `revision` in calling code for reproducibility. Authentication: the
HF token is read from the `HF_TOKEN` environment variable or the standard
HuggingFace cache. The notebook reads the same token from Colab Secrets.

## Local Smoke Test

Install a CUDA PyTorch build first, then install the project dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy>=2,<3" "causal-conv1d>=1.4.0" --no-build-isolation
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation
pip install -r requirements.txt
```

Polymarket smoke run:

```bash
python -m finmamba3.train --hours-train 1 --hours-val 0.25 --JointTrainAgent.SampleMaxSteps 20
```

FI-2010 smoke run:

```bash
python -m finmamba3.train \
    --config configs/fi2010.yaml \
    --dataset fi2010 \
    --JointTrainAgent.SampleMaxSteps 20
```

Expected Polymarket data layout:

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

Expected FI-2010 data layout:

```text
data/
  train/Train_Dst_NoAuction_DecPre_CF_7.txt
  validation/Test_Dst_NoAuction_DecPre_CF_7.txt
```

## Repository Layout

```text
notebooks/
  colab_lob_pretrain.ipynb
pyproject.toml                  Editable-install packaging (pip install -e .)
configs/
  lob.yaml                      Polymarket default. Mamba-3 (SISO) backbone, 94-dim features.
  fi2010.yaml                   FI-2010 default. Mamba-3 (SISO), 46-dim features.
  lob_em.yaml                   Episodic memory enabled (FIFO writes).
  lob_full_ablation.yaml        Student-t + Hawkes + Settlement + EM-novelty + multi-threshold.
  lob_aggregate_only.yaml       Tick-aggregate features only; per-level depth masked.
  lob_studentt.yaml             Student-t reconstruction likelihood instead of MSE.
  lob_mamba1.yaml               Mamba-1 backbone for the architecture sweep.
  lob_mamba2.yaml               Mamba-2 backbone for the architecture sweep.
  lob_transformer.yaml          Transformer backbone for the architecture sweep.
  lob_diagnose.yaml             Collapse-diagnosis knobs.
src/finmamba3/
  train.py                      LOB pretraining entrypoint (python -m finmamba3.train)
  sequence_builder.py           Build normalized LOB sequences; populate the replay buffer
  train_step.py                 World-model update step
  config.py                     DotDict + dotted CLI config overrides
  training_utils.py             Seeding, wandb / no-op logger, EMA
  hf_hub.py                     HuggingFace dataset download / checkpoint upload
  weight_init.py                Layer / weight initializers
  replay_buffer.py
  envs/
    lob_features.py             94-dim microstructure-aware feature engineering
    fi2010_loader.py            FI-2010 NoAuction DecPre/ZScore CF reader (46-dim)
    kaggle_lob_loader.py        Kaggle high-frequency crypto spot LOB loader (Binance-style books)
    bar_aggregation.py          Time/volume/dollar/tick-imbalance/CUSUM bars
    lob_labels.py               Triple-barrier and multi-threshold direction targets
    lob_env.py                  Gymnasium trading environment with reward variants
  models/
    world_model.py              Mamba-3 world model (core orchestration)
    world_model_heads.py        DistHead / RewardHead / TerminationHead
    lob_encoder.py              Transformer-over-depth-tokens encoder + Student-t decoder
    mamba_backbone.py           FinMamba3 sequence wrapper for upstream Mamba
    transformer.py              Stochastic Transformer backbone
    attention.py                Attention blocks + KV cache
    regime_modulation.py        Regime FiLM modulator
    lob_heads.py                Direction, regime, episodic memory, Hawkes, settlement heads
    losses.py                   Symlog two-hot + categorical KL (free-bits)
    activations.py              Config-name -> activation registry
    laprop.py                   LaProp optimizer
  rl/
    actor_critic.py             ActorCriticAgent (Phase B)
    ppo.py                      PPOAgent (Phase B, currently unused)
    returns.py                  Lambda-return + percentile helpers
    normalization.py            RunningMeanStd + VecNormalize
  baselines/
    deeplob.py                  DeepLOB CNN+LSTM reference baseline
    linear_ar.py                Linear vector autoregression floor baseline
  backtester/
    data_loader.py              SQLite/CSV/Parquet loader -> timeline + settlements
    strategy.py                 BaseStrategy ABC + market dataclasses
    engine/                     Vendored DATAHACKS2026 order-matching engine (the parity oracle)
      engine.py                 Main tick-loop orchestrator
      execution.py              Order matching against recorded order books
      market_manager.py         Active markets, lifecycle state, and settlements
      portfolio.py              Cash, positions, and P&L tracking
    engine_cpp/                 Opt-in native C++ accelerator, parity-checked against the Python engine
      _engine.cpp               pybind11 marshal-once / run-many native source
      build.py                  Standalone in-place build for the native engine
  eval/
    pnl_backtest.py             Settlement-accurate order-matching PnL adapter (produces the paper economic tables)
    survivability.py            Kelly sizing + fat-tail bootstrap statistics (pure NumPy)
    signals.py                  World-model probability / EV inference + causal gate series
    strategies.py               CusumGate, WorldModelStrategy, NaiveLagStrategy
    backtest.py                 Gym-env PnL/Sharpe/MaxDD harness for a frozen world model
    run_backtest_cli.py         CLI wrapper around the gym-env backtest
    predictability.py           Efficiency-ratio predictability regime axis used by the PnL gate
    simple_baseline.py          Logistic + GBT baselines on the spot features (architecture-independence control)
    competition_strategy.py     DATAHACKS BaseStrategy adapter
    regime_split.py             Time and volatility splits for non-stationarity tests
    compare_direction.py        World-model vs LinearAR vs DeepLOB direction benchmark
    benchmark_fi2010.py         FI-2010 macro-F1 benchmark against the LOBCAST paper (Prata et al. 2023)
    eval_backbone_metrics.py    Held-out recon NLL + direction macro-F1 (backbone ablation)
    eval_regime_generalization.py         Cross-regime generalization-gap evaluator (Polymarket)
    eval_regime_generalization_fi2010.py  FI-2010 cross-regime generalization-gap evaluator
    make_paper_figures.py       Generate the model-free paper figures from persisted artifacts
    diag_dirhead.py             Direction-head diagnostics
    diag_regime_labels.py       Regime-label distribution diagnostics
    diag_router.py              Regime-FiLM router-discrimination diagnostic
    diagnose_collapse.py        Temporal-prior collapse diagnostics
    imagination_smoke.py        Phase-B imagination smoke test
tests/
  test_fi2010_pipeline.py             Full FI-2010 path end-to-end
  test_lob_features.py                Feature engineering
  test_lob_aggregation.py             Bar aggregation
  test_lob_labels.py                  Direction / triple-barrier labels
  test_kaggle_lob_loader.py           Kaggle crypto LOB loader
  test_baselines.py                   DeepLOB + LinearAR baselines
  test_polymarket_lob_env.py          Gymnasium trading environment
  test_replay_buffer.py               Replay buffer
  test_world_model.py                 World-model heads + update step
  test_world_model_mamba_backbones.py Mamba-1/2/3 backbone wiring
  test_transformer_modern.py          Pre-norm Transformer backbone
  test_regime_modulation.py           Regime FiLM modulator
  test_lob_heads.py                   Direction / settlement / edge heads
  test_train_integration.py           Training loop integration
  test_cross_interval_context.py      Cross-interval sequence context
  test_window_volatility_split.py     Volatility-window split
  test_predictability.py              Predictability regime axis
  test_regime_split.py                Time / volatility non-stationarity splits
  test_compare_direction.py           Direction benchmark
  test_benchmark_fi2010.py            FI-2010 macro-F1 benchmark
  test_eval_backbone_metrics.py       Backbone-ablation metrics
  test_eval_regime_generalization.py  Cross-regime generalization (Polymarket)
  test_eval_regime_generalization_fi2010.py  Cross-regime generalization (FI-2010)
  test_competition_strategy.py        DATAHACKS strategy adapter
  test_run_backtest_cli.py            Gym-env backtest CLI
  test_pnl_backtest.py                Order-matching PnL adapter + survivability
  test_engine_cpp_parity.py           C++/Python engine parity oracle
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
- We swap Mamba-2 for Mamba-3 (SISO; see below).
- We dropped Drama's "Dynamic Frequency-based Sampling" replay scheme in
  favour of an imagine-counter penalty in `replay_buffer.py:47`. This is an
  undocumented deviation; either bring it back as a baseline or write a
  paragraph defending why tick-frequency, not state-visit-frequency, is the
  bottleneck for LOB sequences.

### Mamba-3 (Lahoti et al., arxiv 2603.15569, ICLR 2026)

We instantiate the dynamics with a Mamba-3 SISO backbone. For reference, the
upstream headline numbers: at 1.5B scale, Mamba-3 + MIMO improves average
downstream accuracy by 1.8 points over Gated DeltaNet (0.6 from M3, 1.2 from
MIMO), and Mamba-3 matches Mamba-2 perplexity at half the state size. No
published LOB / market-microstructure paper uses Mamba-3, and only a handful
even use Mamba-2 (`MambaTS`, `MambaStock`, `CryptoMamba`). The MIMO variant is
outside our hardware envelope and is not claimed; the backbone ablation finds
no backbone dominates, so we treat the sequence model as a substrate.

We ship matched-config yamls so the ablation table can be produced with one
command per backbone: `lob_mamba1.yaml`, `lob_mamba2.yaml`,
`lob_transformer.yaml`, and the Mamba-3 SISO default.

### Episodic and retrieval-augmented memory (arxiv 2506.06326, 2602.16192, 2202.08417)

The repo's `EpisodicMemory` (`src/finmamba3/models/lob_heads.py`) is a CPU-side
top-k cosine retriever with FIFO eviction. New: `UseNovelty` flag turns the
write policy into a KL-novelty filter so the buffer becomes a regime catalog
rather than a sliding window of recent states. The `lob_em.yaml`
ablation enables the FIFO variant; `lob_full_ablation.yaml`
enables the novelty-filtered variant. Both compare against the default (off).

### Modern world-model baselines

- DreamerV3 (Hafner et al., Nature 2025): RSSM with 32x32 categorical latents,
  symlog two-hot reward decoding, single hyperparameter set across domains.
  We borrow the symlog two-hot decoder (`losses.py`) and shrink the
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
baseline at `src/finmamba3/baselines/`. Recent transformer-based competitors worth
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
`src/finmamba3/envs/bar_aggregation.py` module is essential for an honest training
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
denoising tools live in `src/finmamba3/envs/`:

1. **Bar aggregation** (`bar_aggregation.py`). Replace the raw tick stream
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

Most LOB RL papers copy a Cartea/Jaimungal market-making reward or 
retrofit Sharpe-on-PnL. The env exposes three reward kinds
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

- `HawkesIntensityHead` (`lob_heads.py`). Predicts log-intensity for
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

Enable all three at once via `lob_full_ablation.yaml`.

## Evaluation

Three CLIs in `src/finmamba3/eval/` consume a trained world-model checkpoint and emit
the artifacts the paper's evaluation table needs.

### Diagnose a checkpoint

`src/finmamba3/eval/diagnose_collapse.py` regenerates the 32-step imagine rollout, computes
posterior and prior categorical entropy on a val batch, and prints the top
per-feature val MSE. Use after a Phase A run that ended with
`Imagine/mid_norm_std = 0` or an unexpectedly large `val_loss`.

```bash
python -m finmamba3.eval.diagnose_collapse \
    --checkpoint saved_models/lob/LOB/<run_id>/ckpt/world_model.pth \
    --config configs/lob.yaml \
    --data-val data/validation \
    --norm-path saved_models/lob/normalization.json \
    --out-dir notes/
```

Outputs: `notes/diagnose_rollout_<slug>.npy`,
`notes/diagnose_rollout_<slug>.png`, and `notes/diagnose_summary_<slug>.json`.

### Compare against direction-prediction baselines

`src/finmamba3/eval/compare_direction.py` evaluates the world-model direction head,
DeepLOB (trained from scratch on the train split), and a closed-form LinearAR
on the same val split, across one or more direction thresholds. Emits a
markdown table.

```bash
python -m finmamba3.eval.compare_direction \
    --world-checkpoint saved_models/lob/LOB/<run_id>/ckpt/world_model.pth \
    --config configs/lob.yaml \
    --data-train data/train --data-val data/validation \
    --thresholds 0.001,0.005,0.01 \
    --baselines world_model,deeplob,linear_ar \
    --epochs-deeplob 3 \
    --out reports/direction_comparison.md
```

### Two evaluation paths

The repo keeps two backtest stacks, deliberately:

- **Order-matching engine** (`finmamba3.eval.pnl_backtest` over `backtester/engine` plus the
  opt-in `engine_cpp` accelerator) is the settlement-accurate path that produces the paper's
  economic / PnL tables. Run it with `finmamba3-backtest` (or `python -m finmamba3.eval.pnl_backtest`).
- **Gym-env harness** (`finmamba3.eval.backtest` via `run_backtest_cli`, below) is a lighter
  Sharpe / PnL / drawdown sketch over `PolymarketLOBEnv`, handy as a quick standalone check.

Both are live and test-covered; use the order-matching path for any reported economic number.

### Run a backtest with a frozen world model

`src/finmamba3/eval/run_backtest_cli.py` wraps the GreedyDirectionPolicy around a frozen
world model, runs `run_backtest` against `PolymarketLOBEnv`, and writes
`BacktestMetrics` (PnL, Sharpe, MaxDD, win rate, portfolio curve) as JSON.

```bash
python -m finmamba3.eval.run_backtest_cli \
    --world-checkpoint saved_models/lob/LOB/<run_id>/ckpt/world_model.pth \
    --config configs/lob.yaml \
    --data-val data/validation \
    --max-steps 5000 \
    --regime-split none \
    --out reports/backtest_<run_id>.json
```

Pass `--regime-split time:<unix_ts>` to evaluate only on markets resolved
after the cutoff, or `--regime-split volatility:<quantile>` to evaluate on
the high-volatility tail. Reuse the same checkpoint with
`lob_em.yaml` (episodic memory enabled) to produce the
non-stationarity A/B numbers.
