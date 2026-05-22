# FinDrama: Regime-Modulated Mamba World Model for Polymarket Binary LOBs

Research notes for adapting the FinDrama (Drama / Mamba-Atari) world model to Polymarket
BTC binary-outcome limit order books, modeled as a non-stationary POMDP with latent regimes.

Thesis: a Mamba world model whose **selective-scan dynamics are modulated by an inferred
latent regime** generalizes better under distribution shift than an unmodulated sequence
model, especially across volatility regimes.

## 1. Architecture topology

Offline world-model pretraining (DreamerV3-style, Phase A):

```
obs (B, L, 100)                      flat LOB features (K=10 levels x 8 + 20 tick)
   |  LOBEncoder (Transformer over depth tokens + CLS)        src/sub_models/lob_encoder.py
   v
embedding (B, L, 1024)
   |  DistHead.forward_post -> 16x16 categorical posterior    src/sub_models/world_models.py
   v
posterior sample -> flatten -> latent (B, L, 256)
   |  FinMambaSequenceModel  (Mamba3 MIMO, n_layer=4, d_model=512)   src/sub_models/fin_mamba.py
   |    stem:  [latent (+ optional action)] -> RMSNorm -> SiLU -> (B, L, 512)
   |    >>> RegimeFiLMModulator infers regime from the stem summary <<<
   |    per block i:  x = RMSNorm(h);  x = gamma_i * x + beta_i;  h = h + Mamba_i(x)
   v
hidden (B, L, 512)
   |  DistHead.forward_prior -> 16x16 categorical prior  (dynamics / representation KL)
   |  LOBDecoder -> reconstruct obs (B, L, 100)
   |  aux heads: Direction (3-class), Hawkes, Settlement (binary outcome)
   v
losses: reconstruction + reward + termination + dynamics_kl + representation_kl
        + direction + hawkes + settlement + regime  (12-tuple, src/training_steps.py)
```

Phase B (RL): `ActorCriticAgent` / `PPOAgent` (`src/agents.py`) learn in imagination rolled
out by `WorldModel.imagine_data`. Because imagination calls the same FiLM-modulated
`sequence_model`, the agent dreams inside a regime-aware world model with no agent-side change.

## 2. Regime modulation of the Mamba selection mechanism (the core contribution)

The brief asks for the inferred regime to "dynamically alter the Delta, B, or C matrices" of
the Mamba selective scan, rather than concatenating a regime id. The key constraint: the
Mamba blocks come from the upstream `mamba_ssm` package, and FinDrama deliberately refuses
vendored copies (`src/sub_models/fin_mamba.py:34-78`) and runs CUDA/TileLang kernels. Editing
the kernel to scale dt/B/C directly would be fragile and would force rebuilding all five
prebuilt HF arch wheels.

Insight used here: in a *selective* SSM, Delta, B and C are input-dependent functions of each
block's input. So applying a per-block FiLM transform `x -> gamma * x + beta` to the block
input propagates into Delta, B and C **through the selection mechanism itself**, while leaving
the CUDA kernel untouched and the wheels valid. This is the FiLM-on-block-inputs design.

Implementation (`src/sub_models/regime_modulation.py`):
- `RegimeFiLMModulator` wraps a `RegimeHead` (reused from `lob_auxiliary.py`) that infers a
  soft regime distribution and embedding from the stem summary (causal, per timestep), and a
  hypernetwork mapping the embedding to per-block `(gamma, beta)` over `d_model` channels.
- The hypernetwork is **zero-initialized** and gated as `gamma = 1 + tanh(.)`, `beta = tanh(.)`,
  so at init `gamma=1, beta=0` (exact identity). An untrained modulator therefore reproduces
  the unmodulated backbone bit-for-bit, which makes the regime-off baseline a clean control
  and stabilizes warm-start. `tanh` also bounds gamma in (0, 2) for bf16 safety.
- Wired into `FinMambaSequenceModel.forward` (`fin_mamba.py`): regime is inferred once from
  the stem output, then `(gamma_i, beta_i)` are applied to each block's normalized input. A
  `return_regime` flag (only set by `WorldModel.update`) surfaces the regime logits for the loss.

Regime inference is **latent + microstructure-prior**: the structural-break features (below)
are part of the obs, so they are encoded into the latent and the stem summary the regime head
reads. A load-balancing regularizer (`regime_load_balance_loss`, Switch-Transformer style)
maximizes the entropy of the batch-averaged regime distribution to prevent collapse onto a
single regime, while leaving per-step assignments free to be peaky. Weight = `RegimeFiLM.EntropyCoef`.

The pre-existing post-hoc `RegimeConditioner` (`world_models.condition_dist_feat`) is left in
place and independently config-gated; it conditions the stack *output*, whereas this work
conditions the scan *input*. They can be combined or ablated independently.

## 3. Binary-market feature engineering

Polymarket prices are bounded probabilities in [0, 1]; liquidity and information dynamics
change violently near the 0/1 boundaries and are sensitive to time-to-expiry. The generic
94-dim LOB vector ignored this. Six tick features are appended (gated by
`Encoder.BinaryMarketFeatures`, Polymarket only; FI-2010 is untouched) in
`append_binary_market_features` (`src/envs/lob_features.py`), taking F_TICK 14 -> 20 and the
flat dim 94 -> 100:

- `boundary_distance = min(mid, 1-mid)` -- proximity to a probability boundary.
- `boundary_scaled_depth` -- log book depth scaled by boundary proximity (market-maker
  behavior degrades near 0/1).
- `logit_mid_velocity`, `logit_mid_acceleration` -- first/second differences of the log-odds
  `logit(mid)`, the natural unbounded coordinate for an implied probability (implied-prob
  velocity / acceleration).
- `amihud_illiquidity` -- rolling `|logit return| / depth` (Amihud 2002), an illiquidity /
  structural-break signal.
- `variance_ratio` -- rolling VR(2) of logit returns (Lo-MacKinlay); <1 mean-reverting,
  ~1 random walk, >1 trending. A regime/structural-break statistic.

Features are appended (not interleaved), so existing tick indices, `midprice_index`, decoder
size indices, and bar `sum_indices` stay valid. Normalization (`fit/from/to_json`) is
schema-width-aware and pins floors for the 20-wide schema; stats are regenerated on the next
training run (`fit_stats=True` on the train split).

## 4. How to run baseline vs treatment (Colab GPU)

Mamba requires CUDA, so runs happen on Colab (the local box is CPU / Python 3.13). The
prebuilt wheels `sj-hryi/FinDrama-wheels/wheels-py312-torch260-cu124-sm{80,89,90,...}` cover
A100/L4/H100 and need no rebuild; the FiLM change is pure PyTorch and does not recompile the
CUDA extensions.

Baseline (regime off) vs treatment (regime on), both with the new binary features:

```
# Baseline.
python -m src.train_lob --config src/config_files/configure_lob.yaml \
    --data-train data/train --data-val data/validation

# Treatment (in-scan regime FiLM on).
python -m src.train_lob --config src/config_files/configure_lob.yaml \
    --data-train data/train --data-val data/validation \
    --Models.WorldModel.RegimeFiLM.Enabled true
```

Distribution-shift evaluation (held-out high-volatility regime) and directional baselines:

```
python -m eval.run_backtest_cli --world-checkpoint <ckpt>.pth \
    --config src/config_files/configure_lob.yaml --data-val data/validation \
    --regime-split volatility:0.5            # reports return, sharpe, max_drawdown, win_rate

python -m eval.compare_direction --world-checkpoint <ckpt>.pth \
    --config src/config_files/configure_lob.yaml \
    --data-train data/train --data-val data/validation \
    --baselines world_model,deeplob,linear_ar --thresholds 0.001,0.005,0.01
```

Headline metric: the treatment's edge over the baseline should be *larger* on the held-out
high-volatility split than in-distribution (the regime-shift generalization claim).

## 5. Verification status

Implemented and CPU-tested (green; the only failing test in the suite,
`test_fi2010_pipeline.py::test_load_invalid_split_raises`, is pre-existing and unrelated to
this work):

- Binary-market features + width-aware normalization -- `tests/test_lob_features.py`.
- Regime inference + FiLM modulator (identity-at-init, load balance, gradient flow) --
  `tests/test_regime_modulation.py`.
- Full FiLM integration through `FinMambaSequenceModel` + `WorldModel.update` via the
  fake-mamba CPU harness, including init-time baseline parity --
  `tests/test_world_model_mamba_backbones.py::test_regime_film_identity_at_init_and_update_finite`.

Pending (require Colab GPU or are downstream): end-to-end baseline-vs-treatment training and
the distribution-shift numbers; the competition-backtester adapter
(`FinDramaCompetitionStrategy`); the RL Phase B joint loop in `train_lob.py:main()` and a
held-out env evaluation roll.
```
