# FinDrama Handoff: Regime-Modulated Mamba for Polymarket Binary LOBs

Living log of what changed, what was tried, what was pivoted, and what is left.
For the architecture/design rationale see `research_notes.md`.

## Branch & remotes
- Work branch: `regime-film-binary-features` on `fork` = `https://github.com/Ruuudy1/FinDrama`.
- `origin` is the upstream `realwenlongwang/Drama` (do not push there).
- The Colab notebook clones `fork` and self-points `BRANCH`/`DATASET` at this branch.

## Thesis
A Mamba world model whose selective-scan dynamics (Delta/B/C) are modulated by an
inferred latent regime generalizes better under distribution shift (esp. volatility
regimes) than an unmodulated sequence model, on Polymarket binary-outcome LOBs.

## Changes by workstream
| # | Workstream | Status | Key files |
|---|-----------|--------|-----------|
| A | Binary-market features (boundary dist/scaled depth, logit-mid velocity/accel, Amihud, variance ratio); 94->100 dims; width-aware normalization; gated by `Encoder.BinaryMarketFeatures` | Done, CPU-tested | `src/envs/lob_features.py`, `tests/test_lob_features.py`, `configure_lob.yaml`, `src/train_lob.py` |
| B | Regime inference + FiLM modulator (zero-init hypernetwork, identity at init); wired into the Mamba block loop | Done, CPU-tested | `src/sub_models/regime_modulation.py`, `src/sub_models/fin_mamba.py`, `tests/test_regime_modulation.py` |
| C | Load-balance regularizer + 12-loss contract + `RegimeFiLM` config | Done, CPU-tested | `src/sub_models/world_models.py`, `src/training_steps.py`, `configure_lob.yaml` |
| Util | GPU-utilization fixes: batch 64->512, AccumSteps 2->1, BatchLength 32->64, Compile on, log every 50 steps, sampler with-replacement fallback, non-fatal tilelang | Done (config), GPU-validate | `configure_lob.yaml`, `src/training_steps.py`, `src/replay_buffer.py`, notebook |
| LR | LR bumped ~3x for the 4x effective batch (Laprop 4e-5->1.2e-4, Adam 1e-4->3e-4, warmup 500->1000) | Done | `configure_lob.yaml` |
| E | Competition adapter `FinDramaCompetitionStrategy` (reuses `extract_features` over a rolling `TickData` window = zero train/serve skew) | Done; feature path CPU-tested, model forward GPU-pending | `src/eval/competition_strategy.py`, `tests/test_competition_strategy.py` |
| F | Phase B imagination trainer (rewrote the broken `phase_b_smoke.py`) | Scaffold; GPU + prereqs pending | `src/eval/phase_b_smoke.py` |
| G | Architecture/design write-up | Done | `research_notes.md` |
| D | Baseline-vs-treatment + distribution-shift eval | Pending (Colab GPU) | commands in `research_notes.md` |

## Key decisions & pivots
- **Modulation = FiLM on block inputs, not a kernel fork.** Delta/B/C are input-dependent in a
  selective SSM, so FiLM on each block's input reaches them without editing `mamba_ssm`. A kernel
  fork would fight the repo's no-vendored-copy rule and force rebuilding all five HF arch wheels.
- **Zero-init hypernetwork (gamma=1, beta=0).** Untrained modulator == unmodulated backbone, so
  the regime-off baseline is an exact control. Verified by a CPU integration test.
- **Regime source = latent + microstructure prior.** Structural features (Amihud, variance ratio,
  boundary distance) live in the obs, so the regime head reading the stem summary reads them.
- **Features appended, not interleaved**, to keep tick indices, `midprice_index`, decoder size
  indices, and bar sum-indices valid; gated so FI-2010's separate pipeline is untouched.
- **Left `lob_aggregation.py` alone (YAGNI):** `DEFAULT_SUM_INDICES` is only referenced by tests;
  bar aggregation is not in the live training path.
- **Low GPU util was starvation, not I/O or kernel speed** (only 1.8/23 GB used). Fix = bigger
  batch + drop accumulation + log less often (removed a per-step GPU->CPU sync) + torch.compile.
  Added a with-replacement sampler fallback so the 512 batch is safe on short markets.
- **Phase B pivot.** A naive env-rollout Phase B is blocked: (1) the env obs is multi-market 18-dim
  while the world model consumes 100-dim LOB features (schema mismatch), (2) Phase A has no reward
  signal (reward=0), (3) Phase A builds the WM with `action_dim=1`. Pivoted to the **imagination
  path** (DreamerV3 style), which runs entirely in the WM latent space and avoids the obs mismatch.
  The old `phase_b_smoke.py` was broken (stale agent/checkpoint signatures, wrong obs); it was
  rewritten to the imagination path.
- **Competition adapter realization:** FinDrama's `src/lob/backtester/strategy.py` *is* the
  competition `BaseStrategy` interface, so the adapter targets it directly and reuses the training
  feature pipeline for zero skew.

## Verification status
- CPU suite: **89 passed, 4 skipped, 1 failed**. The single failure
  `tests/test_fi2010_pipeline.py::test_load_invalid_split_raises` is **pre-existing** (fails on a
  clean tree with our changes stashed; FI-2010 split validation, untouched by this work).
- Skipped: 4 CUDA-only tests (no GPU locally; this box is Python 3.13 / CPU).
- The first real GPU execution was the L4 Colab run; it trained, re-fit 20-tick normalization, and
  checkpointed. GPU-pending: baseline-vs-treatment numbers, backtest metrics, competition-harness
  scoring, the Phase B imagination run.

## Known gaps & next steps
1. **Run D on Colab**: baseline (`RegimeFiLM.Enabled=false`) vs treatment (`true`); compare on
   `run_backtest_cli --regime-split volatility:0.5`. Claim: treatment's edge is larger on the
   held-out high-vol split.
2. **Make Phase B learn**: build the Phase-A WM with `action_dim` = agent action space (so imagined
   actions feed the dynamics), and enable the reward head (`Reward.Enabled`) trained on an env
   reward signal. Until then the imagined reward is zero and the agent has nothing to optimize.
3. **Tune LR/batch** to the GPU: confirm util climbs and VRAM fills; adjust `BatchSize` and LR if
   val loss regresses (the pre-tuning run plateaued at val 368 and early-stopped at 18k).
4. **TileLang/MIMO**: tilelang now installs (non-fatal). To try the MIMO fast path set
   `Mamba3.is_mimo true` and measure; on L4 it may still fall back (tuned for H100).
5. Optional: fix the pre-existing FI-2010 split-validation test.

## Iteration log (Colab L4)
- Run 1 (pre-tuning): batch 64, eff 128, LR 4e-5, len 32 -> GPU util ~38%, mem 1.8/23 GB,
  best val 368, early-stopped 18k. Diagnosis: GPU was starved, not I/O- or kernel-bound.
- Util fix: batch 512, AccumSteps 1, len 64, Compile on, log every 50 -> Run 2 hit GPU util
  100%, mem 13/22 GB. (Compute util 100% with memory headroom is the goal; 13/22 GB is fine,
  could push batch higher but there is no need once compute-bound.)
- Run 2 CRASHED at step 7189 with a CUDA device-side assert ("probability tensor contains
  inf/nan"). Root cause: the replay buffer's non-imagine sampler used an unstable manual
  softmax over visit counts; on the big batch the counts grow large, exp underflows to
  all-zeros -> NaN probs -> torch.multinomial assert (surfaced async in the encoder, a red
  herring). Fixed with a numerically stable torch.softmax. The batch-64 run never hit it
  because counts grew 8x slower and it early-stopped first.
- LR re-tuned 3x -> 2x (sqrt-scaling, Laprop 8e-5). Run 2 best 415 > Run 1 best 368 indicated
  the 3x bump was too hot for the 4x batch (it also crashed before converging).
- Cosmetic: tqdm throttled to every 50 steps (miniters); datetime.utcnow -> now(UTC);
  TransformerEncoder enable_nested_tensor=False to silence the startup warning.
- Ignore: the W&B "Weave" suggestion (generic wandb promo, irrelevant to non-LLM training)
  and the tvm_ffi "Field duplicates ancestor" warnings (harmless tilelang/tvm import noise;
  re-comment the tilelang install if staying on SISO to drop them and save install time).

## How to run (Colab)
- Open the notebook from this branch and Run All (baseline, high-util defaults already set).
- Treatment: add `'--Models.WorldModel.RegimeFiLM.Enabled', 'true'` to the notebook's `run_train`
  extra args, or set it in `configure_lob.yaml`.
- Phase B (after a Phase-A checkpoint exists, GPU):
  `python -m eval.phase_b_smoke --checkpoint <ckpt>.pth --config src/config_files/configure_lob.yaml --data-train data/train --steps 200`
