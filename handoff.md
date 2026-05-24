# FinMamba3 Handoff: Regime-Modulated Mamba for Polymarket Binary LOBs

Living log of what changed, what was tried, what was pivoted, and what is left.
For the architecture/design rationale see `research_notes.md`.

## 2026-05-23 - Code-quality rework (branch `code-quality-rework`)
Full pass to the PEP8-based style rules + SOLID/YAGNI principles, in six verified
stages (CPU suite held at 89 passed / 4 CUDA-skipped / 1 pre-existing FI-2010 fail
throughout):
1. **Packaging.** Everything moved under `src/finmamba3/`; added `pyproject.toml`
   (editable install). All intra-repo imports are absolute `finmamba3.*`; the ~17
   `sys.path.insert` bootstraps are gone. Train with `python -m finmamba3.train`
   (was `python src/train_lob.py`); the notebook now does `pip install -e .`.
2. **Renames.** `sub_models/`->`models/`, `lob/backtester/`->`backtester/`,
   `src/config_files/configure_*.yaml`->`configs/*.yaml`; modules renamed for
   clarity (`fin_mamba`->`mamba_backbone`, `lob_auxiliary`->`lob_heads`,
   `tools`->`weight_init`, `utils`->`training_utils`, `utils_hf`->`hf_hub`,
   `config_utils`->`config`, `training_steps`->`train_step`,
   `polymarket_lob_env`->`lob_env`, `lob_aggregation`->`bar_aggregation`,
   `diagnose_run`->`diagnose_collapse`, `phase_b_smoke`->`imagination_smoke`).
3. **Style.** Region-imported, no trailing whitespace, no blank lines in bodies.
4. **Rule-8 purge.** ~70 `getattr(cfg,'X',d)`->`cfg.get('X',d)`; weight-init
   isinstance/hasattr -> type registry + `bias is not None`; `getattr(nn,name)`
   -> `ACTIVATION_BY_NAME`; wandb try-import -> `WandbLogger`/`NoOpLogger` chosen
   by a `UseWandb` flag; orjson try-import dropped; sets -> dict-keyset/list.
   Carve-outs kept (documented in commits): `mamba_backbone.py` mamba_ssm/triton
   import machinery (GPU-only, fail-fast); `config.py` isinstance for the
   auto-wrapping DotDict + argparse type dispatch; boundary I/O try/except.
5. **God-module splits.** `world_models.py`->`world_model.py` + `world_model_heads.py`
   (KL loss to `losses.py`, dead `imagine_data2` removed); `train_lob.py`->`train.py`
   + `sequence_builder.py`; `agents.py`->`rl/{actor_critic,ppo,returns,normalization}.py`.
   Fixed typos `stright_throught_gradient`->`straight_through_gradient`,
   `comput_loss`->`compute_loss`.
6. **Docs/notebook** reconciled to the new paths and the HF 3-repo split.

**Flagged for the owner (rule-8 "ask, don't guess"):** the `lob_env` reward kind
`settlement_calibrated` reads `Settlement.payoff`/`position_value`, which the
dataclass does not define (dead placeholder). `PPOAgent` is unused, and its dead
`calc_gae_and_reward_to_go` references `self.lamb` (should be `self.lambd`).

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
| A | Binary-market features (boundary dist/scaled depth, logit-mid velocity/accel, Amihud, variance ratio); 94->100 dims; width-aware normalization; gated by `Encoder.BinaryMarketFeatures` | Done, CPU-tested | `src/finmamba3/envs/lob_features.py`, `tests/test_lob_features.py`, `lob.yaml`, `src/finmamba3/train.py` |
| B | Regime inference + FiLM modulator (zero-init hypernetwork, identity at init); wired into the Mamba block loop | Done, CPU-tested | `src/finmamba3/models/regime_modulation.py`, `src/finmamba3/models/mamba_backbone.py`, `tests/test_regime_modulation.py` |
| C | Load-balance regularizer + 12-loss contract + `RegimeFiLM` config | Done, CPU-tested | `src/finmamba3/models/world_model.py`, `src/finmamba3/train_step.py`, `lob.yaml` |
| Util | GPU-utilization fixes: batch 64->512, AccumSteps 2->1, BatchLength 32->64, Compile on, log every 50 steps, sampler with-replacement fallback, non-fatal tilelang | Done (config), GPU-validate | `lob.yaml`, `src/finmamba3/train_step.py`, `src/finmamba3/replay_buffer.py`, notebook |
| LR | LR bumped ~3x for the 4x effective batch (Laprop 4e-5->1.2e-4, Adam 1e-4->3e-4, warmup 500->1000) | Done | `lob.yaml` |
| E | Competition adapter `FinMamba3CompetitionStrategy` (reuses `extract_features` over a rolling `TickData` window = zero train/serve skew) | Done; feature path CPU-tested, model forward GPU-pending | `src/finmamba3/eval/competition_strategy.py`, `tests/test_competition_strategy.py` |
| F | Phase B imagination trainer (rewrote the broken `imagination_smoke.py`) | Scaffold; GPU + prereqs pending | `src/finmamba3/eval/imagination_smoke.py` |
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
- **Left `bar_aggregation.py` alone (YAGNI):** `DEFAULT_SUM_INDICES` is only referenced by tests;
  bar aggregation is not in the live training path.
- **Low GPU util was starvation, not I/O or kernel speed** (only 1.8/23 GB used). Fix = bigger
  batch + drop accumulation + log less often (removed a per-step GPU->CPU sync) + torch.compile.
  Added a with-replacement sampler fallback so the 512 batch is safe on short markets.
- **Phase B pivot.** A naive env-rollout Phase B is blocked: (1) the env obs is multi-market 18-dim
  while the world model consumes 100-dim LOB features (schema mismatch), (2) Phase A has no reward
  signal (reward=0), (3) Phase A builds the WM with `action_dim=1`. Pivoted to the **imagination
  path** (DreamerV3 style), which runs entirely in the WM latent space and avoids the obs mismatch.
  The old `imagination_smoke.py` was broken (stale agent/checkpoint signatures, wrong obs); it was
  rewritten to the imagination path.
- **Competition adapter realization:** FinMamba3's `src/finmamba3/backtester/strategy.py` *is* the
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

- Run 3 (crash fixed, LR 8e-5): completed all 20000 steps in ~3h09m, best val **390** @ 19k.
  **Key finding / pivot:** 390 is worse than Run 1's **368** (batch 64) AND ~5x slower wall
  clock. For this small model + single-market dataset, 100% GPU util was a vanity metric: the
  large batch reduced the gradient noise that was helping and wasted hours. **Reverted** batch
  64 / AccumSteps 2 / len 32 / LR 4e-5 / Compile off (the proven config). Kept the crash fix,
  LogEvery=50, tqdm throttle, and warning fixes. Lesson: optimize time-to-target-val-loss, not
  utilization %.

## How to run (Colab)
- Open the notebook from this branch and Run All (baseline, high-util defaults already set).
- Treatment: add `'--Models.WorldModel.RegimeFiLM.Enabled', 'true'` to the notebook's `run_train`
  extra args, or set it in `lob.yaml`.
- Phase B (after a Phase-A checkpoint exists, GPU):
  `python -m finmamba3.eval.imagination_smoke --checkpoint <ckpt>.pth --config configs/lob.yaml --data-train data/train --steps 200`

## 2026-05-23 (post-restructure) - Parity verification + volatility-split wiring
The repo was fully restructured (separation-of-concerns / SRP / DRY / open-closed).
Package is now `finmamba3` under `src/finmamba3/`; train entry `python -m
finmamba3.train --config configs/lob.yaml --dataset polymarket` (console script
`finmamba3-train`); backtest CLI `python -m finmamba3.eval.run_backtest_cli`. This
session interpreted the batch-64 baseline run, verified the restructure changed no
behavior, and unblocked Experiment D.

### Baseline run interpretation (batch-64 rerun, PRE-restructure code)
- Reproduced the prediction exactly: early-stopped at step 18k, **best val 368.1977 @
  step 13k**, ~38 min, GPU util ~43%. Beats the batch-512 run (val 390) on both loss and
  wall-clock (~5x); re-confirms "optimize time-to-target-val-loss, not utilization %".
- Early-stop fired because nothing improved after 13k -> 20k SampleMaxSteps is more than
  this config needs (effectively ~13k-bound).
- This was the **baseline** (`RegimeFiLM.Enabled: False`). The val-368 checkpoint on HF
  (sj-hryi/FinDrama-checkpoints) was produced by PRE-restructure code, so the baseline
  arm of Exp D will be retrained on the restructured code (owner's choice) to rule out
  drift before any A/B.
- Feature-MSE diagnostics (FeatureDim 100 = 10 levels x 8 [idx 0-79] + 20 tick [80-99],
  binaries at 94-99): top-error features 59/64/67/72/75 are all deep book levels 7-9 --
  expected (top-of-book reconstructs best, deep/sparse levels worst). The binary-market
  features (94-99) are absent from the top-5, so Workstream A's features learn cleanly.
  Divergent trends (feature_72 worsening, 75 improving) = benign capacity reallocation.
  Cosmetic: pre-restructure code logged `feature_NN`; restructured train.py resolves
  human names for known dims.

### Restructure parity: PASS (and improved)
- CPU suite scoped to `tests/`: **93 passed, 1 skipped, 0 failed** (handoff baseline was
  89 / 4-skip / 1-fail; same 94-test total). The restructure fixed the pre-existing
  FI-2010 split-validation failure and made 3 of the 4 formerly CUDA-skipped tests
  CPU-runnable; the one remaining skip is `test_train_integration.py:123` (WorldModel
  needs CUDA-built mamba_ssm). No regressions.
- NOTE: run pytest scoped to `tests/`. A bare `pytest` aborts during collection on the
  untracked `tmppolymarket-bot/` dir (`ImportError` from its `config`); that is a
  separate vendored bot, unrelated to this package.

### Volatility split WIRED (Experiment D unblocker)
- BUG: `run_backtest_cli --regime-split volatility:Q` passed `realized_vol={}` to
  `volatility_split`, so every market landed in the train half and the high-vol TEST set
  was empty -- Exp D's distribution-shift comparison literally could not run.
- FIX: added `realized_vol_from_timeline(timeline)` in `src/finmamba3/eval/regime_split.py`.
  It reads each market's YES-book mid once per distinct book snapshot (forward-filled
  repeats collapsed via `book_ts` so idle ticks don't deflate vol) and takes the std of
  one-step fractional mid-returns. `run_backtest_cli` now computes it from `bt.timeline`
  and feeds it to `volatility_split`. Semantics: `volatility:0.5` keeps the high-vol half
  (vol > median) as the backtest set; threshold is over the eval data's markets.
- Tests: `tests/test_regime_split.py` (5 new): vol ranking, forward-fill collapse,
  single-observation omission, split routing, CLI filter. Full suite now **98 passed, 1
  skipped**.

### Next steps (revised)
1. **Retrain both arms on restructured code (Colab).** Baseline (`RegimeFiLM.Enabled:
   false`) -- confirm it reproduces val ~368 (training-level parity) -- then treatment
   (`true`) and compare val loss. First cheap in-distribution signal.
2. **Run Experiment D.** Per checkpoint: `python -m finmamba3.eval.run_backtest_cli
   --world-checkpoint <ckpt> --config configs/lob.yaml --data-val data/validation
   --regime-split volatility:0.5 --out reports/<arm>_highvol.json`. Claim: treatment's
   edge over baseline is larger on the high-vol split than on the full set
   (`--regime-split none`).
3. Optional: the CLI currently keeps only the high-vol tail; add a matched low-vol arm if
   a controlled low-vs-high comparison is wanted.
