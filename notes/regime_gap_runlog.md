# Regime-FiLM generalization-gap — raw run log (4080, SISO pilots)

Raw experimental log banking the numbers behind the cross-regime generalization-gap study.
This is a data log, not the formal writeup. Conclusion up front: **the regime-FiLM
reconstruction-gap effect is within seed noise on Polymarket — the thesis is not supported here.**

## Setup
- Hardware: local RTX 4080 (12 GB), WSL Ubuntu-22.04, torch 2.7.0+cu126.
- Model: FinMamba3 world model, **SISO** (`Mamba3.is_mimo=false`), default 6 layers unless noted.
- Data: Polymarket multi-market, `--max-markets 8` from `data/train` (hours 6), batch 128, 6000 steps.
- A/B: baseline = `RegimeFiLM.Enabled false`, treatment = `true`. Same seed/data/batch per pair.
- Metric: `eval/eval_regime_generalization.py --metric prediction_mse` — decoder NLL (Gaussian/mse =
  MSE) on the **prior-predicted next tick**, which flows through the FiLM-modulated sequence model
  (the posterior reconstruction is FiLM-blind, so it is the wrong quantity).
- Held-out split: `data/validation`, top-64 markets by length, partitioned at the realized-vol
  median (`volatility_split`, q=0.5) into low-vol (reference) and high-vol (regime-shifted) groups.
- gap = baseline_degradation - treatment_degradation, where degradation = high_vol_MSE - low_vol_MSE
  (positive gap = treatment degrades less = thesis-supporting).

## Headline: 5-seed 6-layer gap (the decisive result)
| seed | gap | note |
|---|---|---|
| 0 | +0.0276 | |
| 1 | +0.0160 | |
| 2 | +0.0216 | |
| 3 | **-0.0404** | training healthy (recon 146->79 / 144->73, dir_acc ~0.37, no NaN) — a genuine sample |
| 4 | +0.0132 | |

**mean = +0.0076 ± 0.0123 (SEM), 95% CI [-0.027, +0.042], p ≈ 0.57 (t, 4 dof) — not distinguishable
from zero.** The first three seeds (+0.022 ± 0.006, 3/3 positive) were a small-sample fluke; seeds
3–4 surfaced a strong negative, collapsing the mean.

## Capacity does not amplify (8-layer A/B)
| config | seed 0 | seed 1 | mean |
|---|---|---|---|
| 6L / 6k | +0.0276 | +0.0160 | +0.022 |
| 8L / 6k | +0.0614 | +0.0044 | +0.033 (undertrained: prediction MSE rose 0.69->0.73 while posterior recon improved 76->73) |
| 8L / 10k | -0.0026 | +0.0392 | +0.018 |

Depth adds variance, not a larger mean. Across all 9 SISO seed-runs the gaps span -0.040 to +0.061.

## Robustness sweeps (CAVEAT: run on seeds 0–2 only, before the seed-3 negative)
These predate seeds 3–4, so they over-represent the initially-positive cluster; read them as
"conditional on the lucky first 3 seeds," superseded by the 5-seed null above.
- **Vol-quantile sweep** (q=0.5..0.8, eval-only re-partition of the 3 seeds): positive in 21/21
  (seed×quantile) cells, stable ~+0.020, no dose-response with severity.
- **Horizon sweep** (open-loop rollout, H=1..12, 3 seeds): positive at the mean across all horizons
  (34/36 cells); an upward trend in seed 0 only, not seeds 1–2 → no robust horizon amplification.

## Direction metric is degenerate (separate, robust finding)
Direction macro-F1 is not usable here: the 3-class direction head collapses to ~97% constant "flat"
(129k windows; treatment flips ~2) even at 6000 steps, dir_acc plateaus ~0.38. Triple-barrier labels
are well balanced (~[40% down, 23% flat, 37% up]). So Polymarket tick-direction is ~unpredictable — a
clean instance of the LOBCAST (Prata et al. 2023) "models collapse out-of-distribution" story.

## Honest conclusion
- The "regime-FiLM improves generalization under volatility-regime shift" thesis is **not supported**
  on this Polymarket setup (reconstruction gap within noise; no capacity amplification).
- This is a fair test (the world model reconstructs well; recon converges ~76), not a no-signal
  artifact for the reconstruction metric.
- Salvageable contributions: (1) the evaluation framework (`eval_regime_generalization.py` + the
  multi-market substrate + the FiLM-sensitive prediction-MSE metric), (2) the direction-collapse / OOD
  diagnostic, (3) the rigorous negative result on regime-FiLM.
- Open question a reviewer will raise: is the mechanism dead everywhere, or only on signal-poor
  Polymarket? → fair-test on FI-2010 (predictable short-horizon signal) resolves it; pending.

## Reproduction
```
python -m finmamba3.eval.eval_regime_generalization \
  --config configs/lob.yaml \
  --baseline-checkpoint <FiLM-off>/world_model_final.pth \
  --treatment-checkpoint <FiLM-on>/world_model_final.pth \
  --data-val data/validation --norm-path <train-norm>.json \
  --metric prediction_mse --volatility-quantile 0.5 --max-markets 64 --hours-val 12
```
Both checkpoints must be trained with `action_dim=1` (train.py default), SISO, and the same
`BinaryMarketFeatures`/depth; the evaluator rebuilds the matching architecture per arm (FiLM off for
baseline, on for treatment) and fails fast on any state-dict key mismatch.
