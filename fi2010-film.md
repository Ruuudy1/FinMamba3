# FI-2010 — regime-FiLM generalization-gap A/B (G1, predictability-primary replication)

## 1. Header

- **Dataset.** FI-2010 (Ntakaris et al. 2018), NoAuction DecPre CF_7, horizon k=10. Loaded via
  `fi2010_loader.py` (K=10, F_level=4, F_tick=6, FeatureDim=46). Single concatenated event stream;
  train/validation are the published split files.
- **Hypothesis (G1 null confirmation).** Same as the Kaggle sheet: regime-FiLM does not improve
  out-of-regime forecasting; pre-registered disconfirmable. This is the predictability-primary
  replication of the prior FI-2010 vol-axis result, for cross-dataset evidence.
- **Metric.** Held-out **direction macro-F1** (primary) and full-channel Student-t **reconstruction NLL**
  (secondary), on the **predictability** (ER, primary) and **spot-vol** (secondary) window splits.
- **Sign convention.**
  ```
  degradation = metric(reference_regime) - metric(shifted_regime)
  gap         = degradation(baseline, FiLM off) - degradation(treatment, FiLM on)   # + = FiLM more robust
  ```
- **Load-bearing gate.** FiLM activation: `film_g` clearly > 0 with `reg_H` off its uniform max `ln 4`.
- **Regime-split validation (ER primary; permutation entropy / Hurst reported).** The ER split separates
  forecastable from random-walk windows on all three measures: the high-ER **reference** bucket has ER 0.186,
  permutation entropy 0.380, Hurst 0.540; the low-ER **shifted** bucket has ER 0.054, PE 0.423 (higher = more
  random), Hurst 0.504 (closer to the 0.5 random-walk value). FI-2010's PE ≈ 0.4 is far below crypto's ≈ 0.9,
  i.e. FI-2010 windows carry real ordinal structure — consistent with DeepLOB's 0.628 macro-F1 — so the FiLM
  null here is a negative on genuinely forecastable data, not a no-signal artifact.

## 2. Setup

- **Config:** `configs/fi2010_studentt.yaml` (Student-t decoder + direction head, SISO, 4 layers).
  Baseline `RegimeFiLM.Enabled False`; treatment `True`.
- **Train (per seed S, arm):**
  `python3 -m finmamba3.train --config configs/fi2010_studentt.yaml --data-train data/fi2010/train
   --data-val data/fi2010/validation --dataset fi2010 --BasicSettings.Seed S
   [--Models.WorldModel.RegimeFiLM.Enabled True] --norm-path <arm norm json>`.
- **Eval (per seed, axis, metric):**
  `python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/fi2010_studentt.yaml
   --dataset fi2010 --regime-axis {predictability|spot_vol} --metric {direction_macro_f1|recon_nll}
   --baseline-checkpoint <base> --treatment-checkpoint <treat> --data-val data/fi2010/validation
   --norm-path <arm norm> --window-len 512 --out reports/<...>.md`.
- **Seeds:** 0, 1 (paired: baseline and treatment share the seed, so the zero-init FiLM is an exact
  control — any arm difference is attributable to FiLM alone). Throughput ~3.6 it/s, ~14 min/arm on the 4080.

## 3. Per-seed table

**Primary — direction macro-F1, predictability axis (window-512 ER split):**

| seed | arm | reference-F1 | shifted-F1 | degradation | gap | film_g | film_b | reg_H |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline  | 0.29149 | 0.28477 | +0.00672 | —        | 0.0000 | 0.0000 | 0.000 |
| 0 | treatment | 0.29149 | 0.28477 | +0.00672 | +0.00000 | 0.0043 | 0.0040 | 1.386 |
| 1 | baseline  | 0.29149 | 0.28477 | +0.00672 | —        | 0.0000 | 0.0000 | 0.000 |
| 1 | treatment | 0.29149 | 0.28477 | +0.00672 | +0.00000 | 0.0040 | 0.0040 | 1.386 |

The treatment direction predictions are byte-identical to the baseline's (gap exactly 0): with FiLM
zero-initialized and inert at convergence (`film_g ≈ 0.004`), the modulated backbone reproduces the
unmodulated one. `reg_H = 1.386 = ln 4` on both treatment seeds — the router sits at its uniform maximum.

**Secondary — full-channel recon NLL (lower is better), both axes:**

| seed | axis | base degr | treat degr | gap | film_g (treat) | reg_H (treat) |
|---|---|---:|---:|---:|---:|---:|
| 0 | predictability | -0.01225 | -0.01158 | -0.00068 | 0.0043 | 1.386 |
| 1 | predictability | -0.01084 | -0.01176 | +0.00093 | 0.0040 | 1.386 |
| 0 | spot_vol       | +0.01119 | +0.00934 | +0.00185 | 0.0043 | 1.386 |
| 1 | spot_vol       | +0.00566 | +0.00792 | -0.00227 | 0.0040 | 1.386 |

Direction macro-F1, spot-vol axis: gap = +0.00000 on both seeds (reference 0.29715, shifted 0.27871,
degradation +0.01844 for both arms) — same exact-control collapse as the predictability axis.

## 4. Running aggregate (gap mean ± sd, SE, t, p, 95% CI over seeds 0,1)

| metric | axis | gaps | mean | sd | t | p | 95% CI | sign | CI excl. 0 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| direction_macro_f1 | predictability | [0, 0]            | +0.00000 | 0.00000 | — | — | [0, 0]                 | 00 | no |
| direction_macro_f1 | spot_vol       | [0, 0]            | +0.00000 | 0.00000 | — | — | [0, 0]                 | 00 | no |
| recon_nll          | predictability | [-0.00068,+0.00093] | +0.00013 | 0.00114 | +0.16 | 0.90 | [-0.01010, +0.01035] | -+ | no |
| recon_nll          | spot_vol       | [+0.00185,-0.00227] | -0.00021 | 0.00291 | -0.10 | 0.94 | [-0.02638, +0.02596] | +- | no |

Every CI includes 0; every recon-NLL axis flips sign across seeds. No gap is distinguishable from zero.

## 5. Backbone ablation (G2)

Matched parameter budgets (~10.3M), seed 0, 3000 steps, `fi2010_*` configs. Mamba-3 SISO reuses the
FiLM-off baseline checkpoint. Recon NLL is the held-out full-channel Student-t NLL (lower is better).

| backbone | params | recon NLL | direction macro-F1 | accuracy |
|---|---:|---:|---:|---:|
| Mamba-1 | 11,440,013 | **-0.54033** | 0.28886\* | 0.76457 |
| Mamba-3 SISO | 10,281,997 | -0.53462 | 0.28886\* | 0.76457 |
| Mamba-2 | 10,163,277 | -0.53087 | 0.28886\* | 0.76457 |
| Transformer | 9,881,485 | -0.51749 | 0.28886\* | 0.76457 |
| Mamba-3 MIMO | — | _A100-only (flagged). Verified on this 4080: module builds (12.0M params) then the TileLang MIMO kernel core-dumps at the first forward — dynamic SMEM > the ~100 KB/block cap._ | | |

\* The world-model next-tick direction head collapses identically across all backbones (macro-F1 = 0.28886,
seed-invariant) — a degenerate metric here, so it cannot discriminate backbones. The DeepLOB benchmark below
is the real in-distribution direction reference.

**DeepLOB / majority / linear-AR benchmark (published horizon-10 labels, LOBCAST-comparable):**

| method | accuracy | macro-F1 | brier |
|---|---:|---:|---:|
| DeepLOB | 0.7398 | **0.6284** | 0.3806 |
| majority floor | 0.6447 | 0.2613 | 0.7105 |
| linear-AR (next-tick floor) | 0.1449 | 0.1568 | 1.7102 |

**Reading.** On recon NLL the Mamba family (Mamba-1/2/3) all beat the Transformer; Mamba-3 SISO is
competitive (≈ Mamba-1, both ahead of Mamba-2 and the Transformer) but does **not** strictly dominate under
matched parameters, so the pre-registered MIMO-advantage claim is **untested on the 4080** and rests on the
flagged A100 MIMO cell. DeepLOB's 0.628 macro-F1 vastly exceeds the majority (0.261) and linear-AR (0.157)
floors, reproducing the LOBCAST in-distribution range and confirming the FI-2010 task carries real signal
(so the FiLM null is not a no-signal artifact).

## 6. Architecture-change log

- Baseline A/B (the gate is decided on the activation diagnostic).
- **Predictability-supervised escalation (the strongest-null test):** forced-active FiLM (`InitScale 0.1`,
  `film_g` starts at 0.250) + router supervised on the Efficiency-Ratio bucket (`SuperviseAxis predictability`,
  `SupervisionWeight 30`, `LRMult 50`, decoupled, `FeedObsVol`, `EntropyCoef 0`). **Result: `film_g` decays
  monotonically** 0.250 → 0.229 (500) → 0.210 (1000) → 0.193 (1500) → 0.178 (2000) → 0.161 (2500, still
  falling), with `reg_H ≈ ln 4` (1.377–1.386, router essentially uniform — the supervision barely nudges it
  off uniform and does not save the modulation). Same signature as the Kaggle escalation: forced active and
  directly ER-supervised, the optimizer still drives the modulation to identity. The escalation null now
  holds on the predictability axis for **both** datasets. A post-hoc router probe with matched ER
  conditioning shows the router is uniform per-step (per-sample entropy 1.377 ≈ ln 4) with argmax-vs-ER-bucket
  agreement 0.314 — only modestly above the 0.250 chance rate, i.e. `SupervisionWeight 30` made the joint
  router only weakly regime-aware and did not keep `film_g` from decaying.
- **Direction-metric caveat (honest):** the FI-2010 synthetic mid is *unchanged on 76% of consecutive
  events* (`|Δmid_norm| = 0`), so at threshold 0.0 the next-tick label is a 76%-flat majority and the
  unweighted head collapses to predicting flat (macro-F1 ≈ 0.29 = an all-flat predictor's score,
  seed-invariant). Its gap of exactly 0 is thus uninformative as a signal-bearing measure. The verdict rests
  on (a) the FiLM-activation diagnostic and (b) the continuous recon-NLL gap, both healthy and both null. A
  class-balanced re-run (below) de-collapses the head to confirm the gap stays null with a non-degenerate metric.
- **Class-balanced direction re-run (resolves the caveat).** With `Direction.ClassBalanced=True` (inverse-
  frequency CE) at threshold 0.0, the head no longer collapses to the flat majority: direction macro-F1 is now
  non-degenerate (0.20-0.27, and baseline ≠ treatment, since the inert-but-nonzero FiLM now perturbs a
  non-collapsed head's predictions). The head is still **near-chance** (`dir ≈ ln 3`, macro-F1 < 0.33 — FI-2010
  next-tick direction is near-random for the world-model auxiliary head, a far harder task than DeepLOB's
  horizon-10). Crucially **the gap stays null**: predictability mean -0.0015 (95% CI [-0.006, +0.003], p=0.15,
  treatment slightly *worse*), spot-vol +0.0007 (CI [-0.057, +0.059]) — both include 0, FiLM inert
  (`film_g ≈ 0.004`, `reg_H = ln 4`). So the flat-collapse was not masking a real effect; the FiLM null holds
  with a non-degenerate primary metric. The secondary recon-NLL is also null on these CB models (gaps
  -0.0007/+0.0008 predictability, +0.0018/-0.0023 spot-vol — tiny, signs flipping across seeds).
  (Checkpoints `xxnzonal`/`0qcpxyh9` s0, `sod5psh0`/`c8qfhxs6` s1.)

## 7. Conclusion — NULL (third-setting confirmation)

Regime-FiLM is a **decisive null** on FI-2010, replicating the prior Polymarket result on a dataset with
real short-horizon LOB structure. The load-bearing gate fails: `film_g ≈ 0.004` (the zero-init modulation
never left identity) and `reg_H = ln 4` (the router is pinned at its uniform maximum) on both treatment
seeds. With FiLM inert, the generalization gaps are artifacts — and indeed every gap is statistically
indistinguishable from zero (all CIs include 0) with recon-NLL signs flipping across seeds. The WIN
criterion is not met (it requires an active FiLM, which does not exist), so this is the **pre-registered
NULL** — the expected third-setting confirmation (the null was established on Polymarket and now FI-2010)
that the optimizer drives the in-scan modulation to
identity under joint training regardless of dataset.

## 8. STATUS

done: **NULL** (G1, seeds 0,1, predictability primary + spot-vol secondary) + predictability-supervised
escalation (`film_g` 0.250→0.161 monotone, the strongest null) + class-balanced direction re-run
(non-degenerate primary metric, gap still null) + G2 backbone ablation complete (Mamba-1/2/3-SISO/Transformer
+ DeepLOB/majority/linear-AR; Mamba-3 MIMO flagged A100-only).
