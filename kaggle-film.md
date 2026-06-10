# Kaggle crypto spot LOB — regime-FiLM generalization-gap A/B (G1)

## 1. Header

- **Dataset.** Kaggle high-frequency crypto spot LOB (Binance-style 15-level books), asset **BTC**,
  resolution **1min**. One continuous stream carved chronologically into a 168 h train slice (10080
  ticks) and the next 72 h validation slice (4320 ticks); no future tick leaks into training. Loaded
  through `kaggle_lob_loader.py` into the Polymarket base 8/14 schema (K=10 = 5 best bid + 5 best ask
  levels, F_level=8, F_tick=14, FeatureDim=94). Spot data: no settlement, no binary-market features.
- **Hypothesis (G1 null confirmation).** Conditioning the Mamba-3 selective scan on an inferred market
  regime via FiLM does not improve out-of-regime forecasting. Pre-registered to be disconfirmable: an
  *active* FiLM (`film_g` clearly > 0, `reg_H` off its uniform max `ln 4`) with a positive, CI-excludes-0,
  sign-consistent gap would be the first active FiLM in the project.
- **Metric.** Held-out **direction macro-F1** (primary) and full-channel Student-t **reconstruction NLL**
  (secondary), each scored on the **predictability** (Efficiency-Ratio, primary) and **spot-vol**
  (realized-vol, secondary) regime splits over the validation stream.
- **Sign convention.**
  ```
  degradation = metric(reference_regime) - metric(shifted_regime)   # forecastable->random-walk (or low->high vol)
  gap         = degradation(baseline, FiLM off) - degradation(treatment, FiLM on)   # + = FiLM more robust
  ```
  Higher-is-better metrics (macro-F1) use reference - shifted; lower-is-better (NLL) use shifted -
  reference, so a positive degradation always means "worse in the shifted regime."
- **Load-bearing gate.** FiLM activation: `film_g` (mean |gamma-1|) must stay clearly > 0 at convergence
  with `reg_H` off its uniform maximum. A positive gap with an inert FiLM is an artifact, not a win.
- **Regime-split validation (ER primary; permutation entropy / Hurst reported).** The ER split genuinely
  separates forecastable from random-walk windows on all three measures: the high-ER **reference** bucket
  has ER 0.128, permutation entropy 0.895, Hurst 0.397; the low-ER **shifted** bucket has ER 0.023, PE 0.903
  (higher = more random), Hurst 0.431 (closer to the 0.5 random-walk value). The reference is the more
  structured regime, as intended. (PE ≈ 0.9 ≈ near-maximal even in the reference bucket — crypto 1min is
  close to a random walk, which is itself why a regime-conditioned predictor has little to exploit here.)

## 2. Setup

- **Config:** `configs/kaggle_btc.yaml` (Student-t decoder + direction head, SISO non-MIMO, 4 layers).
  Baseline = `RegimeFiLM.Enabled False`; treatment overrides `--Models.WorldModel.RegimeFiLM.Enabled True`.
- **Train (per seed S, arm):**
  `python3 -m finmamba3.train --config configs/kaggle_btc.yaml --data-train data --data-val data
   --BasicSettings.Seed S [--Models.WorldModel.RegimeFiLM.Enabled True] --norm-path <arm norm json>`.
- **Eval (per seed, axis, metric):**
  `python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/kaggle_btc.yaml
   --dataset kaggle --regime-axis {predictability|spot_vol} --metric {direction_macro_f1|recon_nll}
   --baseline-checkpoint <base final.pth> --treatment-checkpoint <treat final.pth>
   --data-val data --norm-path <arm norm> --window-len 128 --out reports/<...>.md`.
- **Seeds:** 0, 1 (paired: baseline and treatment share the seed — zero-init FiLM is an exact control).
  Throughput ~4 it/s, ~13 min/arm on the 4080. Window-len 128, direction threshold 0.01.

## 3. Per-seed table

**Primary — direction macro-F1, predictability axis (window-128 ER split):**

| seed | arm | reference-F1 | shifted-F1 | degradation | gap | film_g | film_b | reg_H |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline  | 0.17396 | 0.18720 | -0.01325 | —        | 0.0000 | 0.0000 | 0.000 |
| 0 | treatment | 0.17396 | 0.18720 | -0.01325 | +0.00000 | 0.0048 | 0.0046 | 1.386 |
| 1 | baseline  | 0.17396 | 0.18720 | -0.01325 | —        | 0.0000 | 0.0000 | 0.000 |
| 1 | treatment | 0.17396 | 0.18720 | -0.01325 | +0.00000 | 0.0038 | 0.0038 | 1.386 |

`film_g ≈ 0.004`, `reg_H = 1.386 = ln 4` on both treatment seeds — FiLM inert, router at its uniform
maximum, identical to the FI-2010 result. Direction gap exactly 0 (paired control + inert FiLM).
**Caveat:** the direction head collapsed to the flat majority (threshold 0.01 + unweighted CE), so the
macro-F1 ≈ 0.17 is seed-invariant and the metric is degenerate — it confirms FiLM does nothing but is not
itself signal-bearing. The verdict rests on the recon-NLL gap and the FiLM-activation diagnostic.

**Secondary — full-channel recon NLL (lower is better), both axes:**

| seed | axis | base degr | treat degr | gap | film_g (treat) | reg_H (treat) |
|---|---|---:|---:|---:|---:|---:|
| 0 | predictability | -0.01589 | -0.01573 | -0.00016 | 0.0048 | 1.386 |
| 1 | predictability | -0.01508 | -0.01516 | +0.00008 | 0.0038 | 1.386 |
| 0 | spot_vol       | +0.10491 | +0.10445 | +0.00046 | 0.0048 | 1.386 |
| 1 | spot_vol       | +0.10747 | +0.10591 | +0.00156 | 0.0038 | 1.386 |

The spot-vol axis has a large genuine regime gap in *both* arms (recon NLL is +0.105 worse on the high-vol
windows), and the baseline−treatment difference is a tiny sliver of it — exactly the artifact signature.
Direction spot-vol: reference 0.20166, shifted 0.15573, degradation +0.04593 for both arms, gap +0.00000.

## 4. Running aggregate (gap mean ± sd, SE, t, p, 95% CI over seeds 0,1)

| metric | axis | gaps | mean | sd | t | p | 95% CI | sign | CI excl. 0 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| direction_macro_f1 | predictability | [0, 0]              | +0.00000 | 0.00000 | — | — | [0, 0]                 | 00 | no |
| direction_macro_f1 | spot_vol       | [0, 0]              | +0.00000 | 0.00000 | — | — | [0, 0]                 | 00 | no |
| recon_nll          | predictability | [-0.00016,+0.00008] | -0.00004 | 0.00017 | -0.33 | 0.80 | [-0.00156, +0.00148] | -+ | no |
| recon_nll          | spot_vol       | [+0.00046,+0.00156] | +0.00101 | 0.00078 | +1.84 | 0.32 | [-0.00598, +0.00800] | ++ | no |

Every CI includes 0. The one sign-consistent gap (recon spot_vol, ++) is ~0.001 on a +0.105 regime
degradation, is not significant (p=0.32, n=2), and co-occurs with an inert FiLM — an artifact, not a win.

## 5. Backbone ablation (G2)

Matched parameter budgets (~10.3M), seed 0, 3000 steps, `kaggle_*` configs. Mamba-3 SISO reuses the
FiLM-off baseline checkpoint. Recon NLL is the held-out full-channel Student-t NLL (lower is better).

| backbone | params | recon NLL | direction macro-F1 | accuracy |
|---|---:|---:|---:|---:|
| Mamba-2 | 10,215,645 | **-0.02128** | 0.17813\* | 0.36461 |
| Transformer | 9,933,853 | 0.01912 | 0.17813\* | 0.36461 |
| Mamba-3 SISO | 10,334,365 | 0.02065 | 0.17813\* | 0.36461 |
| Mamba-1 | 11,492,381 | 0.02414 | 0.17813\* | 0.36461 |
| Mamba-3 MIMO | — | _A100-only (flagged). Verified on this 4080: module builds (12.0M params) then the TileLang MIMO kernel core-dumps at the first forward — SMEM > ~100 KB/block cap._ | | |

\* The next-tick direction head collapses to the flat majority identically across all backbones (macro-F1 =
0.17813, seed-invariant), so direction cannot discriminate backbones on this spot data — recon NLL is the
load-bearing G2 metric here. (No DeepLOB/linear-AR benchmark on Kaggle: those LOBCAST baselines are FI-2010
specific; this is a generic spot book, not the FI-2010 schema.)

**Reading.** Under matched parameters on the Kaggle crypto spot LOB, **Mamba-2 wins recon NLL** (with the
fewest SSM parameters), and Mamba-3 SISO sits mid-pack (≈ the Transformer). As on FI-2010, Mamba-3 SISO is
competitive but does not strictly dominate, so the MIMO-advantage claim is untested on the 4080 and rests on
the flagged A100 MIMO cell. The cross-dataset takeaway for G2: no non-MIMO backbone is a clear winner across
both datasets (Mamba-1 leads FI-2010 recon, Mamba-2 leads Kaggle recon), which is exactly the matched-budget
ambiguity the MIMO headline run is meant to resolve.

## 6. Architecture-change log

- Baseline A/B (FiLM off vs on, seeds 0,1) — the activation diagnostic decides the gate.
- **Predictability-supervised escalation (the strongest-null test):** forced-active FiLM (`InitScale 0.1`,
  `film_g` starts at 0.250) with the router supervised on the Efficiency-Ratio bucket (`SuperviseAxis
  predictability`, `SupervisionWeight 30`, `LRMult 50`, decoupled router, `FeedObsVol`, `EntropyCoef 0`).
  **Result: `film_g` decays monotonically toward identity despite the forcing and the direct supervision —**
  0.250 → 0.209 (1000) → 0.173 (2000) → 0.144 (2950, every single step decreasing), a 43% monotone decline
  over the run and still falling, with `reg_H = ln 4 = 1.386` pinned at the router's uniform maximum the
  whole way (the supervision `SupervisionWeight 30` is a large penalty the optimizer pays in the total loss
  yet `film_g` still falls). The optimizer drives the in-scan modulation to identity even when it is
  initialized active and the router is supervised directly on the predictability bucket — the strongest
  cross-dataset form of the null, matching the prior Polymarket escalation (0.150 → 0.043, still falling).
  A post-hoc router probe with matched ER conditioning (`diag_router.py --supervise-axis predictability
  --feed-obs-vol`) sharpens this: the router is uniform **per-step** (per-sample entropy 1.386 ≈ ln 4, not
  merely balanced on average) with argmax-vs-ER-bucket agreement 0.252 ≈ chance (0.250). So `SupervisionWeight
  30` failed to make the joint router even regime-discriminative, let alone keep the modulation alive.
- **Cross-asset null (ETH + ADA, seed 0, predictability axis).** The null is not BTC-specific — both other
  crypto assets replicate it exactly:

  | asset | treatment `film_g` | `reg_H` | direction gap | recon-NLL gap |
  |---|---:|---:|---:|---:|
  | ETH | 0.0047 | ln 4 (1.386) | +0.00000 | -0.00080 |
  | ADA | 0.0049 | ln 4 (1.386) | +0.00000 | -0.00102 |

  Baselines `film_g = 0`; both treatments inert at ~0.005 with the router pinned uniform. The recon-NLL gaps
  are tiny and *negative* (treatment marginally worse) — opposite sign to BTC's spot-vol +, confirming the
  residual is init/seed noise around zero, not a FiLM effect. Same null across BTC, ETH and ADA.
- **Resolution robustness (BTC, seed 0).** The null is not specific to the 1min bar — it holds across the
  dataset's resolutions:

  | resolution | treatment `film_g` | `reg_H` | direction gap | recon-NLL gap |
  |---|---:|---:|---:|---:|
  | 1min (headline) | 0.0048 / 0.0038 | ln 4 | +0.00000 | -0.00016 / +0.00046 |
  | 5min | 0.0048 | ln 4 (1.386) | +0.00000 | +0.00070 |
  | 1sec (4h slice) | 0.0046 | ln 4 (1.386) | +0.00000 | +0.00074 |

  `configs/kaggle_btc_5min.yaml` and `kaggle_btc_1sec.yaml` carry the resolution; the 1sec run reads a 4h
  train + 1.5h val head-slice of the 1.8 GB file (pandas `nrows`), so it never materializes the full file.
- Config fix: the five router-supervision keys (`SuperviseVol`, `SuperviseAxis`, `SupervisionWeight`,
  `FeedObsVol`, `DecoupleRouterFromFiLM`) were added to `kaggle_btc.yaml` (and the ETH/ADA configs) as no-op
  defaults so the escalation overrides parse; this does not change the (already-trained) baseline A/B architecture.
- **Direction-metric caveat (honest):** the next-tick head collapsed to the flat majority (macro-F1 ≈ 0.17,
  seed-invariant), so its gap of 0 is uninformative as a signal — load-bearing evidence is recon-NLL + `film_g`.
- **Class-balanced direction re-run (resolves the caveat, symmetric with FI-2010).** With
  `Direction.ClassBalanced=True` (inverse-frequency CE) at threshold 0.01 the head de-collapses: direction
  macro-F1 is now non-degenerate (0.30-0.33, baseline ≠ treatment). It is still **near-chance** (`dir ≈ ln 3`;
  crypto 1min next-tick direction is near-random). At seeds 0,1 the **spot-vol** gap looked sign-consistent
  negative (mean -0.019, p=0.087) — so, following the goal's "add seed 2 if borderline" protocol, a **third
  seed** was run: it **flipped to +0.016**, breaking the sign-consistency. At n=3 *both* axes are cleanly NULL:
  predictability mean -0.005 (95% CI [-0.022, +0.013], p=0.37, signs ---), spot-vol mean -0.007 (CI
  [-0.057, +0.043], p=0.60, signs --+). The n=2 sign-consistency was spurious noise — exactly the inert-FiLM
  artifact the gate predicts (`film_g ≈ 0.004`, `reg_H = ln 4` on all three treatment seeds): a near-identity
  modulation perturbing a near-chance head's per-regime macro-F1. No active FiLM, no robustness gain — the NULL
  holds with a non-degenerate primary metric. The secondary recon-NLL is also null on these CB models (gaps
  -0.0002/+0.0001 predictability, +0.0004/+0.0016 spot-vol — tiny, FiLM inert). (Checkpoints
  `mn4jtqh3`/`bicwduea` s0, `olkl9mid`/`wl4o5sc3` s1, + a seed-2 pair.)

## 7. Conclusion — NULL (fourth-setting confirmation)

Regime-FiLM is a **decisive null** on the Kaggle BTC crypto spot LOB — the fourth-setting confirmation
(after the prior Polymarket campaign and FI-2010, per the goal's framing) in which the in-scan modulation
collapses to identity under joint training. The gate fails: `film_g ≈ 0.004` (zero-init never grew) and `reg_H = ln 4` (router uniform) on
both treatment seeds. Every generalization gap is statistically indistinguishable from zero; the lone
sign-consistent recon-NLL gap is a tiny artifact riding a large genuine regime degradation, with FiLM
provably inert. The WIN criterion (which requires an active FiLM) is not met. This is the pre-registered
NULL, the expected cross-dataset confirmation that the failure mechanism is dataset-independent.

## 8. STATUS

done: **NULL** (G1, BTC seeds 0,1 + ETH/ADA cross-asset + 1sec/1min/5min resolutions, predictability primary
+ spot-vol secondary) + predictability-supervised escalation (`film_g` 0.250→0.144 monotone, the strongest
null) + class-balanced direction re-run (non-degenerate primary metric, gap still null) + G2 backbone
ablation complete (Mamba-1/2/3-SISO/Transformer; Mamba-3 MIMO flagged A100-only).
