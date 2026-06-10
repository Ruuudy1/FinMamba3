# FinMamba3 — Results

This document is the curated, reproducible summary of the regime-FiLM campaign. The campaign reached a **dual
outcome**: the regime-conditioning architecture it set out to test is a decisive negative, and the data pipeline +
trading strategy built to test it is a genuine, survivable economic positive.

---

## 0. Headline

- **(A) Regime-FiLM architecture — NULL (decisive).** Conditioning the Mamba-3 selective scan on an inferred market
  regime via FiLM does **not** improve out-of-regime behaviour. The load-bearing diagnostic — does the FiLM gamma
  deviation `film_g` stay clearly `> 0` while the router entropy `reg_H` stays off its uniform maximum — is **NO** in
  every family, every seed, and on both a forecasting objective and a money (PnL) objective. FiLM collapses to
  identity under joint training and cannot be forced active even by direct predictability supervision.
- **(B) Spot-conditioned pipeline + selective participation — POSITIVE (all anti-artifact gates).** The Phase-0
  spot-conditioned world model plus a train-tuned, frozen causal predictability gate earns **+$4729 deterministic**
  (held-out BTC, ~46% on a $10k bankroll), **beats a fair trend-following naive by 6–9×**, and is **fully survivable**
  (bootstrap P(profit) = 1.000, worst-5% drawdown 4.7% < 15%). The model's calibrated settlement probability — not the
  gate's mechanical edge — is the dominant driver. The edge is characterized as a **3-way microstructure conjunction**
  (predictable **and** liquid-but-not-efficient **and** early-market).
- **Reading.** The regime axis a FiLM *router* will not encode is exactly the one a *strategy gate* harvests
  profitably. The contribution is the preprocessing overhaul + the selective-participation strategy, not the
  regime-conditioning architecture.

---

## 1. The regime-FiLM null (the architecture question)

The campaign tested FiLM on four regime-*dependent* forecasting objectives plus a PnL objective. The single key
diagnostic across all of them: does `film_g` stay clearly `> 0` at convergence with `reg_H` not pinned at its uniform
maximum (`ln R`: ln 4 on FI-2010, ln 8 on Polymarket)?

| # | Family | Objective | Verdict | FiLM at convergence |
|---|---|---|---|---|
| 1 | Volatility / scale | StudentT NLL | NULL | activatable but router uniform across 5 escalations |
| 2 | Volume / order-flow | Volume NLL | NULL | router uniform (shared P1 ckpts) |
| 3 | Direction macro-F1 | LOBCAST-style | NULL | inert; supervised `film_g` 0.079→0.018, router ≈ chance |
| 4 | Settlement (Polymarket) | BCE log-loss | NULL | inert: `reg_H = ln 8`, `film_g ≈ 0.008`, both seeds |
| 5 | **PnL (Polymarket)** | simulated trading | **NULL** | inert: `film_g ≈ 0.010`, `reg_H = ln 8`, both seeds |

**Mechanism.** FiLM is zero-initialized, so the modulated backbone is *exactly* the unmodulated baseline at step 0
(an exact control). Under joint training the reconstruction/dynamics gradient consistently drives the modulation back
toward identity, and the load-balance term then pins the router uniform. The capacity is not missing — a *frozen*
router can be probed to 0.92 agreement with the volatility bucket — but joint training will not use it.

**Strongest-null test (predictability-supervised escalation).** Forcing FiLM active (`InitScale 0.1`, `film_g` starts
at 0.150) and supervising the router directly on the spot Efficiency-Ratio axis with the strongest settings
(`LRMult 50`, `SupervisionWeight 30`, decoupled router/FiLM, fed obs-vol) still decays `film_g` monotonically
(0.150 → 0.119 → 0.100 → 0.074 → 0.056 → 0.043, still falling) with `reg_H = ln 8` throughout. The optimizer drives
the modulation to identity regardless of objective, regime axis, or forced supervision.

**Apparent positive gaps are artifacts.** With a live head, the direction gap reads +0.008 and the settlement gap
+0.18 ("thesis supported" by sign), but in every such case FiLM is provably inert, the treatment is *absolutely
worse*, and the positive degradation-gap is a `high−low` distributional artifact (regimes differ in label balance /
calibration difficulty); the sign even flips with head calibration while FiLM stays inert. **The FiLM-activation
diagnostics are load-bearing: the generalization-gap metric can read positive for reasons unrelated to the mechanism
under test.**

---

## 2. The economic edge (the question that paid off)

### 2.1 Setup
Phase-0 gave the model the variable that defines settlement: the underlying Chainlink/Binance **spot path**, a
**time-to-expiry clock**, an **honest TTE-weighted settlement target**, **cross-interval context**, and an **asset
embedding**. The frozen model's spot-conditioned YES probability is precomputed with the exact training feature
pipeline (no train/serve skew) and traded through the decoupled DATAHACKS2026 execution engine (real T+1 matching,
$1 settlement, PnL = ending − $10,000) — see `SETUP.md`.

### 2.2 Headline (held-out BTC, frozen ER gate = 0.60, fit on train and frozen)

| arm | PnL | trades | P(profit) | dd95 |
|---|---:|---:|---:|---:|
| world model (gate + settlement prob) | **+4560.9** | 1605 | **1.000** | **0.047** |
| fair naive (same gate, buy spot trend, no model) | +513.1 | 676 | — | — |
| **model − fair naive** | **+4047.8 (8.9×)** | | | |

Robust across 2 model seeds (single threshold fit once on seed-0 train): seed 0 +$4561 (8.9×), seed 1 +$3141 (6.1×),
both P(profit) = 1.000, both clearing every gate. **Reproducible headline:** `--deterministic-latent` makes two runs
byte-identical at **+$4729.31** (1553 trades, P 1.000, dd95 0.045), alongside the ±$300 stochastic-precompute band.

### 2.3 Where the edge lives — a liquidity band and a 3-way conjunction
- **Not a ticker boundary, a liquidity band.** Above a thin-book noise floor (ungated ETH/SOL overfit their thin
  books; ETH *recovers*, 2-seed robust, gated to BTC-level depth, beating a fair depth-gated naive) and below an
  efficient-market ceiling (BTC's deepest ≥150k books reverse in both seeds — the most-liquid markets have already
  arbitraged the lag). BTC works ungated because its books sit in the band throughout; SOL is robustly marginal
  (real but non-survivable, P 0.49–0.67).
- **The edge is a conjunction.** It concentrates where the book is least informed — **predictable** (ER) **and**
  **liquid-but-not-efficient** (depth band) **and** **early-market** (the oracle-lag reverses near expiry as the book
  converges). The three gates stack:

  | strategy | PnL | per-trade | dd95 |
  |---|---:|---:|---:|
  | predictability only (headline) | +$4729 | +$3.05 | 0.045 |
  | + depth band [60k, 130k] | +$4834 | +$4.21 | 0.035 |
  | **+ both (early + band)** | **+$5209** | **+$4.98 (+63%)** | **0.029 (−36%)** |

### 2.4 Mechanism — calibration
The settlement head is real signal but **over-confident-and-directional**: Brier 0.206 (beats a 0.25 constant), yet
its top decile (mean prob 0.984) realizes 0.785 YES. This single diagnostic explains the economics: trading the
*divergence* from the oracle-lagged book (which still implies ~0.50) harvests the 30–90 s repricing lag using only the
*direction* (right 79% vs the book's ~50%), so **fixed small sizing wins**; **Kelly fails** (even temperature-calibrated
at T = 2.9, dd95 39.9% ≫ 15%) because position concentration sacrifices the diversification the *statistical* edge
depends on.

### 2.5 Honest scope
BTC-centric (the band is widest on BTC); the backtest's walk-the-book fills assume recorded liquidity is takeable, so
live market impact would erode some edge; the min-TTE (early-market) cut is a held-out characterization grounded in the
monotonic per-trade decay toward expiry, not a train-frozen claim (the depth ceiling *is* train-justified). The
cross-asset conclusion was self-corrected four times, each by a deliberate robustness check (per-asset decomposition,
depth gate, second seed, train-tuning); the surviving claims are those that withstood adversarial testing.

---

## 3. Reproducibility

- **Config:** `configs/lob_spot.yaml` (108-dim: 80 level + 28 tick; SpotFeatures on; Settlement TTEWeighted +
  SpotSignAux on; 4080 SISO non-MIMO, batch 128, 6 layers, 6000 steps ≈ 27 min/arm). Baseline =
  `RegimeFiLM.Enabled False`; treatment overrides `--Models.WorldModel.RegimeFiLM.Enabled True`.
- **Train (per seed S, arm):** `python3 -m finmamba3.train --config configs/lob_spot.yaml --data-train data/train
  --data-val data/validation --hours-train 12 --hours-val 1 --max-markets 32 --BasicSettings.Seed S
  [--Models.WorldModel.RegimeFiLM.Enabled True] --norm-path <arm norm json>`.
- **Backtest:** `python3 -m finmamba3.eval.pnl_backtest --config configs/lob_spot.yaml --checkpoint <final.pth>
  --data-val data/validation --norm-path <arm norm> --assets BTC --intervals 15m --regime-axis predictability
  --use-cusum --sizing kelly [--regime-film] --deterministic-latent --out reports/pnl_<arm>_seedS.json`.
- **Thresholds** (ER gate 0.60, edge 0.03, gate window 120, fixed 50-share sizing) are fit on the **train** split and
  frozen before any held-out evaluation, per the honesty rule.
- **Tests:** 195 focused tests (`python -m pytest tests/`). New components covered: spot/TTE/cross-interval features,
  predictability (ER / permutation entropy / Hurst), CUSUM gate, Kelly sizer, `NaiveLagStrategy`, bootstrap
  survivability, depth-gated and asset-correct precompute, settlement calibration + temperature scaling.

---

## 4. Decision-gate compliance

All families reached their pre-registered gates; none met the WIN criteria for FiLM (no family had an active FiLM, so
the activation clause fails regardless of gap sign). **No A100/H100 run was spent** — a scale-up is justified only by a
positive multi-seed family with an active FiLM, which does not exist. The Polymarket MIMO headline config exceeds the
RTX 4080's shared-memory cap and segfaults; all runs used the non-MIMO Mamba-3 path. The MIMO headline model genuinely
needs an A100+ — flagged, not spent.

---

## 5. Cross-dataset confirmation (Kaggle crypto spot LOB + FI-2010 predictability axis)

A follow-up campaign replicated the regime-FiLM generalization-gap A/B on two more real LOB datasets, judged by
forecasting metrics only (no PnL — these are spot books with no settlement). The regime axis is now **predictability
primary** (Efficiency-Ratio window split via `eval/predictability.py`) with **spot-vol secondary**; the metric is
held-out **direction macro-F1** (primary) + full-channel Student-t **reconstruction NLL** (secondary). Both arms share
the seed, so the zero-init FiLM is an exact paired control. Detailed sheets: `fi2010-film.md`, `kaggle-film.md`.

| dataset | config | treatment `film_g` | `reg_H` | direction gap (pred/vol) | recon-NLL gap | verdict |
|---|---|---:|---:|---:|---|---|
| FI-2010 (seeds 0,1) | `fi2010_studentt.yaml` | 0.0043 / 0.0040 | ln 4 (1.386) | +0.000 / +0.000 | 95% CI includes 0 both axes, signs flip | **NULL** |
| Kaggle BTC 1min (seeds 0,1) | `kaggle_btc.yaml` | 0.0048 / 0.0038 | ln 4 (1.386) | +0.000 / +0.000 | 95% CI includes 0 both axes | **NULL** |
| Kaggle ETH 1min (seed 0) | `kaggle_eth.yaml` | 0.0047 | ln 4 (1.386) | +0.000 | -0.0008 (≈ 0) | **NULL** |
| Kaggle ADA 1min (seed 0) | `kaggle_ada.yaml` | 0.0049 | ln 4 (1.386) | +0.000 | -0.0010 (≈ 0) | **NULL** |

- **Mechanism (identical to the Polymarket families).** From an exact identity init, joint training leaves `film_g`
  ≈ 0.004 (the modulation never grows) and the load-balance pins `reg_H` at its uniform maximum `ln 4`. With FiLM inert,
  the treatment reproduces the baseline, so every direction gap is *exactly* 0 and every recon-NLL gap straddles 0 with
  signs flipping across seeds. The lone sign-consistent gap (Kaggle recon spot-vol, ~+0.001) rides a +0.105 genuine
  regime degradation and co-occurs with an inert FiLM — an artifact, not a win, exactly as the prior settlement/direction
  gaps were. (The *exactly*-0 direction gap is specific to the degenerate flat-collapsed head; class-balanced re-runs on
  *both* datasets de-collapse the head to a non-degenerate macro-F1 — FI-2010 0.20-0.27, Kaggle 0.30-0.33 — with the
  gap's 95% CI still including 0. The head stays near-chance, `dir ≈ ln 3` — next-tick direction is genuinely hard for
  the world-model auxiliary head. The Kaggle CB spot-vol gap *looked* sign-consistent negative at n=2 (-0.019, p=0.087),
  so per the goal's "add seed 2 if borderline" protocol a third seed was run — it flipped to +0.016, so at n=3 the gap
  is mean -0.007, signs --+, CI [-0.057, +0.043]: the apparent consistency was spurious inert-FiLM noise on a near-chance
  head, not an effect. See `fi2010-film.md`/`kaggle-film.md` §6.)
- **Strongest-null test (predictability-supervised escalation, both datasets).** Forcing FiLM active (`InitScale 0.1`,
  `film_g` starts at 0.250) and supervising the router directly on the Efficiency-Ratio bucket (`SuperviseAxis
  predictability`, `SupervisionWeight 30`, `LRMult 50`, decoupled, `FeedObsVol`, `EntropyCoef 0`) still decays `film_g`
  **monotonically every logged step** with `reg_H ≈ ln 4` throughout — Kaggle BTC 0.250 → 0.144 (2950) and FI-2010
  0.250 → 0.161 (2500), both still falling. An exponential fit `a·exp(-t/τ)+c` nails both trajectories (R² > 0.999) with
  near-identical dynamics (a ≈ 0.30, τ ≈ 7000 steps) and an asymptote `c ≤ 0` (Kaggle -0.049, FI-2010 -0.050); since
  `film_g ≥ 0`, the decay extrapolates *through* identity — it provably converges to 0, not to a positive near-identity
  plateau, so "still falling at 2950 steps" is the early phase of a clean decay-to-identity, not an arbitrary stopping
  point. The optimizer drives the in-scan modulation to identity even when it is
  initialized active and the router is supervised on the primary axis — the strongest cross-dataset form of the null,
  mirroring the Polymarket escalation (0.150 → 0.043, still falling). Note `EntropyCoef 0` here removes the load-balance
  term, so the persistent `reg_H ≈ ln 4` is the router collapsing to uniform *on its own* (not a load-balance artifact),
  which strengthens the null beyond the baseline A/B (where `EntropyCoef 0.01` could be credited for the uniformity).
  Figure: `reports/film_g_escalation_decay.png` (top: `film_g` decay to identity both datasets; bottom: `reg_H` at `ln 4`).
- **Cross-asset (ETH, ADA).** The Kaggle null is not BTC-specific: ETH and ADA both leave `film_g` inert (~0.005) with
  `reg_H = ln 4` and gaps ≈ 0 (recon-NLL gaps tiny and *negative*, opposite sign to BTC — init/seed noise, not an effect).
- **Cross-resolution (BTC).** Nor is it specific to the 1min bar: 5min (`film_g` 0.0048) and 1sec (4h slice, `film_g`
  0.0046) both leave FiLM inert with `reg_H = ln 4` and gaps ≈ 0 — the null holds across all three resolutions.
- **Router capacity (predictability axis, honest nuance).** Two probes on the escalation/baseline checkpoints
  (`diag_router.py`, a frozen linear probe) localize *why* FiLM gains nothing: (a) the jointly-supervised router is
  uniform **per-step** (per-sample entropy ≈ ln 4) with near-chance ER-bucket agreement (Kaggle 0.252, FI-2010 0.314
  vs 0.250) despite `SupervisionWeight 30`; (b) a frozen linear probe on the baseline features decodes the per-step ER
  bucket only **weakly** — and only modestly better with a cleaner (longer) ER window: at probe window-len 128 the
  val accuracy is Kaggle 0.291 / FI-2010 0.290 for a 16-event ER target and Kaggle 0.355 / FI-2010 0.332 for a
  64-event one (chance 0.250). So a noisier target explains only ~+0.04-0.06 of the gap to chance; the weak signal is
  **largely real** — the unmodulated representation genuinely under-encodes the predictability regime (well below the
  strong decodability the prior Polymarket *vol* axis showed, ~0.92 via a frozen router, albeit a softer comparison
  across probe/axis/target). So on the predictability axis the null is **overdetermined**: the per-step regime is only
  faintly present in the representation *even with a cleaner target*, and the modulation decays to identity regardless —
  there is little for a regime-conditioner to exploit, and what little exists, joint training does not harness. A
  clean-regime escalation would feed only a marginally-stronger-but-still-weak signal into a modulation that decays
  regardless, so it would reproduce the null; it is not run. (The *window*-level ER split the gap is measured on is
  clean — validated by the PE/Hurst contrast — so the gap-null itself is not a weak-split artifact.)
- **Which regime is encoded — vol vs predictability (rules out the under-representation escape).** A frozen linear
  probe on the same baseline features decodes the realized-vol bucket *better* than the ER bucket on both datasets
  (Kaggle 0.468 vs 0.359, FI-2010 0.426 vs 0.334, chance 0.250) — so the goal's chosen predictability axis is the
  *harder* one to read from the representation. Crucially, the **spot-vol A/B gap is also null**: FiLM is inert even on
  the axis the representation encodes best, so the null does **not** hinge on the regime being under-represented — it is
  the in-scan modulation collapsing, not merely a missing signal.
- **Reading.** Regime-FiLM is now a decisive null across **Polymarket** (vol/settlement/PnL), **FI-2010**, and **Kaggle
  crypto spot (BTC, ETH, ADA)** — five-plus settings spanning prediction-market and spot LOBs and three crypto assets.
  The failure mechanism — the dynamics gradient drives the modulation to identity and the load-balance pins the router
  uniform under joint training — is dataset- and asset-independent. Figure `reports/film_g_across_settings.png`:
  treatment `film_g` inert at ~0.004 across all 10 A/B settings (FI-2010, BTC/ETH/ADA, 5min/1sec, class-balanced),
  far below an active FiLM.

### 5.1 Backbone ablation (G2)

Mamba-1, Mamba-2, Mamba-3 SISO, and a Transformer under matched parameter budgets (~10.3M, actuals reported), seed 0,
3000 steps, on FI-2010 and Kaggle BTC. Held-out full-channel Student-t reconstruction NLL (lower is better):

| backbone | params | FI-2010 recon NLL | Kaggle recon NLL |
|---|---:|---:|---:|
| Mamba-1      | ~11.45M | **-0.5403** | 0.0241 |
| Mamba-2      | ~10.20M | -0.5309 | **-0.0213** |
| Mamba-3 SISO | ~10.30M | -0.5346 | 0.0207 |
| Transformer  | ~9.90M  | -0.5175 | 0.0191 |
| Mamba-3 MIMO | — | _A100-only — flagged, not launched_ | _A100-only — flagged_ |

- **Honest G2 read.** Under matched parameters on the 4080, **no non-MIMO backbone wins both datasets**: Mamba-1 leads
  FI-2010 recon NLL, Mamba-2 leads Kaggle recon NLL, and Mamba-3 SISO is competitive but does **not** strictly dominate.
  The pre-registered MIMO-advantage claim is therefore **untested on the 4080** and rests entirely on the flagged A100
  Mamba-3 MIMO cell. **Verified on this 4080 (not just cited):** the MIMO module *builds* (12,005,533 params) but the
  TileLang MIMO CUDA kernel **segfaults / dumps core at the first forward** (`CUDAModuleNode::GetFunc`), its dynamic SMEM
  exceeding the RTX 4080's ~100 KB/block cap. A four-corner config sweep shows the A100 requirement is **structural, not
  incidental**: the kernel has a feasibility band where the only SMEM-feasible config fails the warp-tiling constraint
  and every warp-valid config overflows SMEM —

  | config (rank/chunk) | warp tiling (`num_warps=4`) | dynamic SMEM | 4080 result |
  |---|---|---|---|
  | 2 / 8  | **invalid** (`m_warp*n_warp=1≠4`) | ~60 KB (fits) | TVM warp-tiling error |
  | 4 / 8  | valid | ~120 KB (over cap) | core-dump |
  | 2 / 16 | valid | ~120 KB (over cap) | core-dump |
  | 4 / 16 (headline) | valid | ~219 KB (over cap) | core-dump |

  So **no 4080-feasible MIMO config exists** — a larger-SMEM card (L4/A100, ~164–228 KB/block) is genuinely required.
  This is the one remaining headline-hardware run; the user decides whether to spend the A100.
- **Direction macro-F1 is degenerate for the ablation:** the world-model next-tick head collapses identically across all
  backbones (FI-2010 0.289, Kaggle 0.178, seed-invariant), so it cannot discriminate architectures. The FI-2010 DeepLOB
  benchmark on the published horizon-10 labels is the real direction reference: **DeepLOB macro-F1 0.628** vs majority
  0.261 and a linear-AR next-tick floor 0.157 — reproducing the LOBCAST in-distribution range and confirming FI-2010
  carries genuine short-horizon signal (so the FiLM null is a real negative, not a no-signal artifact).
