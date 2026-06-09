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
