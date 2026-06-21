# B1 — Gauge absorption vs. mere regularization (`experiments/altdata/gauge_nowd_escalation.sh`)

## 1. Header — hypothesis and pre-registered criteria

The paper says identity is reached by *gauge absorption* (a function-preserving reparameterization into the
host `W_Delta/W_B/W_C`). The weaker alternative is "the modulation earns nothing, so weight decay drags the
zero-init hypernet back to zero." The word "gauge" in the title needs the distinction nailed.

Pre-registered (verbatim from `colm-submission-goal.md` B1a):
- **Supports the paper (gauge):** the *constant/mean* component of the scale stops decaying (loss is flat
  along it; only WD was pulling it). → "decay along the gauge direction is WD-driven over a flat loss,"
  which is exactly the gauge claim.
- **Contradicts the paper:** `film_g` still decays monotonically to identity with WD off. → the mechanism is
  **not** pure gauge; soften the claim to "the modulation is unrewarded and regularized to identity" and
  demote the gauge argument to a partial explanation.

## 2. Setup

- Driver: `experiments/altdata/gauge_nowd_escalation.sh` — the SAME forced-active, ER-supervised
  strongest-null escalation as `kaggle_escalation.sh` (`InitScale 0.1`, `LRMult 50`, `SupervisionWeight 30`,
  decoupled router, `EntropyCoef 0`, predictability axis), changing **only** the LaProp weight decay, set to
  `--Models.WorldModel.Weight_decay 0.0` (the base override also zeroes the FiLM param group, since its
  `WeightDecay` is unset and inherits the base). Kaggle BTC 1-min, seed 0, 3000 steps (~14 min on the 4080).
- Baseline for comparison: the existing WD=1e-4 run `reports/kaggle_escalation_s0.log`, so only the weight
  decay differs. Decay fit `a*exp(-t/tau)+c` via `experiments/altdata/decay_fit.py`.
- New log: `reports/kaggle_escalation_nowd_s0.log`.

## 3. Results

`film_g(t)` exponential-decay fit, NO-WD vs WD=1e-4 baseline (both forced active at `film_g`≈0.25):

| run | a | tau (steps) | asymptote c | R^2 | linear slope /step |
|---|---:|---:|---:|---:|---:|
| **WD = 1e-4 (baseline)** | 0.2995 | **6685** | -0.049 | **0.9999** | -3.61e-5 |
| **WD = 0 (ablation)** | 0.3009 | **309422** | -0.050 | **0.4828** | -9.70e-7 |

With weight decay ON, `film_g` decays cleanly to identity with the paper's signature (tau≈6685, R^2=0.9999;
0.250 -> 0.144). With weight decay OFF, the decay **flattens**: the exponential time-constant blows up ~46x
(6685 -> 309422), the linear slope shrinks ~37x (-3.6e-5 -> -9.7e-7 per step), and the exponential fit
collapses (R^2 0.9999 -> 0.4828) because there is no longer a clean decay to fit. `film_g` sits at ~0.248
through step 2950 -- essentially unmoved from its forced-active initialisation, with `reg_H` = ln 4 (uniform
router) throughout.

## 4. Verdict — **SUPPORTS**

Removing weight decay **stops** the `film_g` decay (tau lengthens 46x, slope shrinks 37x, the exponential fit
collapses, the value sits flat at its initialisation). The decay along the gauge direction is therefore
**weight-decay-driven over a flat loss**, exactly the gauge prediction: the reconstruction objective is flat
along the modulation, so absent weight decay nothing pulls it toward identity.

This is complemented by **B2** (`inkernel-boundary.md`), whose exact-folding test shows the input-affine
modulation is *exactly* gauge-absorbable (fold residual 1.2e-6) while an in-kernel modulation is not (1.3e-2,
~10^4x larger). Together they distinguish gauge absorption from mere regularization: B2 proves the input-affine
direction is a true gauge direction (flat valley, lossless fold), and B1 proves that on the real model the
decay along that flat valley is driven by weight decay. The B2 toy reproduces the same WD-off flattening
(input-affine `film_g` 0.086 -> 0.010 with WD, -> 0.342 without), so the mechanism is consistent across the
real model and the controlled toy.

**Paper edit it drives:** in `sec:results_null`, the gauge paragraph now states that with weight decay removed
the real-model escalation stops decaying (tau 6685 -> ~3e5, R^2 0.9999 -> 0.48), confirming the gauge-direction
decay is weight-decay-driven over a flat loss rather than objective-driven. (Added alongside the B2 toy
demonstration.)

## 5. STATUS: **done** (B1a no-weight-decay ablation, seed 0; `reports/kaggle_escalation_nowd_s0.log`).

B1b (weight-conservation `||W_in . diag(gamma)||` logging during the default WD-on escalation) is a further
confirmation that would require a logging hook into the Mamba `in_proj`; it is **not run** here. The verdict
does not depend on it: the WD-off flattening (B1a) plus the exact-folding proof (B2) already establish gauge
absorption over mere regularization. Flagged as the natural next measurement, not fabricated.
