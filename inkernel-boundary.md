# B2 — In-kernel boundary, demonstrated at toy scale (`experiments/inkernel/inkernel_boundary.py`)

## 1. Header — hypothesis and pre-registered criteria

The paper ends the gauge-absorption mechanism on an **asserted, untested** boundary: "a surviving
modulation would have to act *inside* the fused kernel, which our hardware cannot run." The hardware limit
is about the *fused, performant* kernel, not the mechanism — `mamba_ssm` ships a pure-PyTorch reference
scan. At toy scale we modulate Δ/B/C *directly inside* that reference scan and test whether **that** form
survives, turning the asserted boundary into a demonstrated one.

Pre-registered (verbatim from `colm-submission-goal.md` B2):
- **SUPPORTS the paper:** the in-kernel modulation **persists** (`film_g`-analog stays clearly `> 0`,
  optionally helps the task). → a *constructive counterpart* to the null; rewrite the boundary paragraph
  from "where a surviving modulation would have to live" to "a modulation that lives there and survives,
  demonstrated at toy scale."
- **CONTRADICTS the paper:** the in-kernel modulation **also decays**. → the boundary claim is wrong and
  the null is *broader* than gauge; revise `sec:results_null` to report the stronger, more surprising null.

## 2. Setup

- Script: `experiments/inkernel/inkernel_boundary.py` (pure PyTorch, CPU, no `mamba_ssm`/CUDA; runs in
  ~5 s). Command: `python experiments/inkernel/inkernel_boundary.py`.
- A tiny SISO selective state-space block on the **reference recurrence** `h_t = exp(Δ_t A) h_{t-1} +
  Δ_t B_t x_t`, `y_t = (h_t · C_t)`, with `d_model=8`, `d_state=4`, sequence length 24, batch 64 — small
  enough that the slow path is exact and we can manipulate the projected Δ/B/C directly.
- Two modulation **placements**, identical modulator, only the application site differs (mirroring the
  paper's input-affine vs post-scan control):
  - `input_affine`: `x̃ = (1+g)·x + β` applied to the block **input** (the gauge-absorbable form).
  - `in_kernel`: `Δ ← (1+g_Δ)·Δ`, `B ← (1+g_B)·B`, `C ← (1+g_C)·C` applied to the projected scan
    parameters **inside** the reference scan (the asserted-boundary form).
  - Both store the modulation as a raw deviation `g` with scale `γ = 1+g` and shift `β = g`, so the
    parameter zero is the identity exactly as in the paper's zero-init hypernet (weight decay therefore
    pulls toward identity, not toward a zero scale). Both are **forced active** at init (`film_g ≈ 0.09`).
- Two analyses: (1) an exact-folding test that measures whether the active modulation can be folded into
  primed host projections `W_Δ/W_B/W_C` with no change in output; (2) joint training on a causal
  EMA-reconstruction task, logging `film_g = mean|γ−1|`, with weight decay ON (1e-2) and OFF (0), seeds 0–3.

## 3. Results

**Exact-folding test (max |output − folded output|; ~0 ⇒ gauge-absorbable):**

| placement | fold residual | reading |
|---|---:|---|
| `input_affine` | **1.19e-06** | absorbable — the scale folds as `W_•·diag(γ)`, the shift into the bias, **losslessly** |
| `in_kernel` | **1.33e-02** | **not** absorbable — ~10^4× larger; a per-channel scale on the post-softplus Δ cannot be matched by any linear reparameterisation of the pre-softplus projection |

The input-affine fold is *analytic and exact* (the gauge move the paper describes). For the in-kernel
placement we fit the best primed projection by 400 Adam steps; the residual floor stays four orders of
magnitude above the input-affine fold — a numerical witness that the in-kernel placement is **not** a
gauge direction of the bounding projections.

**Joint-training `film_g` (start → end over 600 steps):**

| weight decay | `input_affine` | `in_kernel` |
|---|---:|---:|
| **1e-2 (ON)** | 0.086 → **0.010** (decays to identity) | 0.093 → **0.175** (persists / grows) |
| 0 (OFF) | 0.086 → 0.342 | 0.093 → 0.257 |

Seed robustness (WD=1e-2, seeds 1/2/3): `input_affine` end `film_g` 0.010/0.011/0.010 (ratio ≈ 0.12);
`in_kernel` end `film_g` 0.173/0.176/0.175 (ratio ≈ 1.88). The pattern is seed-stable.

## 4. Verdict — **SUPPORTS**

The boundary is demonstrated, two independent ways:
1. **Absorbability (deterministic).** The input-affine modulation folds into the host projections to
   floating-point precision (1.19e-06); the in-kernel modulation provably cannot (1.33e-02, ~11,000×
   larger). The in-kernel placement is **not** a gauge direction.
2. **Training dynamics (under the paper's own weight decay).** With weight decay ON, the gauge-absorbable
   input-affine modulation **decays to identity** (`film_g` 0.086 → 0.010), reproducing the paper's null in
   miniature, while the non-absorbable in-kernel modulation **persists** (0.093 → 0.175) under the *same*
   weight decay — it does real work the host cannot undo, so the optimiser cannot cheaply remove it.

Mechanistic reading: a modulation that lives **inside the kernel** (between the bounding projections)
survives joint training precisely because it escapes gauge absorption — the constructive counterpart to the
input-affine null. The WD-OFF rows additionally cross-confirm **B1**: with weight decay removed even the
gauge-absorbable input-affine modulation does not decay to identity (0.086 → 0.342), so the decay along the
gauge direction is **weight-decay-driven over a flat loss**, exactly the gauge claim.

**Paper edit it drives:** in `sec:results_null`, rewrite the closing boundary sentence from "the
location — inside the kernel — a surviving modulation would have to occupy" to add a demonstrated clause:
at toy scale on the reference scan, an in-kernel modulation of Δ/B/C is non-absorbable (fold residual ~10^4×
the input-affine fold) and *survives* joint training under the same weight decay that drives the
input-affine form to identity — the boundary is demonstrated, not merely asserted. Report the toy as a
reference-scan demonstration (not the fused kernel), consistent with the hardware-envelope scoping.

## 5. STATUS: **done** (folding + dynamics, seeds 0–3; `reports/inkernel_boundary.json`).
