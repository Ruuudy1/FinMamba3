# CAMPAIGN — master index & cross-family status

Driver prompt: `CAMPAIGN_GOAL.md`. Prior null: `NEW_FILM.md`. Each family logs to its own file.

**Premise:** the prior null tested FiLM on the regime-*invariant* conditional mean (`prediction_mse`). This
campaign tests FiLM on regime-*dependent* objectives. Key diagnostic across all families: **does `film_g`
stay > 0 at convergence under the new objective?**

## Families (priority order: most → least likely to be paper-beneficial)

| # | Family | File | Status | Mean gap (±sd) | CI excludes 0 | FiLM active | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Volatility / scale (StudentT NLL) | `vol-scale.md` | **done: NULL** | plain −0.00012, enh −0.00053 (2-seed) | no | activatable but router uniform across 5 escalations | **NULL** (mechanism fully characterized) |
| 2 | Volume / order-flow magnitude | `volume-order-flow.md` | **done: NULL** | plain −0.00055, enh −0.00078 (2-seed) | no | shared P1 ckpts (router uniform) | **NULL** |
| 3 | Direction macro-F1 (LOBCAST) | `direction-macro-f1.md` | **done: NULL** | plain-cb +0.0079 (3-seed, artifact); regsup −0.0247 | no (sign flips w/ head calib) | inert across plain/cb/regsup; supervised film_g 0.079→0.018, router agree 0.29≈chance | **NULL** (gap is a label-balance artifact) |
| 4 | Settlement (Polymarket) | `settlement.md` | **done: NULL** | 2-seed +0.179 (artifact, CI straddles 0) | no | inert: `reg_H=ln8`, `film_g≈0.008` both seeds | **NULL** (gap is a distributional artifact, cross-dataset) |
| 5 | **PnL (economic objective, predictability axis)** | `pnl-film.md` | **done: FiLM NULL + economic WIN** | 2-seed gap −736, −519 (artifact) | no | inert: `film_g≈0.010`, `reg_H=ln8` both seeds | **FiLM NULL** (inert on the money objective too), **but a deployable economic POSITIVE**: spot-conditioned model + train-tuned predictability gate earns +$4.6k held-out (P(profit)=1.0, dd95=4.7%), beating a fair trend-following naive 8.9x |

## Phase 0 prerequisites

- [x] Multi-metric FI-2010 evaluator (`--metric {prediction_mse, studentt_nll, volume_nll, direction_macro_f1}`).
  Added `world_model_prediction_nll` (Student-t, channel-masked) to `compare_direction.py`; generalized
  `eval_regime_generalization_fi2010.py` with `--metric` + `_METRIC_SPEC` dispatch and FI-2010 price/scale
  & volume masks (`_feature_indices`, derived from `enc_cfg`, not hardcoded). Tests:
  `tests/test_eval_regime_generalization_fi2010.py` (12 pass). Full suite 142 pass / 2 skip.
  Smoke-validated all four metric paths on 5-step checkpoints (split non-degenerate, masks disjoint,
  gap sign correct, prediction_mse correctly rejects a studentt ckpt). `configs/fi2010_studentt.yaml`
  created (copy of `fi2010.yaml`, `Decoder.Kind: studentt`).
- [x] (Phase 3) Direction head: the unweighted inline-label CE collapses to the 76% flat majority (all-flat
  predictions, `diag_dirhead`). Fixed with gated inverse-frequency class weighting (`Direction.ClassBalanced`,
  `world_model._direction_class_weight`, tested; 155 pass). Head revived (predicts all 3 classes). The
  published-k-horizon plumbing was unnecessary — class weighting is the actual lever.
- [ ] (Optional) LR sanity: Laprop 4e-5 vs 3e-4. Deferred — prior null converged fine at 4e-5; revisit
  only if Phase 1 diagnostics show undertraining (recon not plateauing).

## Vol-scale deep-dive (mechanism, in progress)

The vol-scale family became the campaign's mechanistic core. Findings so far (all in `vol-scale.md`):
1. Plain + enhanced FiLM on studentt_nll: 2-seed gap centered at zero, sign-flips, activation-invariant.
2. Router pinned uniform (per-step) in all arms; batch-mean reg_H=ln4 hides this (added per-sample entropy
   + `diag_router`).
3. Regime supervision (decision tree's top lever) at weight 1.0: router still uniform. Root-caused two
   stacked bugs — (a) router input lacked the obs-vol axis → fed it (`FeedObsVol`); (b) the fed feature was
   numerically negligible (~3e-4 vs O(1) hidden) → batch-standardized it. Freeze-probe then reaches 0.92
   router-vs-vol agreement, so the mechanism is now genuinely fixable.
4. Yet under *joint* training the optimizer drives the router back to uniform (recon-dominance): the
   StudentT-NLL gradient through FiLM overpowers supervision at weight 1.0. → testing SupervisionWeight 30
   to force discrimination and measure whether a forced regime-discriminative FiLM moves the gap at all.

## Cross-family conclusion — BOUNDED NULL (all four families screened; no WIN)

**Headline:** Regime-FiLM conditioning of the Mamba selective scan does **not** improve out-of-regime
forecasting on FI-2010 or Polymarket, on *any* of the four regime-dependent objectives tested
(volatility/scale StudentT-NLL, order-flow volume NLL, direction macro-F1, settlement log-loss). The
campaign's single key diagnostic — *does `film_g` stay clearly > 0 under a regime-dependent objective?* —
is **NO**, every family, every seed, every dataset. This strengthens the prior `prediction_mse` null
(`NEW_FILM.md`): the failure is not that the old objective rewarded the conditional mean — even objectives
that explicitly reward the regime-dependent quantity cannot keep FiLM active.

**The mechanism, now fully characterized (the paper's contribution):**
1. **FiLM collapses to identity under joint training.** Across all four families the FiLM gamma deviation
   `film_g` decays toward 0 (≈0.002–0.018) and the router entropy `reg_H` sits at its uniform maximum
   (`ln R`: ln4 on FI-2010, ln8 on Polymarket). Reconstruction dominance + the adaptive optimizer discard
   regime-discriminative conditioning; the load-balance term then pins the router uniform.
2. **The strongest escalation does not rescue it.** Regime supervision (SuperviseVol) at weight 30, with
   FeedObsVol + router/​FiLM decoupling + LRMult 50 + positive InitScale, forces `film_g=0.079` at init but
   the optimizer drives it monotonically back to ≈identity (→0.018) while the router stays uniform
   (per-step argmax-vs-vol-bucket agreement 0.29 ≈ chance). A *frozen* router can be probed to 0.92
   agreement (vol-scale.md), so the capacity exists — joint training simply will not use it.
3. **Apparent positive gaps are artifacts, not regime conditioning.** With a *live* head (Phase 3's
   class-balance fix) the direction gap reads +0.008 and the settlement gap +0.18, "thesis supported" by
   sign — but in every such case FiLM is provably inert, the treatment is *absolutely worse*, and the
   positive degradation-gap is a `high−low` distributional artifact (regimes differ in label balance /
   calibration difficulty). The sign even flips (+0.008 → −0.025) with the head's calibration while FiLM
   stays inert. **Methodological upshot for the paper: the generalization-gap metric can read positive for
   reasons unrelated to the mechanism under test, so the FiLM activation diagnostics are load-bearing.**

**Secondary, genuinely useful findings:**
- The FI-2010 direction head collapses to the 76% flat-majority class under unweighted CE; gated
  inverse-frequency class weighting (`Direction.ClassBalanced`) revives it — a reusable correction.
- Explicit volatility-feature input (FeedObsVol, *not* FiLM) lifts absolute direction macro-F1 ~0.27→0.32 —
  a small positive worth a sentence, orthogonal to the FiLM thesis.
- The Polymarket MIMO headline config (`d_state 128`, `is_mimo true`) exceeds the RTX 4080's ~100 KB SMEM
  and segfaults; the settlement screen ran on the non-MIMO path. The MIMO headline model genuinely needs an
  A100+ — flagged, not spent.

**Decision-gate compliance:** all four families reached their pre-registered gates; none met the WIN
criteria (no family had an active FiLM, so the activation clause fails regardless of gap sign). No A100/H100
run was spent. This is the campaign's complete, publishable result: a strengthened, mechanistic negative —
regime conditioning fails to improve OOD forecasting even on regime-dependent objectives, across two
datasets, with the optimizer/router behavior that explains why.

**Total local cost:** FI-2010 ~10M-param non-MIMO runs ≈37 min each; Polymarket ~55M non-MIMO runs ≈84 min
each; all on the 4080 via WSL. No paid compute spent. A100 scale-up is **not** justified (no positive
multi-seed family to scale).

## PnL campaign — the economic-objective extension (`newgoal.md` + `newgoal-2.md` -> `pnl-film.md`)

The successor campaign rebuilt the Polymarket preprocessing (expose the causal Chainlink-spot path, a
time-to-expiry clock, an honest TTE-weighted settlement target, and an honest spot-vol / predictability
regime split) and replaced the forecasting proxy with **simulated PnL** from the DATAHACKS2026
execution engine, judged on a **predictability (Efficiency-Ratio)** regime axis with a CUSUM event
gate, quarter-Kelly sizing, a mandatory `NaiveLagStrategy` anti-artifact baseline, and a fat-tailed
bootstrap survivability gate.

**Result: a dual outcome — the FiLM null holds, but the economic objective yields a genuine WIN on the
data pipeline + strategy.** (A) Across 2 seeds the FiLM treatment is inert (`film_g ≈ 0.010`,
`reg_H = ln 8` router-collapsed); the PnL gap (−736, −519) is a weight artifact, not FiLM conditioning.
A predictability-supervised FiLM escalation (the strongest-null test) **confirms the null maximally**:
forced active and supervised directly on the spot Efficiency-Ratio axis with the strongest settings,
`film_g` still decays toward identity (0.150 → 0.043) and `reg_H` stays at ln 8. Across all five
families and two datasets, the key diagnostic (*does film_g stay clearly > 0?*) is **NO**, now
including the economic objective. (B) **But the predictability regime axis a FiLM router would not
encode is exactly the one a strategy gate harvests profitably.** The Phase-0 spot-conditioned model
plus a *train-tuned* causal predictability gate (selective participation, ER = 0.60 fit on train and
frozen) earns **+$4.6k (~46%) on held-out validation, P(profit) = 1.000, dd95 = 4.7%, beating a fair
trend-following naive by 8.9x** — passing every pre-registered anti-artifact gate (absolute profit,
beats a meaningful naive, bootstrap survivability, train-frozen threshold, fully causal). The model's
calibrated settlement probability, not the gate's mechanical oracle-lag edge, is the dominant driver.
**The campaign's contribution is the preprocessing overhaul + the selective-participation strategy
(economically vindicated), not the regime-conditioning architecture (a confirmed negative).**

**Exhaustively characterized + generalized.** The +$4.5k BTC edge is robust across 2 seeds, 2 intervals
(5m/15m), 1¢/share slippage, both FiLM arms, and all 3 strategy hyperparameters (ER tuned, edge robust,
window optimal at its default); the calibration keystone (over-confident-but-directional head) explains
why fixed sizing wins and Kelly fails even after temperature-scaling. **But it is BTC-specific:** a
per-asset held-out decomposition shows the BTC-trained model *loses* on ETH (−$1757, ≈ the naive) and
SOL (−$386, worse than the naive), so the edge does **not** generalize across assets — an earlier
combined-pool result that looked like generalization (+$3414 over a losing combined naive) was a
BTC-dominated aggregation artifact that the per-asset split overturned. Per-asset training (N=2 retrains,
asset filter through `pick_top_markets` → `Encoder.Assets`) finds ETH/SOL train edges that *overfit*
ungated — but the boundary is **book depth, not the ticker**: ungated they mostly trade *thin* ETH/SOL
books (BTC books are ~2× deeper: median depth 82,873 vs 40,109/32,157). **A book-depth gate recovers the
edge** — deep-book ETH (BTC-level depth) earns +$1325 held-out, P=0.992, beating a *fair depth-gated*
naive by +$3020 (the model, not trend-following). So the edge is a **deep-book microstructure
phenomenon**: robustly survivable on BTC (deep throughout), recoverable on ETH (gated to BTC-level
depth), marginal on SOL (thinnest). The positive is real, survivable, mechanistically explained, and
naive-beating; its boundary is *liquidity*, with BTC the asset deep enough throughout.

**Deferred follow-ons (note only, do not build — YAGNI):** live execution infrastructure
(`py-clob-client`, the RTDS WebSocket relay, Chainlink Data Streams, co-location) belongs to a future
live deployment, not this backtest-only research campaign. Threshold tuning (ER / CUSUM / edge / Kelly)
must be fit on the train split and frozen before any held-out backtest. No A100/H100 justified.

---

# COLM submission campaign (driver: `colm-submission-goal.md`, 2026-06-20)

Take the paper from "strong null" to submitted. Frozen vintage (`w2-reconciliation.md`);
`finmamba3-paper.tex` is the current-vintage source of truth.

## Track A — workshop-ready (DONE)

| item | status | notes |
|---|---|---|
| A1 reconcile `.tex`/`RESULTS.md`/`readme.md` | **done** | every economic figure traces to `reports/recon_*.json`; grep-clean of recorded-vintage numbers; settlement low-pred reframed "+$51 marginal" not "−$826 catastrophic"; EV "comparable total, 6.2× collapse" not "+41%/12×" |
| A2 bibliography + figures + clean compile | **done** | `nagy2023lobs5` key/year/arXiv fixed; flagged arXiv ids verified; Table 1 mixed-units annotated; 4 figures resolve; `pdflatex` clean twice |
| A3 workshop drafts (user chose 3 venues) | **done** | `sufm.md` (~9pp), `actionable-interpretability.md` (~9pp), `moss-paper.tex` (8pp, compiles); double-blind anonymized, frozen numbers |

Deadlines (verified): SUFM Jun 23 AoE (9/4pp, double-blind, non-archival); Actionable Interpretability
Jun 24 AoE (9/5pp, double-blind, non-archival); MOSS on the COLM-26 list (CFP unconfirmed).

## Track B — conference-grade demonstration (DONE; for COLM 2027)

| item | verdict | sheet | drives |
|---|---|---|---|
| B1 gauge vs. regularization | **SUPPORTS** | `gauge-mechanism.md` | WD-off flattens decay (τ 6685→3e5, R² .9999→.48) ⇒ gauge decay is weight-decay-driven over a flat loss |
| B2 in-kernel boundary (toy) | **SUPPORTS** | `inkernel-boundary.md` | input-affine folds exactly (1.2e-6) vs in-kernel non-absorbable (1.3e-2, 1e4×); in-kernel persists under WD ⇒ boundary demonstrated |
| B3 Mamba-LM control | **CONTRADICTS (refines)** | `lm-film.md` | strongly-decodable regime (code vs prose, router .95) ⇒ FiLM persists (0.11→0.52) ⇒ scope the null to *weakly-decodable* regimes |
| B4 scale-redundancy lit | **done** | `sec:related` | van Laarhoven 2017 / Hoffer 2018 / Li & Arora 2020 + novelty delta |
| B5 de-bundle / restructure | **done** | this file + workshop drafts | 9-page de-bundled builds = the workshop drafts; master keeps finance as a "demonstration substrate" |

Each Track B sheet carries a pre-registered SUPPORTS/CONTRADICTS verdict; the B3 contradiction is reflected
honestly in the paper (the "Language-model control" paragraph + scoped strongest-null sentence), not buried.
No A100/H100 launched. `pdflatex` clean (24pp). Host `pytest`: 273 passed, 2 skipped, 1 pre-existing
MIMO-deprecation failure (no `src/` touched this campaign).

## Split-option recommendation (user decides; not executed)

The paper carries two loosely-coupled stories: the gauge-absorption null + mechanism (B1/B2) + boundary (B3),
and the spot-conditioned economic edge (which self-deflates — a logistic beats the world model).
**Recommend splitting for 2027:** Paper 1 (COLM/SSM) = the null + B1/B2/B3, finance as a one-paragraph
substrate (start from `sufm.md`, fold in B1/B2/B3); Paper 2 (finance, e.g. ICAIF) = the spot pipeline + edge +
EV head + simple-model deflation. The master `finmamba3-paper.tex` stays the combined comprehensive record.
