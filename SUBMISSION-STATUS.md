# Submission status: backtester engine and reproducibility artifact

**Date:** 2026-06-21  **Branch:** `workshop-submissions-finishing` (off `architecture-review-fixes`).

Purpose: one place to see what is finished on the engineering / reproducibility track (native engine,
parallelization, packaging, Docker) so the remaining submission gaps are easy to spot. The paper track at
the bottom now carries an **independent build/page-limit verification** (§4, fresh `pdflatex` rebuild
2026-06-21); paper *content* and per-table number reconciliation remain the author's domain.

Legend: ✅ done · ⚠️ done with a caveat · ☐ open.

---

## 0. Map: where things live

| artifact | location | state |
|---|---|---|
| Native C++ engine + parallel pass | `src/finmamba3/backtester/engine_cpp/`, `eval/pnl_backtest.py` | committed engine (`3d344fd`); parallel pass uncommitted |
| Engine engineering record | `cpp-engine-optimization.md` (repo root, **untracked** note) | §1-§9 engine, §10 parallel pass |
| Architecture-review fixes | repo-wide, 12 commits `f4e241c..HEAD` | committed |
| CPU reproducer image | `Dockerfile.backtest`, `docker-compose.yml`, `docker/BACKTEST.md` | this plan, **uncommitted** |
| Profiling harnesses | `benchmarks/engine/` | this plan, **uncommitted** |
| Image tarball | `C:\tmp\finmamba3-backtest.tar.gz` (491 MB) | built |
| Papers | `sufm.tex`, `actionable-interpretability.tex`, `moss-paper.tex` | uncommitted, author reports submission-ready |

---

## 1. Done before this plan

### 1a. Native C++ backtester engine (parity-exact) ✅
Commits `2c5fb2c` (vendor the DATAHACKS2026 engine), `863e875` (pybind11 prototype), `3d344fd` (usable
end-to-end), on `vendor-backtester-engine` and reachable from the current branch.

- Opt-in `--engine cpp` in `pnl_backtest`; the pure-Python `finmamba3.backtester.engine` stays the
  canonical **oracle**.
- Sparse CSR over present `(tick, market)` cells (dense `T×M×K` was 269 GB-infeasible at hours=12; sparse
  is ~75 MB), ladders truncated at cumulative 500 shares, full strategy-gate coverage (Kelly, calibration
  temperature, predictability gate in live-window and per-market-ER forms, depth band, edge mode).
- Speedups, parity-exact (pnl / trades / Sharpe / bootstrap to `0.00e+00`): single-run ~10.5x, gated
  sweep ~44x, pure compute ~107x. Guarded by `tests/test_engine_cpp_parity.py` (5 tests).
- Record: `cpp-engine-optimization.md` §1-§9.

### 1b. Parallel + vectorized-bootstrap pass ✅ (verified on GPU)
Uncommitted on `vendor-backtester-engine`, dated 2026-06-19.

- Profiling showed the C++ engine was no longer the bottleneck (~39 ms/value); the GIL-bound Python
  market-block bootstrap was (~2020 ms/value).
- Shipped: (1) `py::gil_scoped_release` around the C++ tick loop; (2) thread-parallel sweep + dual-arm via
  `--engine-threads` (0=auto, 1=serial, cpp-path only); (3) vectorized `bootstrap_survivability_by_market`
  (numpy, bitwise-identical to the old loop). ~5.4x combined on a hours=6 / 10-value sweep; now
  marshalling-bound.
- Verified on the 4080 via WSL: parity tests pass on the real Linux `.so`, and a single-process check
  feeds one GPU forward pass's edge-EV probs over all 645 BTC markets to both engines for a
  bitwise-identical result.
- Record: `cpp-engine-optimization.md` §10.

### 1c. Architecture-review fixes (F-1 .. F-13, committed) ✅
12 commits `f4e241c..HEAD`. These reshaped the repo into a submittable package and are what this plan
built on rather than duplicated:

- `Dockerfile.backtest` (CPU-only engine reproducer) + de-forked training `Dockerfile` (`e4f300d`, `1a32408`).
- Packaging: `[cpp]` (pybind11) and `[test]` (pytest) optional extras + `finmamba3-backtest` console
  script (`cef4529`); README layout regen (`fcb0537`, `d9280a3`).
- CPU CI that builds `engine_cpp` and runs the parity test (`2fb09c9`, `.github/workflows/ci.yml`).
- `pnl_backtest.py` split into `survivability` / `signals` / `strategies`, **re-exported so its public
  import surface is unchanged** (`7f12c49`, `d714c80`); two documented backtest paths (`cb5e5b1`).
- Code-quality: try/except + isinstance/getattr audit (`8f6cc16`), em-dash strip in `engine_cpp`
  (`742383e`), vendored-engine subpackage exemption (`d3a13d4`).
- COLM null demonstration (B1/B2/B3) and logistic/GBT refit on the current vintage (`f4e241c`, `fd60728`).

---

## 2. This plan: Docker reproducibility artifact

Goal: make the native + parallel C++ backtester reproducible as an image any machine can build, prove
parity, and re-measure the speedups. Built on `Dockerfile.backtest` (did not touch the CUDA image, by
design). All changes uncommitted on `workshop-submissions-finishing`.

### What changed
- `benchmarks/engine/` (new): `engine_harness`, `gate_parity`, `sweep_profile`, `verify_bootstrap_vec`,
  and a shared `bench_common.py`; house-style, `FINMAMBA_DATA_VAL`-driven, GPU/checkpoint-free.
- `benchmarks/engine/fetch_data.sh` (new): pulls the 345 MB `data/polymarket/validation.tar.gz` from the
  public HF dataset `sj-hryi/FinMamba3`.
- `Dockerfile.backtest`: `COPY benchmarks` and `COPY configs` added.
- `docker-compose.yml` (new): single `backtest` service that encodes the data / checkpoint mounts.
- `docker/BACKTEST.md` (new): build / fetch / profiling / distribution runbook with recorded numbers.

### Bugs found and fixed during verification ✅
1. **`.dockerignore` build-break:** the compiled `_engine*.{pyd,so,obj}` from the host tree were copied by
   `COPY src`, and the in-image `python -m engine_cpp.build` import crashed on an ABI mismatch before the
   build ran. Now excluded, so the image compiles a fresh `.so`. (CI never hit this because it builds from
   a clean checkout.)
2. **Missing `configs/`:** the full test suite failed 4 world-model tests with
   `FileNotFoundError: configs/lob.yaml` (they stub `mamba_ssm` but still load a real config). `COPY configs`
   fixed it and also unblocks the `--config configs/...` entrypoint usage.

### Verification, in-container (Linux / gcc / Python 3.12, 22-core box) ✅
| step | result |
|---|---|
| Build parity gate | **5 passed** inside the build |
| `verify_bootstrap_vec` | 40 cases + empties **bitwise-identical** |
| `gate_parity 12` (T=43201, M=582) | **12/12 PASS**, every delta `0e+00` |
| `engine_harness 12` | PARITY PASS; single-run **12.92x**, sweep×20 **140.7x** |
| `sweep_profile 6 10` (`--cpus 8`) | rows **byte-identical** across threads {1,2,4,8}; 1.00 / 1.34 / 1.96 / 2.03x; marshalling-bound |
| full `pytest tests/` | **273 passed, 2 skipped, 1 deselected** (mirrors CI) |

### Distribution
- Offline tarball saved: `C:\tmp\finmamba3-backtest.tar.gz` (491 MB), `docker load`-able on any machine. ✅
- GHCR push (`ghcr.io/ruuudy1/finmamba3-backtest`) documented in the runbook, not executed. ☐

---

## 3. Caveats / known limits ⚠️

- **The CPU image cannot reproduce the world-model PnL tables.** The entrypoint loads the frozen model and
  runs a Mamba forward needing `mamba_ssm` (CUDA). The engine is proven identical Python-vs-C++ on
  identical probabilities; the table-generating model arm stays on the GPU / wheels flow (verified on the
  4080, §10). The image reproduces the engine (parity + speedup), not the model forward.
- **Absolute timings are host-dependent** (CPU, MSVC vs gcc). The submittable invariants are the parity
  deltas (~0) and the thread-determinism (rows byte-identical), both of which hold exactly.
- **`cpp-engine-optimization.md` and this file are untracked research notes** on this branch (the repo
  convention here). Decide whether either should be tracked for the submission.

---

## 4. Submission readiness and open items

### Engineering / reproducibility (this track)
- ✅ Native engine, parallel pass, parity guard, CPU CI, packaging extras, console script.
- ✅ Portable CPU image builds clean, parity gate green, profiling + full suite verified, tarball saved.
- ☐ **Commit the Docker work** (uncommitted: `.dockerignore`, `Dockerfile.backtest`, `benchmarks/`,
  `docker-compose.yml`, `docker/`). Currently interleaved with uncommitted paper edits on the same branch.
- ☐ **GHCR push** (needs a PAT with `write:packages`).
- ☐ Optional: a cached-probabilities path so the CPU image could reproduce a PnL table without a GPU
  (today that needs `mamba_ssm`). Only worth it if a reviewer must regenerate a table on a CPU box.

### Paper track (independently verified 2026-06-21 — fresh from-scratch rebuild)
Three COLM-2026 workshop papers. Re-verified here, not transcribed: each was rebuilt by wiping all `.aux`
and running a clean 2-pass `pdflatex` (inline `thebibliography`, so no bibtex). Page boundaries below come
from the rebuilt `.aux` (section->page) cross-checked against the boundary pages of each PDF.
(Note: `latexmk` aborts in this shell on an unrelated MiKTeX PATH quirk — `did not succeed ... cannot
retrieve attributes for ... claude.exe` — direct `pdflatex` builds clean, all passes exit 0.)

**pdflatex-clean confirmed for all three** ✅ — 0 undefined refs/citations, 0 overfull hboxes, 0
overfull/underfull vboxes. The only log warnings are the benign `h`->`ht` float-placement notes
(cosmetic) in Actionable and MOSS; SUFM logs zero warnings.

| paper | file | deadline (AoE) | limit | verified page layout | within limit? |
|---|---|---|---|---|---|
| SUFM | `sufm.tex` | **Jun 23** | 9 pp main, refs unlimited | body pp 1-9 (§7 Conclusion ends on p9), refs p10; 10 pp total | ✅ |
| Actionable Interpretability | `actionable-interpretability.tex` | **Jun 24** | 9 pp excl. refs + appendix | body pp 1-9 (§8 Conclusion ends on p9), appendix A/B p10, refs pp11-12; 12 pp total | ✅ |
| MOSS | `moss-paper.tex` | **Jun 30** | 4 pp main + unlimited supp. | §7 Discussion now ends on p4; appendices A-D pp5-6, refs p7; 7 pp total | ✅ (trimmed 2026-06-21) |

Open / to confirm on the paper side:
- ☐ **MOSS .sty proxy:** typeset against the COLM-2026 template as a stand-in; swap in the MOSS-specific
  style for camera-ready if it differs (author flagged). Not independently verifiable here without the
  MOSS style file.
- ✅ **MOSS 4-page core — FIXED (2026-06-21).** §7 Discussion previously wrapped ~2 lines onto p5 before
  App.A. Removed its redundant closing sentence ("The claim is scoped to weakly-decodable regimes; the
  economic coda is a deflated, architecture-independent backtest" — both clauses already stated in the
  abstract, contributions, and §4/§5), so §7 now ends on p4. Rebuilt: §7 on p4, App.A on p5, still 7 pp,
  0 undefined/overfull/underfull. Main body is now cleanly ≤4 pp.
- ☐ Commit the paper edits (`sufm.tex`, `actionable-interpretability.tex`, `moss-paper.tex` + COLM
  `.sty`/`.bst`, `fancyhdr.sty`, `natbib.sty`), currently uncommitted.
- ✅ Cross-track (matched-Transformer row): SUFM Table 3 carries `Transformer (pre-norm) 9.61M, recon NLL
  -0.5041 / 0.0224` — the exact 4080 re-run vintage recorded in memory — non-dominant as claimed; MOSS
  Table 3's caption states the same ("the Mamba-3-matched pre-norm Transformer is competitive but
  non-dominant"). The matched-Transformer ablation row has landed in the paper tables. (Settlement
  market-block CI is the author's to confirm row-by-row.)
- ✅ **Table reconciliation (every number, all 3 papers) — done 2026-06-21.** Cross-checked each cell
  against the source-of-truth result files:
  - *Table 1* (RegimeFiLM null, 5 rows): dataset labels correct (vol/scale + volume + direction on
    FI-2010 per `vol-scale.md`/`volume-order-flow.md`; settlement + PnL on Polymarket); gaps (≈0, +0.008,
    +0.18, −$627) and `film_g` (ident, 0.018, 0.008, 0.010) match `RESULTS.md §1` + `pnl-film.md §4`
    (PnL gap −627.4). ✅
  - *Table 2* (cross-dataset, 6 rows): `film_g` 0.004–0.005 match `RESULTS.md §5` (rounded from
    0.0040–0.0049). ✅
  - *Table 3* (backbone): every param + recon NLL matches `RESULTS.md §5.1` exactly (SUFM/Actionable keep
    the post-norm Transformer row; MOSS omits it — editorial, the listed numbers are identical). ✅
  - *Economic coda/substrate prose*: +$5,147, +$6,914 / Brier 0.146 / P 0.9994 / CI [+$2,844,+$10,880]/243,
    +$4,779, [+$2,445,+$7,747]/127, −$1,008, +$51/P0.63, 6.2× (5,044→814), +$4,729 / 1,553 trades / 408 mkts
    — all match `w2-reconciliation.md` (the current pinned vintage) + `pnl-film.md`. ✅
  - **Stale-number sweep:** grep for every superseded figure (−$826, +$2,548, +41%, 12×, +$3,946, +$7,016,
    [+$2,686,+$7,951], +$4,643) returns **0 hits** in all three papers — none of the old vintage survives.
  - **One disclosed nuance (not an error):** reliability deciles read 0.787 top / 0.109 bottom in the
    papers vs 0.785 / 0.108 for the 0.1-wide bins in `pnl-film.md §3e` — a top/bottom-decile vs 0.1-bin
    definitional difference of ~0.002, and all three papers already label this the "recorded-vintage
    structural diagnostic" (deliberately not reconciled to current). Brier 0.206 and 214k probs match exactly.
  - **Memory caveat resolved:** the high-pred CI recorded in memory as [+$2.7k,+$7.9k]/129 is the
    *superseded* vintage; the papers correctly use the current [+$2,445,+$7,747]/127 (`w2` Finding 1).
    `RESULTS.md §7` rounds the logistic P to 0.999; the papers' "exactly 0.9994" matches `w2` Finding 4.

### Consolidated "what's missing" checklist
- ☐ Decide branch/commit strategy: separate the Docker commit from the paper commits, or one PR.
- ☐ GHCR push (if remote distribution is wanted for reviewers) + add the pull command to the runbook.
- ✅ Confirmed each paper's tables/numbers match the current data vintage — full row-by-row reconciliation
  done 2026-06-21 (see paper track): Tables 1/2/3 + all economic prose trace to RESULTS.md /
  w2-reconciliation.md / pnl-film.md / vol-scale.md / volume-order-flow.md on the current pinned vintage,
  with a 0-hit stale-number sweep. One disclosed recorded-vintage diagnostic nuance (reliability deciles),
  not an error.
- ✅ MOSS 4-page balance FIXED (trimmed the redundant §7 closing sentence; §7 now ends on p4). MOSS `.sty`
  proxy still open (typeset against the COLM template stand-in, not independently verifiable here).
- ☐ Decide whether to track `cpp-engine-optimization.md` / `SUBMISSION-STATUS.md` for reviewers.
