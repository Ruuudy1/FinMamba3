"""Evaluation harnesses for the FinMamba3 world model.

This package groups four distinct concerns. The modules are kept at a flat
``finmamba3.eval.<module>`` path on purpose: those paths are referenced by the
packaged ``finmamba3-backtest`` console script, the CI workflow, the test suite,
and the ``experiments/altdata`` reproduction scripts, so a physical split into
subpackages would change every one of those call sites (and the reproduction
commands recorded in the campaign logs). The map below gives the directory the
cohesion a subpackage layout would, without breaking those paths.

PnL / backtest stack (the economic results):
    pnl_backtest          Settlement-accurate order-matching PnL adapter + CLI (the paper tables).
    survivability         Kelly sizing + fat-tail bootstrap statistics (pure NumPy).
    signals               World-model probability / EV inference + causal gate series.
    strategies            CusumGate, WorldModelStrategy, NaiveLagStrategy.
    backtest              Gym-env Sharpe/PnL sketch harness (the other evaluation path).
    run_backtest_cli      CLI wrapper around the gym-env backtest.
    predictability        Efficiency-ratio predictability regime axis used by the gate.
    regime_split          Time / volatility non-stationarity splits.
    simple_baseline       Logistic + GBT baselines on spot features (architecture-independence control).
    competition_strategy  DATAHACKS BaseStrategy adapter.

Generalization evaluation:
    eval_regime_generalization          Cross-regime generalization-gap evaluator (Polymarket).
    eval_regime_generalization_fi2010   The FI-2010 counterpart.
    eval_backbone_metrics               Held-out recon NLL + direction macro-F1 (backbone ablation).
    benchmark_fi2010                    FI-2010 macro-F1 vs the LOBCAST paper.
    compare_direction                   World-model vs LinearAR vs DeepLOB direction benchmark.

Diagnostics:
    diag_dirhead          Direction-head diagnostics.
    diag_regime_labels    Regime-label distribution diagnostics.
    diag_router           Regime-FiLM router-discrimination diagnostic.
    diagnose_collapse     Temporal-prior collapse diagnostics.

Figures / smoke:
    make_paper_figures    Model-free paper figures from persisted artifacts.
    imagination_smoke     Phase-B imagination smoke test.
"""
