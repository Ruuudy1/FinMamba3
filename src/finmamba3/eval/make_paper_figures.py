"""Generate the model-free paper figures from persisted backtest artifacts.

Two figures need no world-model forward pass, only numbers already written to reports/ or reported in the
campaign sheets, so they regenerate deterministically on any machine:
  (1) the settlement-head reliability diagram (over-confident-but-directional), from the headline
      deterministic sweep's calibration bins;
  (2) the regime-FiLM inertness bar across every A/B setting, showing the treatment gamma deviation sits
      at ~0.004 everywhere, far below an active modulation.
Run: python -m finmamba3.eval.make_paper_figures
"""
# region imports
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# endregion


def reliability_figure(calib_json: Path, out_path: Path) -> None:
    """Plot mean predicted probability vs realized YES per decile; the gap to the diagonal is miscalibration.

    Marker area scales with the bin's sample count so the eye weights the populous extreme deciles, which
    is where the divergence trade lives. Points below the diagonal for mean > 0.5 are the over-confidence
    that makes Kelly sizing fail while fixed small sizing on the (still-correct) direction wins.
    """
    calib = json.loads(calib_json.read_text())["calibration"]
    bins = [b for b in calib["bins"] if b["n"] > 0]
    mean_prob = np.array([b["mean_prob"] for b in bins])
    realized = np.array([b["realized_yes"] for b in bins])
    counts = np.array([b["n"] for b in bins], dtype=np.float64)
    sizes = 40.0 + 360.0 * (np.log10(counts) - np.log10(counts.min())) / max(np.log10(counts.max()) - np.log10(counts.min()), 1e-9)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1.0, ls="--", label="perfect calibration", zorder=1)
    ax.axhline(0.5, color="0.85", lw=0.8, zorder=0)
    scatter = ax.scatter(mean_prob, realized, s=sizes, c=mean_prob, cmap="coolwarm", vmin=0, vmax=1, edgecolors="k", linewidths=0.6, zorder=3)
    for b in bins:
        ax.annotate(f"{b['n']/1000:.0f}k", (b["mean_prob"], b["realized_yes"]), textcoords="offset points", xytext=(5, -7), fontsize=6, color="0.3")
    ax.set_xlabel("mean predicted P(YES)")
    ax.set_ylabel("realized YES frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Settlement-head reliability (Brier {calib['brier']:.3f})", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.text(0.97, 0.06, "below diagonal =\nover-confident", ha="right", va="bottom", fontsize=6.5, color="0.3", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def film_inertness_figure(out_path: Path) -> None:
    """Bar of treatment film_g across every A/B setting against an active-modulation reference.

    Every jointly-trained setting leaves film_g near 0.004 (the modulation never grows from its exact
    identity init); the dashed references mark a forced-active init (0.150) and the strongest-null
    escalation init (0.250), so the visual gap is the whole story: the optimizer keeps the modulation at
    identity regardless of dataset, asset, or resolution.
    """
    settings = [
        "FI-2010", "Kaggle BTC 1m", "Kaggle ETH 1m", "Kaggle ADA 1m",
        "Kaggle BTC 5m", "Kaggle BTC 1s", "Polymarket settle", "Polymarket PnL",
    ]
    film_g = [0.0043, 0.0048, 0.0047, 0.0049, 0.0048, 0.0046, 0.008, 0.010]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    positions = np.arange(len(settings))
    ax.bar(positions, film_g, color="#4c72b0", width=0.62, label=r"treatment $\mathrm{film}_g$ (jointly trained)")
    ax.axhline(0.150, color="#c44e52", lw=1.2, ls="--", label="forced-active init (0.150)")
    ax.axhline(0.250, color="#8172b3", lw=1.2, ls=":", label="strongest-null escalation init (0.250)")
    ax.set_xticks(positions)
    ax.set_xticklabels(settings, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel(r"$\mathrm{film}_g = \mathrm{mean}\,|\gamma-1|$", fontsize=9)
    ax.set_ylim(0, 0.28)
    ax.set_title("RegimeFiLM is inert across every setting (router entropy at $\\log R$ throughout)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    for pos, value in zip(positions, film_g):
        ax.annotate(f"{value:.3f}", (pos, value), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5, color="0.25")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def equity_figure(baseline_json: Path, out_path: Path) -> None:
    """Plot terminal-PnL bootstrap distributions for the simple-model arms vs the world-model headline.

    Reads the simple_baseline report's market-block bootstrap CI for the logistic and GBT arms and draws
    them against the world-model headline's reported PnL, so the figure shows directly whether a simple
    model on the same features lands anywhere near the world model. No-op when the report is absent.
    """
    if not baseline_json.exists():
        return
    report = json.loads(baseline_json.read_text())
    arms = report["arms"]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    # World-model reference PnLs on the same 645-market set (persisted: tab:ev_arms / tab:ev_seeds).
    ax.axvline(5147.0, color="#c44e52", lw=1.1, ls="--", label="WM settlement head (+5147)")
    ax.axvline(4779.0, color="#dd8452", lw=1.1, ls=":", label="WM EV head, 3-seed mean (+4779)")
    ax.axvline(0.0, color="0.8", lw=0.8)
    labels = []
    for index, (name, arm) in enumerate(arms.items()):
        # The split report nests the full-set arm under "total"; the flat report is the arm itself.
        total = arm.get("total", arm)
        block = total["bootstrap_market"]
        labels.append(f"{name}\n({block['n_markets']} mkts)")
        ax.errorbar(total["pnl"], index, xerr=[[total["pnl"] - block["ci_low"]], [block["ci_high"] - total["pnl"]]],
                    fmt="o", color="#4c72b0", capsize=4, ms=7, zorder=3)
        ax.scatter(total["naive_pnl"], index, marker="|", s=160, color="0.5", zorder=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("held-out PnL on 645 BTC markets (\\$); bars = market-block 95\\% CI, $|$ = fair naive")
    ax.set_title("A logistic on the same features beats the world model", fontsize=9)
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate model-free paper figures from persisted artifacts")
    p.add_argument("--calib-json", type=Path, default=Path("reports/sweep_heldout_btc_det1.json"))
    p.add_argument("--baseline-json", type=Path, default=Path("reports/simple_baseline_btc645.json"))
    p.add_argument("--figdir", type=Path, default=Path("figures"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    reliability_figure(args.calib_json, args.figdir / "reliability_settlement.pdf")
    film_inertness_figure(args.figdir / "film_g_across_settings.pdf")
    equity_figure(args.baseline_json, args.figdir / "simple_baseline_pnl.pdf")
    print(f"figures written to {args.figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
