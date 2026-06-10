"""Summary figure: treatment film_g is inert (~0.004) in every baseline-A/B setting — the universal
regime-FiLM null at a glance. The zero-init modulation starts at 0 (identity) and a WIN would require it
to grow clearly above 0; instead it sits at ~0.004 everywhere, dataset/asset/resolution-independent.
"""
# region imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# endregion
# Treatment-arm film_g at convergence per baseline-A/B setting (from the run [loss] lines / manifests).
SETTINGS = [
    ("FI-2010 s0", 0.0043), ("FI-2010 s1", 0.0040),
    ("Kaggle BTC s0", 0.0048), ("Kaggle BTC s1", 0.0038),
    ("Kaggle ETH", 0.0047), ("Kaggle ADA", 0.0049),
    ("BTC 5min", 0.0048), ("BTC 1sec", 0.0046),
    ("FI-2010 CB s0", 0.0044), ("FI-2010 CB s1", 0.0041),
]


def main():
    labels = [s[0] for s in SETTINGS]
    values = np.array([s[1] for s in SETTINGS])
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    x = np.arange(len(labels))
    ax.bar(x, values, color="C0", width=0.62, label="treatment film_g (zero-init)")
    ax.axhline(0.0, color="0.4", lw=1.0, ls="-", label="identity (init)")
    ax.axhline(0.05, color="C3", lw=1.1, ls="--", label="≈ a plausibly-active FiLM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("film_g  (mean |gamma - 1|)")
    ax.set_ylim(0, 0.06)
    ax.set_title("Zero-init regime-FiLM stays inert (~0.004) in every setting — the universal null")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.0015, f"{v:.4f}", ha="center", va="bottom", fontsize=7.0)
    fig.tight_layout()
    fig.savefig("reports/film_g_across_settings.png", dpi=150)
    print(f"saved reports/film_g_across_settings.png ({len(SETTINGS)} settings, all film_g in "
          f"[{values.min():.4f}, {values.max():.4f}])")


if __name__ == "__main__":
    main()
