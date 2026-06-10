"""Paper-quality (title-less, vector PDF) version of the escalation film_g decay figure, into the tracked
figures/ dir. The title goes in the LaTeX \\caption, not the figure.
"""
# region imports
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# endregion
PATTERN = re.compile(r"step=(\d+).*?film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+)")


def parse(path):
    value_by_step = {}
    for line in open(path, errors="ignore"):
        for match in PATTERN.finditer(line):
            value_by_step[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))
    steps = sorted(value_by_step)
    return (np.array(steps),
            np.array([value_by_step[s][0] for s in steps]),
            np.array([value_by_step[s][1] for s in steps]))


def main():
    Path("figures").mkdir(exist_ok=True)
    kaggle_steps, kaggle_g, kaggle_h = parse("reports/kaggle_escalation_s0.log")
    fi_steps, fi_g, fi_h = parse("reports/fi2010_escalation_s0.log")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)
    ax_top.plot(kaggle_steps, kaggle_g, color="C0", lw=1.6, label=f"Kaggle BTC (0.250$\\to${kaggle_g[-1]:.3f})")
    ax_top.plot(fi_steps, fi_g, color="C1", lw=1.6, label=f"FI-2010 (0.250$\\to${fi_g[-1]:.3f})")
    ax_top.axhline(0.0, color="0.5", ls="--", lw=0.9, label="identity")
    ax_top.set_ylabel(r"$\mathrm{film}_g$  (mean $|\gamma-1|$)")
    ax_top.set_ylim(bottom=0.0)
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(alpha=0.3)
    ax_bot.plot(kaggle_steps, kaggle_h, color="C0", lw=1.6, label="Kaggle BTC")
    ax_bot.plot(fi_steps, fi_h, color="C1", lw=1.6, label="FI-2010")
    ax_bot.axhline(float(np.log(4)), color="0.5", ls="--", lw=0.9, label=r"$\log 4$ (uniform)")
    ax_bot.set_ylabel(r"$\mathrm{reg}_H$  (router entropy)")
    ax_bot.set_xlabel("training step")
    ax_bot.legend(loc="lower right", fontsize=8)
    ax_bot.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/film_g_escalation_decay.pdf")
    print("saved figures/film_g_escalation_decay.pdf")


if __name__ == "__main__":
    main()
