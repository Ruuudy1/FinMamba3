"""Plot the predictability-supervised escalation diagnostics: the decisive regime-FiLM null evidence.

Top panel: film_g (mean |gamma-1|) decays monotonically to identity on both datasets despite forced
activation + direct ER-bucket supervision. Bottom panel: reg_H stays pinned at ln 4 (uniform router).
"""
# region imports
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# endregion
PATTERN = re.compile(r"step=(\d+).*?film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+)")


def parse(path):
    value_by_step = {}
    for line in open(path, errors="ignore"):
        for match in PATTERN.finditer(line):
            value_by_step[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))
    steps = sorted(value_by_step)
    film_g = np.array([value_by_step[s][0] for s in steps])
    reg_h = np.array([value_by_step[s][1] for s in steps])
    return np.array(steps), film_g, reg_h


def main():
    kaggle_steps, kaggle_g, kaggle_h = parse("reports/kaggle_escalation_s0.log")
    fi_steps, fi_g, fi_h = parse("reports/fi2010_escalation_s0.log")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    ax_top.plot(kaggle_steps, kaggle_g, color="C0", lw=1.6, label=f"Kaggle BTC (0.250 -> {kaggle_g[-1]:.3f})")
    ax_top.plot(fi_steps, fi_g, color="C1", lw=1.6, label=f"FI-2010 (0.250 -> {fi_g[-1]:.3f})")
    ax_top.axhline(0.0, color="0.5", ls="--", lw=0.9, label="identity (film_g = 0)")
    ax_top.set_ylabel("film_g  (mean |gamma - 1|)")
    ax_top.set_title("Forced-active, ER-supervised FiLM decays monotonically to identity")
    ax_top.set_ylim(bottom=0.0)
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(alpha=0.3)
    ax_bot.plot(kaggle_steps, kaggle_h, color="C0", lw=1.6, label="Kaggle BTC")
    ax_bot.plot(fi_steps, fi_h, color="C1", lw=1.6, label="FI-2010")
    ax_bot.axhline(float(np.log(4)), color="0.5", ls="--", lw=0.9, label="ln 4 (uniform router)")
    ax_bot.set_ylabel("reg_H  (router entropy)")
    ax_bot.set_xlabel("training step")
    ax_bot.legend(loc="lower right", fontsize=9)
    ax_bot.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/film_g_escalation_decay.png", dpi=150)
    print(f"saved reports/film_g_escalation_decay.png (kaggle {len(kaggle_steps)} pts, fi2010 {len(fi_steps)} pts)")


if __name__ == "__main__":
    main()
