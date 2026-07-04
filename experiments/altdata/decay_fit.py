"""Quantify the escalation film_g decay: does it extrapolate to identity (asymptote c ~ 0) or to a small
near-identity plateau? Fit film_g(t) = a*exp(-t/tau) + c to each escalation trajectory and report the
asymptote c (the extrapolated end-state of the modulation) with R^2.
"""
# region imports
import re
import numpy as np
from scipy.optimize import curve_fit
# endregion
PATTERN = re.compile(r"step=(\d+).*?film_g=([0-9.]+)")


def parse(path):
    value_by_step = {}
    for line in open(path, errors="ignore"):
        match = PATTERN.search(line)
        if match is not None and "film_g=" in line:
            value_by_step[int(match.group(1))] = float(match.group(2))
    steps = np.array(sorted(value_by_step))
    film_g = np.array([value_by_step[s] for s in steps])
    return steps, film_g


def decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit(name, path):
    steps, film_g = parse(path)
    keep = film_g > 0.0
    steps, film_g = steps[keep], film_g[keep]
    popt, _ = curve_fit(decay, steps, film_g, p0=[0.2, 1500.0, 0.0],
                        bounds=([0.0, 1.0, -0.05], [1.0, 1e6, 0.3]), maxfev=20000)
    a, tau, c = popt
    residual = film_g - decay(steps, *popt)
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((film_g - film_g.mean()) ** 2)
    print(f"=== {name} ({len(steps)} pts, {steps[0]}->{steps[-1]}) ===")
    print(f"  exp-decay fit: a={a:.4f} tau={tau:.0f} steps  asymptote c={c:.4f}  R^2={r2:.4f}")
    print(f"  -> extrapolated end-state film_g = {c:.4f} ({'approaches identity' if c < 0.02 else 'near-identity plateau'})")
    print(f"  linear slope = {np.polyfit(steps, film_g, 1)[0]:.2e}/step (monotone-decreasing check)")


def main():
    fit("Kaggle BTC escalation", "reports/kaggle_escalation_s0.log")
    fit("FI-2010 escalation", "reports/fi2010_escalation_s0.log")


if __name__ == "__main__":
    main()
