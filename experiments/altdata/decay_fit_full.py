"""Persisted, audit-grade refit of the escalation film_g (and post-scan film_b) decays.

Two jobs the bare decay_fit.py does not do:
  1. Refit a*exp(-t/tau)+c with the asymptote lower bound RELAXED (c in [-1, 0.3]) alongside the original
     c>=-0.05 floor, so the reported asymptote is the true unconstrained estimate rather than a value
     pinned at the bound. This settles whether "converges to identity within horizon" is an observation
     or an extrapolation.
  2. Compute and PERSIST the post-scan (output-side) tau fits, which the original driver never stored
     (it crashed on a missing scipy before reaching the fit). Output is written to a JSON artifact so the
     papers cite a reproducible number.
"""
# region imports
import json
import re
import numpy as np
from scipy.optimize import curve_fit
# endregion
STEP = re.compile(r"step=(\d+)")
FILM_G = re.compile(r"film_g=([0-9.]+)")
FILM_B = re.compile(r"film_b=([0-9.]+)")


def parse(path, key):
    pattern = FILM_G if key == "film_g" else FILM_B
    value_by_step = {}
    for line in open(path, errors="ignore"):
        step_match = STEP.search(line)
        value_match = pattern.search(line)
        if step_match is not None and value_match is not None:
            value_by_step[int(step_match.group(1))] = float(value_match.group(1))
    steps = np.array(sorted(value_by_step))
    values = np.array([value_by_step[s] for s in steps])
    return steps, values


def decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_one(steps, values, c_lower):
    popt, _ = curve_fit(decay, steps, values, p0=[0.2, 1500.0, 0.0],
                        bounds=([0.0, 1.0, c_lower], [1.0, 1e6, 0.3]), maxfev=40000)
    a, tau, c = popt
    residual = values - decay(steps, *popt)
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((values - values.mean()) ** 2)
    pinned = abs(c - c_lower) < 1e-3
    return {"a": float(a), "tau": float(tau), "c": float(c), "r2": float(r2), "c_pinned_at_bound": bool(pinned)}


def fit_trajectory(name, path, key):
    steps, values = parse(path, key)
    keep = values > 0.0
    steps, values = steps[keep], values[keep]
    return {
        "name": name, "log": path, "metric": key, "n_points": int(len(steps)),
        "step_first": int(steps[0]), "step_last": int(steps[-1]),
        "value_first": float(values[0]), "value_last": float(values[-1]),
        "linear_slope_per_step": float(np.polyfit(steps, values, 1)[0]),
        "fit_c_floor_neg005": fit_one(steps, values, -0.05),
        "fit_c_relaxed_neg1": fit_one(steps, values, -1.0),
    }


def main():
    trajectories = [
        ("Kaggle BTC escalation (input-side)", "reports/kaggle_escalation_s0.log", "film_g"),
        ("FI-2010 escalation (input-side)", "reports/fi2010_escalation_s0.log", "film_g"),
        ("FI-2010 post-scan escalation film_g", "reports/fi2010_postscan_escalation_s0.log", "film_g"),
        ("Kaggle BTC post-scan escalation film_g", "reports/kaggle_postscan_escalation_s0.log", "film_g"),
        ("FI-2010 post-scan escalation film_b", "reports/fi2010_postscan_escalation_s0.log", "film_b"),
        ("Kaggle BTC post-scan escalation film_b", "reports/kaggle_postscan_escalation_s0.log", "film_b"),
    ]
    records = [fit_trajectory(name, path, key) for name, path, key in trajectories]
    for record in records:
        floor = record["fit_c_floor_neg005"]
        relaxed = record["fit_c_relaxed_neg1"]
        print(f"=== {record['name']} ({record['n_points']} pts, {record['step_first']}->{record['step_last']}) ===")
        print(f"  endpoints: {record['value_first']:.4f} -> {record['value_last']:.4f}   linear slope = {record['linear_slope_per_step']:.2e}/step")
        print(f"  c-floor=-0.05 : a={floor['a']:.4f} tau={floor['tau']:.0f} c={floor['c']:.4f} R2={floor['r2']:.4f} pinned={floor['c_pinned_at_bound']}")
        print(f"  c-relaxed     : a={relaxed['a']:.4f} tau={relaxed['tau']:.0f} c={relaxed['c']:.4f} R2={relaxed['r2']:.4f} pinned={relaxed['c_pinned_at_bound']}")
    output = "reports/decay_fit_results.json"
    with open(output, "w") as handle:
        json.dump(records, handle, indent=2)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
