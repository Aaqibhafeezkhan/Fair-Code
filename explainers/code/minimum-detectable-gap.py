# Generated from explainers/minimum-detectable-gap.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd
from scipy.stats import norm
from faircode.significance import permutation_test

def minimum_detectable_gap(n_a, n_b, rate=0.5, alpha=0.05, power=0.80):
    """Smallest true difference in selection rates that a two-sided test at `alpha` detects with probability `power`
    (normal approximation; `rate` = 0.5 is the worst case, where the variance is largest)."""
    se = np.sqrt(rate * (1 - rate) * (1 / n_a + 1 / n_b))
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se

print("minimum detectable gap in selection rate (80% power, alpha 0.05, base rate 0.5), two equal groups")
for n in (30, 100, 200, 500, 1000, 5000):
    print(f"  {n:5d} per group: {minimum_detectable_gap(n, n):.3f}")

# Check the formula with the repo's own permutation test: plant a gap of the predicted size and count detections.
rng = np.random.default_rng(0)
n, gap, reps = 200, minimum_detectable_gap(200, 200), 300
hits = 0
for r in range(reps):
    a = (rng.random(n) < 0.5 + gap / 2).astype(float)
    b = (rng.random(n) < 0.5 - gap / 2).astype(float)
    hits += permutation_test(a, b, n_permutations=199, random_state=r) < 0.05
print(f"\nplanted a gap of {gap:.3f} between two groups of {n}: the permutation test flagged it in {hits / reps:.0%} of {reps} runs (target 80%)")

# What the benchmark's own audits can and cannot detect.
results = pd.read_csv("results/results_fairness.csv")
rows = results[(results.strategy == "unawareness_proxy_removal") & (results.model == "logistic_regression")
               & (results.metric == "demographic_parity_diff") & ~results.protected_attribute.str.contains("_x_")]
print(f"\n{'audit / attribute':36s} {'n (disadv, adv)':>16} {'detectable gap':>15} {'observed gap':>13} {'p-value':>8}")
for r in rows.sort_values("n_disadvantaged").itertuples():
    mdg = minimum_detectable_gap(r.n_disadvantaged, r.n_advantaged)
    print(f"{r.audit + ' / ' + r.protected_attribute:36s} {f'{r.n_disadvantaged}, {r.n_advantaged}':>16} {mdg:15.3f} {r.value:+13.3f} {r.p_value:8.3f}")
