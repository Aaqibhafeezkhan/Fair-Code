> *A non-significant gap is not a small gap. It might be a gap the audit could never have seen.*

## The One-Sentence Definition

The **minimum detectable gap** is the smallest true difference in selection rates between two groups that a test of a given size would detect with a stated probability (the power, usually 80%), at a stated significance level (usually 0.05).

## Why It Matters

Every audit in this repo reports a p-value for each gap, and a p-value above 0.05 is easily read as "no gap". It means the audit did not find one. With groups of 69 and 131, as in German Credit Lending, the audit cannot find a gap of less than about 21 percentage points, so a result of "not significant" says almost nothing about a gap of 9 points.

Power analysis answers the question before you look: how large must a gap be for this audit to detect it most of the time? It should be reported with every non-significant result, and used to decide how many rows to label in the first place ([Stratified Test Sets for Fairness](stratified-test-sets-for-fairness.md)).

## Core Concept: The Gap That Stands Out From the Noise

For two groups of sizes `n_a` and `n_b` with selection rate near `p`, the standard error of the difference is `sqrt(p (1 - p) (1/n_a + 1/n_b))`. To detect a gap with a two-sided test at level `alpha` and power `1 - beta`, the gap has to be about

```
MDG = (z(1 - alpha/2) + z(1 - beta)) * sqrt(p (1 - p) (1/n_a + 1/n_b))
```

With `alpha = 0.05` and 80% power the multiplier is 2.80. Using `p = 0.5` gives the largest standard error and so the most conservative answer. The formula is a normal approximation, so it is worth checking with the actual test.

## Concrete Example: What the Benchmark's Audits Can Detect

First the formula, then a check with this repo's permutation test, then the benchmark's own groups.

```
minimum detectable gap in selection rate (80% power, alpha 0.05, base rate 0.5), two equal groups
     30 per group: 0.362
    100 per group: 0.198
    200 per group: 0.140
    500 per group: 0.089
   1000 per group: 0.063
   5000 per group: 0.028

planted a gap of 0.140 between two groups of 200: the permutation test flagged it in 77% of 300 runs (target 80%)

audit / attribute                     n (disadv, adv)  detectable gap  observed gap  p-value
german_credit_lending / age                   69, 131           0.208        -0.091    0.171
insurance_denial / age                       106, 161           0.175        -0.073    0.262
insurance_denial / gender                    134, 133           0.171        -0.056    0.400
benefits_denial / national_origin           688, 5825           0.056        +0.005    0.673
benefits_denial / race                      690, 5823           0.056        -0.042    0.000
benefits_denial / age                       944, 5569           0.049        +0.028    0.006
compas / race                              1804, 1444           0.049        +0.149    0.000
benefits_denial / sex                      2153, 4360           0.037        -0.051    0.000
tenant_screening / race                    2943, 2224           0.039        +0.114    0.000
healthcare_readmission / race             4541, 15358           0.024        -0.001    0.076
ai_fair_recruitment / gender              8410, 14652           0.019        -0.007    0.136
healthcare_readmission / age              9010, 10889           0.020        -0.001    0.002
healthcare_readmission / gender           10661, 9238           0.020        +0.001    0.001
```

- **The formula holds up.** Planting the predicted gap (0.140) between two groups of 200, the permutation test from `faircode.significance` flagged it in 77% of 300 runs against the 80% target; a normal approximation is a little optimistic.
- **Small audits cannot see moderate gaps.** At 100 per group the detectable gap is 0.198; at 30 it is 0.362.
- **Against the benchmark's own rows** (S2 logistic regression, demographic parity gap, from `results/`, the current results): German Credit age has groups of 69 and 131, can only detect a gap of 0.208, and observed -0.091 with p = 0.171. Insurance Denial age and gender (0.175 and 0.171 detectable) observed -0.073 and -0.056, also not significant. Those results are consistent with real gaps of that size. At the other end, Benefits Denial by sex can detect 0.037 and observed -0.051, p < 0.001.

Healthcare Readmission and AI Fair Recruitment can detect gaps of 0.02 or less, so their near-zero gaps are informative: the data would have shown a gap of two points.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/minimum-detectable-gap.py`; that file is generated from the block below).

```python
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
```

## Limitations

1. **It is a normal approximation.** The simulation fell about three points short of the target at 200 per group. For very small groups or rates near 0 or 1, simulate with the real test.
2. **The base rate is an assumption.** `p = 0.5` is the worst case. A rate near 0.1 makes smaller gaps detectable, a fact the table above does not use.
3. **It is for one comparison.** Many metrics or many groups need a stricter `alpha`, see [Multiple Comparisons](multiple-comparisons.md), and the detectable gap grows with it.
4. **It does not make a small gap unimportant.** A gap below the detectable size can still be real and harmful, which is the point of knowing the limit.
5. **It describes the test, not the model.** A model's gap also varies with the split, see [Seed and Split Sensitivity](seed-and-split-sensitivity.md).

## Related Concepts

* [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md) - the test whose power this measures
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - an interval is another way to see what the data cannot rule out
* [Stratified Test Sets for Fairness](stratified-test-sets-for-fairness.md) - collecting enough rows for the groups that matter
* [Multiple Comparisons](multiple-comparisons.md) - why the detectable gap grows with the number of tests
* [Disaggregated Evaluation](disaggregated-evaluation.md) - small cells and what they can show

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the smallest groups
* [`results/`](../results/) - the benchmark rows quoted above

## Further Reading

* Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum - the standard reference on power.
* [Ji, D., Smyth, P., Steyvers, M. (2020): Can I Trust My Fairness Metric? Assessing Fairness with Unlabeled Data and Bayesian Inference](https://arxiv.org/abs/2010.09851) - NeurIPS 2020; uncertainty in fairness metrics with limited labels.
* Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley - confidence intervals for differences, ratios and odds ratios of proportions.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
