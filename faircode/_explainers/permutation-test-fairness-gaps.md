> *A fairness gap is a number. Whether it is a finding or a coincidence depends on how many people it was measured on.*

## The One-Sentence Definition

A **permutation test for a fairness gap** asks how often a gap this large would appear if group membership carried no information at all, by shuffling the group labels many times and counting.

## Why It Matters

An audit reports "the positive rate for women is 5.9% and for men 10.9%". Is that a real disparity or sampling noise? On a 6,500-row test set it is overwhelming. On the 200-row test set of a small audit, the same gap often is not, and reading either one without a test leads to the wrong call: ignoring a real gap, or acting on an accident.

Every row in `results/results_fairness.csv` carries a `p_value`, a bootstrap interval (`ci_low`, `ci_high`) and a `significant` flag for exactly this reason. This explainer shows how that p-value is produced.

## Core Concept: Shuffle the Labels

The null hypothesis is that group label is irrelevant. If so, every assignment of "group A" and "group B" to the same rows was equally likely. So:

1. Compute the observed gap in positive rate.
2. Pool the predictions from both groups.
3. Shuffle the group labels, split, and recompute the gap. Repeat thousands of times.
4. The p-value is the share of shuffled gaps at least as large (in absolute value) as the observed one.

It is distribution-free and valid on the small, unbalanced groups audits keep running into. It complements the [bootstrap interval](bootstrap-confidence-intervals.md), which says how large the gap might be, while the permutation p-value says how surprising it is under no effect.

## Concrete Example: Benefits Denial

The [Benefits Denial](../Benefits%20Denial/) audit, logistic regression on core features, benchmark split. The model flags 5.9% of women and 10.9% of men as ineligible, a gap of -0.0509. First the full test set, with a hand-written test checked against `faircode.significance`. Then the same model and the same gap, audited on random subsamples of 200 and 1,000 rows.

```
test rows 6513: 2153 women, 4360 men | positive rate 0.059 vs 0.109 | gap -0.0509
hand-written p: 0.0 | faircode.significance p: 0.0
  200 rows: p < 0.05 in 17% of 200 random samples (median p 0.273)
 1000 rows: p < 0.05 in 72% of 200 random samples (median p 0.013)
```

On the full test set the p-value is below 1/2000 (the resolution of 2,000 shuffles), and the interval is -0.064 to -0.037. On 1,000 rows, a 5-point gap clears p < 0.05 in only 72% of samples. On 200 rows it clears it in only 17%. The gap did not change; the evidence did. The benchmark's German Credit S2 row has a gap of -0.0914 with a p-value of 0.17 in `results/` (the current results): a bigger gap than here, measured on about 200 test rows, and not significant.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 permutation.py`).

```python
import numpy as np
from sklearn.model_selection import train_test_split
from faircode.manifest import load_manifest
from faircode.benchmark import _load_dataset      # the loader the benchmark harness uses
from faircode.strategies import encode_features
from faircode.models import build_model

def load_audit(folder):
    manifest = load_manifest(f"{folder}/audit.yaml")
    df, y, masks = _load_dataset(manifest)        # masks[name] is 1 for the disadvantaged group
    columns = list(dict.fromkeys(manifest.core_features))   # S2 feature set: no protected columns, no proxies
    X = encode_features(df, columns)
    train, test = train_test_split(np.arange(len(df)), test_size=manifest.test_size,
                                   random_state=manifest.random_state, stratify=y)
    return manifest, X, y, {k: v.to_numpy() for k, v in masks.items()}, train, test
from faircode.significance import permutation_test

manifest, X, y, masks, train, test = load_audit("Benefits Denial")
sex = masks["sex"][test]                            # 1 = female
model = build_model("logistic_regression", manifest.random_state).fit(X.iloc[train], y.iloc[train])
pred = model.predict(X.iloc[test]).astype(float)
women, men = pred[sex == 1], pred[sex == 0]
print(f"test rows {len(pred)}: {len(women)} women, {len(men)} men | positive rate {women.mean():.3f} vs {men.mean():.3f} | gap {women.mean() - men.mean():+.4f}")

def permutation_p(a, b, n_permutations=2000, seed=42):
    """Two-sided p: how often does a random relabelling give a gap at least this big?"""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)                          # the group label carries no information (the null)
        hits += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed
    return hits / n_permutations

print("hand-written p:", permutation_p(women, men), "| faircode.significance p:", permutation_test(women, men, n_permutations=2000))

# The same model and the same 5-point gap, audited on smaller and smaller samples.
rng = np.random.default_rng(0)
for n in (200, 1000):
    p_values = []
    for i in range(200):
        pick = rng.choice(len(pred), n, replace=False)
        a, b = pred[pick][sex[pick] == 1], pred[pick][sex[pick] == 0]
        if min(len(a), len(b)) >= 5:
            p_values.append(permutation_test(a, b, n_permutations=300, random_state=i))
    p_values = np.array(p_values)
    print(f"{n:5d} rows: p < 0.05 in {np.mean(p_values < 0.05):.0%} of {len(p_values)} random samples (median p {np.median(p_values):.3f})")
```

`faircode.significance.significance_report` returns the gap, the bootstrap interval, the p-value and a `small_sample_warning` for a group under 30 rows in one call.

## Limitations

1. **A p-value is not an effect size.** With enough rows, a 0.2 point gap is "significant". Read it next to the gap and its interval.
2. **A p of 0 means smaller than 1/n_permutations.** Report "p < 0.0005", not zero.
3. **It tests one test split, not the model.** Retraining with a different seed moves the gap. The test does not capture that variance.
4. **Many gaps means many chances.** The benchmark reports dozens of metrics per audit. See [Multiple Comparisons](multiple-comparisons.md).
5. **Exchangeability is an assumption.** If rows are duplicated, clustered by household or correlated in time, shuffling labels over-states the evidence. Use a grouped permutation.

## Related Concepts

* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the size of the gap, the companion to this test
* [Multiple Comparisons](multiple-comparisons.md) - what to do when you run many of these
* [Demographic Parity](demographic-parity.md) - the gap being tested here
* [Sampling Bias](sampling-bias.md) - why the sample may not represent the population
* [Subgroup Fairness](subgroup-fairness.md) - small intersections make the power problem worse

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above
* [`German Credit Lending/`](../German%20Credit%20Lending/) - a small audit where the test matters most
* [`results/`](../results/) - every benchmark row has a `p_value`

## Further Reading

* Fisher, R. A. (1935). *The Design of Experiments.* Oliver and Boyd - the origin of the permutation test.
* Ernst, M. D. (2004). Permutation Methods: A Basis for Exact Inference. Statistical Science 19(4); a clear modern account of why and when it is exact.
* Efron, B., Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall; the companion interval method.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
