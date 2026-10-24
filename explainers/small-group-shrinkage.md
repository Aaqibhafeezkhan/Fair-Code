> *With 30 rows in a group, the raw rate is mostly noise. Shrinkage trades a little bias for a lot less of it, and it can hide the one gap you were looking for.*

## The One-Sentence Definition

**Small-group shrinkage** is estimating a group's rate (for example its accuracy) as a weighted average of the group's own raw rate and the pooled rate across all groups, with a weight that depends on how many rows the group has, so that small groups are pulled strongly toward the pool and large groups barely at all.

## Why It Matters

A disaggregated audit reports a number for every group and intersection, and the smallest groups produce the most extreme numbers by chance. A group of 30 can show 70% or 100% accuracy with no real difference from the rest. Reading those raw numbers as findings creates false alarms, and discarding small groups hides the people an audit exists to protect.

Empirical-Bayes shrinkage is the standard way to estimate many small rates at once. It improves the estimate of each group on average, and it is the same idea as the Stein estimator. The catch for fairness is that "pulled toward the pool" means "pulled away from the group's own distinct level", so a real gap in a small group is partly hidden by the method meant to stabilize it.

## Core Concept: A Weighted Average With a Data-Driven Weight

```
shrunk_rate(group) = (successes + k * pooled_rate) / (rows + k)
```

`k` acts like a number of imaginary rows at the pooled rate. It is estimated from how much the raw group rates vary beyond what binomial noise explains: if they vary a lot, `k` is small and groups keep their own rates; if they vary little, `k` is large and everything is pulled to the pool. Groups with more rows than `k` are barely moved.

## Concrete Example: Healthcare Readmission

[Healthcare Readmission](../Healthcare%20Readmission/): a logistic regression's accuracy on 20,000 test rows is the "population", split into age band by race cells. Cells with fewer than 30 population rows are left out, leaving 29 whose true accuracy is known. Then 300 small audits of 600 randomly chosen rows each estimate every cell's accuracy two ways and are scored against the truth.

```
29 cells of at least 30 rows; sizes from 30 to 4124; true accuracy from 0.819 to 1.000

error (RMSE against the true cell accuracy) from a 600-row audit, averaged over 300 audits
  all 29 cells:        raw 0.144 | shrunk 0.043
  the smallest 8 cells:   raw 0.152 | shrunk 0.062

the cell with the lowest true accuracy, [70-80)|Hispanic (72 rows): true 0.819; average estimate raw 0.796, shrunk 0.884 (pooled accuracy 0.887)
```

- **Shrinkage cuts the error by two thirds.** The root-mean-square error against the true cell accuracy falls from 0.144 (raw) to 0.043 (shrunk), and for the eight smallest cells from 0.152 to 0.062.
- **It also hides a gap.** The cell with the lowest true accuracy (age 70 to 80, Hispanic patients, 72 rows) has a true accuracy of 0.819, 6.8 points below the pooled 0.887. Raw estimates average 0.796 (noisy, but centred on the right level) while shrunk estimates average 0.884, almost exactly the pool. In an audit of this size the raw estimate flags the cell and the shrunk one does not.

Use shrinkage to rank or screen many small groups with less noise, and keep the raw numbers and their intervals next to it for any cell that looks different.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/small-group-shrinkage.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Healthcare Readmission", strategy="unawareness")
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
model = LogisticRegression(max_iter=2000).fit(X[train], y[train])
correct = (model.predict(X[test]) == y[test]).astype(float)

# 50 groups: age band x race. The 20,000 test rows are the "population"; each cell's true accuracy is known.
age = data.df["age"].to_numpy()[test]
race = data.df["race"].to_numpy()[test]
cell = np.array([f"{a}|{r}" for a, r in zip(age, race)])
cells, inverse, sizes = np.unique(cell, return_inverse=True, return_counts=True)
keep = sizes >= 30                                # cells with at least 30 rows in the population have a trustworthy "true" accuracy
cells, sizes = cells[keep], sizes[keep]
remap = -np.ones(len(keep), dtype=int); remap[keep] = np.arange(keep.sum())
inverse = remap[inverse]
in_cells = inverse >= 0
correct, inverse = correct[in_cells], inverse[in_cells]
truth = np.array([correct[inverse == i].mean() for i in range(len(cells))])
print(f"{len(cells)} cells of at least 30 rows; sizes from {sizes.min()} to {sizes.max()}; true accuracy from {truth.min():.3f} to {truth.max():.3f}")

def shrink(successes, counts):
    """Empirical-Bayes estimate: pull each cell's rate toward the pooled rate by a weight n / (n + k), where k is the
    prior strength estimated from how much the cells' raw rates vary beyond what binomial noise explains."""
    pooled = successes.sum() / counts.sum()
    raw = successes / np.maximum(counts, 1)
    noise = np.average(pooled * (1 - pooled) / np.maximum(counts, 1), weights=counts)
    between = max(np.average((raw - pooled) ** 2, weights=counts) - noise, 1e-6)
    k = pooled * (1 - pooled) / between
    return (successes + k * pooled) / (counts + k), k

rng = np.random.default_rng(0)
raw_err, shrunk_err, small_raw, small_shrunk, estimates = [], [], [], [], []
small = sizes <= np.quantile(sizes, 0.25)
worst = int(np.argmin(truth))
for _ in range(300):
    rows = rng.choice(len(correct), 600, replace=False)          # a small audit: 600 labelled rows
    counts = np.bincount(inverse[rows], minlength=len(cells)).astype(float)
    successes = np.bincount(inverse[rows], weights=correct[rows], minlength=len(cells))
    raw = np.where(counts > 0, successes / np.maximum(counts, 1), successes.sum() / counts.sum())
    shrunk, k = shrink(successes, counts)
    raw_err.append(np.sqrt(np.mean((raw - truth) ** 2))); shrunk_err.append(np.sqrt(np.mean((shrunk - truth) ** 2)))
    small_raw.append(np.sqrt(np.mean((raw[small] - truth[small]) ** 2))); small_shrunk.append(np.sqrt(np.mean((shrunk[small] - truth[small]) ** 2)))
    estimates.append((raw[worst], shrunk[worst]))

print(f"\nerror (RMSE against the true cell accuracy) from a 600-row audit, averaged over 300 audits")
print(f"  all {len(cells)} cells:        raw {np.mean(raw_err):.3f} | shrunk {np.mean(shrunk_err):.3f}")
print(f"  the smallest {small.sum()} cells:   raw {np.mean(small_raw):.3f} | shrunk {np.mean(small_shrunk):.3f}")
mean_raw, mean_shrunk = np.mean(estimates, axis=0)
print(f"\nthe cell with the lowest true accuracy, {cells[worst]} ({sizes[worst]} rows): true {truth[worst]:.3f}; "
      f"average estimate raw {mean_raw:.3f}, shrunk {mean_shrunk:.3f} (pooled accuracy {correct.mean():.3f})")
```

## Limitations

1. **It biases every group toward the pool.** That is the point, and it is a problem when the question is whether one group differs from the pool.
2. **The weight is estimated from the data.** With few groups, or groups of very similar size, the estimate of `k` is itself noisy.
3. **It assumes groups are exchangeable.** If a group is different in kind (a different site, a different measurement), pooling it with the others is the wrong prior.
4. **The cells here were chosen to have 30 or more rows.** Cells below that were excluded so a true value exists to compare with. A real audit has to say what it does with the rest.
5. **It does not give an interval.** A shrunk rate needs an interval of its own, from a hierarchical model or a bootstrap of the whole procedure.

## Related Concepts

* [Disaggregated Evaluation](disaggregated-evaluation.md) - the per-group reporting this stabilizes
* [Subgroup Fairness](subgroup-fairness.md) - why small intersections matter and are hard to measure
* [Minimum Detectable Gap](minimum-detectable-gap.md) - what a small group can and cannot show
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the interval to report beside any rate
* [Intersectional Bias](intersectional-bias.md) - the small cells that compound

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit used for the example above

## Further Reading

* Efron, B., Morris, C. (1977). Stein's Paradox in Statistics. Scientific American 236(5) - why estimating many rates together beats estimating each alone.
* [Kearns, M., Neel, S., Roth, A., Wu, Z. S. (2018): Preventing Fairness Gerrymandering](https://arxiv.org/abs/1711.05144) - ICML 2018; auditing many small subgroups.
* [Gelman, A. et al. (2013): Bayesian Data Analysis (3rd ed.), chapter 5](https://sites.stat.columbia.edu/gelman/book/) - Chapman and Hall; hierarchical models for many groups.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
