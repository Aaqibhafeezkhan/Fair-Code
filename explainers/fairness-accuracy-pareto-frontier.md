> *A trade-off is not a single curve you slide along. It is the set of choices nothing else beats, and sometimes it is shorter than you expect.*

## The One-Sentence Definition

A **fairness-accuracy Pareto frontier** is the set of models (or decision rules) for which no other candidate has both higher accuracy and a smaller fairness gap, so that moving along it improves one at the cost of the other.

## Why It Matters

[Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) shows one mitigation moving a gap and an accuracy. A single before-and-after hides two questions. Was the "after" the best available at its gap, or would a different method have kept more accuracy? And how steep is the trade-off: how many points of accuracy does each point of gap cost?

The frontier answers both. Enumerate many candidate rules, keep the non-dominated ones, and plot or tabulate them. A method that lies well inside the frontier is leaving accuracy on the table; a frontier that is nearly flat says the trade-off is small. Martinez, Bertran and Sapiro (2020) treat this as multi-objective optimization.

## Core Concept: Non-Dominated Points

```
a point (accuracy a, gap g) is on the frontier if no other point has  a' >= a  and  g' <= g  with one strict
```

For a fixed score you can generate candidates by sweeping the decision rule: a threshold for each group on a grid. A frontier of a fixed score only covers what thresholds can do; a different training method (a different score) can lie off it.

## Concrete Example: Tenant Screening

[Tenant Screening](../Tenant%20Screening/): the S2 logistic regression scores, a 41 by 41 grid of threshold pairs (one per group, 0.20 to 0.80), evaluated on the benchmark's test rows, giving 1,681 candidate rules. Then the benchmark's five strategies from `results/` (the current results) are placed against it.

```
1681 threshold pairs, 8 on the accuracy-gap frontier
single threshold 0.5:  accuracy 0.639, gap 0.114

the frontier, smallest gap first
  |parity gap|  accuracy  threshold, white  threshold, black
        0.0004     0.617              0.32              0.38
        0.0004     0.630              0.35              0.41
        0.0020     0.636              0.38              0.44
        0.0023     0.639              0.43              0.49
        0.0032     0.640              0.49              0.55
        0.0047     0.645              0.46              0.52
        0.1185     0.645              0.47              0.59
        0.1468     0.646              0.46              0.59

the benchmark's strategies on the same audit (benchmark split, so the numbers can differ slightly from the sweep):
  baseline                     accuracy 0.690, |gap| 0.054; best sweep accuracy at that gap 0.645 (+0.045)
  unawareness                  accuracy 0.689, |gap| 0.119; best sweep accuracy at that gap 0.645 (+0.044)
  unawareness_proxy_removal    accuracy 0.639, |gap| 0.114; best sweep accuracy at that gap 0.645 (-0.006)
  in_processing                accuracy 0.626, |gap| 0.017; best sweep accuracy at that gap 0.645 (-0.019)
  post_processing              accuracy 0.639, |gap| 0.005; best sweep accuracy at that gap 0.645 (-0.005)
```

- **The frontier has 8 points and is nearly flat.** Accuracy runs from 0.617 at a gap of 0.0004 to 0.645 at 0.0047, and then stays at 0.645 to 0.646 all the way to a gap of 0.147. The most accurate rules and the fair ones are almost the same rules.
- **Equalizing is almost free here.** The single threshold of 0.5 has accuracy 0.639 and a gap of 0.114; a pair of thresholds (0.46 and 0.52) has accuracy 0.645 and a gap of 0.0047. The default threshold was not the accuracy-maximizing one, so a small shift improves both.
- **S2, S4 and S3 sit on or just inside the frontier.** S2 (0.639, gap 0.114) is 0.006 below the frontier's accuracy at its gap level, S4 post-processing (0.639, 0.005) 0.005 below, S3 in-processing (0.626, 0.017) 0.019 below.
- **S0 and S1 are off it, above it.** S0 and S1 reach 0.690 and 0.689 because they use extra columns (the protected attribute and proxies), which is a different score, not a better rule on this one. A frontier belongs to a score.

So for this audit, with this score, the frontier says the cost of fairness is under a point of accuracy. In other audits it is steeper, which is why the shape matters more than any single point.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/fairness-accuracy-pareto-frontier.py`; that file is generated from the block below).

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no race, no proxies
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
black = data.masks["race"].to_numpy().astype(int)
score = LogisticRegression(max_iter=1000).fit(X[train], y[train]).predict_proba(X)[:, 1]
s_te, y_te, b_te = score[test], y[test], black[test]

# Sweep one threshold per group and keep (accuracy, |parity gap|) for each pair: 41 x 41 candidate rules.
grid = np.linspace(0.2, 0.8, 41)
points = []
for t0 in grid:
    for t1 in grid:
        pred = np.where(b_te == 1, s_te >= t1, s_te >= t0)
        points.append((float((pred == y_te).mean()), abs(float(pred[b_te == 1].mean() - pred[b_te == 0].mean())), t0, t1))
points = np.array(points)

def non_dominated(p):
    """A point is on the frontier if no other point has accuracy at least as high and a gap at most as large (and one is strictly better)."""
    keep = []
    for i, (acc, gap, *_ ) in enumerate(p):
        dominated = np.any((p[:, 0] >= acc) & (p[:, 1] <= gap) & ((p[:, 0] > acc) | (p[:, 1] < gap)))
        keep.append(not dominated)
    return p[np.array(keep)]
frontier = non_dominated(points)
frontier = frontier[np.argsort(frontier[:, 1])]
print(f"{len(points)} threshold pairs, {len(frontier)} on the accuracy-gap frontier")
print(f"single threshold 0.5:  accuracy {points[np.argmin(np.abs(points[:, 2] - 0.5) + np.abs(points[:, 3] - 0.5)), 0]:.3f}, "
      f"gap {points[np.argmin(np.abs(points[:, 2] - 0.5) + np.abs(points[:, 3] - 0.5)), 1]:.3f}")
print("\nthe frontier, smallest gap first")
print(f"  {'|parity gap|':>12} {'accuracy':>9} {'threshold, white':>17} {'threshold, black':>17}")
for acc, gap, t0, t1 in frontier:
    print(f"  {gap:12.4f} {acc:9.3f} {t0:17.2f} {t1:17.2f}")

# Where do the benchmark's five strategies sit? Logistic regression on this audit, from results/ (the current results).
fair = pd.read_csv("results/results_fairness.csv"); perf = pd.read_csv("results/results_performance.csv")
f = fair[(fair.audit == "tenant_screening") & (fair.model == "logistic_regression") & (fair.protected_attribute == "race")
         & (fair.metric == "demographic_parity_diff")].set_index("strategy")
p = perf[(perf.audit == "tenant_screening") & (perf.model == "logistic_regression") & (perf.metric == "accuracy")].set_index("strategy")
print("\nthe benchmark's strategies on the same audit (benchmark split, so the numbers can differ slightly from the sweep):")
for strategy in f.index:
    acc, gap = p.loc[strategy, "value"], abs(f.loc[strategy, "value"])
    better = frontier[frontier[:, 1] <= gap]
    best = better[:, 0].max() if len(better) else float("nan")
    print(f"  {strategy:28s} accuracy {acc:.3f}, |gap| {gap:.3f}; best sweep accuracy at that gap {best:.3f} ({acc - best:+.3f})")
```

## Limitations

1. **The frontier belongs to the candidates you tried.** More thresholds, other scores or other methods can move it outward.
2. **It is measured on one test split.** The accuracies differ by a few tenths of a point between rules, which is inside the split noise, see [Seed and Split Sensitivity](seed-and-split-sensitivity.md). Choose on training or validation data and report on test data.
3. **One gap, one measure of accuracy.** Frontiers differ by metric, and equalized odds, calibration and parity do not share one.
4. **A frontier is a menu, not an answer.** Which point to take is a policy decision.
5. **Group thresholds use the group at prediction time,** with the legal issues in [Group-Specific Thresholds](group-specific-thresholds.md).

## Related Concepts

* [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) - the trade-off a frontier maps
* [Group-Specific Thresholds](group-specific-thresholds.md) - the rule family swept here
* [Randomized Post-Processing](randomized-post-processing.md) - reaching points between thresholds
* [Mitigation Strategies](mitigation-strategies.md) - the five strategies placed against the frontier
* [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) - an objective that picks a point differently

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows the strategies are read from

## Further Reading

* [Martinez, N., Bertran, M., Sapiro, G. (2020): Minimax Pareto Fairness: A Multi Objective Perspective](https://proceedings.mlr.press/v119/martinez20a.html) - ICML 2020; the fairness-accuracy trade-off as multi-objective optimization.
* [Wei, S., Niethammer, M. (2022): The Fairness-Accuracy Pareto Front](https://arxiv.org/abs/2008.10797) - Statistical Analysis and Data Mining 15(3); tracing the front for several methods.
* [Dutta, S. et al. (2020): Is There a Trade-Off Between Fairness and Accuracy? A Perspective Using Mismatched Hypothesis Testing](https://arxiv.org/abs/1910.07870) - ICML 2020; when the trade-off disappears.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
