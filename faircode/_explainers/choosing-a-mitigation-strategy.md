> *Five strategies, seven audits. The table of what each one costs and what it needs is already in the benchmark's results.*

## The One-Sentence Definition

**Choosing a mitigation strategy** is picking among the benchmark's five approaches (use everything, drop the protected column, drop it and its proxies, constrain training, adjust predictions) by what each needs, what it costs in accuracy, and how much of the gap it leaves.

## Why It Matters

[Mitigation Strategies](mitigation-strategies.md) describes the five rungs of the ladder. A practitioner needs a different answer: which one should I use? That depends on constraints you have before you look at any numbers. If the protected attribute is not available at prediction time, post-processing is out. If you cannot retrain, in-processing is out. If the legal environment forbids using the attribute in a decision, so are group thresholds.

The benchmark can say something about the rest. Over the seven audits it records each strategy's accuracy and parity gap for the same model, so the cost and the typical effect of each can be read off.

## Core Concept: Constraints First, Then the Numbers

1. **What does each strategy need?** Group labels while training, or at prediction time.
2. **What does it cost?** The change in accuracy against the simplest honest baseline.
3. **What does it leave?** The size of the remaining gap and whether it is still significant.

The baseline for comparison is S2, the same model without the protected column or its proxies, because that is what most careful teams start with.

## Concrete Example: The Benchmark's Seven Audits

The logistic regression rows of `results/` (the current results) for every audit and protected attribute, 13 pairs (intersections left out), with S2 as the reference.

```
logistic regression, 13 audit/attribute pairs, compared with S2 (the same model without the protected column or proxies)
strategy              median accuracy change  median |parity gap|  gap smaller than S2  still significant   needs group labels: training / prediction
S0 baseline                           +0.000                0.054                 4/13              10/13   no (it uses the column as a feature) / yes, as a feature
S1 unawareness                        +0.000                0.041                 4/13               6/13   no / no
S2 proxy removal                      +0.000                0.042                    -               7/13   no / no
S3 in-processing                      -0.027                0.009                 8/13               4/13   yes / no
S4 post-processing                    +0.000                0.007                 9/13               3/13   yes / yes
```

- **S3 in-processing and S4 post-processing close the gap; the rest barely do.** The median absolute parity gap is 0.009 for S3 and 0.007 for S4 against 0.042 for S2 and 0.041 for S1. The gap is smaller than S2's in 8 of 13 pairs for S3 and 9 of 13 for S4.
- **S3 costs accuracy; S4 does not, in the median.** S3's median accuracy change is -0.027 and S4's is +0.000. The medians hide the spread, so check the audit you care about.
- **Fewer pairs stay significant:** 4 of 13 for S3 and 3 of 13 for S4, against 7 of 13 for S2 and 10 of 13 for S0 (the model that uses the protected attribute directly).
- **The needs differ.** S1 and S2 need no group labels anywhere. S3 needs them while training. S4 needs them while training and again at prediction, which is the constraint that most often rules it out.

A short decision rule follows: if you cannot use the attribute at prediction time, choose S3; if you can and may, S4 gives the smallest gaps for the least accuracy; if neither is possible, S2 with a proxy audit is the floor, and its remaining gaps need explaining.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/choosing-a-mitigation-strategy.py`; that file is generated from the block below). It reads the benchmark's current results.

```python
import numpy as np
import pandas as pd

fair = pd.read_csv("results/results_fairness.csv")          # the benchmark's current results
perf = pd.read_csv("results/results_performance.csv")
STRATEGIES = ["baseline", "unawareness", "unawareness_proxy_removal", "in_processing", "post_processing"]
LABEL = dict(zip(STRATEGIES, ["S0 baseline", "S1 unawareness", "S2 proxy removal", "S3 in-processing", "S4 post-processing"]))
NEEDS = {  # (group labels while training, group labels when predicting)
    "baseline": ("no (it uses the column as a feature)", "yes, as a feature"), "unawareness": ("no", "no"),
    "unawareness_proxy_removal": ("no", "no"), "in_processing": ("yes", "no"), "post_processing": ("yes", "yes")}

gap = fair[(fair.model == "logistic_regression") & (fair.metric == "demographic_parity_diff") & ~fair.protected_attribute.str.contains("_x_")]
acc = perf[(perf.model == "logistic_regression") & (perf.metric == "accuracy")].set_index(["audit", "strategy"]).value
pairs = gap.set_index(["audit", "protected_attribute", "strategy"])
rows = []
for strategy in STRATEGIES:
    changes, remaining, closed, significant, acc_change = [], [], 0, 0, []
    for (audit, attribute), g in gap.groupby(["audit", "protected_attribute"]):
        base = pairs.loc[(audit, attribute, "unawareness_proxy_removal")]
        row = pairs.loc[(audit, attribute, strategy)]
        remaining.append(abs(row.value)); changes.append(abs(row.value) - abs(base.value))
        significant += bool(row.significant); closed += abs(row.value) < abs(base.value)
        acc_change.append(acc[(audit, strategy)] - acc[(audit, "unawareness_proxy_removal")])
    rows.append((strategy, np.median(acc_change), np.median(remaining), closed, significant, len(changes)))

print(f"logistic regression, {len(rows) and rows[0][5]} audit/attribute pairs, compared with S2 (the same model without the protected column or proxies)")
print(f"{'strategy':20s} {'median accuracy change':>23} {'median |parity gap|':>20} {'gap smaller than S2':>20} {'still significant':>18}   needs group labels: training / prediction")
for strategy, d_acc, med_gap, closed, significant, n in rows:
    smaller = '-' if strategy == 'unawareness_proxy_removal' else f'{closed}/{n}'
    print(f"{LABEL[strategy]:20s} {d_acc:+23.3f} {med_gap:20.3f} {smaller:>20} {f'{significant}/{n}':>18}   {NEEDS[strategy][0]} / {NEEDS[strategy][1]}")
```

## Limitations

1. **Medians over 13 pairs hide a lot.** The audits differ in size, base rate and how much accuracy there is to lose. Look at the audit closest to yours.
2. **Only logistic regression is shown.** The benchmark also runs random forests and gradient boosting, which behave differently.
3. **All five use demographic parity as the target.** The benchmark constrains S3 and S4 on one fixed criterion, so a different criterion would give a different table, see [Choosing a Fairness Metric](choosing-a-fairness-metric.md).
4. **The gaps come from single splits.** See [Seed and Split Sensitivity](seed-and-split-sensitivity.md) for how much that matters on small audits.
5. **A smaller gap is not a fair outcome.** The strategies change the model's behaviour; whether the remaining or removed gap is justified is a separate question.

## Related Concepts

* [Mitigation Strategies](mitigation-strategies.md) - what the five strategies are
* [Fairness-Accuracy Pareto Frontier](fairness-accuracy-pareto-frontier.md) - how much accuracy the best rule at a given gap keeps
* [Group-Specific Thresholds](group-specific-thresholds.md) - what S4 does in its simplest form
* [Choosing a Fairness Metric](choosing-a-fairness-metric.md) - the choice that comes before this one
* [Fairness Through Unawareness](fairness-through-unawareness.md) - why S1 and S2 are not enough

## Related Projects in This Repo

* [`results/`](../results/) - the numbers in the table
* [Benchmark Results Dashboard](../benchmark.html) - the same table, filterable

## Further Reading

* [Bird, S. et al. (2020): Fairlearn: A toolkit for assessing and improving fairness in AI](https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/) - Microsoft technical report MSR-TR-2020-32; the toolkit behind the benchmark's S3 and S4.
* [Agarwal, A. et al. (2018): A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453) - ICML 2018; the in-processing method used for S3.
* [Friedler, S. A. et al. (2019): A Comparative Study of Fairness-Enhancing Interventions in Machine Learning](https://arxiv.org/abs/1802.04422) - FAT* 2019; how interventions compare across datasets and metrics.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
