> *Cross-validation asks whether the model works on new rows. Leave-one-group-out asks whether it works on new kinds of people.*

## The One-Sentence Definition

**Leave-one-group-out evaluation** trains the model with one group removed from the training data and tests it on that group, to measure how well it generalizes to people the training data never contained.

## Why It Matters

A random test split is drawn from the same population as the training data, so every group is represented in both. Real deployments are not so tidy: a model built in one region gets used in another, a model trained before a policy change meets applicants after it, a group that was rare in training turns out to be common in use.

Holding a group out simulates that. It is a close relative of the [distribution shifts](distribution-shift.md) that wreck models in production, and of the worst-group evaluations behind benchmarks like WILDS. It is also an honest check on a quiet assumption: that the groups you did see are enough to cover the groups you did not.

## Core Concept: Hold Out the Group, Not Random Rows

```
for each group g:
    train on everyone except g
    test on g, and compare with the model trained on everyone
```

The comparison that matters is the drop in accuracy on `g`: the cost of never having seen it. A large drop means the model leans on the group's presence. A drop near zero means either that other groups cover it, or that the model barely depends on whatever distinguishes it. The groups should be real axes of variation (a region, a site, a time period), not an arbitrary label.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): a random forest on every column of the audit (race and native country included, so the model can use the group), the benchmark's split, with each race and each of the four most common native countries held out in turn.

```
race (random_forest)
  held-out group       test rows  with it  without it   change
  Amer-Indian-Eskimo          67    0.910       0.866   -0.045
  Asian-Pac-Islander         196    0.847       0.837   -0.010
  Black                      579    0.924       0.927   +0.003
  Other                       44    0.955       0.955   +0.000
  White                     5627    0.843       0.837   -0.006

native.country (random_forest)
  held-out group       test rows  with it  without it   change
  ?                          136    0.875       0.868   -0.007
  Mexico                     144    0.944       0.938   -0.007
  Philippines                 30    0.800       0.767   -0.033
  United-States             5825    0.848       0.828   -0.020
```

For most groups, removing them from training changes accuracy on them by a point or less. The largest drops are 4.5 points for the 67 American Indian or Eskimo test rows, 3.3 points for the 30 Philippines rows, and 2.0 for United States, the largest group. The two largest drops are in groups of 67 and 30 rows, where one or two rows move accuracy by more than a point, so treat them as a prompt to check, not a result.

The overall reading: on this data a forest generalizes to a held-out group about as well as it does to a random sample. That is a finding worth stating, because it supports the assumption that groups seen cover groups unseen. On an audit whose groups differ more, it would not.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/leave-one-group-out-evaluation.py`; that file is generated from the block below).

```python
import numpy as np
import pandas as pd
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial", strategy="baseline")     # baseline: every feature, race and native.country included
train, test = data.idx_train, data.idx_test
X, y = data.X, data.y.to_numpy()

def leave_one_group_out(column, labels, model_name="random_forest", min_rows=30):
    """For each group: accuracy on its test rows from a model trained with everyone, and from a model
    trained on every group EXCEPT that one."""
    codes = data.X_all[column].to_numpy()
    everyone = build_model(model_name, data.random_state).fit(X.iloc[train], y[train])
    print(f"{column} ({model_name})")
    print(f"  {'held-out group':20s} {'test rows':>9} {'with it':>8} {'without it':>11} {'change':>8}")
    for code, label in labels.items():
        held = test[codes[test] == code]
        if len(held) < min_rows:
            continue
        kept = train[codes[train] != code]
        without = build_model(model_name, data.random_state).fit(X.iloc[kept], y[kept])
        a, b = everyone.score(X.iloc[held], y[held]), without.score(X.iloc[held], y[held])
        print(f"  {label:20s} {len(held):9d} {a:8.3f} {b:11.3f} {b - a:+8.3f}")

raw = pd.read_csv("Benefits Denial/adult.csv", usecols=["race", "native.country"])
race_names = dict(enumerate(sorted(raw["race"].unique())))          # the harness codes text columns alphabetically
leave_one_group_out("race", race_names)
print()
country_names = dict(enumerate(sorted(raw["native.country"].unique())))
common = raw["native.country"].value_counts().index[:4]
leave_one_group_out("native.country", {c: n for c, n in country_names.items() if n in set(common)})
```

Change the column and the label dictionary to hold out something else, such as a region or an age band.

## Limitations

1. **Small held-out groups are noisy.** A drop measured on 30 rows can be chance. Report the group size and an interval ([Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md)).
2. **It measures dependence on the group, not unfairness.** A model can generalize perfectly and still be unfair to the group. It is a robustness check next to [Disaggregated Evaluation](disaggregated-evaluation.md), not a replacement.
3. **A simple model may not care.** When we held out whole age bands, weekly-hours bands and education levels from a logistic regression on the same data, accuracy on the held-out group changed by at most 1.4 points, because other groups covered the same feature values. The check is most informative for flexible models.
4. **The groups must be meaningful.** Holding out a random label tests nothing.
5. **It trains the model many times.** One fit per group, which is costly for large models.

## Related Concepts

* [Distribution Shift](distribution-shift.md) - the real-world version of an unseen group
* [Subgroup Fairness](subgroup-fairness.md) - why group-level behaviour needs group-level checks
* [Disaggregated Evaluation](disaggregated-evaluation.md) - the companion check on groups the model did see
* [Sampling Bias](sampling-bias.md) - when a group is missing from training by accident
* [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) - an objective that targets the worst group

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Koh, P. W. et al. (2021): WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421) - ICML 2021; benchmarks built around held-out groups and worst-group accuracy.
* [Sagawa, S., Koh, P. W., Hashimoto, T., Liang, P. (2020): Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731) - ICLR 2020; the worst-group objective that this evaluation motivates.
* Quinonero-Candela, J. et al. (eds.) (2009). Dataset Shift in Machine Learning. MIT Press - the standard reference on how and why test data differs from training data.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
