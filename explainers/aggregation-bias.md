> *One model for everyone assumes the same features mean the same thing for everyone. When they do not, the pooled model is wrong for each group in a different way.*

## The One-Sentence Definition

**Aggregation bias** occurs when a single model is fitted to data from groups whose relationship between features and outcome differs, so the pooled model fits each group worse than a model fitted to that group alone.

## Why It Matters

A pooled model learns the average relationship. When the groups differ, the average is the right answer for nobody, and the group with less data is fitted the worst because it counts for less in the average. Medicine has long examples: the same lab value can mean different things by sex or ancestry, and a model fitted to the majority can be least accurate for the minority.

It is also the mirror of [Simpson's Paradox](simpsons-paradox.md) and of [Subgroup Fairness](subgroup-fairness.md): a trend in the aggregate that is not the trend inside each group.

## Core Concept: Pooled Versus Separate Models

The check is cheap. Fit one model on everyone and one model per group, and compare on each group's own test rows:

```
pooled model on group g's test rows      vs      model fitted on group g's training rows only
```

If the separate models are clearly better for a group, the relationship differs by group and a single set of coefficients cannot express both. Comparing the coefficients shows how: a coefficient that is positive in one group and negative in the other is the extreme case.

## Concrete Example: COMPAS

[COMPAS](../COMPAS/): the target is whether the COMPAS risk text is High or Medium, the features are the audit's core and proxy columns with race left out, logistic regression, the benchmark's split. Groups are White and Black defendants.

```
accuracy on the test rows of each group
  White defendants  pooled model 0.569 | model fitted on that group only 0.680
  Black defendants  pooled model 0.575 | model fitted on that group only 0.631

coefficient per standard deviation: pooled | White defendants only | Black defendants only
  Sex_Code_Text   +0.03 | -0.12 | +0.12
  MaritalStatus   +0.53 | +0.45 | +0.54
  CustodyStatus   -0.11 | -0.05 | -0.05
```

Fitting each group separately lifts accuracy from 0.569 to 0.680 for White defendants (11.1 points) and from 0.575 to 0.631 for Black defendants (5.6 points). The coefficient on `Sex_Code_Text` is -0.12 in one group and +0.12 in the other, while the pooled model puts it near zero (+0.03): the average of two opposite effects.

Two cautions. This model has only three features, so its absolute accuracy is low whatever the grouping; the comparison is the point. And a separate model per race needs the race at prediction time, which brings the legal problems discussed in [Group-Specific Thresholds](group-specific-thresholds.md).

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/aggregation-bias.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("COMPAS", strategy="unawareness")   # core + proxy columns, no race column
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()                             # 1 = scored High or Medium
black = data.masks["race"].to_numpy()             # 1 = Black defendants
columns = list(data.X.columns)
names = {0: "White defendants", 1: "Black defendants"}

def fit(rows):
    return LogisticRegression(max_iter=1000).fit(X[rows], y[rows])

pooled = fit(train)
separate = {g: fit(train[black[train] == g]) for g in (0, 1)}

print("accuracy on the test rows of each group")
for g in (0, 1):
    rows = test[black[test] == g]
    print(f"  {names[g]:17s} pooled model {pooled.score(X[rows], y[rows]):.3f} | model fitted on that group only {separate[g].score(X[rows], y[rows]):.3f}")

print("\ncoefficient per standard deviation: pooled | White defendants only | Black defendants only")
for i, col in enumerate(columns):
    print(f"  {col:15s} {pooled.coef_[0][i]:+.2f} | {separate[0].coef_[0][i]:+.2f} | {separate[1].coef_[0][i]:+.2f}")
```

Run it on any audit by changing the folder name and the attribute. On some audits there is little to find: on Healthcare Readmission the pooled and separate models score within a tenth of a point of each other, so one model is fine there.

## Limitations

1. **Separate models may not be allowed.** Using the group at decision time can be unlawful in hiring, lending and housing. Check before acting on the finding.
2. **Smaller groups get noisier models.** A group with a few hundred rows fitted alone can overfit, so the gain above is partly a larger-sample gain for the bigger group.
3. **It is a test, not a cure.** The finding tells you the relationship differs, not why. The cause may be a [measurement problem](measurement-bias.md) in one group.
4. **Interactions can do the same job.** A single model with group-by-feature interactions can capture the difference without a model per group, and may be easier to defend.
5. **Pooled can be the better choice.** If the groups really share the relationship, pooling gives more data per coefficient. The check exists to tell the cases apart.

## Related Concepts

* [Simpson's Paradox](simpsons-paradox.md) - an aggregate trend that reverses inside the groups
* [Subgroup Fairness](subgroup-fairness.md) - why group-level behaviour needs group-level checks
* [Measurement Bias](measurement-bias.md) - a common reason the relationship differs
* [Group-Specific Thresholds](group-specific-thresholds.md) - the legal limits of treating groups differently
* [Class Imbalance](class-imbalance.md) - why the smaller group counts for less in a pooled fit

## Related Projects in This Repo

* [`COMPAS/`](../COMPAS/) - the audit used for the example above

## Further Reading

* [Suresh, H., Guttag, J. (2021): A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle](https://arxiv.org/abs/1901.10002) - EAAMO 2021; defines aggregation bias in the taxonomy of harms.
* [Buolamwini, J., Gebru, T. (2018): Gender Shades](https://proceedings.mlr.press/v81/buolamwini18a.html) - FAT* 2018; a single aggregate accuracy hiding large differences between intersectional groups.
* Pearl, J. (2014). Understanding Simpson's Paradox. The American Statistician 68(1) - the aggregation reversal and when to trust which level.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
