> *A model can be worth using for one group at a threshold and worse than treating everyone for another group at the same threshold.*

## The One-Sentence Definition

**Decision curve analysis** plots the *net benefit* of acting on a model's predictions at each threshold probability, against the two defaults of treating everyone and treating no one; drawn per group, it shows the thresholds at which the model helps each group.

## Why It Matters

Accuracy, AUC and error rates describe a model without saying whether using it is better than the alternatives. Vickers and Elkin (2006) proposed net benefit for exactly that question in medicine: given that a false positive has a cost, at what threshold is the model worth acting on? The threshold is not a technical setting. It encodes how many false positives you will tolerate to catch one true positive.

Per group, the same curve can tell a different story. A threshold that is useful for one group can be no better than flagging everyone for another, even when the model is "the same" for both. That is a fairness finding that a single aggregate curve, or a metric at 0.5, cannot show.

## Core Concept: Net Benefit

At threshold probability `t`, the model flags everyone whose score is at least `t`. The net benefit is

```
net benefit(t) = TP / n  -  FP / n * t / (1 - t)
```

True positives count as gains. Each false positive is a loss weighted by the odds `t / (1 - t)`: at `t = 0.2` a false positive costs a quarter of a true positive's gain, at `t = 0.5` it costs the same. Choosing `t` is therefore the same as stating the cost ratio: acting at `t = 0.2` says a missed case is four times worse than a false alarm.

Two reference lines come with it:

* **Treat everyone:** `prevalence - (1 - prevalence) * t / (1 - t)`, a model that flags every person.
* **Treat no one:** zero by definition.

The model is useful at `t` only if its net benefit is above both. Computing all three per group, with each group's own prevalence in its treat-everyone line, gives a curve per group.

## Concrete Example: COMPAS, Race

The [COMPAS](../COMPAS/) audit: the label is whether the COMPAS risk score text is High or Medium, a logistic regression on the core features (sex and marital status) predicts it, the disadvantaged group is African-American defendants. This audit models the score COMPAS issued, not whether anyone reoffended, so "net benefit" here measures how well the model reproduces that score. Prevalence differs a lot: 31% of the advantaged group's test rows are positive against 59% for the disadvantaged group.

```
COMPAS - race: net benefit per 100 people at each threshold probability
  threshold | advantaged: model  treat-all | disadvantaged: model  treat-all
       0.15 |   19.02    19.02      |   51.81    51.81
       0.20 |   14.72    13.95      |   48.48    48.79
       0.25 |   12.30     8.22      |   44.53    45.38
       0.30 |    7.79     1.66      |   41.31    41.48
       0.35 |    2.90    -5.90      |   37.46    36.98
       0.40 |   -2.89   -14.73      |   33.24    31.73
       0.45 |   -9.29   -25.16      |   27.35    25.52
       0.50 |  -17.24   -37.67      |   21.56    18.07
advantaged    beats treat-all and treat-none at thresholds: 0.18 to 0.37
disadvantaged beats treat-all and treat-none at thresholds: 0.32 to 0.52
```

At a threshold of 0.25 the model adds 12.30 net benefit per 100 people for the advantaged group, against 8.22 for treating them all. For the disadvantaged group at the same threshold it adds 44.53 against 45.38 for treating all of them: the model is slightly worse than flagging everyone. The useful thresholds hardly overlap: the model beats both defaults from 0.18 to 0.37 for the advantaged group and from 0.32 to 0.52 for the disadvantaged group, so below 0.32 the model is not better than "flag everyone" for the disadvantaged group, and above 0.37 it is not worth using for the advantaged group at all (net benefit falls below zero at 0.40).

The takeaway is not that the model is fine in one range: it is that any single threshold serves one group better than the other, and that the right one depends on the cost ratio each group faces.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/decision-curve-analysis-by-group.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit
from faircode.models import build_model

def net_benefit(y, p, t):
    """TP/n - FP/n * t/(1-t): true positives, minus false positives weighted by the odds of the threshold."""
    pred = p >= t
    return (pred & (y == 1)).mean() - (pred & (y == 0)).mean() * t / (1 - t)

def treat_all(y, t):
    return y.mean() - (1 - y.mean()) * t / (1 - t)

def useful_ranges(y, p, grid):
    """Threshold ranges where the model beats both treating everyone and treating no one."""
    ok = [net_benefit(y, p, t) > max(treat_all(y, t), 0) for t in grid]
    ranges, start, last = [], None, None
    for t, flag in zip(grid, ok):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            ranges.append((start, last)); start = None
        last = t
    if start is not None:
        ranges.append((start, grid[-1]))
    return ranges

data = load_audit("COMPAS")
tr, te = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
p = model.predict_proba(data.X.iloc[te])[:, 1]
y = data.y.iloc[te].to_numpy()
s = data.group("race")[te]                           # 1 = disadvantaged group
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

print("COMPAS - race: net benefit per 100 people at each threshold probability")
print("  threshold | advantaged: model  treat-all | disadvantaged: model  treat-all")
for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    row = [f"{100 * f(y[s == g], p[s == g], t) if f is net_benefit else 100 * f(y[s == g], t):7.2f}"
           for g in (0, 1) for f in (net_benefit, lambda yy, *a: treat_all(yy, a[-1]))]
    print(f"  {t:9.2f} | {row[0]}  {row[1]}      | {row[2]}  {row[3]}")
for g, name in ((0, "advantaged"), (1, "disadvantaged")):
    spans = ", ".join(f"{a:.2f} to {b:.2f}" for a, b in useful_ranges(y[s == g], p[s == g], grid))
    print(f"{name:13s} beats treat-all and treat-none at thresholds: {spans}")
```

## Limitations

1. **The threshold stands in for a cost ratio you have to choose.** Net benefit is only as meaningful as that ratio. A table at one threshold is a claim about costs; say what you assumed. See [Cost-Sensitive Evaluation](cost-sensitive-evaluation.md).
2. **It assumes calibrated probabilities.** The `t / (1 - t)` weighting is exact only when scores are probabilities. An uncalibrated score gives net benefit that is not comparable across thresholds or groups. See [Calibration](calibration.md).
3. **It is noisy for small groups.** Net benefit is a mean of a few outcomes; the curves for a group of a few hundred rows can cross by chance. Add bootstrap intervals before reading a crossing as a finding.
4. **Different prevalence changes the treat-everyone line.** A group with 59% positives has a much higher default line than one with 31%, so the same model looks less useful for it. That is a real feature of the data, not a flaw of the method, but it is easy to mistake for model bias.
5. **A label that is itself a model output means the curve measures agreement with that model.** In the COMPAS audit the label is the issued score. See [Label Bias](label-bias.md).

## Related Concepts

* [Cost-Sensitive Evaluation](cost-sensitive-evaluation.md) - choosing the threshold from costs
* [False Positives vs. False Negatives](false-positives-vs-false-negatives.md) - the two errors the threshold trades off
* [Precision-Recall Curve](precision-recall-curve.md) - another threshold view, also sensitive to prevalence
* [ROC Curve and AUC](roc-curve-auc.md) - the threshold-free ranking summary
* [Group-Specific Thresholds](group-specific-thresholds.md) - using a different cutoff per group
* [Calibration](calibration.md) - what net benefit assumes

## Related Projects in This Repo

* [`COMPAS/`](../COMPAS/) - the audit used for the example above

## Further Reading

* [Vickers, A. J., Elkin, E. B. (2006): Decision Curve Analysis: A Novel Method for Evaluating Prediction Models](https://doi.org/10.1177/0272989X06295361) - Medical Decision Making 26(6); the original method.
* [Vickers, A. J., van Calster, B., Steyerberg, E. W. (2016): Net Benefit Approaches to the Evaluation of Prediction Models, Molecular Markers, and Diagnostic Tests](https://doi.org/10.1136/bmj.i6) - BMJ 352; a short practical guide.
* [Angwin, J. et al. (2016): Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) - ProPublica; the COMPAS analysis behind the audit's data.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
