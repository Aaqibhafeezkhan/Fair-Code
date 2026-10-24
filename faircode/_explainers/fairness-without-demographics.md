> *If you do not know who the worst-off group is, you can still train against the people the model is already getting wrong.*

## The One-Sentence Definition

**Fairness without demographics** is a family of training methods that improve the accuracy of the worst-off groups without ever using group labels, usually by putting more weight on the examples the model finds hardest.

## Why It Matters

[Distributionally Robust Optimization](distributionally-robust-optimization.md) and [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) both need group labels during training, which many teams do not have, are not allowed to use, or do not trust. They also only protect the groups you thought to define.

The label-free idea is that groups a model fails are usually groups with high loss, so high loss stands in for membership. Hashimoto and colleagues showed a worst-case objective can be optimized without labels, and Liu and colleagues made it practical with a two-step recipe: train once, find the rows it got wrong, train again with those rows weighted up.

## Core Concept: Upweight the Errors

```
1. train a first model
2. find the training rows it misclassifies
3. train a second model, giving those rows a weight greater than 1
```

No group appears anywhere. Group labels are used only afterwards, to *evaluate* whether the worst group improved. The weight is the dial: a modest value shifts attention toward hard rows, a large one makes the model chase noise and the rare errors of a class, and the model degrades.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): logistic regression on the core columns, the benchmark's split. The four evaluation groups are sex crossed with age 55 and over, and they are used only to report results.

```
baseline                           overall 0.811 | male <55 0.776 male 55+ 0.712 female <55 0.900 female 55+ 0.901 | worst group 0.712
upweight the errors x1.5           overall 0.811 | male <55 0.778 male 55+ 0.717 female <55 0.898 female 55+ 0.891 | worst group 0.717
upweight the errors x2             overall 0.815 | male <55 0.789 male 55+ 0.725 female <55 0.889 female 55+ 0.874 | worst group 0.725
upweight the errors x3             overall 0.773 | male <55 0.765 male 55+ 0.742 female <55 0.792 female 55+ 0.816 | worst group 0.742
upweight the errors x5             overall 0.607 | male <55 0.626 male 55+ 0.702 female <55 0.526 female 55+ 0.670 | worst group 0.526
```

The baseline is the benchmark's S2 row in `results/` (the current results): 0.811 accuracy, with the worst group, men 55 and over, at 0.712.

- **A small upweight helps.** At 2x the worst group improves to 0.725 and overall accuracy to 0.815.
- **A larger one helps the worst group more and costs the rest.** At 3x the worst group is at 0.742, but overall accuracy is down to 0.773 and women under 55 drop from 0.900 to 0.792.
- **Past that it breaks.** At 5x the model collapses to 0.607 overall and the worst group, now women under 55, to 0.526.

The "worst group" can change identity: at 5x it is a different group from the baseline's. A label-free method has no way to prefer one group's accuracy over another's; it moves all of them.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/fairness-without-demographics.py`; that file is generated from the block below). Group labels appear only in the evaluation lines.

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X, y = data.X, data.y.to_numpy()
sex, aged = data.masks["sex"].to_numpy(), data.masks["age"].to_numpy()
group = sex * 2 + aged                           # used ONLY to evaluate, never to train
labels = ["male <55", "male 55+", "female <55", "female 55+"]

def fit(weights=None):
    return build_model("logistic_regression", data.random_state).fit(X.iloc[train], y[train], sample_weight=weights)

def report(name, model):
    correct = model.predict(X.iloc[test]) == y[test]
    per_group = [correct[group[test] == k].mean() for k in range(4)]
    print(f"{name:34s} overall {correct.mean():.3f} | " + " ".join(f"{l} {a:.3f}" for l, a in zip(labels, per_group))
          + f" | worst group {min(per_group):.3f}")

baseline = fit()
report("baseline", baseline)

# Just Train Twice: upweight the training rows the first model got wrong. No group labels are used.
wrong = baseline.predict(X.iloc[train]) != y[train]
for upweight in (1.5, 2, 3, 5):
    report(f"upweight the errors x{upweight}", fit(np.where(wrong, upweight, 1.0)))
```

## Limitations

1. **The weight is a free parameter, and the cliff is steep.** 2x helped, 3x traded accuracy for the worst group, 5x broke the model. Choose it on held-out data with group labels, which defeats part of the point.
2. **You still need labels to know it worked.** The method trains without them, but evaluation needs them, see [Disaggregated Evaluation](disaggregated-evaluation.md).
3. **High loss is not the same as a protected group.** The rows a model gets wrong may be label noise or outliers, which upweighting amplifies. See [Label Bias](label-bias.md).
4. **It cannot target a legal standard.** Nothing in it aims at demographic parity or equalized odds.
5. **The gains are modest.** Here the best worst-group change was 3 points, at a cost elsewhere. A method with labels, like [Distributionally Robust Optimization](distributionally-robust-optimization.md), can do more when labels are available.

## Related Concepts

* [Distributionally Robust Optimization](distributionally-robust-optimization.md) - the worst-group objective that needs labels
* [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) - the hard-error version, also needing labels
* [Inferring Protected Attributes](inferring-protected-attributes.md) - estimating the groups instead of avoiding them
* [Class Imbalance](class-imbalance.md) - the same weighting mechanics
* [Disaggregated Evaluation](disaggregated-evaluation.md) - how to check the worst group afterwards

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above
* [`results/`](../results/) - the benchmark row quoted for S2

## Further Reading

* [Hashimoto, T., Srivastava, M., Namkoong, H., Liang, P. (2018): Fairness Without Demographics in Repeated Loss Minimization](https://arxiv.org/abs/1806.08010) - ICML 2018; a worst-case objective optimized without group labels.
* [Liu, E. Z. et al. (2021): Just Train Twice: Improving Group Robustness without Training Group Information](https://arxiv.org/abs/2107.09044) - ICML 2021; the two-step upweighting recipe used above.
* [Lahoti, P. et al. (2020): Fairness without Demographics through Adversarially Reweighted Learning](https://arxiv.org/abs/2006.13114) - NeurIPS 2020; learns the weights with an adversary instead of fixing them.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
