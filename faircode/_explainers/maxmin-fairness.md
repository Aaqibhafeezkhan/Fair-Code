> *Parity asks whether groups are treated alike. Max-min fairness asks a different question: how well off is the worst-off group, and can we make that better?*

## The One-Sentence Definition

**Max-min (Rawlsian) fairness** judges a model by its worst-off group, not by whether groups match each other: among all models, prefer the one that makes the minimum group-level outcome as good as possible.

## Why It Matters

Every parity metric in this repo ([Demographic Parity](demographic-parity.md), [Equalized Odds](equalized-odds.md), [Predictive Parity](predictive-parity.md)) compares groups with each other. A model can pass all of them by making everyone equally badly off. Parity has no opinion on the level, only on the gap.

Max-min fairness comes from John Rawls' difference principle in *A Theory of Justice* (1971): inequalities are acceptable only if they work to the benefit of the least advantaged. It flips the priority. A model that lifts the worst group from 75.7% to 77.0% accuracy is better under max-min, even if the groups end up further apart. That is a real conflict with the parity family, and [Fairness Metric Conflicts](fairness-metric-conflicts.md) does not cover it, because it only compares parity-style metrics against each other.

## Core Concept: Minimize the Maximum Group Loss

Demographic parity constrains a rate: the positive-prediction rate must match across groups. Max-min changes the objective instead of adding a constraint.

```
parity:   minimize overall loss   subject to   rate(group A) = rate(group B)
max-min:  minimize   max over groups g of   loss(g)
```

The second line has no fairness constraint at all. It has a different definition of "best": the model whose worst group does best. Two choices decide everything:

1. **What "loss" means.** Error rate, log-loss and false-negative rate can rank the same groups in different orders.
2. **Which groups are allowed to be "worst".** A group of 12 rows will have a wild error rate by chance. Without a minimum group size, the objective chases noise.

A simple way to optimize it is iterative reweighting: train, find the group with the highest error, give its rows more weight, retrain, repeat.

## Concrete Example: Benefits Denial

This uses the [Benefits Denial](../Benefits%20Denial/) audit (UCI Adult, income above 50K as the eligibility proxy), a logistic regression on the audit's core features with sex and race left out of the inputs, and four groups: sex crossed with race. The test split is the benchmark's own (20%, `random_state=42`).

```
baseline  overall 0.811 | male, White/Asian 0.757 | male, other race 0.872 | female, White/Asian 0.894 | female, other race 0.938 | worst group 0.757 | sex parity gap -0.0509
max-min   overall 0.811 | male, White/Asian 0.770 | male, other race 0.852 | female, White/Asian 0.874 | female, other race 0.914 | worst group 0.770 | sex parity gap -0.0563
group weights [11.39  1.    1.    1.  ]
```

Read it against the benchmark's own rows in `results/` (the current results, not the frozen snapshot). The baseline here is the benchmark's S2 strategy (`unawareness_proxy_removal`): 0.811 accuracy and a sex parity gap of -0.0509. The in-processing strategy S3 (Fairlearn's `ExponentiatedGradient`, a parity constraint) reaches a gap of -0.0219 at 0.784 accuracy.

| Model | Overall accuracy | Worst group accuracy | Sex parity gap |
|-------|:-:|:-:|:-:|
| Baseline (S2) | 0.811 | 0.757 | -0.0509 |
| Max-min reweighting | 0.811 | 0.770 | -0.0563 |
| Parity constraint (S3, from `results/`) | 0.784 | not computed | -0.0219 |

Three things to notice:

- **The worst-off group by accuracy is the largest one** (male, White or Asian: 4,008 of 6,513 test rows), not a small or marginalized one. Accuracy is the loss here, so that is what "worst off" means. Pick a different loss and the answer changes.
- **The worst group gained 1.3 points and everyone else lost some**: the female groups dropped from 0.894 and 0.938 to 0.874 and 0.914. Overall accuracy did not move.
- **The parity gap got slightly wider** (-0.0509 to -0.0563). The max-min model did better by its own measure and worse by demographic parity's.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 maxmin.py`). `_load_dataset` is the loader the benchmark harness uses, so the split matches `results/`.

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

manifest, X, y, masks, train, test = load_audit("Benefits Denial")
group = masks["sex"] * 2 + masks["race"]     # 0 male White/Asian, 1 male other race, 2 female White/Asian, 3 female other race
labels = ["male, White/Asian", "male, other race", "female, White/Asian", "female, other race"]
X_tr, X_te = X.iloc[train], X.iloc[test]
y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
g_tr, g_te = group[train], group[test]

def fit(sample_weight=None):
    model = build_model("logistic_regression", manifest.random_state)
    return model.fit(X_tr, y_tr, sample_weight=sample_weight)

def group_error(model, floor=30):
    wrong = model.predict(X_tr) != y_tr
    # groups under `floor` training rows are never "the worst off": a tiny group's error is noise
    return np.array([wrong[g_tr == k].mean() if (g_tr == k).sum() >= floor else -1.0 for k in range(4)])

def maxmin_fit(rounds=6, step=1.5):
    weight = np.ones(4)
    model = fit()
    for _ in range(rounds):
        weight[group_error(model).argmax()] *= step      # push weight onto whoever is worst off right now
        w = weight[g_tr]
        model = fit(w / w.mean())
    return model, weight

def summarise(model, name):
    pred = model.predict(X_te)
    correct = pred == y_te
    by_group = [correct[g_te == k].mean() for k in range(4)]
    parity_gap = pred[masks["sex"][test] == 1].mean() - pred[masks["sex"][test] == 0].mean()
    print(f"{name:9s} overall {correct.mean():.3f} | " +
          " | ".join(f"{lab} {acc:.3f}" for lab, acc in zip(labels, by_group)) +
          f" | worst group {min(by_group):.3f} | sex parity gap {parity_gap:+.4f}")

summarise(fit(), "baseline")
model, weight = maxmin_fit()
summarise(model, "max-min")
print("group weights", np.round(weight, 2))
```

`floor=30` mirrors the `small_sample_warning` threshold in `faircode/significance.py` (fewer than 30 rows): a group that small is never allowed to be "the worst".

## Limitations

1. **Tiny groups dominate without a floor.** The worst-case group is the noisiest one. A minimum group size, as in the profiler's small-group warning, is not optional.
2. **"Worst off" needs a loss, and the loss is a choice.** Accuracy, false-negative rate and log-loss can crown different groups. The Rawlsian argument says nothing about which to use.
3. **It can widen a parity gap, as it did here.** If your legal or policy standard is a parity metric, max-min can move you the wrong way. This is a trade-off, not a free improvement.
4. **It needs group labels at training time**, and the groups have to be defined up front. [Distributionally Robust Optimization](distributionally-robust-optimization.md) relaxes this, at the cost of a free parameter.
5. **The gain is small when the model is already near its ceiling.** A 1.3-point lift on the worst group cost the other groups 2 to 2.4 points. Whether that is a good deal is a values question, not a statistics one.

## Related Concepts

* [Fairness Metric Conflicts](fairness-metric-conflicts.md) - why parity metrics cannot all hold, and why max-min adds one more axis
* [Mitigation Strategies](mitigation-strategies.md) - the benchmark's constraint-based in-processing strategy that max-min is contrasted with
* [Class Imbalance](class-imbalance.md) - the same reweighting mechanics, applied to groups instead of labels
* [Distributionally Robust Optimization](distributionally-robust-optimization.md) - the smooth, soft version of the same worst-case idea
* [Subgroup Fairness](subgroup-fairness.md) - why the choice of groups matters

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows quoted for S2 and S3

## Further Reading

* Rawls, J. (1971). *A Theory of Justice.* Harvard University Press - the difference principle behind max-min fairness.
* [Hashimoto, T., Srivastava, M., Namkoong, H., Liang, P. (2018): Fairness Without Demographics in Repeated Loss Minimization](https://arxiv.org/abs/1806.08010) - ICML 2018; controls the worst-case group risk without needing group labels.
* [Martinez, N., Bertran, M., Sapiro, G. (2020): Minimax Pareto Fairness: A Multi Objective Perspective](https://proceedings.mlr.press/v119/martinez20a.html) - ICML 2020; minimizes the worst group's risk along the Pareto frontier.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
