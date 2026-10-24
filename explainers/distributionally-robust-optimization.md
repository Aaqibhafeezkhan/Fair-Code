> *A fairness constraint needs you to name the metric and the groups in advance. Distributionally robust optimization hedges against the worst group instead, and asks only that you can find it.*

## The One-Sentence Definition

**Distributionally robust optimization (DRO)** trains a model to minimize its worst-case loss over a set of plausible data distributions, and in its group form that set is "any reweighting of the groups", so the model is pushed to do well on whichever group it is currently failing.

## Why It Matters

[Mitigation Strategies](mitigation-strategies.md) describes this repo's in-processing strategy, S3: Fairlearn's `ExponentiatedGradient`. It optimizes accuracy subject to a fairness constraint such as demographic parity. That has two requirements: you decide on the metric before training, and the constraint must be about the group you thought to check.

If the group that is actually harmed is not the one you constrained, a parity constraint does nothing for it. DRO has no parity target at all. It asks for a model whose loss does not blow up on any group, which can help when the harmed subgroup is one you did not think of, as long as it is among the groups you can define and measure.

## Core Concept: Min-Max Instead of Constrained Minimization

```
constraint (S3):  minimize loss   subject to   parity gap <= epsilon
group DRO:        minimize   max over weightings q of   sum_g q_g * loss(g)
```

The inner maximum is an adversary that moves weight onto the group with the highest loss. The outer minimum is the model fitting against that adversary. In the simplified loop below, the adversary updates its weights exponentially:

```
q_g  <-  q_g * exp(eta * loss_g)      then renormalize
```

`eta` is the free parameter. A small `eta` is a mild hedge. A large `eta` lets the adversary put almost all the weight on one group, and the model over-hedges against it. A related idea, [Max-Min (Rawlsian) Fairness](maxmin-fairness.md), uses hard error and fixed weight steps. DRO uses a smooth loss and a soft, exponentially updated weighting.

## Concrete Example: Benefits Denial

Same audit and feature set as the other explainers in this group: [Benefits Denial](../Benefits%20Denial/), logistic regression on the core features, benchmark split. Four groups this time: sex crossed with age (55 and over).

```
baseline     accuracy 0.811 | per-group log-loss male <55 0.482 male 55+ 0.593 female <55 0.320 female 55+ 0.315 | worst 0.593 | sex parity gap -0.0509
DRO eta=0.5  accuracy 0.807 | per-group log-loss male <55 0.481 male 55+ 0.549 female <55 0.375 female 55+ 0.360 | worst 0.549 | sex parity gap -0.0569
             adversary's group weights [0.325 0.573 0.054 0.048]
DRO eta=1.0  accuracy 0.810 | per-group log-loss male <55 0.485 male 55+ 0.542 female <55 0.396 female 55+ 0.379 | worst 0.542 | sex parity gap -0.0697
             adversary's group weights [0.271 0.707 0.013 0.009]
DRO eta=2.0  accuracy 0.809 | per-group log-loss male <55 0.489 male 55+ 0.539 female <55 0.409 female 55+ 0.390 | worst 0.539 | sex parity gap -0.0664
             adversary's group weights [0.166 0.833 0.001 0.   ]
test group sizes [3710  650 1859  294]
```

The baseline is the benchmark's S2 strategy in `results/` (the current results): accuracy 0.811, sex parity gap -0.0509. S3, the parity-constrained model, gets a gap of -0.0219 at 0.784 accuracy.

| Model | Accuracy | Worst-group log-loss | Sex parity gap |
|-------|:-:|:-:|:-:|
| Baseline (S2) | 0.811 | 0.593 (male 55+) | -0.0509 |
| Group DRO, eta 0.5 | 0.807 | 0.549 | -0.0569 |
| Group DRO, eta 1.0 | 0.810 | 0.542 | -0.0697 |
| Group DRO, eta 2.0 | 0.809 | 0.539 | -0.0664 |
| Parity constraint (S3, from `results/`) | 0.784 | not computed | -0.0219 |

What it did and did not do:

- **The worst group improved** (male 55+ log-loss 0.593 to 0.539-0.549) at almost no cost to overall accuracy.
- **The other groups paid for it.** Female under 55 went from 0.320 to 0.375-0.409. The adversary put 83% of the weight on male 55+ at `eta=2.0` and nearly 0% on women, which is what over-hedging looks like.
- **Parity got worse** (-0.0509 to as far as -0.0697). DRO never looked at it. If sex parity is the legal standard, S3's constraint is the tool, and DRO is not.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/distributionally-robust-optimization.py`; that file is generated from the block below).

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
from sklearn.metrics import log_loss

manifest, X, y, masks, train, test = load_audit("Benefits Denial")
group = masks["sex"] * 2 + masks["age"]      # 0 male under 55, 1 male 55+, 2 female under 55, 3 female 55+
labels = ["male <55", "male 55+", "female <55", "female 55+"]
X_tr, X_te = X.iloc[train], X.iloc[test]
y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
g_tr, g_te = group[train], group[test]

def fit(sample_weight=None):
    model = build_model("logistic_regression", manifest.random_state)
    return model.fit(X_tr, y_tr, sample_weight=sample_weight)

def group_loss(model, X_, y_, g_):
    p = model.predict_proba(X_)[:, 1]
    return np.array([log_loss(y_[g_ == k], p[g_ == k], labels=[0, 1]) for k in range(4)])

def group_dro(rounds=25, eta=1.0):
    n = np.bincount(g_tr, minlength=4)
    q = np.ones(4) / 4                                  # weight on each group: the adversary's choice
    model = fit()
    for _ in range(rounds):
        q = q * np.exp(eta * group_loss(model, X_tr, y_tr, g_tr))   # raise weight where loss is high
        q /= q.sum()
        w = (q / n)[g_tr]                               # per-row weight = group weight / group size
        model = fit(w / w.mean())
    return model, q

def summarise(model, name):
    pred = model.predict(X_te)
    loss = group_loss(model, X_te, y_te, g_te)
    parity_gap = pred[masks["sex"][test] == 1].mean() - pred[masks["sex"][test] == 0].mean()
    print(f"{name:12s} accuracy {(pred == y_te).mean():.3f} | per-group log-loss " +
          " ".join(f"{lab} {v:.3f}" for lab, v in zip(labels, loss)) +
          f" | worst {loss.max():.3f} | sex parity gap {parity_gap:+.4f}")

summarise(fit(), "baseline")
for eta in (0.5, 1.0, 2.0):
    model, q = group_dro(eta=eta)
    summarise(model, f"DRO eta={eta}")
    print("             adversary's group weights", np.round(q, 3))
print("test group sizes", np.bincount(g_te, minlength=4))
```

## Limitations

1. **The size of the uncertainty set is a free parameter.** Too small and nothing changes. Too large and the model hedges against a group that is not really that bad. The table above shows the loss plateauing from `eta=1.0` to `2.0` while the other groups keep paying.
2. **It still needs group labels during training.** DRO does not optimize a parity metric, but the groups in the inner maximum are defined by labels you must have. Methods that infer groups exist, but this simplified version does not.
3. **It optimizes loss, not a fairness definition.** Lower worst-group loss does not imply a smaller parity gap, a smaller equalized-odds gap or any legal standard being met.
4. **Overparameterized models make it harder.** The original group-DRO paper found that on large neural networks that fit the training data perfectly, worst-group performance needs strong regularization to improve. A logistic regression, as here, does not have that problem, but a deep model will.
5. **Small groups produce noisy adversaries.** A group with a few dozen rows has an unstable loss estimate, and the exponentiated weights will chase it. Pair it with a minimum group size, as in [Class Imbalance](class-imbalance.md).

## Related Concepts

* [Mitigation Strategies](mitigation-strategies.md) - S3, the constraint-based in-processing approach this is contrasted with
* [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) - the price of every method here
* [Class Imbalance](class-imbalance.md) - the same reweighting mechanics, applied to groups
* [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) - the hard-error version of the worst-group objective
* [Distribution Shift](distribution-shift.md) - the reason a worst-case distribution is worth hedging against

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows quoted for S2 and S3

## Further Reading

* [Sagawa, S., Koh, P. W., Hashimoto, T., Liang, P. (2020): Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731) - ICLR 2020; the group-DRO formulation and the regularization finding.
* [Duchi, J., Namkoong, H. (2021): Learning Models with Uniform Performance via Distributionally Robust Optimization](https://arxiv.org/abs/1810.08750) - Annals of Statistics 49(3); the general theory of DRO as a route to uniform performance.
* [Hashimoto, T., Srivastava, M., Namkoong, H., Liang, P. (2018): Fairness Without Demographics in Repeated Loss Minimization](https://arxiv.org/abs/1806.08010) - ICML 2018; DRO used to limit worst-group error when groups are unknown.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
