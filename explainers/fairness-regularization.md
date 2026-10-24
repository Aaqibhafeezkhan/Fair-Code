> *Add the parity gap to the loss with a weight. The gap on probabilities is easy to shrink; the gap on decisions is the one that counts.*

## The One-Sentence Definition

A **fairness regularizer** is a penalty term for a fairness gap that is added to the training loss and weighted against accuracy by one number, so a single model fit trades the two off directly.

## Why It Matters

Constraint-based methods such as the benchmark's S3 (`ExponentiatedGradient`) enforce a gap limit. A regularizer is the soft version: no hard limit, just a price. That makes the trade-off visible as a curve you can read before choosing a point on it, and it works with any model you can train with gradients.

The catch is that decisions are thresholded, and a threshold is not differentiable. The penalty has to be written on something smooth, usually the predicted probabilities. A smooth gap and the real gap are different numbers, and the difference decides whether the method works.

## Core Concept: Loss Plus a Weighted Gap

```
total loss  =  log-loss  +  lambda * (soft gap)^2

soft gap    =  mean predicted probability for the disadvantaged group
               - mean predicted probability for the advantaged group
```

Squaring the gap makes it differentiable and treats a positive gap and a negative one alike. `lambda` is the price: at 0 you get the ordinary model, and as it grows the optimizer is willing to give up more accuracy to shrink the gap. The idea is the "prejudice remover" regularizer of Kamishima and colleagues, applied here to demographic parity.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): predicted above $50K income is the "ineligible" flag, women are the disadvantaged group, logistic regression on the audit's core features, the benchmark's split. Implemented with plain NumPy gradient descent.

```
  lambda  accuracy  soft gap  thresholded gap
       0     0.811   -0.0340          -0.0509
       1     0.811   -0.0332          -0.0516
      10     0.810   -0.0272          -0.0454
     100     0.794   -0.0090          -0.0212
    1000     0.770   +0.0043          +0.0010
   10000     0.759   +0.0013          +0.0000
```

The `lambda = 0` row is the benchmark's S2 baseline in `results/` (the current results): 0.811 accuracy and a thresholded gap of -0.0509. The in-processing strategy S3, a hard constraint, reaches -0.0219 at 0.784 accuracy.

Three things to read off the table:

- **Small penalties do nothing.** At `lambda = 1` the thresholded gap is -0.0516, no better than the baseline.
- **The two gaps differ.** At `lambda = 0` the soft gap is -0.0340 and the thresholded gap is -0.0509. The penalty shrinks the first one directly and the second one only through it.
- **Equalizing costs accuracy, and it accelerates.** The thresholded gap reaches +0.0010 at `lambda = 1000`, at 0.770 accuracy, 4.1 points below the baseline, and past that the gap does not move while accuracy keeps falling.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/fairness-regularization.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy(dtype=float)
s = data.masks["sex"].to_numpy(dtype=float)      # 1 = women, the disadvantaged group
sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def fit(lam, steps=3000, lr=0.5):
    """Minimise  log-loss + lam * (soft parity gap)^2, where the soft gap is the difference in the
    group means of the predicted PROBABILITY (differentiable), not of the thresholded decision."""
    Xtr, ytr, str_ = X[train], y[train], s[train]
    w, b = np.zeros(X.shape[1]), 0.0
    g1, g0 = str_ == 1, str_ == 0
    for _ in range(steps):
        p = sigmoid(Xtr @ w + b)
        soft_gap = p[g1].mean() - p[g0].mean()
        dp = (p - ytr) / len(ytr)                                  # d log-loss / d z
        dgap = np.where(g1, 1 / g1.sum(), -1 / g0.sum()) * p * (1 - p)   # d soft_gap / d z
        dz = dp + lam * 2 * soft_gap * dgap
        w -= lr * Xtr.T @ dz
        b -= lr * dz.sum()
    return w, b

print(f"{'lambda':>8} {'accuracy':>9} {'soft gap':>9} {'thresholded gap':>16}")
for lam in (0, 1, 10, 100, 1000, 10000):
    w, b = fit(lam)
    p = sigmoid(X[test] @ w + b)
    pred = (p >= 0.5).astype(int)
    s_te = s[test]
    print(f"{lam:8d} {(pred == y[test]).mean():9.3f} {p[s_te == 1].mean() - p[s_te == 0].mean():+9.4f} "
          f"{pred[s_te == 1].mean() - pred[s_te == 0].mean():+16.4f}")
```

## Limitations

1. **The penalty is a stand-in for the real gap.** The soft gap and the thresholded gap move together but are not the same, so always report the thresholded gap on held-out data.
2. **The useful range of `lambda` is narrow and data-dependent.** Little happens until about 10, the gap closes between 100 and 1000, and nothing further is gained after. Scan it on a log scale.
3. **It targets one metric.** This penalty is for demographic parity. Equalized odds needs a different term, and the two conflict, see [Fairness Metric Conflicts](fairness-metric-conflicts.md).
4. **It needs group labels while training.** The penalty is computed from group membership.
5. **You pick the point on the curve.** The method does not say which `lambda` is right. That is a policy choice about how much accuracy a smaller gap is worth.

## Related Concepts

* [Mitigation Strategies](mitigation-strategies.md) - S3 as a hard constraint, in contrast to this soft penalty
* [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) - the curve this method draws
* [Demographic Parity](demographic-parity.md) - the gap being penalized
* [Fairness Metric Conflicts](fairness-metric-conflicts.md) - why one penalty cannot cover every metric
* [Adversarial Debiasing](adversarial-debiasing.md) - another soft, objective-level in-processing method

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows quoted for S2 and S3

## Further Reading

* Kamishima, T., Akaho, S., Asoh, H., Sakuma, J. (2012). Fairness-Aware Classifier with Prejudice Remover Regularizer. ECML PKDD - the regularizer approach to a discrimination-aware classifier.
* [Zafar, M. B. et al. (2017): Fairness Constraints: Mechanisms for Fair Classification](https://arxiv.org/abs/1507.05259) - AISTATS 2017; a covariance-based constraint that makes the same relaxation precise.
* [Agarwal, A. et al. (2018): A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453) - ICML 2018; the hard-constraint counterpart used by the benchmark's S3 strategy.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
