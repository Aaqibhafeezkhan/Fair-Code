> *Train a predictor while a second model tries to read race off its output, and punish the predictor whenever the second model succeeds.*

## The One-Sentence Definition

**Adversarial debiasing** is an in-processing mitigation that trains a predictor together with an adversary that tries to recover the protected attribute from the predictor's output, and updates the predictor to fit the label while making the adversary fail.

## Why It Matters

The repo's in-processing strategy, S3, hands Fairlearn a fairness constraint and asks it to find the most accurate model that satisfies it. That needs the constraint written down in advance. Adversarial debiasing states the goal differently: the model's scores should carry no usable information about the protected attribute. If the adversary cannot tell which group a score came from, the scores cannot be producing group-dependent decisions.

It also generalizes. The adversary can read a score, a hidden layer or a whole representation, which is why the idea is the basis of many neural-network fairness methods.

## Core Concept: A Two-Player Game

There are two models.

```
predictor:  p = sigmoid(w . x + b)          trained to predict the label y
adversary:  a = sigmoid(u * z + c)          trained to predict the group s from the predictor's logit z
```

The adversary minimizes its own loss. The predictor minimizes its task loss minus `alpha` times the adversary's loss:

```
predictor step:   d(task loss) / dz   -   alpha * d(adversary loss) / dz
```

`alpha` is the single dial. At `alpha = 0` it is ordinary logistic regression. As `alpha` grows, the predictor is pushed harder to produce scores the adversary cannot use. For demographic parity the adversary sees only the score. For equalized odds Zhang, Lemoine and Mitchell also feed it the true label.

## Concrete Example: Tenant Screening

[Tenant Screening](../Tenant%20Screening/): the model flags an applicant as likely to reoffend within three years, the disadvantaged group is Black applicants, the features are the audit's core columns, and the split is the benchmark's. Everything is plain NumPy: a predictor and an adversary updated in turn by gradient descent.

```
alpha  0.0: accuracy 0.639 | parity gap +0.1136 | race recoverable from the score, AUC 0.598
alpha  0.5: accuracy 0.639 | parity gap +0.0927 | race recoverable from the score, AUC 0.580
alpha  1.0: accuracy 0.637 | parity gap +0.0739 | race recoverable from the score, AUC 0.566
alpha  2.0: accuracy 0.637 | parity gap +0.0469 | race recoverable from the score, AUC 0.547
alpha  4.0: accuracy 0.628 | parity gap +0.0296 | race recoverable from the score, AUC 0.529
```

The `alpha = 0` row is the benchmark's S2 baseline in `results/` (the current results): 0.639 accuracy and a parity gap of +0.1139 (this run: +0.1136, because it is trained by gradient descent rather than scikit-learn's solver). Raising `alpha` to 4.0 cuts the gap to +0.0296 for 1.1 points of accuracy, and the adversary's ability to recover race from the score falls from an AUC of 0.598 toward chance (0.5). For comparison, the benchmark's S3 row (Fairlearn's `ExponentiatedGradient`) has a gap of +0.0172 at 0.626 accuracy, and its S4 row (`ThresholdOptimizer`) has -0.0054 at 0.639.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/adversarial-debiasing.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-9
X = (X - mean) / std
y = data.y.to_numpy(dtype=float)
s = data.masks["race"].to_numpy(dtype=float)      # 1 = Black applicants, the disadvantaged group
sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def train_predictor(alpha, steps=1500, lr=0.5):
    """Predictor p = sigmoid(X w + b). Adversary a = sigmoid(u * z + c) tries to read race from the
    predictor's own logit z. The predictor minimises task loss - alpha * adversary loss."""
    Xtr, ytr, str_ = X[train], y[train], s[train]
    w, b, u, c = np.zeros(X.shape[1]), 0.0, 0.1, 0.0
    for _ in range(steps):
        z = Xtr @ w + b
        p, a = sigmoid(z), sigmoid(u * z + c)
        # adversary step: get better at predicting race from z
        u -= lr * np.mean((a - str_) * z)
        c -= lr * np.mean(a - str_)
        # predictor step: fit the label, and push the adversary's loss UP
        dz = (p - ytr) - alpha * (a - str_) * u
        w -= lr * Xtr.T @ dz / len(ytr)
        b -= lr * dz.mean()
    return w, b

def report(alpha):
    w, b = train_predictor(alpha)
    z = X[test] @ w + b
    pred = (sigmoid(z) >= 0.5).astype(int)
    s_te, y_te = s[test], y[test]
    gap = pred[s_te == 1].mean() - pred[s_te == 0].mean()
    leak = roc_auc_score(s_te, z)                 # how well a threshold-free reader recovers race from the score
    print(f"alpha {alpha:4.1f}: accuracy {(pred == y_te).mean():.3f} | parity gap {gap:+.4f} | race recoverable from the score, AUC {leak:.3f}")

for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
    report(alpha)
```

The last column is the audit of the method itself: after training, a fresh reader tries to recover the attribute from the score. If its AUC is still high, the adversary lost.

## Limitations

1. **It is hard to train.** Two models chasing each other can oscillate. Step sizes, the number of steps and the initial adversary all matter, and a result at one setting says little about another.
2. **The adversary only removes what it can find.** Here it is a one-parameter logistic reader of the score. A stronger reader of the same score, or a reader of the inputs, may still find group information. Test with a different, stronger model.
3. **No guarantee for any single metric.** The gap fell steadily with `alpha`, but no `alpha` is "the one that satisfies parity". You pick a point on a curve.
4. **It costs accuracy, gradually.** 0.639 to 0.628 here, and the cost accelerates at larger `alpha`. See [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md).
5. **It needs the protected attribute during training**, even though the final model does not use it as an input.

## Related Concepts

* [Mitigation Strategies](mitigation-strategies.md) - where this sits among pre-, in- and post-processing
* [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) - the cost curve every `alpha` traces
* [Fairness Through Unawareness](fairness-through-unawareness.md) - why not feeding the attribute in is not enough
* [Proxy Variables](proxy-variables.md) - the leakage the adversary is trying to find
* [Demographic Parity](demographic-parity.md) - the criterion the adversary above is built for

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows quoted for S2, S3 and S4

## Further Reading

* [Zhang, B. H., Lemoine, B., Mitchell, M. (2018): Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/abs/1801.07593) - AIES 2018; the predictor-adversary game and its gradient update.
* [Edwards, H., Storkey, A. (2016): Censoring Representations with an Adversary](https://arxiv.org/abs/1511.05897) - ICLR 2016; the earlier adversarial formulation for hiding an attribute.
* [Beutel, A. et al. (2017): Data Decisions and Theoretical Implications when Adversarially Learning Fair Representations](https://arxiv.org/abs/1707.00075) - practical notes on what the adversary needs to see.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
