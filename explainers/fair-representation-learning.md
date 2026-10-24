> *Hiding an attribute from one classifier is not the same as removing it. A representation can look clean to a linear reader and still be an open book to a forest.*

## The One-Sentence Definition

**Fair representation learning** transforms the inputs into a new representation that keeps the information needed for the task and drops the information about the protected attribute, so any model trained on the representation inherits the removal.

## Why It Matters

Dropping the protected column does not drop its information, because proxies carry it ([Proxy Variables](proxy-variables.md)). Fair representation learning attacks the proxies directly: it changes every row so the attribute cannot be read from what is left. Its attraction is separation of concerns. One team builds the cleaned representation, and downstream teams can train whatever model they like on it.

The risk is the word "cannot". It is only as strong as the reader you tested against, and that makes this the method where testing the claim matters most.

## Core Concept: Remove the Directions That Predict the Group

The simplest version is a linear projection. Fit a linear reader that predicts the group from the features, take its direction, and project every row onto the space orthogonal to it. Repeat on the result: the first projection removes the most obvious signal and the next reader finds what was left. This is iterative nullspace projection (Ravfogel and colleagues), and it is a linear-algebra cousin of the learned representations of Zemel and colleagues.

```
for k rounds:
    reader = linear model predicting the group from the current features
    remove the reader's direction from every row
```

After `k` rounds a linear reader can do no better than chance. Whether any reader can is the question the example answers.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the features are the audit's core and proxy columns with the sex column removed (marital status, relationship, occupation and the rest, which do carry sex), standardized. The group is sex; the model is logistic regression on the cleaned rows.

```
columns: ['workclass', 'education', 'education.num', 'capital.gain', 'capital.loss', 'marital.status', 'occupation', 'relationship', 'hours.per.week', 'fnlwgt']
0 directions removed: accuracy 0.821 | parity gap -0.1336 | sex recoverable, AUC: linear reader 0.862, random forest 0.915
1 directions removed: accuracy 0.806 | parity gap -0.0480 | sex recoverable, AUC: linear reader 0.598, random forest 0.912
2 directions removed: accuracy 0.800 | parity gap -0.0259 | sex recoverable, AUC: linear reader 0.513, random forest 0.897
4 directions removed: accuracy 0.776 | parity gap -0.0070 | sex recoverable, AUC: linear reader 0.500, random forest 0.859
```

Two readers try to recover sex from each representation: a linear model, and a random forest that can find non-linear structure.

- **The linear reader is beaten quickly.** Its AUC falls from 0.862 to 0.598 after one direction and to 0.500 (chance) after four.
- **The random forest is not.** Its AUC is still 0.859 after four directions are removed, against 0.915 before. The group is still there; it is just no longer linearly separable.
- **The parity gap falls with it**, from -0.1336 to -0.0070, for 4.5 points of accuracy (0.821 to 0.776).

So the downstream logistic regression is nearly gap-free while the representation is not clean. The model benefits only because it is linear. A stronger downstream model could use what the forest finds.

The baseline here is standardized and fitted by scikit-learn on the S1 columns, so it is not the benchmark's S1 row (a gap of -0.1003 in `results/`, the current results).

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/fair-representation-learning.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial", strategy="unawareness")   # core + proxy columns, no sex column
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
s = data.masks["sex"].to_numpy()                  # 1 = women
print("columns:", list(data.X.columns))

def remove_directions(k):
    """Iterative nullspace projection: repeatedly fit a linear reader of sex on the training rows
    and project its direction out of every row."""
    P = np.eye(X.shape[1])
    for _ in range(k):
        reader = LogisticRegression(max_iter=1000).fit(X[train] @ P, s[train])
        w = reader.coef_[0] @ P
        w = w / np.linalg.norm(w)
        P = (np.eye(len(w)) - np.outer(w, w)) @ P
    return X @ P

def report(k):
    Z = remove_directions(k)
    task = LogisticRegression(max_iter=1000).fit(Z[train], y[train])
    pred = task.predict(Z[test])
    gap = pred[s[test] == 1].mean() - pred[s[test] == 0].mean()
    linear = LogisticRegression(max_iter=1000).fit(Z[train], s[train])
    forest = RandomForestClassifier(n_estimators=100, random_state=data.random_state, n_jobs=-1).fit(Z[train], s[train])
    print(f"{k} directions removed: accuracy {(pred == y[test]).mean():.3f} | parity gap {gap:+.4f} | "
          f"sex recoverable, AUC: linear reader {roc_auc_score(s[test], linear.predict_proba(Z[test])[:, 1]):.3f}, "
          f"random forest {roc_auc_score(s[test], forest.predict_proba(Z[test])[:, 1]):.3f}")

for k in (0, 1, 2, 4):
    report(k)
```

The test of the claim is the last two numbers on each line: always try a reader stronger than the one you removed.

## Limitations

1. **"Removed" is relative to a reader.** A representation that hides the attribute from a linear model can still leak to a non-linear one, as the 0.859 random-forest AUC shows. Report the strongest reader you tried.
2. **It removes task signal too.** The directions that separate the groups are also directions that predict the label when the label itself is correlated with the group. Accuracy fell 4.5 points.
3. **It needs the attribute during training**, to fit the readers.
4. **It is fit on one dataset.** A new population has different correlations, and the projection no longer removes them. Refit and retest.
5. **It cleans the features, not the labels.** If the labels carry historical bias ([Label Bias](label-bias.md)), a model trained on clean features still learns it.

## Related Concepts

* [Proxy Variables](proxy-variables.md) - the signal the projection is removing
* [Fairness Through Unawareness](fairness-through-unawareness.md) - the starting point: dropping the column alone
* [Mitigation Strategies](mitigation-strategies.md) - the pre-processing family this belongs to
* [Adversarial Debiasing](adversarial-debiasing.md) - learns the removal instead of computing it
* [Label Bias](label-bias.md) - a bias a clean representation cannot fix

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Zemel, R., Wu, Y., Swersky, K., Pitassi, T., Dwork, C. (2013): Learning Fair Representations](https://proceedings.mlr.press/v28/zemel13.html) - ICML 2013; the learned-representation formulation.
* [Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., Goldberg, Y. (2020): Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://arxiv.org/abs/2004.07667) - ACL 2020; the projection method used above and its limits.
* [Edwards, H., Storkey, A. (2016): Censoring Representations with an Adversary](https://arxiv.org/abs/1511.05897) - ICLR 2016; learning the representation against an adversary instead.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
