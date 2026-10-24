> *You cannot audit a gap you cannot see. When the group is missing, you estimate it, and the estimate has its own error.*

## The One-Sentence Definition

**Inferring protected attributes** means estimating each person's group membership from other fields (a surname and address, a name list, a trained classifier) so a fairness gap can be measured when the true attribute was never collected.

## Why It Matters

Many organizations are not allowed to collect race, or never did: lenders in the United States are generally barred from asking about race on most consumer credit applications, and many datasets simply do not have the column. They still need to know whether a model treats groups differently. The usual answer is to estimate the attribute, which is how regulators and lenders have measured disparity in practice.

That swaps one problem for another. The estimate is wrong for some people, and the errors are not random: they follow the same correlations the proxies do. How wrong the measured gap becomes is the question.

## Core Concept: Hard Labels or Probabilities

Given an estimated probability `p` that each person is in the disadvantaged group, there are two ways to compute a group gap in the model's decisions `d`:

```
hard label:           assign each person to their most likely group, then compare group means of d
probability-weighted: gap = sum(p * d) / sum(p)  -  sum((1 - p) * d) / sum(1 - p)
```

The weighted version uses every person's uncertainty instead of rounding it away. Both are biased toward zero when the estimate is noisy, because mislabelled people blur the two groups together. That attenuation is the thing to measure.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the audited model is logistic regression on the core and proxy columns, with sex removed. We pretend sex was never recorded, train a reader on sex labels for only 5% of the training rows (1,302 rows), and use it to estimate sex for the test rows. The truth is known, so the error of each estimate is visible.

```
true sex available: parity gap -0.1003  (women 0.052, men 0.152)
inferred from 1302 labelled rows: accuracy 0.823, AUC 0.916
most likely group (hard label): parity gap -0.0920
probability-weighted:           parity gap -0.0885
```

With the true sex available the gap is -0.1003 (women flagged 5.2%, men 15.2%), which is the benchmark's S1 row in `results/` (the current results). The reader is decent, with 0.823 accuracy and 0.916 AUC, and the estimates of the gap are:

- **Hard labels:** -0.0920, about 8% smaller in size than the truth.
- **Probability-weighted:** -0.0885, about 12% smaller.

Both point the right way and are the right order of magnitude, and both understate the gap. In this run the hard labels happened to do slightly better than the weighted estimate; that is one dataset and one reader, not a rule.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/inferring-protected-attributes.py`; that file is generated from the block below). The true attribute is used only to score the result.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial", strategy="unawareness")   # core + proxy columns, no sex column
train, test = data.idx_train, data.idx_test
women = data.masks["sex"].to_numpy()                          # the true group, which we pretend is missing
y = data.y.to_numpy()
rng = np.random.default_rng(0)

# 1. The audited model (the same S1 model whose gap we want to measure).
audited = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], y[train])
pred = audited.predict(data.X.iloc[test])
truth = women[test]
true_gap = pred[truth == 1].mean() - pred[truth == 0].mean()
print(f"true sex available: parity gap {true_gap:+.4f}  (women {pred[truth == 1].mean():.3f}, men {pred[truth == 0].mean():.3f})")

# 2. Infer sex from the other columns, using sex labels for only 5% of the TRAINING rows.
labelled = rng.choice(train, size=int(0.05 * len(train)), replace=False)
reader = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=data.random_state, n_jobs=-1)
reader.fit(data.X.iloc[labelled], women[labelled])
p_woman = reader.predict_proba(data.X.iloc[test])[:, 1]
print(f"inferred from {len(labelled)} labelled rows: accuracy {((p_woman >= 0.5) == truth).mean():.3f}, AUC {roc_auc_score(truth, p_woman):.3f}")

# 3. Two ways to use the inferred group to estimate the gap.
hard = p_woman >= 0.5
hard_gap = pred[hard].mean() - pred[~hard].mean()
soft_gap = (p_woman * pred).sum() / p_woman.sum() - ((1 - p_woman) * pred).sum() / (1 - p_woman).sum()
print(f"most likely group (hard label): parity gap {hard_gap:+.4f}")
print(f"probability-weighted:           parity gap {soft_gap:+.4f}")
```

## Limitations

1. **The estimate understates the gap.** Misclassification blurs the groups, so the measured gap is usually smaller than the true one. Treat it as a lower bound unless corrected.
2. **Errors are not random.** The reader is wrong in structured ways, often for the same people the model is wrong for, which can bias the estimate in either direction.
3. **It needs some labels, or a trustworthy external source.** Here 5% of rows had the attribute. With none, you rely on a name list or a census join, which carry their own error rates.
4. **Inferring a protected attribute is itself sensitive.** It creates a record that did not exist. Check what law and policy allow, and keep the inferred column out of the decision system.
5. **Do not use the inferred attribute in the model.** It is a measurement tool for the audit, not an input, see [Fairness Through Unawareness](fairness-through-unawareness.md).

## Related Concepts

* [Protected Attribute](protected-attribute.md) - what is being estimated
* [Proxy Variables](proxy-variables.md) - the same correlations the estimate relies on
* [Fairness Through Unawareness](fairness-through-unawareness.md) - why not having the attribute does not make a model fair
* [Fairness Without Demographics](fairness-without-demographics.md) - improving the model when you cannot measure the groups at all
* [Measurement Bias](measurement-bias.md) - how an error in measuring a group blurs a gap

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above
* [`results/`](../results/) - the benchmark row quoted for S1

## Further Reading

* Elliott, M. N. et al. (2009). Using the Census Bureau's Surname List to Improve Estimates of Race/Ethnicity and Associated Disparities. Health Services and Outcomes Research Methodology 9(2) - the method behind Bayesian Improved Surname Geocoding.
* [Chen, J., Kallus, N., Mao, X., Svacha, G., Udell, M. (2019): Fairness Under Unawareness: Assessing Disparity When Protected Class Is Unobserved](https://arxiv.org/abs/1811.11154) - FAT* 2019; when a proxy-based estimate of disparity is and is not reliable.
* [Kallus, N., Mao, X., Zhou, A. (2022): Assessing Algorithmic Fairness with Unobserved Protected Class Using Data Combination](https://arxiv.org/abs/1906.00285) - Management Science; bounds on disparity when the attribute and the outcome sit in different datasets.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
