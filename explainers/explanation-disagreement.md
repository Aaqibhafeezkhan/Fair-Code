> *Two explanation methods on the same model can name different top features. When a fairness claim rests on one of them, that is not a detail.*

## The One-Sentence Definition

**Explanation disagreement** is the situation in which different feature-attribution methods (permutation importance, SHAP, a surrogate's coefficients) rank the inputs of the same model differently, so a statement about what the model uses depends on which method you ask.

## Why It Matters

Fairness work leans on explanations in specific sentences: "the model does not use race", "the main driver is income", "the top feature is a proxy for sex". Each is a claim that a feature is or is not important, and each is only as good as the method behind it. Methods answer different questions: how much does the score drop when this column is scrambled, how much does this column move the average prediction away from the baseline, how large is this coefficient in a different model.

When the methods agree, the claim is robust. When they disagree, the disagreement is the finding, and it deserves to be reported with the claim.

## Core Concept: Rank the Features Three Ways, Then Compare the Ranks

1. **Permutation importance:** drop in accuracy when a column is shuffled.
2. **SHAP:** mean absolute contribution of the column to the model's predictions ([SHAP Values](shap-values.md)).
3. **Standardized logistic coefficients:** magnitude of the weight in a linear model fitted to the same data. (This is a different model, so it asks a different question.)

Compare with a rank correlation (Spearman) overall and per group, and look at the specific features a fairness claim is about.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the audit's core and proxy columns with sex and race left out, a gradient-boosted model, the benchmark's split. The ranks are of 10 features (1 is most important).

```
feature           permutation   SHAP  |logistic coef|   (rank of 10, everyone)
relationship                3      1                3
education.num               2      2                2
capital.gain                1      3                1
marital.status              6      4                4
occupation                  7      5                9
hours.per.week              5      6                5
capital.loss                4      7                6
workclass                   8      8                8
fnlwgt                     10      9               10
education                   9     10                7

Spearman correlation between the rankings
  permutation vs SHAP, everyone: 0.83
  permutation vs logistic coefficients: 0.90; SHAP vs logistic: 0.78
  permutation vs SHAP, men: 0.83
  permutation vs SHAP, women: 0.66
```

- **Broad agreement.** The rankings correlate at 0.83 (permutation against SHAP) and 0.90 (permutation against the logistic coefficients), and the least important features (`workclass`, `fnlwgt`) are ranked near the bottom by all three, so a claim that the model ignores them holds.
- **The top spot is contested.** `capital.gain` is first by permutation and by the logistic coefficients but third by SHAP, and `relationship` is first by SHAP and third by the other two.
- **That matters for fairness.** `relationship` encodes sex (Husband and Wife are almost fully male and female). A claim that "the model's main driver is a proxy for sex" is true by SHAP and weaker by the other two. A claim that "capital gain is the main driver" has the opposite pattern.
- **The disagreement is larger for women.** The correlation between permutation and SHAP is 0.83 on men's rows and 0.66 on women's, so the explanation is less settled for the group with fewer positives.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/explanation-disagreement.py`; that file is generated from the block below). It needs `shap`, which is not a dependency of this package (`pip install shap`).

```python
import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial", strategy="unawareness")   # core + proxy columns, no protected columns
train, test = data.idx_train, data.idx_test
X = data.X.astype(float)
y = data.y.to_numpy()
women = data.masks["sex"].to_numpy().astype(int)
columns = list(X.columns)
model = GradientBoostingClassifier(random_state=data.random_state).fit(X.iloc[train], y[train])

def permutation(rows):
    return permutation_importance(model, X.iloc[rows], y[rows], scoring="accuracy", n_repeats=10, random_state=0).importances_mean

def shap_mean_abs(rows):
    return np.abs(shap.TreeExplainer(model).shap_values(X.iloc[rows])).mean(axis=0)

methods_by_group = {}
for name, rows in (("everyone", test), ("men", test[women[test] == 0]), ("women", test[women[test] == 1])):
    methods_by_group[name] = {"permutation": permutation(rows), "SHAP": shap_mean_abs(rows)}

# A third, non-tree view: standardized logistic regression coefficients (the model is different, so this is a different question).
Z = (X - X.iloc[train].mean()) / (X.iloc[train].std() + 1e-9)
logistic = np.abs(LogisticRegression(max_iter=1000).fit(Z.iloc[train], y[train]).coef_[0])

def rank(values):
    return len(values) - np.argsort(np.argsort(values))     # 1 = most important

print(f"{'feature':16s} {'permutation':>12} {'SHAP':>6} {'|logistic coef|':>16}   (rank of 10, everyone)")
perm, sh = rank(methods_by_group["everyone"]["permutation"]), rank(methods_by_group["everyone"]["SHAP"])
lg = rank(logistic)
for i in np.argsort(sh):
    print(f"{columns[i]:16s} {perm[i]:12d} {sh[i]:6d} {lg[i]:16d}")

print("\nSpearman correlation between the rankings")
everyone = methods_by_group["everyone"]
print(f"  permutation vs SHAP, everyone: {spearmanr(everyone['permutation'], everyone['SHAP'])[0]:.2f}")
print(f"  permutation vs logistic coefficients: {spearmanr(everyone['permutation'], logistic)[0]:.2f}; SHAP vs logistic: {spearmanr(everyone['SHAP'], logistic)[0]:.2f}")
for name in ("men", "women"):
    m = methods_by_group[name]
    print(f"  permutation vs SHAP, {name}: {spearmanr(m['permutation'], m['SHAP'])[0]:.2f}")
```

## Limitations

1. **Different methods answer different questions.** Disagreement is not an error in one of them. Report which question each answers rather than picking the one that supports the claim.
2. **The logistic coefficients are from another model.** They describe the linear fit, not the gradient-boosted model, so agreement or not says something about the data's structure, not about that model.
3. **Rank correlation hides magnitude.** Two features can be close in importance and far in rank, and the reverse.
4. **Correlated features split credit unpredictably.** Permutation and SHAP treat a pair of correlated columns differently, which is part of the `relationship` and `marital.status` difference above.
5. **Ten features is a small ranking.** A correlation of 0.66 on ten items has wide uncertainty.

## Related Concepts

* [SHAP Values](shap-values.md) - the attribution method used above
* [What Is LIME?](lime.md) - a local surrogate, another way to explain
* [Feature Importance by Group](feature-importance-by-group.md) - per-group permutation importance
* [Proxy Variables](proxy-variables.md) - why the rank of `relationship` matters here
* [Counterfactual Explanation](counterfactual-explanation.md) - an explanation of one decision, not of the model

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Krishna, S. et al. (2022): The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective](https://arxiv.org/abs/2202.01602) - the study of when explanation methods disagree and how practitioners cope.
* [Lundberg, S., Lee, S.-I. (2017): A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) - NeurIPS 2017; the SHAP method.
* [Rudin, C. (2019): Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/abs/1811.10154) - Nature Machine Intelligence 1; the case against relying on post hoc explanations.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
