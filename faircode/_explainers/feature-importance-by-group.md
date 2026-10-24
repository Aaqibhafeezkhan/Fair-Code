> *One importance chart for the whole model assumes the model reasons the same way for everyone. Compute it per group and see.*

## The One-Sentence Definition

**Feature importance by group** is the importance of each input measured separately on each group's rows, so you can see whether the model leans on different features for different people.

## Why It Matters

A global importance ranking says which features matter on average. If one feature drives decisions for one group and barely touches another, the average hides it, and that is where aggregation effects and proxy use tend to show up ([Aggregation Bias](aggregation-bias.md)). It is also a check on a fairness claim: "the model doesn't rely on X" should hold for every group, not just overall.

The measurement is permutation importance (Breiman, 2001): shuffle one column among the rows you are looking at and see how much the score drops. Run it on a group's rows and you get that group's importance. It is model-agnostic and needs no assumptions about the model's form.

## Core Concept: Shuffle One Column, On One Group's Rows

```
for each group:
    for each feature:
        repeat 20 times: shuffle that column within the group's test rows, record the drop in accuracy
    importance = mean drop (with its standard deviation over the shuffles)
```

Two things to keep in mind. The importance is in units of the score, so it depends on how much score there is to lose: a group where the model is nearly always right has less room to drop. And the shuffles give a standard deviation, which tells you whether a difference between two features or two groups is large enough to mean anything.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): a gradient-boosted model on the audit's five core columns (sex and race are not inputs), the benchmark's split, test rows split into men and women. The score is accuracy.

```
gradient boosting, accuracy 0.832; test rows: 4360 men, 2153 women
feature           everyone                men              women
workclass          +0.0016    +0.0012 +/-0.0009    +0.0006 +/-0.0010
education          -0.0000    +0.0001 +/-0.0003    +0.0001 +/-0.0003
education.num      +0.0250    +0.0308 +/-0.0023    +0.0100 +/-0.0020
capital.gain       +0.0698    +0.0792 +/-0.0023    +0.0417 +/-0.0022
capital.loss       +0.0261    +0.0302 +/-0.0018    +0.0121 +/-0.0011
```

- **The same two columns drive both groups,** `capital.gain` first, but the size differs a lot. Shuffling `capital.gain` costs 0.079 accuracy on men and 0.042 on women; `education.num` costs 0.031 against 0.010; `capital.loss` costs 0.030 against 0.012. For women the model leans on these columns about a third to a half as much.
- **The order of the lower two changes,** with `capital.loss` ahead of `education.num` for women and the reverse for men, although for women the two are within one standard deviation of each other.
- **`workclass` and `education` are near zero for both,** consistent with the claim that the model does not use them.

Part of the gap is mechanical. Women are a minority of the positive class in this data, a model can get most of them right by predicting "no", and accuracy has less to lose. That does not make the difference unimportant, but it is a reason to also check with a measure that does not depend on the base rate, such as AUC, before concluding the model reasons differently.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/feature-importance-by-group.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.inspection import permutation_importance
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X, y = data.X, data.y.to_numpy()
women = data.masks["sex"].to_numpy().astype(int)
model = build_model("gradient_boosting", data.random_state).fit(X.iloc[train], y[train])

def importance(rows):
    """Drop in accuracy when one column is shuffled, on `rows` only; mean and standard deviation over 20 shuffles."""
    result = permutation_importance(model, X.iloc[rows], y[rows], scoring="accuracy", n_repeats=20, random_state=0)
    return result.importances_mean, result.importances_std

everyone = importance(test)
by_group = {name: importance(test[women[test] == g]) for g, name in ((0, "men"), (1, "women"))}
print(f"gradient boosting, accuracy {model.score(X.iloc[test], y[test]):.3f}; test rows: {(women[test] == 0).sum()} men, {(women[test] == 1).sum()} women")
print(f"{'feature':15s} {'everyone':>10} {'men':>18} {'women':>18}")
for i, column in enumerate(X.columns):
    men, women_ = by_group["men"], by_group["women"]
    print(f"{column:15s} {everyone[0][i]:+10.4f} {men[0][i]:+10.4f} +/-{men[1][i]:.4f} {women_[0][i]:+10.4f} +/-{women_[1][i]:.4f}")
```

## Limitations

1. **Importance depends on the score.** A group with a lower base rate or a higher accuracy can show smaller drops for the same reliance, so compare with more than one score.
2. **Correlated features share importance.** Shuffling one of two near-duplicates can leave the other to carry the signal, so both look unimportant. See [Proxy Entanglement](proxy-entanglement.md).
3. **Shuffling creates unrealistic rows.** A shuffled column can combine values that never occur together, and the model is judged on them.
4. **Small groups have wide intervals.** The standard deviation over shuffles measures only the shuffling, not the sampling noise of a small group.
5. **Importance is not effect.** A feature can matter equally to two groups while pushing in opposite directions, see [Partial Dependence by Group](partial-dependence-by-group.md).

## Related Concepts

* [Partial Dependence by Group](partial-dependence-by-group.md) - the direction of a feature's effect, per group
* [Explanation Disagreement](explanation-disagreement.md) - when two importance methods give different answers
* [SHAP Values](shap-values.md) - an attribution for a single prediction
* [Aggregation Bias](aggregation-bias.md) - when one model fits groups with different relationships
* [Proxy Variables](proxy-variables.md) - why "the model does not use X" needs a check per group

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Breiman, L. (2001): Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) - Machine Learning 45; permutation importance.
* [Fisher, A., Rudin, C., Dominici, F. (2019): All Models Are Wrong, but Many Are Useful](https://www.jmlr.org/papers/v20/18-760.html) - JMLR 20; model class reliance and its uncertainty.
* [Molnar, C. (2022): Interpretable Machine Learning, permutation feature importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html) - a worked treatment with the caveats listed above.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
