> *A model can respond to a feature the same way for two groups or not at all. The curve per group shows which, as long as you only trust it where the group has data.*

## The One-Sentence Definition

**Partial dependence by group** is the model's average predicted probability as one feature is set to each value in a grid, computed separately over each group's rows, so the curve for one group can be compared with the curve for another.

## Why It Matters

[Feature Importance by Group](feature-importance-by-group.md) says how much a model uses a feature for each group. Partial dependence says what the model does with it. Two groups can have the same importance for schooling and very different curves: one rises steeply with years of education, the other barely, or one starts higher and stays higher.

The curve also tells you something importance cannot: the size of a gap at a given value of the feature. If the model predicts a lower probability for one group at every level of education with the other columns averaged over that group's own rows, the difference is the model's, not only the data's.

## Core Concept: Set the Feature, Average Over the Group

```
for each value v in the grid:
    set the feature to v for every row of the group, leaving the other columns as they are
    average the model's predicted probability over those rows
```

One implementation detail matters a great deal. scikit-learn's default for tree models (`method="recursion"`) does not use the rows you pass; it averages over the training data's leaf weights, so the curve is identical for every group. Use `method="brute"` to average over the rows of the group.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the same gradient-boosted model as the previous explainer, with years of schooling (`education.num`) varied from 6 to 16.

```
average predicted probability of 'income above $50K' as education.num varies
 education.num       6       9      10      12      13      14      16
           men   0.123   0.196   0.222   0.267   0.381   0.486   0.621
         women   0.079   0.152   0.179   0.225   0.342   0.452   0.596
gap (women-men)  -0.044  -0.044  -0.043  -0.042  -0.039  -0.033  -0.025

test rows actually observed at each value (the curve is only supported where there is data)
           men     133    1420     880     138     767     224      64
         women      59     699     576      71     327      95      17
```

- **The curves have the same shape.** Probability of income above $50K climbs from 0.123 (men) and 0.079 (women) at 6 years to 0.621 and 0.596 at 16 years.
- **The gap shrinks as schooling rises.** Women's curve is 0.044 below men's at 6 years and 0.025 below at 16, so more schooling closes some of the gap but not all of it. The model gives a woman with a doctorate-level education less than a man with the same education, averaged over each group's own capital gains and losses and other columns.
- **The ends are thinly supported.** There are 17 women in the test rows at 16 years and 59 at 6 years, against 1,420 men at 9 years, so the right end of the women's curve rests on very few rows.

The final block of the output is the check that should accompany any curve like this: the number of rows behind each point.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/partial-dependence-by-group.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.inspection import partial_dependence
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X, y = data.X.astype(float), data.y.to_numpy()   # float columns: partial dependence refuses integer ones
women = data.masks["sex"].to_numpy().astype(int)
model = build_model("gradient_boosting", data.random_state).fit(X.iloc[train], y[train])
# method="brute" averages over the rows you pass (here, one group). The default "recursion" for trees ignores them and
# uses the training rows, which would draw the same curve for every group.

grid = [6, 9, 10, 12, 13, 14, 16]                # years of schooling (education.num)
print("average predicted probability of 'income above $50K' as education.num varies")
print(f"{'education.num':>14} " + " ".join(f"{g:>7}" for g in grid))
rows = {}
for g, name in ((0, "men"), (1, "women")):
    subset = X.iloc[test[women[test] == g]]
    curve = partial_dependence(model, subset, ["education.num"], custom_values={"education.num": np.array(grid)},
                               kind="average", method="brute", response_method="predict_proba")["average"][0]
    rows[name] = curve
    print(f"{name:>14} " + " ".join(f"{v:7.3f}" for v in curve))
print(f"{'gap (women-men)':>14} " + " ".join(f"{w - m:+7.3f}" for w, m in zip(rows["women"], rows["men"])))

print("\ntest rows actually observed at each value (the curve is only supported where there is data)")
for g, name in ((0, "men"), (1, "women")):
    values = X["education.num"].to_numpy()[test[women[test] == g]]
    print(f"{name:>14} " + " ".join(f"{(values == v).sum():7d}" for v in grid))
```

## Limitations

1. **It averages over the group's own other columns.** The gap above mixes the model's treatment of sex with the different distribution of the other features in each group, so it is not a pure effect of sex (the model does not even see it).
2. **It assumes the feature can be set freely.** Setting `education.num` to 16 for a row whose other columns do not fit that education creates an unrealistic row.
3. **Thin regions are unreliable.** The 17-row end of the women's curve is where the model has the least to learn from and the curve is least trustworthy.
4. **Averages hide heterogeneity.** An average curve can be flat while individual curves rise and fall, so individual conditional expectation curves are a useful companion.
5. **The default can silently give the same curve for every group.** Check that the curves can differ, as in the note above.

## Related Concepts

* [Feature Importance by Group](feature-importance-by-group.md) - how much the model uses a feature, per group
* [Explanation Disagreement](explanation-disagreement.md) - when two ways of explaining the same model differ
* [Aggregation Bias](aggregation-bias.md) - one model and different relationships by group
* [Counterfactual Explanation](counterfactual-explanation.md) - the one-person version of "set the feature and see"
* [Algorithmic Recourse](algorithmic-recourse.md) - when the feature is one a person can change

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Friedman, J. H. (2001): Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) - Annals of Statistics 29(5); the introduction of partial dependence.
* [Goldstein, A., Kapelner, A., Bleich, J., Pitkin, E. (2015): Peeking Inside the Black Box](https://arxiv.org/abs/1309.6392) - Journal of Computational and Graphical Statistics 24(1); individual conditional expectation plots.
* [Apley, D. W., Zhu, J. (2020): Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models](https://arxiv.org/abs/1612.08468) - Journal of the Royal Statistical Society B 82(4); accumulated local effects, which avoid unrealistic rows.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
