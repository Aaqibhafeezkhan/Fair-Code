> *An overall accuracy is an average over people. It tells you nothing about who is in the tail.*

## The One-Sentence Definition

**Disaggregated evaluation** reports every performance metric separately for each group, and for each intersection of groups, with an interval, instead of one number for the whole population.

## Why It Matters

A model can have a good headline number and fail one group badly, because the groups it fails are small. The Gender Shades study made this concrete: commercial face classifiers with high overall accuracy had error rates for darker-skinned women many times those for lighter-skinned men. Every fairness metric in this repo is a comparison of per-group numbers, so the discipline underneath them is simply to look at the per-group numbers first.

It has a cost that is easy to miss: small groups give wide intervals. A reported difference between two groups of 33 and 73 people can be pure chance, and an apparent absence of a gap in a tiny group can be the same.

## Core Concept: Slice, Then Put an Interval on Every Slice

1. **Pick the slices.** The protected attributes one at a time, and the intersections of them ([Intersectional Bias](intersectional-bias.md), [Subgroup Fairness](subgroup-fairness.md)).
2. **Compute each metric per slice**, not just accuracy: error rates, selection rate, calibration.
3. **Attach an interval** to each, by bootstrap ([Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md)), and state a minimum group size below which you only report the interval, not the point estimate.
4. **Compare slices to the whole, and to each other,** using the interval, not the gap alone ([Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md)).

## Concrete Example: Insurance Denial

[Insurance Denial](../Insurance%20Denial/): logistic regression on the audit's core features, the benchmark's split (267 test rows), groups by age and gender and their four intersections. Each row has a bootstrap 95% interval from 2,000 resamples.

```
overall accuracy 0.644

everyone                     n= 267  accuracy 0.644  95% CI [0.584, 0.700]  width 0.116
age: advantaged              n= 161  accuracy 0.671  95% CI [0.596, 0.745]  width 0.149
age: disadvantaged           n= 106  accuracy 0.604  95% CI [0.509, 0.698]  width 0.189
gender: advantaged           n= 133  accuracy 0.602  95% CI [0.519, 0.684]  width 0.165
gender: disadvantaged        n= 134  accuracy 0.687  95% CI [0.604, 0.761]  width 0.157
age 0 x gender 0             n=  60  accuracy 0.633  95% CI [0.500, 0.750]  width 0.250
age 0 x gender 1             n= 101  accuracy 0.693  95% CI [0.604, 0.782]  width 0.178
age 1 x gender 0             n=  73  accuracy 0.575  95% CI [0.466, 0.699]  width 0.233
age 1 x gender 1             n=  33  accuracy 0.667  95% CI [0.485, 0.818]  width 0.333
```

The overall figure, 0.644 with an interval 0.116 wide, hides the spread: age-disadvantaged is 0.604 against 0.671 for the advantaged group, gender-disadvantaged is 0.687 against 0.602 for the advantaged. The intersections are more extreme in both directions, and their intervals are far wider: 0.250 for 60 people and 0.333 for 33. Every one of those intervals contains the overall accuracy, so this test set cannot establish a gap between any two of them. That is the lesson, not a failing of the method: with 267 rows the audit cannot say much about any intersection.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/disaggregated-evaluation.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Insurance Denial")            # S2 feature set; groups are age x gender
train, test = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], data.y.iloc[train])
pred = model.predict(data.X.iloc[test])
y_te = data.y.iloc[test].to_numpy()
older = data.masks["age"].to_numpy()[test]       # 1 = the disadvantaged age group
female = data.masks["gender"].to_numpy()[test]   # 1 = the disadvantaged gender group
rng = np.random.default_rng(0)

def row(label, mask):
    correct = (pred == y_te)[mask]
    boots = [rng.choice(correct, len(correct)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label:28s} n={mask.sum():4d}  accuracy {correct.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")

print(f"overall accuracy {(pred == y_te).mean():.3f}\n")
row("everyone", np.ones(len(y_te), dtype=bool))
for a, an in ((0, "age: advantaged"), (1, "age: disadvantaged")):
    row(an, older == a)
for g, gn in ((0, "gender: advantaged"), (1, "gender: disadvantaged")):
    row(gn, female == g)
for a in (0, 1):
    for g in (0, 1):
        row(f"age {a} x gender {g}", (older == a) & (female == g))
```

## Limitations

1. **More slices mean more chances for an alarming number.** Twenty slices will throw up a bad-looking one by chance. See [Multiple Comparisons](multiple-comparisons.md).
2. **Small slices cannot be evaluated.** The honest output for a 33-row slice is "unknown, wide interval", and a minimum group size makes that rule explicit.
3. **Intersections explode.** Three attributes with four values each is 64 cells, and most will be too small to report.
4. **The slices must match the harm.** Slicing on a column that does not correspond to anyone's real group membership produces tidy tables and no insight.
5. **Per-group accuracy is one metric.** Equal accuracy can coexist with very different error types, see [Accuracy Equality](accuracy-equality.md) and [Treatment Equality](treatment-equality.md).

## Related Concepts

* [Subgroup Fairness](subgroup-fairness.md) - why group-level checks need group-level data
* [Intersectional Bias](intersectional-bias.md) - the slices that cross attributes
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the interval on each slice
* [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md) - the test for a gap between two slices
* [Multiple Comparisons](multiple-comparisons.md) - what to do when you report many slices
* [What Is an AUC Gap Between Groups?](auc-gaps-by-group.md) - the same curve computed per group, with an interval for the gap

## Related Projects in This Repo

* [`Insurance Denial/`](../Insurance%20Denial/) - the audit used for the example above
* [Open Dataset Profiler](../profiler.html) - reports group sizes and warns on small groups before you model

## Further Reading

* [Barocas, S. et al. (2021): Designing Disaggregated Evaluations of AI Systems: Choices, Considerations, and Tradeoffs](https://arxiv.org/abs/2103.06076) - AIES 2021; the design decisions in a disaggregated evaluation.
* [Buolamwini, J., Gebru, T. (2018): Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification](https://proceedings.mlr.press/v81/buolamwini18a.html) - FAT* 2018; the study that made per-group reporting standard.
* [Barocas, S., Hardt, M., Narayanan, A. (2023): Fairness and Machine Learning](https://fairmlbook.org) - MIT Press; the chapter on evaluation and the case for looking at group-level error.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
