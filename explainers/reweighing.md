> *Fix the training data before you train the model: give each (group, label) cell the weight it would have if group and label were independent.*

## The One-Sentence Definition

**Reweighing** is a pre-processing mitigation that assigns each training row a weight so that, in the weighted data, group membership and the label are statistically independent, without changing any feature or any label.

## Why It Matters

In a biased dataset, the disadvantaged group is under-represented among positive labels. A model trained on it learns that base-rate gap and reproduces it. Dropping the protected column ([Fairness Through Unawareness](fairness-through-unawareness.md)) does not fix it when proxies carry the signal, and in-processing methods such as `ExponentiatedGradient` need you to swap the training procedure.

Reweighing needs neither. It is one line before `fit`: pass `sample_weight`. Every estimator that accepts weights can use it, and the data itself is not edited, which is why it is popular when you cannot change the model or may not alter records.

## Core Concept: Weights From Expected Over Observed

For group `s` and label `y`, the weight is the count you would expect if the two were independent divided by the count you actually see:

```
weight(s, y) = P(s) * P(y) / P(s, y)
```

A cell that is over-represented gets weight below 1. An under-represented cell gets weight above 1. In the weighted data the positive rate is the same in both groups, so a model that only learns from the labels has no gap to copy. The method is from Kamiran and Calders (2012).

## Concrete Example: German Credit Lending

The [German Credit Lending](../German%20Credit%20Lending/) audit: positive is a good credit rating, the disadvantaged group is age under 30, logistic regression, core features only, benchmark split.

```
German Credit Lending - age
  weights per (group, label): {(0, 0): 1.176, (0, 1): 0.94, (1, 0): 0.802, (1, 1): 1.119}
  baseline   accuracy 0.700 | positive rate disadvantaged 0.710 advantaged 0.802 | parity gap -0.0914
  reweighing accuracy 0.695 | positive rate disadvantaged 0.826 advantaged 0.809 | parity gap +0.0169
Benefits Denial - sex
  weights per (group, label): {(0, 0): 1.094, (0, 1): 0.787, (1, 0): 0.852, (1, 1): 2.206}
  baseline   accuracy 0.811 | positive rate disadvantaged 0.059 advantaged 0.109 | parity gap -0.0509
  reweighing accuracy 0.811 | positive rate disadvantaged 0.059 advantaged 0.111 | parity gap -0.0523
```

Reweighing took the parity gap from -0.0914 to +0.0169 for 0.5 points of accuracy (0.700 to 0.695). The baseline row matches the benchmark's S2 strategy in `results/` (the current results): accuracy 0.700 and a gap of -0.0914. For comparison, the same file has S3 `in_processing` at a gap of +0.068 and accuracy 0.715, and S4 `post_processing` at +0.0467 and 0.720.

That is one data set of 1,000 rows. The test split is about 200 rows, and the benchmark's own p-value for the S2 gap is 0.17, so neither the baseline gap nor the improvement is statistically firm. See [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md).

## Where It Does Not Work

The second block in the output is the same method on [Benefits Denial](../Benefits%20Denial/), sex as the attribute. The weights are doing something (women with high income get 2.206x weight), but the parity gap does not move: -0.0509 before, -0.0523 after. The features still separate the groups, and the model keeps using them.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 reweighing.py`).

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

def reweighing_weights(s, y):
    """Kamiran and Calders: weight = P(group) * P(label) / P(group, label), per (group, label) cell."""
    w = np.empty(len(y))
    for g in (0, 1):
        for label in (0, 1):
            cell = (s == g) & (y == label)
            w[cell] = (s == g).mean() * (y == label).mean() / cell.mean()
    return w

def run(folder, attribute):
    manifest, X, y, masks, train, test = load_audit(folder)
    s_tr, s_te = masks[attribute][train], masks[attribute][test]
    y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
    w = reweighing_weights(s_tr, y_tr)
    print(folder, "-", attribute)
    print("  weights per (group, label):", {(g, l): round(float(w[(s_tr == g) & (y_tr == l)][0]), 3) for g in (0, 1) for l in (0, 1)})
    for name, weight in (("baseline", None), ("reweighing", w)):
        model = build_model("logistic_regression", manifest.random_state).fit(X.iloc[train], y_tr, sample_weight=weight)
        pred = model.predict(X.iloc[test])
        print(f"  {name:10s} accuracy {(pred == y_te).mean():.3f} | positive rate disadvantaged {pred[s_te == 1].mean():.3f} "
              f"advantaged {pred[s_te == 0].mean():.3f} | parity gap {pred[s_te == 1].mean() - pred[s_te == 0].mean():+.4f}")

run("German Credit Lending", "age")
run("Benefits Denial", "sex")
```

## Limitations

1. **It only touches the label-group association.** If other features separate the groups, as in Benefits Denial, the weights cannot stop the model from using them. Pair it with [proxy removal](proxy-variables.md).
2. **It can lift the disadvantaged group past parity.** The German Credit gap went from -0.0914 to +0.0169. Weights target the training distribution, not a chosen test-set gap.
3. **Extreme weights make fits unstable.** A cell with few rows and a large weight can dominate. The Benefits weight of 2.206 is mild, but a rare cell gets a much larger one.
4. **It targets demographic parity.** It makes no promise about equalized odds or calibration.
5. **Small data gives noisy estimates.** The weights are estimated from the same few rows, and the evaluation is on a small test set.

## Related Concepts

* [Mitigation Strategies](mitigation-strategies.md) - where pre-, in- and post-processing sit in the benchmark
* [Demographic Parity](demographic-parity.md) - the criterion reweighing is built to reach
* [Class Imbalance](class-imbalance.md) - the same weighting mechanics, applied to labels
* [Label Bias](label-bias.md) - why the observed label association exists
* [Fairness Through Unawareness](fairness-through-unawareness.md) - why dropping the column is not enough

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the audit used for the main example
* [`Benefits Denial/`](../Benefits%20Denial/) - the audit where it does not work
* [`results/`](../results/) - the benchmark rows quoted for S2, S3 and S4

## Further Reading

* [Kamiran, F., Calders, T. (2012): Data Preprocessing Techniques for Classification without Discrimination](https://link.springer.com/article/10.1007/s10115-011-0463-8) - Knowledge and Information Systems 33(1); the reweighing method.
* Calders, T., Kamiran, F., Pechenizkiy, M. (2009). Building Classifiers with Independency Constraints. IEEE ICDM Workshops - the earlier formulation of the independence goal.
* [Bellamy, R. et al. (2019): AI Fairness 360](https://arxiv.org/abs/1810.01943) - IBM Journal of Research and Development 63(4/5); an open-source toolkit that ships reweighing among its pre-processing methods.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
