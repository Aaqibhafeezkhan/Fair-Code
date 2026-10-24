> *A feature is a measurement, not the thing itself. When one group is measured worse than another, a gap appears that the true values do not contain.*

## The One-Sentence Definition

**Measurement bias** occurs when a feature or label is a noisier, or systematically different, measurement of the quantity you care about for one group than for another.

## Why It Matters

Arrest counts stand in for crime, cost stands in for health need ([The Obermeyer Case](obermeyer-cost-proxy.md)), a pulse oximeter reading stands in for blood oxygen. Each is a measurement, and measurements can be worse for some people: less care means fewer recorded diagnoses, a foreign degree may be credited at a lower level, an instrument may be calibrated on one skin tone.

A model trained on the measured values learns the measurement error as if it were signal. It is easy to confuse with two neighbours. [Label Bias](label-bias.md) is a biased target, a decision made by people. [Proxy Variables](proxy-variables.md) are real features that happen to correlate with a group. Measurement bias is a *distortion of a feature that is meant to be neutral*, and it is invisible in the data: the measured column looks fine.

## Core Concept: Systematic Error and Noise Are Different Problems

Two kinds of error behave differently:

1. **Systematic error (bias).** Every value for one group is shifted: recorded lower, rounded down, undercounted. The measured feature now separates the groups, and a model uses it to do so. The gap grows.
2. **Random error (noise).** Values for one group are scattered around the truth. The feature becomes less informative for that group, the model leans on it less for them, and the measured gap often *shrinks* while accuracy falls.

The first creates unfairness the true data does not contain. The second hides the true gap and degrades the group's predictions. Neither shows up as a missing value.

## Concrete Example: Benefits Denial

This is a simulation, because the real measurement process is not in the data: we inject mis-measurement into one column for women only and watch what changes. [Benefits Denial](../Benefits%20Denial/), logistic regression on the audit's core columns, the benchmark's split. The predicted-positive flag means "income above $50K, so ineligible".

```
true values                                    accuracy 0.811 | positive rate women 0.059 men 0.109 | parity gap -0.0509
education.num credited 15% low for women       accuracy 0.822 | positive rate women 0.040 men 0.149 | parity gap -0.1089
education.num noisy for women (sd 1.0x)        accuracy 0.799 | positive rate women 0.075 men 0.088 | parity gap -0.0135
capital.gain recorded 50% low for women        accuracy 0.812 | positive rate women 0.053 men 0.112 | parity gap -0.0592
```

- **True values:** accuracy 0.811 and a parity gap of -0.0509, the benchmark's S2 row in `results/` (the current results).
- **Crediting women 15% less schooling** (`education.num` scaled to 0.85) doubles the gap to -0.1089 and *raises* measured accuracy to 0.822. The error is systematic, so it encodes sex, and sex predicts the label in this data.
- **Noisy schooling for women** (noise with the column's own standard deviation) shrinks the gap to -0.0135 and lowers accuracy to 0.799. The model has stopped trusting the column, for women and for everyone.
- **Halving women's recorded capital gains** moves the gap only to -0.0592, because that column matters less to this model.

The point of the accuracy column: a model can look better on the recorded labels while it is worse for a group, because the labels never saw the true values.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/measurement-bias.py`; that file is generated from the block below). The function below applies a scale and/or noise to one column for one group.

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
y_tr, y_te = data.y.iloc[train].to_numpy(), data.y.iloc[test].to_numpy()
women = data.masks["sex"].to_numpy()
rng = np.random.default_rng(0)

def measured(column, scale=1.0, noise=0.0):
    """Return the feature matrix as if `column` were mis-measured for women only: multiplied by
    `scale` (systematic under-recording) and/or with Gaussian noise (sd = noise * the column's sd)."""
    X = data.X.copy()
    col = X[column].to_numpy(dtype=float)
    bad = col * scale + rng.normal(0, noise * col.std(), len(col))
    X[column] = np.where(women == 1, bad, col)
    return X

def report(label, X):
    model = build_model("logistic_regression", data.random_state).fit(X.iloc[train], y_tr)
    pred = model.predict(X.iloc[test])
    w = women[test]
    print(f"{label:46s} accuracy {(pred == y_te).mean():.3f} | positive rate women {pred[w == 1].mean():.3f} "
          f"men {pred[w == 0].mean():.3f} | parity gap {pred[w == 1].mean() - pred[w == 0].mean():+.4f}")

report("true values", data.X)
report("education.num credited 15% low for women", measured("education.num", scale=0.85))
report("education.num noisy for women (sd 1.0x)", measured("education.num", noise=1.0))
report("capital.gain recorded 50% low for women", measured("capital.gain", scale=0.5))
```

To look for the real thing, compare the distribution of a feature by group for something that should be equal (for example, the same procedure code recorded at different rates), and check whether the feature's relationship to the outcome differs by group.

## Limitations

1. **The data does not show it.** You only see the measured column. You need an independent source of truth, or domain knowledge about how each group is measured, to know it is happening.
2. **The simulation is a guess about the error.** A 15% shift is illustrative. Real errors are not a clean scale or clean Gaussian noise, and they differ by feature.
3. **Higher accuracy is not reassurance.** Systematic error made accuracy go up here. Accuracy against the recorded label cannot detect it.
4. **Noise and bias point opposite ways.** Because noise shrinks the gap, a small measured gap is not proof of a small true gap.
5. **Fixing the model does not fix the measurement.** Mitigations act on the model; the repair is in how the quantity is recorded, or in choosing a different quantity.

## Related Concepts

* [Label Bias](label-bias.md) - when the target, not a feature, is distorted
* [Proxy Variables](proxy-variables.md) - real features that correlate with a group, as opposed to distorted ones
* [The Obermeyer Case](obermeyer-cost-proxy.md) - cost as a mis-measurement of health need
* [Missing Data as Bias in EHRs](missing-data-bias-ehr.md) - unequal access to care showing up as unequal records
* [Race Correction in Clinical Algorithms](race-correction-clinical-algorithms.md) - a measurement adjustment with its own fairness cost

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the simulation
* [`results/`](../results/) - the benchmark row quoted for S2

## Further Reading

* [Suresh, H., Guttag, J. (2021): A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle](https://arxiv.org/abs/1901.10002) - EAAMO 2021; places measurement bias among the other sources of harm.
* [Jacobs, A. Z., Wallach, H. (2021): Measurement and Fairness](https://arxiv.org/abs/1912.05511) - FAccT 2021; treats fairness problems as mismatches between a construct and its measurement.
* [Obermeyer, Z. et al. (2019): Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342) - Science 366; the well-known case of a cost measurement standing in for need.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
