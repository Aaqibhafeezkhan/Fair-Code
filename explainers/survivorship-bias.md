> *The data you have is the data that got through. If a filter chose who you see, the relationships you measure are the relationships among the survivors.*

## The One-Sentence Definition

**Survivorship bias** occurs when your data contains only the people who passed an earlier filter, so what you learn about the whole population from it is distorted.

## Why It Matters

The classic case is Abraham Wald's aircraft: engineers wanted armour where the returning planes were hit, and Wald saw that the planes missing from the sample were the ones hit elsewhere. The same logic runs through hiring and lending. You see job performance only for the people who were hired, repayment only for the people who were approved, retention only for the people who stayed.

A model trained on survivors has never seen the rejected, and it is tested on survivors too, so the damage is invisible in its metrics. The effect is not on rare edge cases: if a score decided who got through, the feature behind that score will look less predictive in the survivors than it is in everyone.

## Core Concept: Range Restriction and Selection

When a gate keeps people with a high value of some feature, two things follow in the survivors:

1. **Range restriction.** The gate's feature varies less, so its apparent link to the outcome weakens.
2. **A changed base rate.** The outcome is more common among survivors than in the population, so rates, gaps and calibration are all measured against a different baseline.

[Selection Bias](selection-bias.md) is the general family. Survivorship is the case where the filter is an earlier stage of the same process, and [Reject Inference](reject-inference.md) is the lending-specific attempt to correct for it.

## Concrete Example: AI Fair Recruitment

[AI Fair Recruitment](../AI%20Fair%20Recruitment/): the data has the hiring decision for every candidate, so the full picture is available for comparison. We then impose a gate: only candidates with a technical test score of 70 or more "reach" the stage we pretend to have data on. Logistic regression on every feature, standardized.

```
candidates 115309, passed the test gate 34695 (30.1%)
hire rate, everyone 0.259 | survivors only 0.473
gender gap in hire rate (women minus men), everyone -0.0292 | survivors only -0.0354

coefficient (per standard deviation), fitted on everyone vs on survivors only
  Experience_Years       +0.01   +0.01
  Technical_Test_Score   +0.97   +0.83
  Age                    +0.00   -0.00
  Gender                 +0.10   +0.08
```

- **The base rate nearly doubles.** The hire rate is 0.259 for everyone and 0.473 among the 30.1% who pass the gate.
- **The gap widens slightly.** The women-minus-men hire-rate gap is -0.0292 for everyone and -0.0354 among survivors.
- **The gate's own feature looks weaker.** The coefficient on `Technical_Test_Score` falls from +0.97 to +0.83 in survivors, the range-restriction effect. The `Gender` coefficient moves from +0.10 to +0.08.

The size of the distortion depends on how hard the gate is. A gate that keeps one in three is already enough to move the base rate and the effect of the gating feature.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/survivorship-bias.py`; that file is generated from the block below). The audit has the outcome for everyone, so we can compare against it; with real survivorship data you cannot, and the check is to ask what the gate was and model it.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("AI Fair Recruitment", strategy="baseline")   # every feature, Gender included
train, test = data.idx_train, data.idx_test
X = data.X
X_std = ((X - X.iloc[train].mean()) / (X.iloc[train].std() + 1e-9)).to_numpy(dtype=float)
y = data.y.to_numpy()
women = data.masks["gender"].to_numpy()
screen = X["Technical_Test_Score"].to_numpy(dtype=float)
columns = list(X.columns)

# The earlier gate: only candidates with a technical test score of 70 or more reach the stage we have data on.
survivor = screen >= 70
print(f"candidates {len(y)}, passed the test gate {survivor.sum()} ({survivor.mean():.1%})")
print(f"hire rate, everyone {y.mean():.3f} | survivors only {y[survivor].mean():.3f}")
print(f"gender gap in hire rate (women minus men), everyone {y[women == 1].mean() - y[women == 0].mean():+.4f} | "
      f"survivors only {y[survivor & (women == 1)].mean() - y[survivor & (women == 0)].mean():+.4f}")

def coefficients(rows):
    return LogisticRegression(max_iter=1000).fit(X_std[rows], y[rows]).coef_[0]

everyone, kept = coefficients(train), coefficients(train[survivor[train]])
print("\ncoefficient (per standard deviation), fitted on everyone vs on survivors only")
for i, col in enumerate(columns):
    print(f"  {col:22s} {everyone[i]:+.2f}   {kept[i]:+.2f}")
```

## Limitations

1. **You usually cannot see the comparison.** Here the full data exists; in practice the rejected have no outcome. The first defence is knowing the gate and its rule.
2. **The gate here is a clean threshold.** Real gates mix scores, judgement and chance, so the shift in base rates and coefficients is harder to predict.
3. **It does not always produce a larger gap.** The gap moved slightly here, and the sign and size depend on how the gate relates to the group.
4. **Reweighting helps only if the gate depends on observed features.** If the selection also depends on something unrecorded, no adjustment recovers the population.
5. **Evaluation inherits it.** Testing on survivors tells you about survivors. See [Disaggregated Evaluation](disaggregated-evaluation.md) for reporting results by group.

## Related Concepts

* [Selection Bias](selection-bias.md) - the general family of filtered samples
* [Reject Inference](reject-inference.md) - estimating outcomes for the applicants a lender turned away
* [Sampling Bias](sampling-bias.md) - when the sample is not drawn from the population at all
* [Label Bias](label-bias.md) - when the decision that produced the label was itself biased
* [Feedback Loop Bias](feedback-loop-bias.md) - when the filter decides what the next model sees

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above

## Further Reading

* Mangel, M., Samaniego, F. J. (1984). Abraham Wald's Work on Aircraft Survivability. Journal of the American Statistical Association 79(386) - the aircraft-armour analysis.
* Lakkaraju, H. et al. (2017). The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables. KDD - outcomes observed only for the people a decision-maker chose.
* [Suresh, H., Guttag, J. (2021): A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle](https://arxiv.org/abs/1901.10002) - EAAMO 2021; where selection fits among the sources of harm.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
