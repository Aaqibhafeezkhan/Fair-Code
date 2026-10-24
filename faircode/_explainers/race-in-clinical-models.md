> *Whether a clinical model should see a patient's race is argued as if it were a statistical question. On real data it is often a values question with a small statistical effect.*

## The One-Sentence Definition

The **race-in-clinical-models question** is whether a prediction model for care should take race (or an ethnicity field) as an input, given that race is a social category that can stand in for differences in access and exposure but is not itself a biological cause.

## Why It Matters

Several clinical formulas used a race term for years, and [Race Correction in Clinical Algorithms](race-correction-clinical-algorithms.md) describes how that delayed care for some patients. The same question now arises for every machine-learning risk model trained on hospital data: leave race in and the model can be checked for calibration by group and can correct for real differences in data quality; take it out and the model cannot use it, but the information often leaks back through proxies ([Proxy Variables](proxy-variables.md)).

Two arguments are common and both are partly empirical. For including it: a model that ignores a group variable can be systematically off for that group, and you cannot see it without the variable. Against: using race as a predictor can hard-code a historical disparity into a score that allocates care. This explainer runs the comparison on the benchmark's hospital data so the empirical half has numbers.

## Core Concept: Three Things to Compare

1. **Accuracy:** does the model with race predict better overall?
2. **Calibration by group:** does the predicted risk match the observed rate in each group, with and without race?
3. **Allocation:** if the score picks who gets a limited resource, how does the share of places and the share of the truly high-risk who get one change?

Only the third is what a patient experiences, and it needs a rule for how many places there are. The example uses the top 10% of scores.

## Concrete Example: Healthcare Readmission

[Healthcare Readmission](../Healthcare%20Readmission/): 30-day readmission after a hospital stay with diabetes, about 99,500 patients, logistic regression on the audit's core and proxy columns, the benchmark's split. One model also receives race, gender and age; the other receives none of them.

```
with race in the model
  AUC 0.635
  African American: mean predicted risk 0.115 vs observed 0.103 | given a place 0.114 | of those who were readmitted, given a place 0.213
  other patients : mean predicted risk 0.112 vs observed 0.114 | given a place 0.097 | of those who were readmitted, given a place 0.203
without race (and without gender and age)
  AUC 0.633
  African American: mean predicted risk 0.114 vs observed 0.103 | given a place 0.111 | of those who were readmitted, given a place 0.213
  other patients : mean predicted risk 0.112 vs observed 0.114 | given a place 0.097 | of those who were readmitted, given a place 0.205
```

- **Accuracy barely moves:** AUC 0.635 with race and 0.633 without. For this model race adds almost no predictive information beyond the clinical columns.
- **Calibration differs by group either way:** African American patients have a mean predicted risk of 0.115 and an observed rate of 0.103, so the model over-predicts for them by about 1.2 points with race and 1.1 without; for other patients predicted 0.112 against observed 0.114. Removing race does not fix it, because the over-prediction comes from the other columns.
- **Allocation shifts by under a point:** the share of African American patients given a place is 11.4% with race and 11.1% without, against 9.7% for others. Of the patients who really were readmitted, 21.3% of African American patients and 20.3% of others receive a place with race, 21.3% and 20.5% without.

So on this model the debate is not a trade between accuracy and fairness: the variable carries almost no signal, and the group-level calibration gap exists with or without it. A model where race carried real signal would tell a different story, and the check above is how to find out.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/race-in-clinical-models.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

with_race = load_audit("Healthcare Readmission", strategy="baseline")      # core + proxy + race, gender and age
without = load_audit("Healthcare Readmission", strategy="unawareness")     # the same without the protected columns
train, test = with_race.idx_train, with_race.idx_test
y = with_race.y.to_numpy()
black = (with_race.df["race"] == "AfricanAmerican").to_numpy().astype(int)[test]

def fit(data):
    X = data.X.to_numpy(dtype=float)
    Z = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
    model = LogisticRegression(max_iter=2000).fit(Z[train], y[train])
    return model.predict_proba(Z[test])[:, 1]

def report(name, p):
    y_te = y[test]
    # Use the score to pick the 10% of patients for a follow-up programme with a fixed number of places.
    chosen = p >= np.quantile(p, 0.90)
    print(f"{name}")
    print(f"  AUC {roc_auc_score(y_te, p):.3f}")
    for g, label in ((1, "African American"), (0, "other patients ")):
        rows = black == g
        print(f"  {label}: mean predicted risk {p[rows].mean():.3f} vs observed {y_te[rows].mean():.3f} "
              f"| given a place {chosen[rows].mean():.3f} | of those who were readmitted, given a place {chosen[rows & (y_te == 1)].mean():.3f}")

report("with race in the model", fit(with_race))
report("without race (and without gender and age)", fit(without))
```

## Limitations

1. **One dataset, one outcome.** Readmission is a weak signal here (AUC about 0.63), which is why race adds little. A model with stronger signal can behave very differently.
2. **The "other patients" group is mixed.** It combines several groups, and a gap for one group can be hidden inside it. Report each group you have enough rows for.
3. **A small allocation shift is not a small harm.** At a few thousand patients per percentage point, the number of people affected can still be large.
4. **Removing race does not remove it.** The model without race still sees the proxy columns, and its calibration gap is unchanged.
5. **Whether to include it is not settled by this table.** The choice also turns on how the score will be used and on what the clinical and ethical consensus is, see [Race Correction in Clinical Algorithms](race-correction-clinical-algorithms.md).

## Related Concepts

* [Race Correction in Clinical Algorithms](race-correction-clinical-algorithms.md) - formulas that took race as an explicit input
* [Proxy Variables](proxy-variables.md) - why removing the column does not remove the information
* [Clinical Score Miscalibration](clinical-score-miscalibration.md) - calibration by group in detail
* [Underdiagnosis Bias](underdiagnosis-bias.md) - how the labels themselves can differ by group
* [Fairness Through Unawareness](fairness-through-unawareness.md) - the general case for dropping an attribute

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit used for the example above

## Further Reading

* [Vyas, D. A., Eisenstein, L. G., Jones, D. S. (2020): Hidden in Plain Sight - Reconsidering the Use of Race Correction in Clinical Algorithms](https://www.nejm.org/doi/full/10.1056/NEJMms2004740) - New England Journal of Medicine 383; the review of race terms across specialities.
* [Obermeyer, Z. et al. (2019): Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342) - Science 366; a score that allocated care and its group-level calibration.
* Delgado, C. et al. (2022). A Unifying Approach for GFR Estimation: Recommendations of the NKF-ASN Task Force on Reassessing the Inclusion of Race in Diagnosing Kidney Disease. American Journal of Kidney Diseases 79(2) - the recommendation to remove the race term from eGFR.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
