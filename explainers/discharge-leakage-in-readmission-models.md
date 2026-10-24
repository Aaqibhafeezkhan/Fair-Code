> *Some columns are only known once the outcome is nearly decided. A model that uses them looks better than it will be on the day it is needed.*

## The One-Sentence Definition

**Discharge leakage** is the use, in a readmission model, of columns recorded at or after discharge (where the patient went, whether they died) that would not be known when the model is meant to make its prediction, and that carry information about the outcome.

## Why It Matters

[Data Leakage](data-leakage.md) explains the general problem: information from the future inflates evaluation scores. A readmission model is typically run at or before discharge to decide who gets follow-up, so what counts as "available at prediction time" depends on when it runs. The discharge disposition (home, nursing facility, hospice, expired) is known at the end of the stay, which is exactly the moment such a model is used, so it may be fair to use. A patient who died during the stay, though, cannot be readmitted, and their row is a built-in negative.

The fairness side is quieter. Rows for patients who died or went to hospice are not spread equally across groups, and they lower the measured base rate and shift every per-group number. Both problems are easy to check.

## Core Concept: Ask Two Questions of Each Column

1. **When is it known?** If it is not available when the model will run, it leaks. If it is, it is allowed but it is carrying information from the end of the stay.
2. **What does it do to the rows?** Some values (expired, hospice) change who can have the outcome at all. Those rows belong in the training data only if the deployed model will see them.

## Concrete Example: Healthcare Readmission

[Healthcare Readmission](../Healthcare%20Readmission/): about 99,500 patients with diabetes, 30-day readmission, a gradient-boosted model on the audit's core and proxy columns (no race, gender or age), the benchmark's split. The audit's two row filters remove unknown race and gender and do not remove patients who died or went to hospice. The model flags the top 11% of scores, about the base rate.

```
rows 99,492; died or went to hospice 2,384 (2.4%); readmitted within 30 days: 1.8% of them, 11.5% of everyone else
share of African American patients among them: 18.4% (all rows: 19.3%)

all rows, with discharge disposition                 AUC 0.671 | flagged rate: African American 0.126, others 0.106, gap +0.0194
all rows, without discharge disposition              AUC 0.640 | flagged rate: African American 0.129, others 0.105, gap +0.0239
patients who left alive only, with disposition       AUC 0.663 | flagged rate: African American 0.125, others 0.106, gap +0.0190
patients who left alive only, without disposition    AUC 0.639 | flagged rate: African American 0.131, others 0.105, gap +0.0259
```

- **2,384 rows (2.4%) are end-of-life discharges.** Only 1.8% of them were readmitted against 11.5% of everyone else. They are in the data as ordinary "not readmitted" examples.
- **The disposition column helps a lot:** AUC 0.671 with it and 0.640 without. Dropping the end-of-life rows removes only part of that, leaving 0.663 against 0.639, so most of the lift comes from other dispositions (for example discharge to a skilled nursing facility, where readmission is more common).
- **The group gap moves a little:** the flagged-rate gap for African American patients is +0.0194 with the column and +0.0239 without, and it is +0.0190 and +0.0259 on the rows that left alive. Dropping the column makes the gap slightly wider here.
- **The rows are not spread evenly:** African American patients are 18.4% of the end-of-life rows against 19.3% of all rows, a small difference.

If the model is meant to run on admission, the disposition column is a leak and the 0.671 is not achievable. If it runs at discharge, the column is legitimate and the 11.5% base rate that excludes end-of-life patients is the one to use.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/discharge-leakage-in-readmission-models.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Healthcare Readmission", strategy="unawareness")   # core + proxy columns, no race, gender or age
train, test = data.idx_train, data.idx_test
y = data.y.to_numpy()
black = (data.df["race"] == "AfricanAmerican").to_numpy().astype(int)
disposition = data.df["discharge_disposition_id"].to_numpy()
END_OF_LIFE = {11, 13, 14, 19, 20, 21}          # expired, or discharged to hospice: readmission within 30 days is very unlikely

dead = np.isin(disposition, list(END_OF_LIFE))
print(f"rows {len(y):,}; died or went to hospice {dead.sum():,} ({dead.mean():.1%}); readmitted within 30 days: "
      f"{y[dead].mean():.1%} of them, {y[~dead].mean():.1%} of everyone else")
print(f"share of African American patients among them: {black[dead].mean():.1%} (all rows: {black.mean():.1%})")

def report(name, columns, rows_train, rows_test):
    X = data.X[columns]
    model = HistGradientBoostingClassifier(random_state=data.random_state).fit(X.iloc[rows_train], y[rows_train])
    p = model.predict_proba(X.iloc[rows_test])[:, 1]
    flagged = p >= np.quantile(p, 0.89)              # flag the top 11%, about the base rate
    b = black[rows_test]
    print(f"{name:52s} AUC {roc_auc_score(y[rows_test], p):.3f} | flagged rate: African American {flagged[b == 1].mean():.3f}, "
          f"others {flagged[b == 0].mean():.3f}, gap {flagged[b == 1].mean() - flagged[b == 0].mean():+.4f}")

everything = list(data.X.columns)
without = [c for c in everything if c != "discharge_disposition_id"]
alive_train, alive_test = train[~dead[train]], test[~dead[test]]
print()
report("all rows, with discharge disposition", everything, train, test)
report("all rows, without discharge disposition", without, train, test)
report("patients who left alive only, with disposition", everything, alive_train, alive_test)
report("patients who left alive only, without disposition", without, alive_train, alive_test)
```

## Limitations

1. **The "when is it known" question has no answer in the data.** It depends on how the model will be deployed, so the same column is a leak in one design and a feature in another.
2. **Disposition codes are coarse.** The grouping into end-of-life codes here is a choice, and other codes (such as left against medical advice) have their own effects.
3. **The fairness effect is small on this data.** The flagged-rate gap moved by a few tenths of a point. In a dataset where groups differ more in discharge destination, it could move more.
4. **Dropping the column hides, not fixes.** Without the disposition, the model leans on other columns that carry part of it.
5. **Excluding rows changes the question.** A model trained and tested only on patients who left alive answers a different question from one that must also score the patients who will not.

## Related Concepts

* [Data Leakage](data-leakage.md) - the general problem of information from the future
* [Survivorship Bias](survivorship-bias.md) - when the rows that remain are the ones that passed a filter
* [Missing Data as Bias in EHRs](missing-data-bias-ehr.md) - how recording depends on care
* [Disaggregated Evaluation](disaggregated-evaluation.md) - how to look at the per-group effect of a design choice
* [Model Cards](model-cards.md) - where the intended time of use should be written down

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit used for the example above

## Further Reading

* [Strack, B. et al. (2014): Impact of HbA1c Measurement on Hospital Readmission Rates](https://doi.org/10.1155/2014/781670) - BioMed Research International; the paper that describes this dataset and its variables.
* [Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. (2012): Leakage in Data Mining: Formulation, Detection, and Avoidance](https://doi.org/10.1145/2382577.2382579) - ACM Transactions on Knowledge Discovery from Data 6(4); a taxonomy of leakage.
* [Wiens, J. et al. (2019): Do No Harm: A Roadmap for Responsible Machine Learning for Health Care](https://doi.org/10.1038/s41591-019-0548-6) - Nature Medicine 25; on defining the time of use for a clinical model.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
