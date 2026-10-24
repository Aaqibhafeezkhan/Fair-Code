> *A label that records who came back to the hospital records who could come back. The check is cheap, and on this data it comes out clean.*

## The One-Sentence Definition

**Utilization as a label** means training a model on a target that counts use of the healthcare system (visits, admissions, cost, readmission) as if it measured need, which makes the label depend on access as well as on illness.

## Why It Matters

The best-known case is cost: [The Obermeyer Case](obermeyer-cost-proxy.md) found that a widely used algorithm gave Black patients lower risk scores at the same level of illness because it was trained to predict spending, and spending is lower where access is lower. Readmission is a softer version of the same label. A patient with no regular care, no transport or a hospital that is hard to reach is less likely to be recorded as returning, however sick.

The test for the problem is not complicated. Compare the label across groups at the same measured level of illness. If the label is lower for one group at the same burden, it is recording access. If it is the same, the label may be doing its job. This explainer runs that test on the benchmark's hospital data and reports what it finds, including that it finds little.

## Core Concept: Compare at the Same Burden

1. **Choose a measure of need that does not depend on access to the specific service.** Here, the number of diagnoses recorded for the stay is the closest column, with the caveat that diagnoses are themselves recorded when care happens.
2. **Stratify on it** and compare the label rate between groups in each stratum.
3. **Average the gaps** with weights for how common each stratum is.
4. **Look at the raw use measures** (visits in the previous year) for the same groups, since a gap there is the mechanism.

A material gap at the same burden is the signature of an access-driven label.

## Concrete Example: Healthcare Readmission

[Healthcare Readmission](../Healthcare%20Readmission/): about 99,500 patients with diabetes, the target is readmission within 30 days, the groups are African American patients against everyone else, and the burden is the number of diagnoses recorded for the stay.

```
what the label records is a hospital return, which needs access to a hospital
readmitted within 30 days: African American patients 0.112, others 0.112

readmission rate by recorded diagnoses (columns: others, African American)
black       0      1
burden              
1-5     0.081  0.097
6-7     0.107  0.106
8       0.122  0.103
9       0.124  0.127
10+     0.152  0.125
patients per cell
black       0     1
burden             
1-5     14929  5121
6-7     16058  4143
8        8573  1867
9       40623  8071
10+        99     8

gap at the same burden, weighted by how common each burden is: +0.0027

visits in the year before: African American patients 1.20, others 1.22
```

On this data the label does not show an access gap. The overall readmission rate is 0.112 for both groups, the gaps within a burden stratum are small and go both ways (+0.016 at 1 to 5 diagnoses, -0.019 at 8), the weighted gap is +0.0027, and visits in the previous year are close (1.20 against 1.22). The 10 or more diagnoses stratum has only 8 African American patients, so its rate says nothing.

That is a finding. It says that for this label the Obermeyer mechanism is not visible in the columns available; it does not say the label is a good measure of need, only that the same-burden comparison is clean. A label based on cost or on total visits could still fail, and the check above should be repeated for any label before it is used.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/utilization-as-a-label.py`; that file is generated from the block below). `data.df` holds the raw rows, so columns the harness does not encode can still be used to define strata.

```python
import numpy as np
import pandas as pd
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Healthcare Readmission")
df = data.df.copy()
df["readmitted_30"] = data.y.to_numpy()
df["black"] = (df["race"] == "AfricanAmerican").astype(int)

print("what the label records is a hospital return, which needs access to a hospital")
print(f"readmitted within 30 days: African American patients {df[df.black == 1].readmitted_30.mean():.3f}, "
      f"others {df[df.black == 0].readmitted_30.mean():.3f}")

# Compare patients with the same illness burden: the same number of recorded diagnoses.
df["burden"] = pd.cut(df["number_diagnoses"], [0, 5, 7, 8, 9, 20], labels=["1-5", "6-7", "8", "9", "10+"])
table = df.groupby(["burden", "black"], observed=True).readmitted_30.agg(["mean", "size"]).unstack("black")
print("\nreadmission rate by recorded diagnoses (columns: others, African American)")
print(table["mean"].round(3).to_string())
print("patients per cell")
print(table["size"].to_string())

weights = df.groupby("burden", observed=True).size() / len(df)
rate = df.groupby(["burden", "black"], observed=True).readmitted_30.mean().unstack("black")
print(f"\ngap at the same burden, weighted by how common each burden is: {(weights * (rate[1] - rate[0])).sum():+.4f}")

# What a utilization label would see: visits in the previous year.
prior = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
print(f"\nvisits in the year before: African American patients {prior[df.black == 1].mean():.2f}, others {prior[df.black == 0].mean():.2f}")
```

## Limitations

1. **The burden measure is itself recorded through care.** A patient with less access may have fewer diagnoses recorded, which would make the strata look healthier than they are and hide a gap.
2. **Only one label was checked.** Readmission within 30 days is not cost, and the result does not transfer to other labels.
3. **Strata are coarse.** Number of diagnoses is a blunt proxy for severity. A finer severity measure could show a gap this one does not.
4. **Small cells.** The top stratum has 8 patients in one group, and the other cells have a few thousand, so differences below a point are within noise. See [Disaggregated Evaluation](disaggregated-evaluation.md).
5. **A clean check is not a proof.** It rules out one mechanism in one measure.

## Related Concepts

* [The Obermeyer Case](obermeyer-cost-proxy.md) - the cost label that failed this test
* [Label Bias](label-bias.md) - labels that carry historical decisions
* [Underdiagnosis Bias](underdiagnosis-bias.md) - unequal recorded disease
* [Missing Data as Bias in EHRs](missing-data-bias-ehr.md) - unequal access showing up as unequal records
* [Measurement Bias](measurement-bias.md) - a quantity recorded differently by group

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit used for the example above

## Further Reading

* [Obermeyer, Z. et al. (2019): Dissecting racial bias in an algorithm used to manage the health of populations](https://www.science.org/doi/10.1126/science.aax2342) - Science 366; the cost-as-need label and how the same-illness comparison exposed it.
* [Jacobs, A. Z., Wallach, H. (2021): Measurement and Fairness](https://arxiv.org/abs/1912.05511) - FAccT 2021; a framework for when a label measures what it claims.
* [Strack, B. et al. (2014): Impact of HbA1c Measurement on Hospital Readmission Rates](https://doi.org/10.1155/2014/781670) - BioMed Research International; the paper that describes this dataset.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
