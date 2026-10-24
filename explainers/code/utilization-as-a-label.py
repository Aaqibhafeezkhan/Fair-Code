# Generated from explainers/utilization-as-a-label.md by scripts/build_explainers.py. Edit the markdown, not this file.
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
