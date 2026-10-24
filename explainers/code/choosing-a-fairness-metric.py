# Generated from explainers/choosing-a-fairness-metric.md by scripts/build_explainers.py. Edit the markdown, not this file.
import pandas as pd

results = pd.read_csv("results/results_fairness.csv")      # the benchmark's current results
base = results[(results.strategy == "unawareness_proxy_removal") & (results.model == "logistic_regression")
               & ~results.protected_attribute.str.contains("_x_")]
rows = []
for (audit, attribute), group in base.groupby(["audit", "protected_attribute"]):
    flagged = sorted(group[group.significant == True].metric)    # noqa: E712 - a boolean column
    rows.append((audit, attribute, len(flagged), len(group), ", ".join(m.replace("_diff", "") for m in flagged)))

print(f"{'audit':24s} {'attribute':16s} flagged  which metrics are significant")
for audit, attribute, n_flagged, n_total, which in rows:
    print(f"{audit:24s} {attribute:16s} {n_flagged}/{n_total}     {which or '-'}")

split = [r for r in rows if 0 < r[2] < r[3]]
print(f"\n{len(split)} of {len(rows)} audit/attribute pairs have some metrics significant and others not.")
