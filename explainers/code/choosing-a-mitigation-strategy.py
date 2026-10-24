# Generated from explainers/choosing-a-mitigation-strategy.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd

fair = pd.read_csv("results/results_fairness.csv")          # the benchmark's current results
perf = pd.read_csv("results/results_performance.csv")
STRATEGIES = ["baseline", "unawareness", "unawareness_proxy_removal", "in_processing", "post_processing"]
LABEL = dict(zip(STRATEGIES, ["S0 baseline", "S1 unawareness", "S2 proxy removal", "S3 in-processing", "S4 post-processing"]))
NEEDS = {  # (group labels while training, group labels when predicting)
    "baseline": ("no (it uses the column as a feature)", "yes, as a feature"), "unawareness": ("no", "no"),
    "unawareness_proxy_removal": ("no", "no"), "in_processing": ("yes", "no"), "post_processing": ("yes", "yes")}

gap = fair[(fair.model == "logistic_regression") & (fair.metric == "demographic_parity_diff") & ~fair.protected_attribute.str.contains("_x_")]
acc = perf[(perf.model == "logistic_regression") & (perf.metric == "accuracy")].set_index(["audit", "strategy"]).value
pairs = gap.set_index(["audit", "protected_attribute", "strategy"])
rows = []
for strategy in STRATEGIES:
    changes, remaining, closed, significant, acc_change = [], [], 0, 0, []
    for (audit, attribute), g in gap.groupby(["audit", "protected_attribute"]):
        base = pairs.loc[(audit, attribute, "unawareness_proxy_removal")]
        row = pairs.loc[(audit, attribute, strategy)]
        remaining.append(abs(row.value)); changes.append(abs(row.value) - abs(base.value))
        significant += bool(row.significant); closed += abs(row.value) < abs(base.value)
        acc_change.append(acc[(audit, strategy)] - acc[(audit, "unawareness_proxy_removal")])
    rows.append((strategy, np.median(acc_change), np.median(remaining), closed, significant, len(changes)))

print(f"logistic regression, {len(rows) and rows[0][5]} audit/attribute pairs, compared with S2 (the same model without the protected column or proxies)")
print(f"{'strategy':20s} {'median accuracy change':>23} {'median |parity gap|':>20} {'gap smaller than S2':>20} {'still significant':>18}   needs group labels: training / prediction")
for strategy, d_acc, med_gap, closed, significant, n in rows:
    smaller = '-' if strategy == 'unawareness_proxy_removal' else f'{closed}/{n}'
    print(f"{LABEL[strategy]:20s} {d_acc:+23.3f} {med_gap:20.3f} {smaller:>20} {f'{significant}/{n}':>18}   {NEEDS[strategy][0]} / {NEEDS[strategy][1]}")
