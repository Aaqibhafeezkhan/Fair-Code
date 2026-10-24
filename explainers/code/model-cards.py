# Generated from explainers/model-cards.md by scripts/build_explainers.py. Edit the markdown, not this file.
import pandas as pd

AUDIT, STRATEGY, MODEL = "benefits_denial", "unawareness_proxy_removal", "logistic_regression"
fair = pd.read_csv("results/results_fairness.csv")          # the benchmark's current results
perf = pd.read_csv("results/results_performance.csv")
f = fair[(fair.audit == AUDIT) & (fair.strategy == STRATEGY) & (fair.model == MODEL)]
p = perf[(perf.audit == AUDIT) & (perf.strategy == STRATEGY) & (perf.model == MODEL)].set_index("metric")

print(f"# Model card: {MODEL} on {AUDIT} (strategy {STRATEGY})\n")
print("## Intended use\nA benchmark model that predicts whether income is above $50K, framed as a means-test proxy for benefits")
print("eligibility. It is a teaching and audit artefact, not for deciding real applications.\n")
print("## Training data\nUCI Adult Census Income, features: the audit's core columns only (no protected attribute, no proxy columns).\n")
print("## Overall performance (test split, 95% bootstrap interval)")
for metric in ("accuracy", "auc", "f1"):
    r = p.loc[metric]
    ci = "" if pd.isna(r.ci_low) else f" [{r.ci_low:.3f}, {r.ci_high:.3f}]"
    print(f"- {metric}: {r.value:.3f}{ci}")
print("\n## Per-group results (gap = disadvantaged minus advantaged)")
for attribute, g in f[~f.protected_attribute.str.contains("_x_")].groupby("protected_attribute"):
    parity = g[g.metric == "demographic_parity_diff"].iloc[0]
    equal_opp = g[g.metric == "equal_opportunity_diff"].iloc[0]
    flag = "significant" if parity.significant else "not significant"
    print(f"- {attribute}: demographic parity gap {parity.value:+.3f} ({flag}, n={int(parity.n_disadvantaged)} vs {int(parity.n_advantaged)}), "
          f"equal opportunity gap {equal_opp.value:+.3f}")
print("\n## Known limits\n- Gaps are measured on one test split; see the intervals and p-values in results/results_fairness.csv.")
print("- Groups smaller than 30 test rows carry small_sample_warning in the results file.")
