# Generated from explainers/algorithmic-impact-assessment.md by scripts/build_explainers.py. Edit the markdown, not this file.
import pandas as pd

AUDIT, STRATEGY, MODEL = "tenant_screening", "unawareness_proxy_removal", "logistic_regression"
fair = pd.read_csv("results/results_fairness.csv")          # the benchmark's current results
g = fair[(fair.audit == AUDIT) & (fair.strategy == STRATEGY) & (fair.model == MODEL) & (fair.protected_attribute == "race")]

print(f"Impact assessment, measurement section: {AUDIT} ({MODEL}, {STRATEGY})\n")
print("1. Who is affected: rental applicants, grouped here by race (Black vs White). Intersections: not measured by this audit.")
print("2. Decision: the model flags an applicant as likely to reoffend within three years; a flag leads to refusal in the framing of the audit.\n")
print("3. Disparity measured on the held-out test split:")
for r in g.itertuples():
    verdict = "SIGNIFICANT" if r.significant else "not significant"
    print(f"   {r.metric:26s} {r.value:+.3f}  95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  p={r.p_value:.3f}  {verdict}")
flagged = g[g.significant == True]                                   # noqa: E712 - a boolean column
print(f"\n4. Finding: {len(flagged)} of {len(g)} metrics are significant at 0.05: " + ", ".join(sorted(flagged.metric)) + ".")
print("5. Mitigation options measured by the benchmark (same audit, same model):")
for strategy, rows in fair[(fair.audit == AUDIT) & (fair.model == MODEL) & (fair.protected_attribute == "race")
                           & (fair.metric == "demographic_parity_diff")].groupby("strategy"):
    r = rows.iloc[0]
    print(f"   {strategy:28s} demographic parity gap {r.value:+.3f}  p={r.p_value:.3f}")
print("\n6. Not measured here, and needed before deployment: who reviews flags, an appeal route, the legal basis for using")
print("   this data, and a repeat of this measurement after deployment.")
