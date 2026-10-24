> *An audit tells you how big the gap is. An impact assessment asks who is on the other side of it, and what happens to them.*

## The One-Sentence Definition

An **algorithmic impact assessment** is a structured review, done before an automated decision system is deployed and repeated afterwards, of who the system affects, what can go wrong for them, how that was measured, and what will be done about it.

## Why It Matters

Everything else in this repo is measurement: a gap, an interval, a p-value. A decision to deploy needs more than measurement. It needs a statement of who is affected, what the alternatives are, who can appeal, and who is accountable. Without a structure, the measurement gets handed over as a number and the judgement around it is skipped.

Impact assessments are that structure. Public bodies have used them for automated decisions, and several jurisdictions require some form of one for high-risk systems. The exact obligations differ by place and change quickly, so rely on the primary text for your own jurisdiction instead of a summary.

## Core Concept: The Measurement Is One Section

A typical assessment has parts that no audit script can supply:

1. **Purpose and alternatives:** what decision, why automate it, what was the non-automated process.
2. **Who is affected:** the people the decision touches, including those who never interact with the system.
3. **Measured disparity:** the audit results, with intervals and the metrics chosen and why ([Choosing a Fairness Metric](choosing-a-fairness-metric.md)).
4. **Mitigation:** what was tried and what it cost.
5. **Safeguards:** human review, appeal, notice, logging.
6. **Monitoring:** what is re-measured after deployment and when it triggers a review.

Section 3 is exactly what the benchmark produces, so it can be generated.

## Concrete Example: Tenant Screening

This is the measurement section of an assessment for the [Tenant Screening](../Tenant%20Screening/) audit, generated from the benchmark's `results/` (the current results) for the S2 logistic regression. Sections 1, 2 and 6 are written by a person; the script prints what it can compute and says what it cannot.

```
Impact assessment, measurement section: tenant_screening (logistic_regression, unawareness_proxy_removal)

1. Who is affected: rental applicants, grouped here by race (Black vs White). Intersections: not measured by this audit.
2. Decision: the model flags an applicant as likely to reoffend within three years; a flag leads to refusal in the framing of the audit.

3. Disparity measured on the held-out test split:
   demographic_parity_diff    +0.114  95% CI [+0.088, +0.138]  p=0.000  SIGNIFICANT
   disparate_impact_ratio     +1.186  95% CI [+1.140, +1.232]  p=0.000  SIGNIFICANT
   equal_opportunity_diff     +0.084  95% CI [+0.053, +0.116]  p=0.000  SIGNIFICANT
   equalized_odds_diff        +0.149  95% CI [+0.109, +0.193]  p=0.000  SIGNIFICANT
   predictive_parity_diff     -0.029  95% CI [-0.061, +0.002]  p=0.072  not significant
   accuracy_equality_diff     -0.012  95% CI [-0.039, +0.015]  p=0.399  not significant

4. Finding: 4 of 6 metrics are significant at 0.05: demographic_parity_diff, disparate_impact_ratio, equal_opportunity_diff, equalized_odds_diff.
5. Mitigation options measured by the benchmark (same audit, same model):
   baseline                     demographic parity gap +0.054  p=0.000
   in_processing                demographic parity gap +0.017  p=0.178
   post_processing              demographic parity gap -0.005  p=0.689
   unawareness                  demographic parity gap +0.119  p=0.000
   unawareness_proxy_removal    demographic parity gap +0.114  p=0.000

6. Not measured here, and needed before deployment: who reviews flags, an appeal route, the legal basis for using
   this data, and a repeat of this measurement after deployment.
```

Reading it as an assessor: four of the six metrics are significant, the flagged-rate gap is +0.114 with an interval of +0.088 to +0.138, and the mitigation table shows what the benchmark's strategies leave behind. Both in-processing (+0.017, p = 0.178) and post-processing (-0.005, p = 0.689) get the demographic parity gap to a level the test cannot distinguish from zero; removing the race column and the proxies alone does not (+0.114).

The last paragraph matters most: who reviews a flag, whether there is an appeal, the legal basis for using this data and when the measurement is repeated are not in any results file, and an assessment that leaves them blank has not been completed.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/algorithmic-impact-assessment.py`; that file is generated from the block below). It reads the benchmark's current results; change the three constants at the top to assess another audit.

```python
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
```

## Limitations

1. **It documents risk; it does not remove it.** A complete assessment can conclude that a system should not be deployed.
2. **Measurement is only one part.** A good fairness number can sit beside an unfair process with no appeal.
3. **The legal landscape moves.** Required content and trigger conditions change, so use the current primary source for your jurisdiction.
4. **It is a snapshot.** Without the monitoring step, the numbers describe the day of the audit, see [Model Drift](model-drift.md).
5. **Affected people are rarely in the room.** An assessment written only by the builders tends to miss harms they would not think of. Involving affected groups is a design choice, not a script.

## Related Concepts

* [Model Cards](model-cards.md) - the model-level document an assessment can cite
* [Datasheets for Datasets](datasheets-for-datasets.md) - the data-level document it can cite
* [Choosing a Fairness Metric](choosing-a-fairness-metric.md) - how section 3's metrics are picked
* [Disparate Impact](disparate-impact.md) - a legal test an assessment often has to address
* [Mitigation Strategies](mitigation-strategies.md) - the options for section 4

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit the assessment above is about
* [`results/`](../results/) - the numbers in its measurement section

## Further Reading

* Reisman, D., Schultz, J., Crawford, K., Whittaker, M. (2018). Algorithmic Impact Assessments: A Practical Framework for Public Agency Accountability. AI Now Institute - the framework for public agencies.
* Moss, E., Watkins, E. A., Singh, R., Elish, M. C., Metcalf, J. (2021). Assembling Accountability: Algorithmic Impact Assessment for the Public Interest. Data and Society - a comparison of impact-assessment regimes.
* [Raji, I. D. et al. (2020): Closing the AI Accountability Gap](https://arxiv.org/abs/2001.00973) - FAT* 2020; an end-to-end internal auditing framework.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
