> *There is no best fairness metric. There is the metric that measures the harm you are trying to prevent, and the others that will disagree with it.*

## The One-Sentence Definition

**Choosing a fairness metric** is deciding which group-level comparison matches the harm at stake and the standard you are held to, before looking at any numbers.

## Why It Matters

The metrics in this repo answer different questions: [Demographic Parity](demographic-parity.md) asks whether groups are selected at the same rate, [Equal Opportunity](equal-opportunity.md) whether qualified people are accepted at the same rate, [Predictive Parity](predictive-parity.md) whether a positive means the same thing for everyone. They are not interchangeable and, outside special cases, they cannot all hold at once ([Fairness Metric Conflicts](fairness-metric-conflicts.md)).

That makes the choice consequential. Pick afterwards and you can find a metric that says what you hoped. The benchmark's own results show how often they diverge.

## Core Concept: Start From the Harm

Work backwards from what goes wrong for a person:

| The harm | Who it hits | The question | Start with |
|----------|-------------|--------------|------------|
| Not selected when you should have been | Qualified people in one group | Are qualified people accepted at the same rate? | [Equal Opportunity](equal-opportunity.md) |
| Wrongly flagged or denied | People who would have been fine | Are false flags equally common? | False positive rate, in [Equalized Odds](equalized-odds.md) |
| Unequal access to a benefit | Everyone in a group | Are groups selected at the same rate? | [Demographic Parity](demographic-parity.md) and [Disparate Impact](disparate-impact.md) |
| A score that means different things | People relying on the score | Does a positive predict the outcome equally? | [Predictive Parity](predictive-parity.md) and [Calibration](calibration.md) |
| One group is simply worse served | The worst-off group | How well off is the worst group? | [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) |

Then check the standard you are held to (a legal threshold such as the 80% rule, a regulator's named metric), and what the label means: if the label is itself biased ([Label Bias](label-bias.md)), the error-rate metrics inherit it and demographic parity may be the safer reading.

## Concrete Example: The Benchmark's Own Results

For every audit and protected attribute, take the benchmark's S2 model (logistic regression, no protected column, no proxies) in `results/` (the current results) and list which of the six metrics are significant. If the metrics agreed, each row would be all or nothing.

```
audit                    attribute        flagged  which metrics are significant
ai_fair_recruitment      gender           1/6     accuracy_equality
benefits_denial          age              3/6     accuracy_equality, demographic_parity, disparate_impact_ratio
benefits_denial          national_origin  1/6     accuracy_equality
benefits_denial          race             3/6     accuracy_equality, demographic_parity, disparate_impact_ratio
benefits_denial          sex              4/6     accuracy_equality, demographic_parity, disparate_impact_ratio, predictive_parity
compas                   race             6/6     accuracy_equality, demographic_parity, disparate_impact_ratio, equal_opportunity, equalized_odds, predictive_parity
german_credit_lending    age              0/6     -
healthcare_readmission   age              3/6     accuracy_equality, demographic_parity, disparate_impact_ratio
healthcare_readmission   gender           3/6     accuracy_equality, demographic_parity, disparate_impact_ratio
healthcare_readmission   race             1/6     accuracy_equality
insurance_denial         age              0/6     -
insurance_denial         gender           0/6     -
tenant_screening         race             4/6     demographic_parity, disparate_impact_ratio, equal_opportunity, equalized_odds

9 of 13 audit/attribute pairs have some metrics significant and others not.
```

They mostly do not agree. COMPAS fails all six. German Credit and Insurance Denial fail none. The other 9 of 13 pairs fail some and pass others: Benefits Denial by sex fails demographic parity, disparate impact, accuracy equality and predictive parity but not equal opportunity or equalized odds, while Tenant Screening by race is the reverse for predictive parity and accuracy equality. The same model, the same audit, and the verdict depends on the metric.

Three worked paths from the repo's audits:

- **Tenant Screening (a flag leads to refusal):** the harm is a wrongful flag, so start with the false positive rate and equalized odds, then check demographic parity.
- **Benefits Denial (a means test):** the harm is wrongly excluding eligible people, so start with equal opportunity.
- **COMPAS (a risk score shown to a judge):** the harm is a score that means different things by group, so start with predictive parity and calibration, then accept that error-rate parity cannot also hold when base rates differ.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/choosing-a-fairness-metric.py`; that file is generated from the block below). It reads the benchmark's current results; change `strategy` or `model` to look at another.

```python
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
```

## Limitations

1. **"Significant" depends on sample size.** A metric can look fine because the group is small, see [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md).
2. **The table assumes the label is right.** If the label is biased, every error-rate row inherits it.
3. **One metric is rarely enough.** Reporting the chosen metric plus the ones that disagree is more honest than reporting one.
4. **Legal standards decide some choices.** Where a regulator names a metric, the choice is made for you.
5. **The conflicts are mathematical.** When base rates differ, no model can satisfy calibration and equal error rates together, see [Fairness Metric Conflicts](fairness-metric-conflicts.md).

## Related Concepts

* [Fairness Metric Conflicts](fairness-metric-conflicts.md) - why you must choose
* [Demographic Parity](demographic-parity.md), [Equal Opportunity](equal-opportunity.md), [Equalized Odds](equalized-odds.md) - the main candidates
* [Predictive Parity](predictive-parity.md) and [Calibration](calibration.md) - the score-meaning family
* [Max-Min (Rawlsian) Fairness](maxmin-fairness.md) - a different kind of answer
* [Disaggregated Evaluation](disaggregated-evaluation.md) - look at the per-group numbers before choosing

## Related Projects in This Repo

* [`results/`](../results/) - the benchmark rows used above
* [Benchmark Results Dashboard](../benchmark.html) - filter the same table interactively

## Further Reading

* [Chouldechova, A. (2017): Fair Prediction with Disparate Impact](https://arxiv.org/abs/1610.07524) - Big Data 5(2); why equal error rates and equal predictive value conflict when base rates differ.
* [Kleinberg, J., Mullainathan, S., Raghavan, M. (2016): Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807) - ITCS 2017; the impossibility result for calibration and balance.
* [Verma, S., Rubin, J. (2018): Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf) - FairWare 2018; a catalogue of the definitions with a worked example for each.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
