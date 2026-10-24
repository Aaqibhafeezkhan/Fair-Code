> *A model card does not make a model fair. It makes it possible to see, before anyone relies on it, who it was built for and who it fails.*

## The One-Sentence Definition

A **model card** is a short, standard document that states what a model is for, what data it was trained on, how it performs overall and for each group, and where it should not be used.

## Why It Matters

A fairness audit produces numbers, and numbers that live in a notebook do not reach the person deciding whether to deploy. A model card is the container that carries them: the same headings every time, so a reader can compare two models, and an explicit place for the uncomfortable results.

It also changes what gets measured. If the card has a per-group table, someone has to fill it in, and an empty row is visible. The card is a disclosure, not a guarantee, and a card that is only a list of reassurances is worse than none.

## Core Concept: Fixed Headings, Per-Group Numbers

The usual sections, from Mitchell and colleagues (2019):

1. **Model details:** what it is, who built it, which version.
2. **Intended use:** the tasks it is for, and the ones it is explicitly not for.
3. **Factors:** the groups and conditions performance was reported across.
4. **Metrics:** what was measured and why.
5. **Evaluation and training data:** where they came from and how they were split.
6. **Quantitative analyses:** the results, **disaggregated** by the factors above ([Disaggregated Evaluation](disaggregated-evaluation.md)).
7. **Ethical considerations and caveats:** what is known to be wrong or unmeasured.

The strongest habit is generating the numeric sections from the audit's output instead of typing them, so the card cannot disagree with the results.

## Concrete Example: Benefits Denial

This card is generated, not hand-written: the numbers are read straight from the benchmark's `results/` (the current results) for the S2 logistic regression on [Benefits Denial](../Benefits%20Denial/).

```
# Model card: logistic_regression on benefits_denial (strategy unawareness_proxy_removal)

## Intended use
A benchmark model that predicts whether income is above $50K, framed as a means-test proxy for benefits
eligibility. It is a teaching and audit artefact, not for deciding real applications.

## Training data
UCI Adult Census Income, features: the audit's core columns only (no protected attribute, no proxy columns).

## Overall performance (test split, 95% bootstrap interval)
- accuracy: 0.811 [0.802, 0.820]
- auc: 0.772
- f1: 0.433 [0.406, 0.459]

## Per-group results (gap = disadvantaged minus advantaged)
- age: demographic parity gap +0.028 (significant, n=944 vs 5569), equal opportunity gap +0.013
- national_origin: demographic parity gap +0.005 (not significant, n=688 vs 5825), equal opportunity gap +0.061
- race: demographic parity gap -0.042 (significant, n=690 vs 5823), equal opportunity gap +0.031
- sex: demographic parity gap -0.051 (significant, n=2153 vs 4360), equal opportunity gap +0.018

## Known limits
- Gaps are measured on one test split; see the intervals and p-values in results/results_fairness.csv.
- Groups smaller than 30 test rows carry small_sample_warning in the results file.
```

Everything under "Overall performance" and "Per-group results" is read from the CSV, including the intervals and the significance flags. The narrative sections (intended use, training data, limits) are the part a person has to write, and they matter as much: the first line of "Intended use" says this is an audit artefact, not for real decisions.

What the card shows that the accuracy alone does not: accuracy is 0.811, but sex and race have significant parity gaps (-0.051 and -0.042), the age gap (+0.028) is significant in the other direction, and national origin is not significant at all.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/model-cards.py`; that file is generated from the block below). It reads the benchmark's current results; change the three constants at the top to document another model.

```python
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
```

## Limitations

1. **A card is a disclosure, not a guarantee.** It tells readers what was measured. It does not make the model safe to use.
2. **It is only as good as its evaluation.** A card built on a small or unrepresentative test set looks authoritative and is not. See [Sampling Bias](sampling-bias.md).
3. **It goes stale.** A model retrained on new data, or a population that has shifted ([Model Drift](model-drift.md)), makes the numbers wrong without changing the card.
4. **Missing groups are invisible.** The card reports only the groups someone chose to evaluate. A group nobody measured has no row.
5. **The caveats are the hard part.** Intended use and known limits need judgement that no script can generate.

## Related Concepts

* [Disaggregated Evaluation](disaggregated-evaluation.md) - the per-group tables a card is built around
* [Datasheets for Datasets](datasheets-for-datasets.md) - the matching document for the training data
* [ML Bias](ml-bias.md) - the sources of bias the caveats section should name
* [Accuracy Is Not Enough in Healthcare AI](accuracy-not-enough-healthcare-ai.md) - why a headline number is not a safety claim
* [Model Drift](model-drift.md) - why a card needs a date and a review
* [Algorithmic Impact Assessment](algorithmic-impact-assessment.md) - the review that a model card feeds into

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit the card describes
* [`results/`](../results/) - where every number on the card comes from

## Further Reading

* [Mitchell, M. et al. (2019): Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) - FAT* 2019; the proposal and the section list.
* [Raji, I. D. et al. (2020): Closing the AI Accountability Gap](https://arxiv.org/abs/2001.00973) - FAT* 2020; where documentation fits in an end-to-end audit.
* [Barocas, S., Hardt, M., Narayanan, A. (2023): Fairness and Machine Learning](https://fairmlbook.org) - MIT Press; on reporting and documenting model behaviour.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
