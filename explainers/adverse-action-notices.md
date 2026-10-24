> *A decline is not an explanation. The reasons a lender must give are the one place a model's inputs are forced into the open, and they can show a proxy the audit missed.*

## The One-Sentence Definition

An **adverse action notice** is the notice a lender must give an applicant who is declined or offered worse terms, and the **reason codes** on it are the specific main factors in the model's score that led to the decision.

## Why It Matters

In the United States the Equal Credit Opportunity Act and its Regulation B (12 CFR 1002.9) require a creditor to tell a declined applicant the specific principal reasons, or how to ask for them, and the Consumer Financial Protection Bureau has said that using a complex algorithm is not an excuse for a vague reason. Check the current text and your jurisdiction before relying on any summary here, including this one.

Reason codes matter for fairness for a second reason. A model has a coefficient for every input, but nobody reads a coefficient table. A reason code is the same information turned into a sentence an applicant and a regulator will read, so a reason like "marital status" or "neighbourhood" stops being a column name and becomes a statement that a protected characteristic influenced the decision.

## Core Concept: Contributions Relative to the Average Applicant

For a linear model, each feature's contribution to an applicant's log-odds is its coefficient times how far the applicant's standardized value is from the average. The reasons are the features with the most negative contributions, in order. For a non-linear model the same idea is carried by Shapley values ([SHAP Values](shap-values.md)).

```
contribution_j  =  coefficient_j * (standardized value_j)      relative to the average applicant
reason codes    =  the 2 to 4 features with the most negative contribution
```

Two properties to keep in mind. The reasons are relative to a reference applicant, which is a choice. And the reasons explain the score, not the world: they say what moved the model, not what the applicant could change ([Algorithmic Recourse](algorithmic-recourse.md)).

## Concrete Example: German Credit Lending

[German Credit Lending](../German%20Credit%20Lending/): logistic regression on the audit's core features with age left out, standardized. The model declines 185 of the 1,000 applicants (the model is fitted on the benchmark's training rows and applied to every row, to get enough declines to count). The main reason for each is its most negative contribution.

```
the model declines 185 of 1000 applicants

age 30 and over: 97 declined (15% of the group); main reason
  checking_status          62%
  credit_amount            18%
  duration                 9%
  personal_status          6%

under 30: 88 declined (24% of the group); main reason
  checking_status          62%
  credit_amount            12%
  personal_status          12%
  purpose                  9%

largest differences in how often a reason is in an applicant's top two (under 30 minus 30 and over):
reason
personal_status    0.15
purpose           -0.06
duration          -0.05
```

Two things stand out. The headline reason is the same for both age groups (`checking_status`, 62% each), so a quick look at the top reason would find nothing. But the second layer differs: `personal_status` is the main reason for 12% of the under-30s declined against 6% of the others, and across an applicant's top two reasons it appears 15 points more often for the under-30s. In this dataset `personal_status` is the Statlog attribute that combines sex and marital status, so for a real lender a reason code of that kind would be a red flag in itself.

The age gap in declines is also visible: 24% of under-30s are declined against 15% of the rest. The benchmark's S2 row for this audit in `results/` (the current results) gives the same direction, a parity gap of -0.0914, but with a 95% interval of -0.218 to +0.041 and a p-value of 0.17, because the test set is only 200 rows.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/adverse-action-notices.py`; that file is generated from the block below).

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("German Credit Lending")       # S2 feature set: core features, no protected column
train = data.idx_train
X = data.X.to_numpy(dtype=float)
Z = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()                            # 1 = good credit
young = data.masks["age"].to_numpy().astype(int) # 1 = applicants under 30, the disadvantaged group
columns = np.array(data.X.columns)

model = LogisticRegression(max_iter=1000).fit(Z[train], y[train])
declined = np.flatnonzero(model.predict(Z) == 0)  # every applicant the model would decline (all 1,000 rows)

# Reason codes: the features that pushed each declined applicant's score down the most.
contribution = Z[declined] * model.coef_[0]       # per-feature contribution to the log-odds, relative to the average applicant
order = np.argsort(contribution, axis=1)          # most negative first
main_reason = columns[order[:, 0]]

print(f"the model declines {len(declined)} of {len(y)} applicants")
for g, name in ((0, "age 30 and over"), (1, "under 30")):
    rows = young[declined] == g
    shares = pd.Series(main_reason[rows]).value_counts(normalize=True).head(4)
    print(f"\n{name}: {rows.sum()} declined ({rows.sum() / (young == g).sum():.0%} of the group); main reason")
    for reason, share in shares.items():
        print(f"  {reason:24s} {share:.0%}")

top_two = pd.DataFrame({"reason": np.concatenate([columns[order[:, 0]], columns[order[:, 1]]]),
                        "young": np.concatenate([young[declined], young[declined]])})
share = top_two.groupby(["reason", "young"]).size().unstack(fill_value=0)
share = share / share.sum()
print("\nlargest differences in how often a reason is in an applicant's top two (under 30 minus 30 and over):")
print((share[1] - share[0]).sort_values(key=abs, ascending=False).head(3).round(2).to_string())
```

## Limitations

1. **The reasons depend on the reference applicant.** Relative to the average applicant, relative to the approval threshold, or relative to a typical approved applicant gives different lists. A notice has to say which, and the regulator's expectations on this are for the primary text to settle.
2. **Correlated features split the blame.** Two features that carry the same signal share a contribution, so each may fall below the cut-off for being a "main reason" while together they decided the outcome.
3. **A reason is not a remedy.** "Checking status" tells the applicant what moved the score, not what to change. See [Algorithmic Recourse](algorithmic-recourse.md).
4. **A clean top reason can hide the problem.** Here the first reason was identical across groups, and the difference sat in the second.
5. **Counts are small.** There are 97 and 88 declined applicants in the two groups; the 15-point difference in top-two share is a flag to investigate, not a tested effect. See [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md).

## Related Concepts

* [SHAP Values](shap-values.md) - contributions for models that are not linear
* [Counterfactual Explanation](counterfactual-explanation.md) - the basis for notices that say what would change the decision
* [Algorithmic Recourse](algorithmic-recourse.md) - reasons the applicant can act on
* [Proxy Variables](proxy-variables.md) - why a reason like marital status is a warning
* [Disparate Impact](disparate-impact.md) - the legal test that a pattern in reasons feeds

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the audit used for the example above
* [`results/`](../results/) - the benchmark row quoted for S2

## Further Reading

* Consumer Financial Protection Bureau (2022). Consumer Financial Protection Circular 2022-03: Adverse action notification requirements in connection with credit decisions based on complex algorithms - what the regulator said about algorithms and specific reasons.
* [Lundberg, S., Lee, S.-I. (2017): A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) - NeurIPS 2017; Shapley contributions for models beyond linear ones.
* Bracke, P., Datta, A., Jung, C., Sen, S. (2019). Machine Learning Explainability in Finance: An Application to Default Risk Analysis. Bank of England Staff Working Paper 816 - reason-style explanations for a credit model.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
