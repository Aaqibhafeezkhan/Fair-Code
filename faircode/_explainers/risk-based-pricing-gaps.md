> *A model can approve everyone and still treat groups differently, by deciding how much to offer or how much to charge. Fairness metrics for yes-or-no decisions do not see it.*

## The One-Sentence Definition

A **pricing gap** is a difference between groups in the amount, rate or term a model quotes to applicants who were all approved, and it is measured as a difference in a continuous outcome instead of a difference in selection rates.

## Why It Matters

Every metric in this repo's benchmark is built on a decision that is yes or no: [Demographic Parity](demographic-parity.md), [Equalized Odds](equalized-odds.md), [Disparate Impact](disparate-impact.md). A lender that approves every applicant passes all of them. It can still offer a smaller line of credit, a higher rate or a shorter term to one group, and that is a decision with the same consequences.

The measurement is simpler than it sounds. The quantity is a number, so the gap is a difference of means, and the uncertainty is a bootstrap interval ([Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md)). The hard parts are choosing the quantity and remembering that the same gap can be fair or unfair depending on whether the difference in price tracks a real difference in risk.

## Core Concept: A Regression, Then a Gap in Its Output

```
1. train a regression for the quoted amount (rate, limit, term) on the non-protected features
2. predict it for the test applicants
3. gap = mean(quoted | disadvantaged) - mean(quoted | advantaged), with a bootstrap interval
4. compare it with the gap in the amounts people actually received, and with the approve or deny gap
```

If the quoted gap is larger than the real-world gap, the model has widened a difference. If it is zero while the approval gap is not, price is not where the unfairness sits.

## Concrete Example: German Credit Lending

[German Credit Lending](../German%20Credit%20Lending/) has a continuous `credit_amount`, which stands in for the amount a lender would offer. Everyone is treated as approved; the model predicts the amount from every other core column, with age left out, using the benchmark's split (200 test applicants, 69 of them under 30).

```
real amounts: under 30 3,277, 30 and over 3,790 (gap -513)
linear regression  quoted amount, under 30 minus 30 and over: -233 (-7%), 95% interval [-802, +308]
gradient boosting  quoted amount, under 30 minus 30 and over: -477 (-13%), 95% interval [-1,039, +94]
```

In the data, under-30s received 513 less on average (3,277 against 3,790). Both models quote a smaller amount to the under-30s too: 233 less (7%) from the linear regression and 477 less (13%) from gradient boosting. The intervals say how much to read into that: [-802, +308] and [-1,039, +94] both include zero. With 200 test rows the data cannot separate a real 13% gap from noise, which is the situation [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md) describes.

For comparison, the same audit's approve-or-deny parity gap in `results/` (the current results, S2 strategy) is -0.0914 with an interval of -0.218 to +0.041. Both are the same direction and both are inconclusive on this sample, and neither alone would show the other.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/risk-based-pricing-gaps.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("German Credit Lending")       # S2 feature set: core features, no protected column
train, test = data.idx_train, data.idx_test
young = data.masks["age"].to_numpy().astype(int)[test]     # 1 = under 30, the disadvantaged group
rng = np.random.default_rng(0)

# Everyone is "approved"; the model decides how much to lend. The quoted amount is predicted from the other
# columns (the one continuous amount a lender sets per applicant in this file).
X = data.X.drop(columns=["credit_amount"])
amount = data.df["credit_amount"].to_numpy(dtype=float)

def bootstrap_gap(values, group, n=2000):
    a, b = values[group == 1], values[group == 0]
    gaps = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(gaps, [2.5, 97.5])

print(f"real amounts: under 30 {amount[test][young == 1].mean():,.0f}, 30 and over {amount[test][young == 0].mean():,.0f} "
      f"(gap {amount[test][young == 1].mean() - amount[test][young == 0].mean():+,.0f})")
for name, model in (("linear regression", LinearRegression()),
                    ("gradient boosting", GradientBoostingRegressor(random_state=data.random_state))):
    quoted = model.fit(X.iloc[train], amount[train]).predict(X.iloc[test])
    gap = quoted[young == 1].mean() - quoted[young == 0].mean()
    lo, hi = bootstrap_gap(quoted, young)
    relative = gap / quoted[young == 0].mean()
    print(f"{name:18s} quoted amount, under 30 minus 30 and over: {gap:+,.0f} ({relative:+.0%}), 95% interval [{lo:+,.0f}, {hi:+,.0f}]")
```

## Limitations

1. **A price gap is not automatically unfair.** If the quoted amount tracks real differences in risk or need, a gap can be justified. The test is whether the group still gets a worse offer at the same risk, which needs a risk estimate that is itself not biased.
2. **The stand-in outcome is imperfect.** `credit_amount` here is what applicants asked for and received, not an offer. Real pricing data would have the rate and term the lender set.
3. **Small samples hide the answer.** Both intervals cross zero. See [Disaggregated Evaluation](disaggregated-evaluation.md) for how fast intervals widen.
4. **One number per group loses the spread.** A mean gap can hide a gap that only exists at the top or bottom of the distribution, so quantiles or a full comparison of distributions may be needed.
5. **There is no standard threshold.** The four-fifths rule is for selection rates. For prices, the standard is a legal and policy question to answer for your setting, from primary sources.

## Related Concepts

* [Demographic Parity](demographic-parity.md) - the yes-or-no counterpart this extends
* [Disparate Impact](disparate-impact.md) - the legal test, written for selection
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the interval on a difference of means
* [Fairness in Ranking](fairness-in-ranking.md) - another outcome that is not a single decision
* [Adverse Action Notices](adverse-action-notices.md) - what a lender owes an applicant who gets worse terms

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the audit used for the example above
* [`results/`](../results/) - the benchmark row quoted for the approve-or-deny gap

## Further Reading

* [Bartlett, R., Morse, A., Stanton, R., Wallace, N. (2022): Consumer-Lending Discrimination in the FinTech Era](https://doi.org/10.1016/j.jfineco.2021.05.047) - Journal of Financial Economics 143(1); price differences by group in algorithmic and face-to-face mortgage lending.
* [Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., Walther, A. (2022): Predictably Unequal? The Effects of Machine Learning on Credit Markets](https://doi.org/10.1111/jofi.13090) - The Journal of Finance 77(1); who gains and loses when lenders move to machine-learning pricing.
* [Barocas, S., Hardt, M., Narayanan, A. (2023): Fairness and Machine Learning](https://fairmlbook.org) - MIT Press; on fairness criteria beyond classification.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
