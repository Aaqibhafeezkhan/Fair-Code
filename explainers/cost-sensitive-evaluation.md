> *A false positive and a false negative almost never cost the same. Evaluate a model as if they did, and the fairness numbers answer a question nobody asked.*

## The One-Sentence Definition

**Cost-sensitive evaluation** scores a classifier by its expected cost, `c_fp` for each false positive and `c_fn` for each false negative, instead of by accuracy or by error counts that treat both mistakes alike.

## Why It Matters

Wrongly flagging a tenant, wrongly denying a benefit, and missing a patient who will be readmitted are different mistakes with different prices, and the price can differ by group. A false positive for someone with fewer options to appeal costs more. Evaluating at 0.5 assumes `c_fp = c_fn` without saying so. Elkan (2001) showed the consequence: for a calibrated score, the cost-minimizing threshold is not 0.5, it is

```
t* = c_fp / (c_fp + c_fn)
```

So a cost ratio is a threshold, and a threshold is a cost ratio. Changing the costs changes which model decisions are good, which groups pay for them, and how large the fairness gap is.

## Core Concept: Expected Cost per Group

For a group `g` and a cutoff `t_g`, the expected cost per person is

```
cost_g = (FP_g * c_fp + FN_g * c_fn) / n_g
```

Three things follow, and the example below shows each:

* **Cost can be compared across groups.** Two groups with the same accuracy can have very different expected costs if one's errors are mostly the expensive kind.
* **The best cutoff per group comes from the costs.** Minimizing `cost_g` over `t_g`, on training scores, gives a cutoff per group. When the cost ratio is the same in both groups and the scores are calibrated, the cutoffs coincide with Elkan's formula and with each other.
* **Always compare with "flag nobody".** If a false positive is expensive enough, the cheapest policy can be to flag almost no one, which is a finding about the model, not a success.

## Concrete Example: Tenant Screening, Three Cost Settings

The [Tenant Screening](../Tenant%20Screening/) audit: positive means the model flags a person as likely to reoffend, so a false positive wrongly denies housing and a false negative lets through someone who then reoffends. Logistic regression, core features, the benchmark split. Costs are per 100 applicants on the test set; the cutoffs for "cost-optimal" are fit on training scores.

```
Tenant Screening - race: expected cost per 100 applicants on the test set (a positive is a flagged applicant)
a false positive costs 1, a false negative costs 1
  threshold 0.5            thresholds 0.500 / 0.500 | flagged 0.612 / 0.725 | cost   35.4 /   36.6
  Elkan c_fp/(c_fp+c_fn)   thresholds 0.500 / 0.500 | flagged 0.612 / 0.725 | cost   35.4 /   36.6
  cost-optimal per group   thresholds 0.470 / 0.500 | flagged 0.667 / 0.725 | cost   34.7 /   36.6
  flag nobody              thresholds 1.010 / 1.010 | flagged 0.000 / 0.000 | cost   57.1 /   58.1
a false positive costs 5, a false negative costs 1
  threshold 0.5            thresholds 0.500 / 0.500 | flagged 0.612 / 0.725 | cost  114.4 /  138.7
  Elkan c_fp/(c_fp+c_fn)   thresholds 0.833 / 0.833 | flagged 0.016 / 0.042 | cost   57.4 /   57.4
  cost-optimal per group   thresholds 0.845 / 0.840 | flagged 0.009 / 0.033 | cost   57.9 /   57.7
  flag nobody              thresholds 1.010 / 1.010 | flagged 0.000 / 0.000 | cost   57.1 /   58.1
a false positive costs 5 for the advantaged group and 8 for the disadvantaged group, a false negative 1
  threshold 0.5            thresholds 0.500 / 0.500 | flagged 0.612 / 0.725 | cost  114.4 /  215.3
  Elkan c_fp/(c_fp+c_fn)   thresholds 0.833 / 0.889 | flagged 0.016 / 0.000 | cost   57.4 /   58.1
  cost-optimal per group   thresholds 0.845 / 0.870 | flagged 0.009 / 0.004 | cost   57.9 /   57.7
  flag nobody              thresholds 1.010 / 1.010 | flagged 0.000 / 0.000 | cost   57.1 /   58.1
```

**Equal costs.** With `c_fp = c_fn = 1` the 0.5 cutoff is already Elkan's. The cost-optimal advantaged-group cutoff moves slightly, to 0.470, and cuts that group's cost from 35.4 to 34.7. The two groups' costs differ by only 1.2 per 100.

**A false positive costs five times a false negative.** Elkan's cutoff jumps to 0.833 and the learned ones land at 0.845 and 0.840, almost the same. Cost at the default 0.5 is 114.4 for White applicants and 138.7 for Black applicants; at the cost-optimal cutoffs it falls to between 57.4 and 57.9 for both. But "flag nobody" costs 57.1 and 58.1, so the model at the right cutoff is no better than flagging nobody (it flags 1.6% and 4.2% with Elkan's cutoff). Under that cost ratio this model should not be making these decisions, a conclusion no accuracy figure would give.

**Costs that differ by group.** If a false positive costs 5 for White applicants and 8 for Black applicants, the cost-optimal cutoffs separate to 0.845 and 0.870, and Elkan's formula says 0.833 and 0.889. At 0.5 the disadvantaged group's cost is 215.3 against 114.4: the default cutoff makes the group with the higher price pay 100 more per 100 applicants. Again the best cost-aware rule is close to flagging nobody.

## How This Relates to Group-Specific Thresholds

The [Group-Specific Thresholds](group-specific-thresholds.md) explainer chooses a cutoff per group to satisfy a fairness criterion (demographic parity gave 0.565 for this same audit, equal opportunity 0.558). Here the cutoffs come from the cost ratios instead, and the two do not coincide:

* With equal costs in both groups, cost-optimal cutoffs are nearly the same for both groups (0.833 and 0.833, or 0.845 and 0.840), so they do **not** equalize positive rates (0.016 and 0.042 flagged at Elkan's cutoff). Cost-optimality is not a fairness criterion.
* With unequal costs, cost-optimal cutoffs *differ* by group, and a rule that treats the groups differently because their costs differ is a decision that needs a stated justification and a legal check (see that explainer's Limitations).
* Both rest on the same dial, one cutoff per group. The difference is what sets it: a fairness constraint, or a stated price per error.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/cost-sensitive-evaluation.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit
from faircode.models import build_model

data = load_audit("Tenant Screening")
tr, te = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
p_tr, p_te = (model.predict_proba(data.X.iloc[i])[:, 1] for i in (tr, te))
y_tr, y_te = data.y.iloc[tr].to_numpy(), data.y.iloc[te].to_numpy()
s = data.group("race")
s_tr, s_te = s[tr], s[te]                            # 1 = disadvantaged group (Black applicants)

def cost(y, p, t, c_fp, c_fn):
    """Expected cost per person: a false positive costs c_fp, a false negative costs c_fn."""
    pred = p >= t
    return ((pred & (y == 0)).sum() * c_fp + (~pred & (y == 1)).sum() * c_fn) / len(y)

grid = np.round(np.arange(0.05, 1.001, 0.005), 3)    # 1.0 flags nobody
def best_threshold(g, c_fp, c_fn):
    """The cutoff that minimizes cost on the TRAINING scores of group g."""
    costs = [cost(y_tr[s_tr == g], p_tr[s_tr == g], t, c_fp, c_fn) for t in grid]
    return grid[int(np.argmin(costs))]

print("Tenant Screening - race: expected cost per 100 applicants on the test set (a positive is a flagged applicant)")
scenarios = [("a false positive costs 1, a false negative costs 1", {0: (1, 1), 1: (1, 1)}),
             ("a false positive costs 5, a false negative costs 1", {0: (5, 1), 1: (5, 1)}),
             ("a false positive costs 5 for the advantaged group and 8 for the disadvantaged group, a false negative 1",
              {0: (5, 1), 1: (8, 1)})]
for name, costs in scenarios:
    print(name)
    rules = {"threshold 0.5": {g: 0.5 for g in (0, 1)},
             "Elkan c_fp/(c_fp+c_fn)": {g: costs[g][0] / sum(costs[g]) for g in (0, 1)},
             "cost-optimal per group": {g: best_threshold(g, *costs[g]) for g in (0, 1)},
             "flag nobody": {g: 1.01 for g in (0, 1)}}
    for label, t in rules.items():
        per = [100 * cost(y_te[s_te == g], p_te[s_te == g], t[g], *costs[g]) for g in (0, 1)]
        flagged = [(p_te[s_te == g] >= t[g]).mean() for g in (0, 1)]
        print(f"  {label:24s} thresholds {t[0]:.3f} / {t[1]:.3f} | flagged {flagged[0]:.3f} / {flagged[1]:.3f} | cost {per[0]:6.1f} / {per[1]:6.1f}")
```

## Limitations

1. **The costs are assumptions, and the result is only as good as they are.** The 5:1 and 5:8 ratios here are illustrations, not estimates of real harms. Real costs should come from the people affected, and be tested over a range.
2. **Elkan's formula needs calibrated probabilities.** An uncalibrated score gives a cutoff that is not cost-optimal; the learned cutoffs here sit within 0.02 of the formula because logistic regression is roughly calibrated on this data. See [Calibration](calibration.md).
3. **Small cost differences between rules are noise.** The gaps of 0.3 to 0.6 per 100 between Elkan's cutoff and the learned ones are within sampling error on a test set of about 5,000 rows. Compare rules with an interval.
4. **Unequal costs by group is a policy choice, not a fact in the data.** Assigning a higher cost to one group's errors can be justified (harm is greater) or can embed a bias. State who sets the number.
5. **The label is a proxy for the outcome.** Rearrest and the recorded reoffence are not the same as reoffending, so even the "false negative" cost is measured against a flawed label. See [Label Bias](label-bias.md).

## Related Concepts

* [False Positives vs. False Negatives](false-positives-vs-false-negatives.md) - the two errors whose prices this explainer sets
* [Group-Specific Thresholds](group-specific-thresholds.md) - a cutoff per group from a fairness criterion instead of a cost
* [Confusion Matrix](confusion-matrix.md) - where FP and FN counts come from
* [Disaggregated Evaluation](disaggregated-evaluation.md) - computing any metric, here cost, per group
* [Decision Curve Analysis by Group](decision-curve-analysis-by-group.md) - the same cost ratio, shown across all thresholds
* [AUC Gaps by Group](auc-gaps-by-group.md) - the threshold-free counterpart
* [Calibration](calibration.md) - what Elkan's formula assumes

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit used for the example above

## Further Reading

* [Elkan, C. (2001): The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf) - IJCAI 2001; the cost-minimizing threshold and why costs and class proportions are interchangeable.
* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - NeurIPS 2016; threshold rules under a fairness constraint, the counterpart to cost-optimal ones.
* [Corbett-Davies, S., Goel, S. (2018): The Measure and Mismeasure of Fairness](https://arxiv.org/abs/1808.00023) - why a fairness constraint and a cost-minimizing rule can pull apart.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
