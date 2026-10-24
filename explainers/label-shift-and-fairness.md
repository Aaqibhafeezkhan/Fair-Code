> *If the base rate changes and nothing else does, some of your fairness metrics stay put and some swing wildly. Knowing which is half of reading a monitoring chart.*

## The One-Sentence Definition

**Label shift** is a change in how common the outcome is in a group (the base rate) while the relationship between the features and the outcome stays the same, and it moves some fairness metrics (selection rate, precision) and leaves others (true and false positive rates) unchanged.

## Why It Matters

Deployed populations change in how often the outcome occurs: more patients are readmitted in winter, a lender enters a market with a higher default rate, one group's applicants change. If the model's score for a person given their features still means what it did, this is label shift, and it has a clean consequence. The error rates computed within each outcome class, the true and false positive rates, do not change, because they condition on the label. Everything that mixes the classes does: the selection rate, the precision, the calibration.

So a monitor that sees a parity gap or a precision gap move after a change in base rate does not have evidence that the model got worse. The fairness metric that looks alarming may simply be tracking the base rate. See [Fairness Metric Conflicts](fairness-metric-conflicts.md) for why metrics that mix classes cannot all hold when base rates differ.

## Core Concept: Reweight the Classes Inside One Group

You can see the effect without waiting for new data. Give each positive row in a group the weight `new rate / old rate` and each negative row `(1 - new rate) / (1 - old rate)`, which changes the group's base rate to the target while keeping the features-given-label distribution. Recompute the metrics on the weighted rows.

```
TPR, FPR           unchanged (they are within-class)
selection rate     moves with the base rate
precision (PPV)    moves with the base rate
```

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the S2 logistic regression and its test rows. Women's base rate for "income above $50K" is 0.111 in the data and men's is 0.305. We change women's base rate from 0.05 to 0.40, leaving men alone, and recompute four group gaps.

```
women's base rate in the test rows 0.111; men's 0.305

 women base rate  parity gap   TPR gap   FPR gap  precision gap
           0.050     -0.0683   +0.0181   -0.0004        -0.4443
           0.111     -0.0509   +0.0181   -0.0004        -0.2329  <- as measured
           0.200     -0.0251   +0.0181   -0.0004        -0.0807
           0.300     +0.0038   +0.0181   -0.0004        +0.0072
           0.400     +0.0326   +0.0181   -0.0004        +0.0594
```

- **True and false positive rate gaps do not move.** The TPR gap is +0.0181 and the FPR gap -0.0004 at every base rate.
- **The parity gap swings.** It runs from -0.0683 at a women's base rate of 0.05 through -0.0509 as measured to +0.0038 at 0.30 and +0.0326 at 0.40. The "gap" crosses zero where women's base rate gets close to men's, because selection follows the base rate.
- **The precision gap swings more.** It is -0.4443 at 0.05, -0.2329 as measured and +0.0072 at 0.30. A model that is "fair" on precision at one base rate is badly unfair at another with no change in the model.

So a model audited on a parity or predictive-parity criterion can fail it after a pure label shift, and one audited on equal opportunity or equalized odds cannot.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/label-shift-and-fairness.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], data.y.iloc[train])
pred = model.predict(data.X.iloc[test])
y = data.y.to_numpy()[test]
women = data.masks["sex"].to_numpy()[test].astype(bool)
observed = y[women].mean()
print(f"women's base rate in the test rows {observed:.3f}; men's {y[~women].mean():.3f}")

def reweighted(target_rate):
    """Label shift for women only: change the share of positives among women to `target_rate`, keeping the
    features-given-label relationship fixed, by weighting positives and negatives."""
    w = np.ones(len(y))
    w[women & (y == 1)] = target_rate / observed
    w[women & (y == 0)] = (1 - target_rate) / (1 - observed)
    return w

def metrics(w):
    def rate(mask):
        return np.average(pred[mask], weights=w[mask])
    sel_w, sel_m = rate(women), rate(~women)
    tpr_w, tpr_m = rate(women & (y == 1)), rate(~women & (y == 1))
    fpr_w, fpr_m = rate(women & (y == 0)), rate(~women & (y == 0))
    ppv_w = np.average(y[women & (pred == 1)], weights=w[women & (pred == 1)])
    ppv_m = y[~women & (pred == 1)].mean()
    return sel_w - sel_m, tpr_w - tpr_m, fpr_w - fpr_m, ppv_w - ppv_m

print(f"\n{'women base rate':>16} {'parity gap':>11} {'TPR gap':>9} {'FPR gap':>9} {'precision gap':>14}")
for target in (0.05, observed, 0.20, 0.30, 0.40):
    sel, tpr, fpr, ppv = metrics(reweighted(target))
    tag = "  <- as measured" if abs(target - observed) < 1e-9 else ""
    print(f"{target:16.3f} {sel:+11.4f} {tpr:+9.4f} {fpr:+9.4f} {ppv:+14.4f}{tag}")
```

## Limitations

1. **It assumes the features-given-label relationship is fixed.** Real shifts often move both, and then the error rates move too. Label shift is a special case that is easy to reason about, not a description of all drift.
2. **The reweighting changes the sample, not the model.** It tells you what the metrics would be under a different base rate, assuming the assumption holds.
3. **You cannot always see the base rate.** If outcomes arrive late or not at all, the shift is invisible until they do, see [Delayed Feedback](delayed-feedback.md).
4. **A change in one group's rate is only one kind of shift.** Both groups can shift, and in different directions.
5. **Choosing the "right" metric does not remove the shift.** It decides which changes in a monitor you will treat as alarms.

## Related Concepts

* [Distribution Shift](distribution-shift.md) - the general family, of which label shift is one case
* [Base Rate Fallacy](base-rate-fallacy.md) - why precision follows the base rate
* [Predictive Parity](predictive-parity.md) - the metric that moves most here
* [Equalized Odds](equalized-odds.md) - the error-rate criterion that does not
* [Monitoring Fairness in Production](monitoring-fairness-in-production.md) - reading a gap that moves

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Lipton, Z., Wang, Y.-X., Smola, A. (2018): Detecting and Correcting for Label Shift with Black Box Predictors](https://arxiv.org/abs/1802.03916) - ICML 2018; the label-shift assumption and its correction.
* [Chouldechova, A. (2017): Fair Prediction with Disparate Impact](https://arxiv.org/abs/1610.07524) - Big Data 5(2); how base rates force fairness metrics apart.
* [Quinonero-Candela, J. et al. (eds.) (2009): Dataset Shift in Machine Learning](https://mitpress.mit.edu/9780262170055/dataset-shift-in-machine-learning/) - MIT Press; covariate, prior (label) and concept shift.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
