> *Closing a fairness gap is rarely free - the same intervention that shrinks a demographic parity gap can quietly cost the model a meaningful share of its accuracy, and neither number tells the whole story alone.*

## The One-Sentence Definition

The **fairness-accuracy trade-off** is the observation that fairness-mitigation strategies which reduce a model's fairness gap often also reduce its predictive accuracy, and the two cannot always be improved together - a model sits somewhere on a Pareto frontier between them, not at an independently-optimal point on each.

## Why It Matters

[Pre-, In-, and Post-Processing Fairness Mitigations](mitigation-strategies.md) walks through this repo's five-strategy ladder and what each stage does to the fairness *gap*. It never mentions accuracy at all. That is the other half of the same story: every one of those strategies also changes model *performance*, and treating "we closed the gap" as the end of the evaluation - without asking what it cost - is exactly how a fairness intervention can look like an unambiguous win when it was actually a trade a team never explicitly agreed to make.

This isn't a reason to avoid fairness mitigation. It's a reason to report both numbers side by side, every time, so the trade being made is visible and can be judged deliberately rather than discovered later.

## Concrete Example: COMPAS - Audit 01

Both tables below use the baseline logistic regression model on race, across all five strategies, using the frozen numbers exactly as `faircode/benchmark.py` computed them (`paper/results-frozen/results_fairness.csv` and `paper/results-frozen/results_performance.csv`).

| Strategy | Demographic Parity Diff | Accuracy | Accuracy vs. S0 |
|---|---:|---:|---:|
| S0 baseline | 0.855 (sig.) | 65.3% | - |
| S1 unawareness | 0.172 (sig.) | 57.1% | -8.2 pts |
| S2 unawareness_proxy_removal | 0.115 (sig.) | 57.2% | -8.1 pts |
| S3 in_processing | -0.014 (n.s.) | 53.9% | -11.4 pts |
| S4 post_processing | 0.023 (n.s.) | 55.7% | -9.7 pts |

The pattern [Mitigation Strategies](mitigation-strategies.md) already shows on the fairness side - a huge S0→S1 improvement, a smaller S1→S2 gain, and S3/S4 converging near zero - has a cost sitting directly underneath it that a fairness-only table hides completely:

**The single largest accuracy drop (S0 → S1, -8.2 points) coincides with the single largest fairness gain (85.5 → 17.2 points closed).** Simply dropping the protected attribute buys most of the fairness improvement this ladder achieves, and most of the accuracy cost too, in the same step.

**S3 in_processing reaches the best fairness number (-1.4 points, not statistically significant) at the worst accuracy (53.9%, an 11.4-point drop from baseline).** This is the sharpest illustration of the trade-off on this audit: the strategy that gets closest to eliminating the demographic parity gap is also the one that costs the model the most predictive accuracy - not a coincidence, since `ExponentiatedGradient` is directly trading classification accuracy against the fairness constraint during training, by design.

**S2 and S4 show it isn't a strict, monotonic trade.** S2 has a smaller fairness gap than S1 (11.5 vs. 17.2 points) at essentially the same accuracy (57.2% vs. 57.1%) - a case where fairness improved with no meaningful accuracy cost. S4 recovers some accuracy back from S3 (55.7% vs. 53.9%) while its fairness gap stays statistically indistinguishable from S3's. The frontier isn't a straight line where every fairness point costs a fixed accuracy price; some steps are close to free, and some are expensive, which is exactly why both numbers need checking at every step rather than assuming the relationship is linear.

## Detection Code

Pairs each strategy's fairness gap with its accuracy so a table can't report one without the other.

```python
import pandas as pd


def fairness_accuracy_tradeoff(fairness_df, performance_df, audit, fairness_metric,
                                protected_attribute, model="logistic_regression"):
    """
    Joins one fairness metric's progression across the five strategies with
    the corresponding accuracy for the same audit/model, so a fairness gain
    can be checked against its accuracy cost at every step.

    Parameters:
        fairness_df: a DataFrame shaped like paper/results-frozen/results_fairness.csv
        performance_df: a DataFrame shaped like paper/results-frozen/results_performance.csv
        audit, fairness_metric, protected_attribute, model: filter values

    Returns a DataFrame with one row per strategy, in S0-S4 order, with the
    fairness value/significance and accuracy side by side, plus an
    `accuracy_delta_from_baseline` column.
    """
    order = ["baseline", "unawareness", "unawareness_proxy_removal",
             "in_processing", "post_processing"]

    fair = fairness_df[
        (fairness_df["audit"] == audit)
        & (fairness_df["metric"] == fairness_metric)
        & (fairness_df["protected_attribute"] == protected_attribute)
        & (fairness_df["model"] == model)
    ].set_index("strategy").reindex(order)[["value", "significant"]]
    fair.columns = ["fairness_value", "fairness_significant"]

    acc = performance_df[
        (performance_df["audit"] == audit)
        & (performance_df["metric"] == "accuracy")
        & (performance_df["model"] == model)
    ].set_index("strategy").reindex(order)[["value"]]
    acc.columns = ["accuracy"]

    joined = fair.join(acc).reset_index()
    joined["accuracy_delta_from_baseline"] = joined["accuracy"] - joined["accuracy"].iloc[0]
    return joined


# Usage example:
# import pandas as pd
# fairness = pd.read_csv("paper/results-frozen/results_fairness.csv")
# performance = pd.read_csv("paper/results-frozen/results_performance.csv")
# print(fairness_accuracy_tradeoff(
#     fairness, performance, "compas", "demographic_parity_diff", "race",
# ))
```

## Limitations

### 1. The trade-off isn't fixed, universal, or always present

As S2 shows above, a fairness gain can arrive at essentially no accuracy cost. The trade-off is a common empirical pattern on this benchmark, not a law - never assume a fairness fix must cost accuracy without checking, and never assume a specific ratio (e.g. "1 fairness point costs 0.1 accuracy points") transfers from one audit or metric to another.

### 2. "Accuracy" is itself only one performance metric, and can hide its own asymmetry

A dropped-accuracy number doesn't say *which* predictions got worse. `faircode/benchmark.py` also computes AUC and F1 alongside accuracy for exactly this reason - a strategy that trades away F1 while accuracy looks stable is a different trade than one that drops all three metrics together. See [What Is Accuracy Equality?](accuracy-equality.md) for the parallel problem on the fairness side: a single blended number can hide two very different underlying stories.

### 3. The Pareto frontier is estimated from a specific dataset and split

The exact numbers above hold for the COMPAS audit's baseline logistic regression model and 80/20 stratified split (`random_state=42`) - a different model family, a different dataset, or a different split can land at a different point on the frontier. Treat the *pattern* (check both numbers, expect some cost, don't assume it's linear) as the transferable lesson, not the specific -8.2/-11.4/-9.7 point figures.

### 4. A statistically insignificant fairness gap is not the same as "fixed"

S3's -1.4-point gap and S4's 2.3-point gap both have confidence intervals crossing zero (see [What Is a Bootstrap Confidence Interval?](bootstrap-confidence-intervals.md)) - "not statistically significant on this test set" is weaker evidence than "the disparity is gone," and shouldn't be reported as though the accuracy cost bought a fully solved fairness problem.

## Related Concepts

* [What Are Pre-, In-, and Post-Processing Fairness Mitigations?](mitigation-strategies.md) - the fairness-only half of the exact same S0-S4 progression used here.
* [What Is Accuracy Equality?](accuracy-equality.md) - a fairness metric built on accuracy itself, and the parallel reminder that one blended accuracy number can hide asymmetry underneath it.
* [What Is Bias-Variance Trade-off?](bias-variance-tradeoff.md) - a different, model-internal trade-off; useful for not confusing "the model overfit" with "the model traded accuracy for fairness."
* [Fairness Regularization](fairness-regularization.md) - a penalty weight traces the trade-off curve
* [What Is Two-Sided Fairness in a Marketplace?](two-sided-fairness.md) - exposure for the ranked and utility for the searcher on one sweep

## Related Projects in This Repo

* [`COMPAS/`](../COMPAS/) - the audit above, where S3 buys the best fairness number at the worst accuracy.
* [`faircode/benchmark.py`](../faircode/benchmark.py) - runs every strategy/model/metric combination in one uniform pipeline, producing both `results_fairness.csv` and `results_performance.csv` from the exact same train/test split so the two are directly comparable, strategy for strategy.

## Further Reading

* [Agarwal, A. et al. (2018): A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453) - the `ExponentiatedGradient` method behind this repo's S3 in-processing strategy, framed explicitly as a constrained optimization trading accuracy against a fairness constraint.
* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - the `ThresholdOptimizer`-style post-processing method behind this repo's S4 strategy.
* [Corbett-Davies, S., Goel, S. (2018): The Measure and Mismeasure of Fairness](https://arxiv.org/abs/1808.00023) - a critical look at the assumptions underlying fairness-accuracy trade-off framings, including when the "trade-off" itself is an artifact of a poorly-chosen decision threshold rather than an unavoidable cost.

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
