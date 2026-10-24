> *A ratio of two rates has a lopsided sampling distribution. When the denominator is nearly zero, no interval is honest.*

## The One-Sentence Definition

A **confidence interval for a ratio** such as the disparate impact ratio accounts for the fact that a ratio of two noisy rates is skewed and unbounded above, usually by working on the log of the ratio.

## Why It Matters

The four-fifths rule is a ratio: a group's selection rate divided by the highest group's rate. An audit that reports a ratio of 0.74 with no interval invites the question of whether 0.74 might be 0.9. Intervals for differences are straightforward ([Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md)). A ratio is not: it can run from 0 to infinity, a ratio of 2 and a ratio of 0.5 describe the same disparity in opposite directions, and as the denominator approaches zero the ratio explodes.

That explosion is not a theoretical worry. In the benchmark's own results a COMPAS model had a ratio of 1,250 with an interval up to 879,158, which is why the harness now reports no ratio when the advantaged group has fewer than 30 selected rows.

## Core Concept: Work on the Log Scale

1. **Percentile bootstrap on the ratio:** resample each group, recompute the ratio, take the 2.5th and 97.5th percentiles. Simple, but unstable when the denominator can be zero in a resample.
2. **Delta method on the log ratio (Katz et al., 1978):** `log(ratio)` is close to normal with standard error `sqrt(1/x_a - 1/n_a + 1/x_b - 1/n_b)`, where `x` is the number selected and `n` the group size. Exponentiate the interval's ends. Symmetric on the log scale, so a ratio of 2 and one of 0.5 get matching intervals.
3. **Percentile bootstrap on the log ratio** gives the same shape as 2 without assuming normality.

All three need enough selected rows in both groups. The standard error above shows why: `1/x_b` grows as the number selected shrinks.

## Concrete Example: Two Real Ratios and One Hypothetical Sweep

First a well-behaved ratio, then the COMPAS gradient-boosting case behind the harness change, then the same arithmetic with the denominator set to 3, 10 and 30 selected rows.

```
Benefits Denial, women vs men                    126/2153  vs   477/4360  ratio     0.535 | percentile bootstrap [   0.441,      0.642] | log-ratio (delta) [   0.443,      0.646]
COMPAS gradient boosting: 1 of 1444 White defendants are selected
COMPAS gradient boosting, Black vs White        1564/1804  vs     1/1444  ratio  1251.894 | percentile bootstrap [ 314.374,   1271.104] | log-ratio (delta) [ 176.451,   8881.992]
3 selected in the advantaged group               800/1804  vs     3/1444  ratio   213.452 | percentile bootstrap [  91.479,    655.563] | log-ratio (delta) [  68.842,    661.825]
10 selected in the advantaged group              800/1804  vs    10/1444  ratio    64.035 | percentile bootstrap [  38.139,    159.293] | log-ratio (delta) [  34.454,    119.014]
30 selected in the advantaged group              800/1804  vs    30/1444  ratio    21.345 | percentile bootstrap [  15.437,     32.903] | log-ratio (delta) [  14.924,     30.529]
```

- **A stable ratio gives agreement.** On Benefits Denial women are selected at 0.535 times the rate of men; the percentile bootstrap gives [0.441, 0.642] and the delta method [0.443, 0.646].
- **A near-zero denominator breaks both.** The gradient-boosting model with race as a feature selects 1 of 1,444 White defendants. The ratio is 1,252. The percentile bootstrap says [314, 1,271] (it cannot exceed what one selected row allows, so its upper end is falsely tight and it drops resamples where the denominator is zero), and the delta method says [176, 8,882]. The truth is that a ratio built on one row is not estimable.
- **The intervals converge as the denominator grows.** At 3 selected rows the two methods give [91, 656] and [69, 662]; at 30 they give [15.4, 32.9] and [14.9, 30.5], close to each other and to a usable answer.

So the fix for the benchmark was not a better interval. It was a rule: with fewer than about 30 selected rows in the denominator group, report no ratio.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/confidence-intervals-for-ratios.py`; that file is generated from the block below).

```python
import numpy as np
from scipy.stats import norm
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

def intervals(selected_a, n_a, selected_b, n_b, level=0.95, n_boot=4000, seed=0):
    """Ratio of selection rates, group a over group b, with three 95% intervals."""
    rng = np.random.default_rng(seed)
    ratio = (selected_a / n_a) / (selected_b / n_b)
    z = norm.ppf(0.5 + level / 2)
    # 1. percentile bootstrap on the ratio itself
    boot_a = rng.binomial(n_a, selected_a / n_a, n_boot) / n_a
    boot_b = rng.binomial(n_b, selected_b / n_b, n_boot) / n_b
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = boot_a / boot_b
    ratios = ratios[np.isfinite(ratios)]
    percentile = np.percentile(ratios, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    # 2. delta method on the log ratio: se = sqrt(1/x_a - 1/n_a + 1/x_b - 1/n_b)  (Katz et al. 1978)
    se = np.sqrt(1 / selected_a - 1 / n_a + 1 / selected_b - 1 / n_b)
    log_interval = np.exp(np.log(ratio) + np.array([-z, z]) * se)
    return ratio, percentile, log_interval, se

def show(label, sa, na, sb, nb):
    ratio, pct, log, se = intervals(sa, na, sb, nb)
    print(f"{label:46s} {sa:5d}/{na:<5d} vs {sb:5d}/{nb:<5d} ratio {ratio:9.3f} | percentile bootstrap [{pct[0]:8.3f}, {pct[1]:10.3f}] "
          f"| log-ratio (delta) [{log[0]:8.3f}, {log[1]:10.3f}]")

# 1. A stable ratio: Benefits Denial, logistic regression on the core features, selection of women against men.
data = load_audit("Benefits Denial")
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[data.idx_train], data.y.iloc[data.idx_train])
pred = model.predict(data.X.iloc[data.idx_test])
women = data.masks["sex"].to_numpy()[data.idx_test] == 1
show("Benefits Denial, women vs men", int(pred[women].sum()), int(women.sum()), int(pred[~women].sum()), int((~women).sum()))

# 2. A ratio with a near-zero denominator: COMPAS, gradient boosting with every feature (race included), Black vs White defendants.
compas = load_audit("COMPAS", strategy="baseline")
gb = build_model("gradient_boosting", compas.random_state).fit(compas.X.iloc[compas.idx_train], compas.y.iloc[compas.idx_train])
pred = gb.predict(compas.X.iloc[compas.idx_test])
black = compas.masks["race"].to_numpy()[compas.idx_test] == 1
print(f"COMPAS gradient boosting: {int(pred[~black].sum())} of {int((~black).sum())} White defendants are selected")
if pred[~black].sum() > 0:
    show("COMPAS gradient boosting, Black vs White", int(pred[black].sum()), int(black.sum()), int(pred[~black].sum()), int((~black).sum()))

# 3. The same arithmetic with a denominator of 3, 10 and 30 selected rows out of 1,444.
for denominator in (3, 10, 30):
    show(f"{denominator} selected in the advantaged group", 800, 1804, denominator, 1444)
```

## Limitations

1. **Both methods need selected rows in both groups.** With zero selected rows in the denominator the ratio is undefined and the delta formula divides by zero.
2. **The delta method is an approximation.** It works well for 30 or more selected rows and degrades below that, as the sweep above shows.
3. **The percentile bootstrap hides the problem.** Dropping resamples where the denominator is zero shrinks the interval toward the point estimate and makes a bad ratio look precise.
4. **It treats each rate as an independent binomial.** If rows are clustered (one person in the data many times), the standard error is too small.
5. **A ratio hides the size of the groups.** Report the counts of selected and total rows next to every ratio.

## Related Concepts

* [Disparate Impact](disparate-impact.md) - the ratio and the four-fifths rule
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the interval for a difference
* [Minimum Detectable Gap](minimum-detectable-gap.md) - how many rows a comparison needs
* [Demographic Parity](demographic-parity.md) - the difference form of the same comparison
* [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md) - a test that does not need a variance formula

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the stable ratio
* [`COMPAS/`](../COMPAS/) - the near-zero denominator
* [`results/`](../results/) - the benchmark's ratio rows

## Further Reading

* Katz, D., Baptista, J., Azen, S. P., Pike, M. C. (1978). Obtaining Confidence Intervals for the Risk Ratio in Cohort Studies. Biometrics 34(3) - the log-ratio interval.
* Efron, B., Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall - bootstrap intervals and their failure modes.
* The Uniform Guidelines on Employee Selection Procedures (1978), 29 CFR Part 1607 - the four-fifths rule that the ratio comes from.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
