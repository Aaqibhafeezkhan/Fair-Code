> *The total gap is one number. Law and ethics ask about the paths through it: which part runs through something we accept, and which does not.*

## The One-Sentence Definition

A **path-specific effect** is the part of the effect of a protected attribute on an outcome that travels along a chosen set of causal paths (for example, through schooling) rather than along others (for example, directly or through occupation), so a gap can be split into a part you may accept and a part you may not.

## Why It Matters

Discrimination law and moral intuition do not treat all routes alike. A pay gap that comes through a difference in qualifications is not argued about the way a pay gap with identical qualifications is. [Causal Graphs for Fairness](causal-graphs-for-fairness.md) shows that the verdict on a gap changes with which variables you adjust for. Path-specific effects state the same thing more carefully: name the paths you consider fair, and estimate what is left.

Nabi and Shpitser (2018) and Chiappa (2019) develop this for fairness; the simplest, mediation analysis with a linear model, is enough to show the idea.

## Core Concept: Total, Direct and Indirect Effects

With a linear model and mediators `M` between the attribute `A` and the outcome `Y`:

```
total effect       = coefficient of A in  Y ~ A + (controls)
direct effect      = coefficient of A in  Y ~ A + M + (controls)
indirect via M_i   = (coefficient of A in  M_i ~ A + controls)  *  (coefficient of M_i in Y ~ A + M + controls)
direct + indirect  = total      (exactly, for a linear model)
```

The direct effect is the part not through the mediators you listed. Whether to call it "unfair" depends on whether you think the listed mediators are the whole of what is acceptable, and that is the assumption to state.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the effect of being a woman on the probability of income above $50K, adjusting for age, with two mediators treated as acceptable paths: years of schooling (`education.num`) and hours worked per week. 300 bootstrap resamples give the intervals.

```
effect of being a woman on P(income above $50K), adjusting for age; linear model; 32,561 rows
  total effect:                      -0.1788  [-0.1861, -0.1702]
  indirect, through education.num    -0.0025  [-0.0056, +0.0005]
  indirect, through hours.per.week   -0.0272  [-0.0294, -0.0249]
  direct effect (not through these): -0.1491  [-0.1562, -0.1407]
  check: direct + indirect = -0.1788
  share of the total that runs through the two mediators: 17%
```

- **Total effect -0.1788** (interval -0.186 to -0.170): women are about 18 points less likely to be above the line after age.
- **Through schooling: -0.0025**, interval -0.006 to +0.0005, which includes zero. In this data women have about as much schooling as men on average, so little of the gap runs through it.
- **Through hours worked: -0.0272** (interval -0.029 to -0.025): about 15% of the total, because women work fewer hours in this data.
- **Direct effect -0.1491:** the 83% left over after removing the two accepted paths. If schooling and hours are the only acceptable routes, that is the gap to explain.

The two routes together are 17% of the gap. The rest is either a direct effect, discrimination, or the effect of mediators not listed, such as occupation, marital status or unmeasured factors, and the analysis cannot say which.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/path-specific-effects.py`; that file is generated from the block below). Only NumPy is needed for the regressions.

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial")
df = data.df
woman = data.masks["sex"].to_numpy().astype(float)
y = data.y.to_numpy().astype(float)                 # 1 = income above $50K
age = df["age"].to_numpy(dtype=float)
mediators = {"education.num": df["education.num"].to_numpy(dtype=float), "hours.per.week": df["hours.per.week"].to_numpy(dtype=float)}
names = list(mediators)
M = np.column_stack([mediators[n] for n in names])

def ols(columns, target):
    X = np.column_stack([np.ones(len(target))] + columns)
    return np.linalg.lstsq(X, target, rcond=None)[0]

def decompose(rows):
    w, a, m, out = woman[rows], age[rows], M[rows], y[rows]
    total = ols([w, a], out)[1]                                         # sex -> income, mediators not adjusted
    full = ols([w, a] + [m[:, i] for i in range(m.shape[1])], out)      # income on sex, age and the mediators
    direct, b = full[1], full[3:]
    a_paths = np.array([ols([w, a], m[:, i])[1] for i in range(m.shape[1])])   # sex -> each mediator
    return total, direct, a_paths * b

rng = np.random.default_rng(0)
everyone = np.arange(len(y))
total, direct, indirect = decompose(everyone)
boots = [decompose(rng.choice(everyone, len(everyone))) for _ in range(300)]
def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5]); return f"[{lo:+.4f}, {hi:+.4f}]"

print(f"effect of being a woman on P(income above $50K), adjusting for age; linear model; {len(y):,} rows")
print(f"  total effect:                      {total:+.4f}  {interval([b[0] for b in boots])}")
for i, name in enumerate(names):
    print(f"  indirect, through {name:15s}  {indirect[i]:+.4f}  {interval([b[2][i] for b in boots])}")
print(f"  direct effect (not through these): {direct:+.4f}  {interval([b[1] for b in boots])}")
print(f"  check: direct + indirect = {direct + indirect.sum():+.4f}")
print(f"  share of the total that runs through the two mediators: {indirect.sum() / total:.0%}")
```

## Limitations

1. **The decomposition needs the graph to be right.** If a mediator affects another mediator, or if an unmeasured cause affects both a mediator and the outcome, the product of coefficients is not the path effect.
2. **The linear-probability model is crude.** A probability outcome with large effects can fall outside 0 to 1, and a logistic model has no exact additive decomposition.
3. **"Acceptable path" is a judgement.** Counting hours worked as acceptable is itself contestable, because hours may be shaped by caregiving burdens, see [Proxy Variables](proxy-variables.md).
4. **Intervals cover sampling noise only.** They do not cover the graph assumptions.
5. **It explains a gap; it does not change a model.** To act on it you still need a mitigation, see [Mitigation Strategies](mitigation-strategies.md).

## Related Concepts

* [Causal Graphs for Fairness](causal-graphs-for-fairness.md) - drawing the assumptions
* [Counterfactual Fairness](counterfactual-fairness.md) - a model fair under a given graph
* [Proxy Variables](proxy-variables.md) - when a mediator is a stand-in for the attribute
* [Simpson's Paradox](simpsons-paradox.md) - adjustment that changes the sign
* [Conditional Demographic Parity](conditional-demographic-parity.md) - a parity check that conditions on an accepted factor

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Nabi, R., Shpitser, I. (2018): Fair Inference on Outcomes](https://arxiv.org/abs/1705.10378) - AAAI 2018; constraining the effect along chosen paths.
* Chiappa, S. (2019). Path-Specific Counterfactual Fairness. AAAI - counterfactual fairness restricted to unfair paths.
* VanderWeele, T. J. (2015). Explanation in Causal Inference: Methods for Mediation and Interaction. Oxford University Press - the assumptions behind a mediation analysis.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
