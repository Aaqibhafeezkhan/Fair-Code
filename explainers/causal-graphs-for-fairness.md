> *Two analysts with the same data can reach opposite fairness verdicts. The difference is usually a drawing they did not show: which variables cause which.*

## The One-Sentence Definition

A **causal graph** for fairness is a diagram of assumptions about what causes what (the protected attribute, the variables it influences, the decision), and it determines which variables you may control for and which you must not when asking whether a gap is fair.

## Why It Matters

A raw gap (women are 0.109 likely to earn above $50K, men 0.306) says nothing about why. People answer that by "controlling for" other variables such as education, hours or occupation, and the choice changes the answer. Whether it should is not a statistical question. Education is plausibly caused by the attribute in part (unequal schooling), and controlling for it removes a path that may be unfair. A variable caused by both the attribute and the outcome can create a gap when you condition on it.

[Counterfactual Fairness](counterfactual-fairness.md) and [Confounding Variables](confounding-variable.md) rely on a graph without showing one. Drawing it down makes the assumptions arguable, which is the point: the data cannot tell two analysts with different graphs who is right.

## Core Concept: Common Causes, Mediators and Colliders

| Role of a variable | In the graph | Adjusting for it |
|---|---|---|
| Common cause of attribute and outcome | `Z -> sex`, `Z -> income` | Needed, or the gap is confounded |
| Mediator | `sex -> M -> income` | Removes the part of the gap that runs through `M`. Right if that path is fair, wrong if it is the discrimination |
| Collider | `sex -> C <- income` | Creates a gap that was not there |

The assumed graph for the example below is `sex -> education, hours, occupation, marital status -> income`, `sex -> income` directly, and `age` as a common cause. It is an assumption, written down so it can be argued about.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): a linear-probability regression of "income above $50K" on being a woman, with different adjustment sets. The coefficient on sex is the gap that remains after adjusting.

```
women 33.1% of the rows; income above $50K: women 0.109, men 0.306

adjusting for                           sex coefficient   what the graph says it is
nothing                                         -0.1963   the total gap
age                                             -0.1788   age is a common cause: legitimate to adjust
education.num                                   -0.1926   a mediator: removes the part of the gap that runs through schooling
hours.per.week                                  -0.1566   a mediator: removes the part through hours worked
occupation                                      -0.2017   a mediator, and a contested one
marital.status                                  -0.0340   a mediator or a proxy; the graph decides
age, education, hours                           -0.1491   the 'qualifications' set
all of them                                     -0.0432   everything we have
```

- **Adjusting for age moves the gap from -0.196 to -0.179.** Age is a common cause, so adjusting is legitimate.
- **Adjusting for education barely changes it (-0.193).** The data say that schooling does not carry much of the gap in this sample.
- **Adjusting for hours worked moves it to -0.157.** About a fifth of the gap runs through hours.
- **Adjusting for marital status moves it to -0.034.** More than 80% of the gap disappears, because marital status in this data is almost a restatement of sex and of being a primary earner. Whether that means "the gap is explained" or "the explanation is the discrimination" depends entirely on whether you draw marital status as a fair mediator or as a proxy for sex.
- **Adjusting for occupation makes the gap larger (-0.202).** Occupation is correlated with the attribute and the outcome in a way that opens a path, which is how adjustment can enlarge a gap.

None of these is the answer. They are the answers under different graphs.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/causal-graphs-for-fairness.py`; that file is generated from the block below). The label column records the role the graph gives each set; change the graph and the interpretation changes with it.

```python
import numpy as np
import pandas as pd
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial")
df = data.df
woman = data.masks["sex"].to_numpy().astype(float)
high_income = data.y.to_numpy().astype(float)      # 1 = income above $50K

def sex_coefficient(adjust):
    """Linear-probability regression of high income on being a woman, adjusting for the columns in `adjust`
    (categorical columns are one-hot encoded). Returns the coefficient on sex: the gap that remains."""
    columns = [np.ones(len(df)), woman]
    for name in adjust:
        if not pd.api.types.is_numeric_dtype(df[name]):
            columns.extend(pd.get_dummies(df[name], drop_first=True).to_numpy(dtype=float).T)
        else:
            columns.append(df[name].to_numpy(dtype=float))
    X = np.column_stack(columns)
    return np.linalg.lstsq(X, high_income, rcond=None)[0][1]

# Assumed graph (an assumption, not a finding):
#   sex -> education.num, hours.per.week, occupation, marital.status -> income;  sex -> income (direct);  age -> everything
print(f"women {woman.mean():.1%} of the rows; income above $50K: women {high_income[woman == 1].mean():.3f}, men {high_income[woman == 0].mean():.3f}")
print(f"\n{'adjusting for':38s} {'sex coefficient':>16}   what the graph says it is")
for label, adjust, meaning in (
        ("nothing", [], "the total gap"),
        ("age", ["age"], "age is a common cause: legitimate to adjust"),
        ("education.num", ["education.num"], "a mediator: removes the part of the gap that runs through schooling"),
        ("hours.per.week", ["hours.per.week"], "a mediator: removes the part through hours worked"),
        ("occupation", ["occupation"], "a mediator, and a contested one"),
        ("marital.status", ["marital.status"], "a mediator or a proxy; the graph decides"),
        ("age, education, hours", ["age", "education.num", "hours.per.week"], "the 'qualifications' set"),
        ("all of them", ["age", "education.num", "hours.per.week", "occupation", "marital.status"], "everything we have")):
    print(f"{label:38s} {sex_coefficient(adjust):+16.4f}   {meaning}")
```

## Limitations

1. **The graph is an assumption.** No amount of data confirms that an arrow is absent. Different defensible graphs give different adjusted gaps, as above.
2. **The model is linear.** A linear-probability regression with one coefficient per variable ignores interactions and non-linearity.
3. **Unmeasured causes remain.** A cause of both sex-related choices and income that is not in the data cannot be adjusted for.
4. **The "legitimate" label is a value judgement.** Whether a path counts as fair is for law, policy and the affected community, not for the analyst alone.
5. **Adjusting is not mitigating.** The adjusted gap tells you about an explanation; it does not change what the model does.

## Related Concepts

* [Path-Specific Effects](path-specific-effects.md) - splitting a gap into the paths through it
* [Counterfactual Fairness](counterfactual-fairness.md) - fairness defined on a causal model
* [Confounding Variables](confounding-variable.md) - the common-cause case
* [Simpson's Paradox](simpsons-paradox.md) - when adjusting reverses the sign
* [Proxy Variables](proxy-variables.md) - when a mediator is really a stand-in for the attribute

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Kilbertus, N. et al. (2017): Avoiding Discrimination through Causal Reasoning](https://arxiv.org/abs/1706.02744) - NeurIPS 2017; resolving and proxy variables in a causal graph.
* [Kusner, M. J., Loftus, J., Russell, C., Silva, R. (2017): Counterfactual Fairness](https://arxiv.org/abs/1703.06856) - NeurIPS 2017; fairness defined on a structural causal model.
* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press - graphs, adjustment sets and colliders.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
