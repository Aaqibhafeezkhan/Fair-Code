> *Where a person lives is the oldest proxy for who they are. A model never has to see race to be told it.*

## The One-Sentence Definition

A **geographic proxy** is a location feature (zip code, census area, neighbourhood) that is correlated enough with a protected attribute that a model using it can reconstruct the attribute, and **redlining** is the historical practice of denying services in areas defined largely by who lived there.

## Why It Matters

In the 1930s federal agencies drew maps grading neighbourhoods by risk, and lenders used them to deny mortgages in the areas graded lowest, which were largely Black neighbourhoods. The maps are gone, but residential segregation is not, so an address still carries information about race. Rothstein's *The Color of Law* traces the history, and later work has measured how the old grades line up with present-day neighbourhood outcomes.

For a modern model the consequence is mechanical. A location feature is cheap to collect, predictive of many outcomes and rarely thought of as sensitive, and it is exactly the kind of column [Proxy Variables](proxy-variables.md) warns about. Whether it moves a fairness gap in practice is an empirical question, and the example below shows it can go either way.

## Core Concept: Two Questions About a Location Column

1. **How much of the attribute does it reveal?** Predict the protected attribute from the location alone, on held-out rows. An AUC well above 0.5 means the location is a proxy. Compare with the model's other features to see whether it is the main route.
2. **What does it do to the gap?** Train the decision model with and without the location and compare the parity gap and accuracy. A strong proxy does not guarantee a wider gap: the location may also carry legitimate signal, and its net effect on decisions can go either way.

## Concrete Example: Tenant Screening

[Tenant Screening](../Tenant%20Screening/) has `Residence_PUMA`, a Census public-use microdata area, for each person. It is not one of the audit's features. The group is Black applicants, the model is logistic regression on the core features, and the split is the benchmark's.

```
25 areas, share of Black residents by area runs from 0.15 to 0.89
race recovered from the area alone: AUC 0.719 (0.5 is chance)
race recovered from the model's own features: AUC 0.655
core features                      accuracy 0.639 | parity gap +0.1136
core features + area               accuracy 0.639 | parity gap +0.0812
```

The 25 areas run from 15% to 89% Black among the people in them. A score that is only the training-set Black share of the person's area recovers race with an AUC of 0.719, better than the model's own features do (0.655). So the area is a strong proxy.

Then the surprise: adding the area as one-hot columns to the recidivism model leaves accuracy unchanged at 0.639 and moves the parity gap from +0.1136 to +0.0812. The proxy is strong and the gap went down. The area carries information about recidivism that the core features were encoding through other, race-correlated columns, and giving the model the area directly changed which columns it leaned on. The lesson is not that location is safe. It is that you cannot read the effect of a proxy from how strong it is, so you have to measure both.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/geographic-proxies-and-redlining.py`; that file is generated from the block below). `data.df` holds the raw rows, so a column the harness does not encode (the area) can still be read.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no race, no proxies
train, test = data.idx_train, data.idx_test
black = data.masks["race"].to_numpy().astype(int)
area = data.df["Residence_PUMA"].to_numpy()      # a Census public-use microdata area: a neighbourhood-sized region
y = data.y.to_numpy()

# 1. How much of race does the area alone reveal? Use each area's Black share in the training rows as the score.
share = pd.Series(black[train]).groupby(area[train]).mean()
score = pd.Series(area[test]).map(share).fillna(black[train].mean()).to_numpy()
print(f"{len(share)} areas, share of Black residents by area runs from {share.min():.2f} to {share.max():.2f}")
print(f"race recovered from the area alone: AUC {roc_auc_score(black[test], score):.3f} (0.5 is chance)")
core = data.X.to_numpy(dtype=float)
core = (core - core[train].mean(axis=0)) / (core[train].std(axis=0) + 1e-9)
reader = LogisticRegression(max_iter=1000).fit(core[train], black[train])
print(f"race recovered from the model's own features: AUC {roc_auc_score(black[test], reader.predict_proba(core[test])[:, 1]):.3f}")

# 2. What does adding the area do to a recidivism model's parity gap?
dummies = pd.get_dummies(pd.Series(area), prefix="area").to_numpy(dtype=float)
def report(name, features):
    model = LogisticRegression(max_iter=2000).fit(features[train], y[train])
    pred = model.predict(features[test])
    gap = pred[black[test] == 1].mean() - pred[black[test] == 0].mean()
    print(f"{name:34s} accuracy {(pred == y[test]).mean():.3f} | parity gap {gap:+.4f}")
report("core features", core)
report("core features + area", np.hstack([core, dummies]))
```

## Limitations

1. **A strong proxy is not a verdict.** The gap fell when the area was added. Treat the AUC and the gap change as two separate measurements.
2. **Area resolution matters.** A PUMA is large; zip codes and census blocks are finer and reveal more, and a coarse region reveals less.
3. **Target encoding on the training rows can overfit.** The area-share score here is fitted on the training rows and tested on held-out rows, which is the safe way round. Doing it on all rows inflates the AUC.
4. **It does not show cause.** Residential segregation, policing patterns and economic differences all feed the correlation, and the model cannot tell them apart.
5. **Dropping the location is not automatically fair.** Removing it may push the model onto other proxies, as [Fairness Through Unawareness](fairness-through-unawareness.md) shows.

## Related Concepts

* [Proxy Variables](proxy-variables.md) - the general mechanism, and a chi-squared check for a categorical proxy
* [Proxy Entanglement](proxy-entanglement.md) - when several proxies reinforce each other
* [Fairness Through Unawareness](fairness-through-unawareness.md) - why dropping the column may not help
* [Disparate Treatment](disparate-treatment.md) - when using a location is a direct, intended use
* [Inferring Protected Attributes](inferring-protected-attributes.md) - the same idea used for measurement, not for decisions

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit used for the example above

## Further Reading

* Rothstein, R. (2017). *The Color of Law: A Forgotten History of How Our Government Segregated America.* Liveright - the history of housing segregation policy.
* Aaronson, D., Hartley, D., Mazumder, B. (2021). The Effects of the 1930s HOLC "Redlining" Maps. American Economic Journal: Economic Policy 13(4) - measures the lasting effects of the maps.
* [Barocas, S., Selbst, A. D. (2016): Big Data's Disparate Impact](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2477899) - California Law Review 104; how data mining can reproduce discrimination through proxies.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
