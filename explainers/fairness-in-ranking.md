> *A model can score two candidates almost the same and still bury one of them on page three. Classification fairness metrics cannot see that harm, because it only exists once you add position.*

## The One-Sentence Definition

**Fairness in ranking** asks whether groups receive their fair share of *exposure* - the attention an item gets from its position in an ordered list - not just whether they receive similar scores.

## Why It Matters

Every metric elsewhere in this repo ([Demographic Parity](demographic-parity.md), [Equalized Odds](equalized-odds.md), [Predictive Parity](predictive-parity.md)) is defined for a binary decision: hired or not, flagged or not, approved or not. A search engine, a job-recommendation feed or a marketplace results page does not make one decision per person. It produces an ordered list, and where someone lands decides how many people ever look at them. Users click near the top; position bias is large and well documented.

So two candidates with nearly identical scores can get very different outcomes, and the damage is a group-level one that only appears when a ranking function and an attention model are stacked on top of the scores.

## Core Concept: Exposure

Exposure is a position-weighted measure of the attention an item receives. A standard choice borrowed from information retrieval discounts by the log of the rank:

```
exposure(rank) = 1 / log2(rank + 1)       rank 1 -> 1.00,  rank 3 -> 0.50,  rank 7 -> 0.33
```

Fairness of exposure (Singh and Joachims, 2018) asks whether groups with the same average relevance get the same average exposure. It is not enough for the groups' average *scores* to match. Compare it with demographic parity: parity looks at a rate over a decision, exposure looks at a distribution over positions.

Where the list is cut off matters. A hiring team that looks at the top 1,000 of 23,062 candidates gives zero exposure to everyone below rank 1,000, and averaging a discount over the full list washes that out.

## Concrete Example: AI Fair Recruitment

The [AI Fair Recruitment](../AI%20Fair%20Recruitment/) audit, logistic regression, the benchmark's own split (23,062 test candidates, 36.5% of them women). Instead of thresholding at 0.5, sort the test candidates by predicted hire probability. Two rankings: the baseline model with `Gender` as a feature, and the same model with `Gender` removed.

```
baseline (Gender is a feature)
  female share of the top 1000: 0.070 (pool 0.365)
  exposure ratio female/male, whole list: 0.955
  exposure ratio female/male, top 1000 only: 0.107
  exposure per unit of hire rate, female/male: 0.116
mitigated (Gender removed)
  female share of the top 1000: 0.356 (pool 0.365)
  exposure ratio female/male, whole list: 1.001
  exposure ratio female/male, top 1000 only: 1.015
  exposure per unit of hire rate, female/male: 1.099
```

The whole-list exposure ratio for the baseline is 0.955, which looks like near parity. Only the top-1,000 cutoff shows what is happening: women get 0.107 of men's exposure and make up 7.0% of the shortlist against 36.5% of the pool. Removing `Gender` brings the shortlist to 35.6% women. Dividing exposure by each group's actual hire rate (24.6% for women, 26.6% for men) gives 1.099 after mitigation: women now get about 10% more exposure per unit of hire rate than men.

For scale, the benchmark's classification rows for the same audit in `results/` (the current results) show a demographic parity gap of -0.0656 for the baseline and -0.0065 without `Gender`. A 6.6 point gap in a threshold decision became a shortlist with 7% women. The classification metric understates it.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/fairness-in-ranking.py`; that file is generated from the block below). The code ranks by score, applies the log discount, and reports exposure ratios with and without a cutoff.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from faircode.manifest import load_manifest
from faircode.benchmark import _load_dataset      # the loader the benchmark harness uses
from faircode.strategies import encode_features
from faircode.models import build_model

def load_audit(folder, with_gender=False):
    manifest = load_manifest(f"{folder}/audit.yaml")
    df, y, masks = _load_dataset(manifest)        # masks[name] is 1 for the disadvantaged group
    columns = list(dict.fromkeys(manifest.core_features + manifest.proxy_features + (['Gender'] if with_gender else [])))
    X = encode_features(df, columns)
    train, test = train_test_split(np.arange(len(df)), test_size=manifest.test_size,
                                   random_state=manifest.random_state, stratify=y)
    return manifest, X, y, {k: v.to_numpy() for k, v in masks.items()}, train, test

manifest, X_biased, y, masks, train, test = load_audit("AI Fair Recruitment", with_gender=True)
_, X_fair, *_ = load_audit("AI Fair Recruitment", with_gender=False)
female = masks["gender"][test]                      # 1 = female candidate (the disadvantaged group)
hired = y.iloc[test].to_numpy()
K = 1000                                            # only the top 1,000 of 23,062 candidates are ever looked at

def scores(X):
    model = build_model("logistic_regression", manifest.random_state)
    return model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])[:, 1]

def exposure(score, k=None):
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(1, len(score) + 1)
    e = 1 / np.log2(rank + 1)                       # position discount: rank 1 -> 1.0, rank 3 -> 0.5, rank 7 -> 0.33
    return e if k is None else np.where(rank <= k, e, 0.0)

def report(score, name):
    top = np.argsort(-score, kind="stable")[:K]
    whole, cut = exposure(score), exposure(score, K)
    print(f"{name}\n  female share of the top {K}: {female[top].mean():.3f} (pool {female.mean():.3f})"
          f"\n  exposure ratio female/male, whole list: {whole[female == 1].mean() / whole[female == 0].mean():.3f}"
          f"\n  exposure ratio female/male, top {K} only: {cut[female == 1].mean() / cut[female == 0].mean():.3f}"
          f"\n  exposure per unit of hire rate, female/male: "
          f"{(cut[female == 1].mean() / hired[female == 1].mean()) / (cut[female == 0].mean() / hired[female == 0].mean()):.3f}")

report(scores(X_biased), "baseline (Gender is a feature)")
report(scores(X_fair), "mitigated (Gender removed)")
```

## Limitations

1. **The discount function is an assumption.** `1 / log2(rank + 1)` is a convention. A steeper or flatter attention curve changes every number above. Estimate it from real click data when you have it.
2. **The cutoff is an assumption too.** The whole-list ratio was 0.955 and the top-1,000 ratio 0.107 for the same model. Report the cutoff you used.
3. **Equal exposure is not always the goal.** Singh and Joachims ask for exposure proportional to merit. The "per unit of hire rate" line is that idea in a crude form, and it overshoots to 1.099 here.
4. **Re-ranking trades off with utility.** Forcing equal exposure moves some higher-scored candidates down. That is a [fairness-accuracy trade-off](fairness-accuracy-tradeoff.md) in a different shape, and it is measured with ranking utility instead of accuracy.
5. **Hire rate is a noisy relevance label.** If the historical hiring decisions are biased, "relevance" inherits the bias. See [Label Bias](label-bias.md).

## Related Concepts

* [Demographic Parity](demographic-parity.md) - the classification counterpart that exposure generalizes
* [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) - re-ranking costs utility
* [Subgroup Fairness](subgroup-fairness.md) - exposure can differ for intersections the aggregate hides
* [Label Bias](label-bias.md) - when the relevance label is itself biased
* [Feedback Loop Bias](feedback-loop-bias.md) - ranked results shape the clicks that retrain the ranker
* [What Is Popularity Bias?](popularity-bias.md) - exposure that compounds when a ranker learns from its own clicks
* [What Is Two-Sided Fairness in a Marketplace?](two-sided-fairness.md) - exposure for the ranked and utility for the searcher on one sweep

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows quoted for the classification gap

## Further Reading

* [Singh, A., Joachims, T. (2018): Fairness of Exposure in Rankings](https://arxiv.org/abs/1802.07281) - KDD 2018; the exposure framework and the proportional-to-merit criterion.
* [Biega, A. J., Gummadi, K. P., Weikum, G. (2018): Equity of Attention: Amortizing Individual Fairness in Rankings](https://arxiv.org/abs/1805.01788) - SIGIR 2018; fair attention for individual items across a series of rankings.
* [Jarvelin, K., Kekalainen, J. (2002): Cumulated Gain-Based Evaluation of IR Techniques](https://dl.acm.org/doi/10.1145/582415.582418) - ACM TOIS 20(4); the log-discount position model used above.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
