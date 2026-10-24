> *People click near the top whatever is there. A model trained on those clicks learns where things were, not what was good.*

## The One-Sentence Definition

**Position bias** is the tendency to click results because of where they sit in a list; a **click model** describes how likely each position is to be looked at, so that clicks can be read as evidence of relevance rather than of position.

## Why It Matters

Click logs look like free relevance labels: a click means somebody wanted it. But a result at rank 1 is looked at far more often than one at rank 20, so it is clicked more often at the same quality. If a ranker is trained on raw clicks, it learns to prefer whatever the previous ranker put on top.

That is a fairness problem as soon as the old order correlated with a sensitive or proxy feature, because the new model turns a legacy ordering into a weight on that feature. [Popularity Bias](popularity-bias.md) is the same mechanism accumulating over time. Here the question is smaller and answerable: how to read one log without being fooled by position.

## Core Concept: The Position-Based Model and Inverse Propensity Weighting

The position-based click model says

```
P(click on item i at rank r) = P(examined at rank r) * P(item i is relevant)
```

Examination depends only on the rank. If it is known, an item's relevance can be estimated without bias by weighting each click by `1 / P(examined at its rank)`: a click at a position nobody looks at counts for more, a click at rank 1 counts for about one. This is inverse propensity weighting, and Joachims, Swaminathan and Schnabel (2017) showed that a ranker trained with it is unbiased for the true relevance if the propensities are right.

Everything depends on that "if". The propensities have to be known or estimated, for example by randomizing a small part of the traffic, and the click model has to be a good description of how people look at a list.

## Concrete Example: AI Fair Recruitment

The [AI Fair Recruitment](../AI%20Fair%20Recruitment/) audit. The 23,062 test candidates are split at random into lists of 20, like pages of results shown to a recruiter. A legacy ranker orders each list mostly by `Age` (plus some experience and test score, plus noise). Clicks are simulated with the position-based model: examination is `rank^-1` (rank 1 always looked at, rank 20 one time in twenty) and relevance is the hire label, so the true relevance is known and can be used to grade the result. Three kinds of ranker are fitted on the click log of 807 lists using the same three features (`Experience_Years`, `Technical_Test_Score`, `Age`), and graded on 346 held-out lists.

```
807 training lists, click rate by position: rank 1: 0.398, rank 2: 0.182, rank 5: 0.066, rank 10: 0.037, rank 20: 0.007 | hire rate at rank 1: 0.379, at rank 20: 0.134
true eta is 1.0
  ranker                    | weight on Age | AUC vs hired | nDCG@5
  raw clicks                |         0.884 |        0.681 |  0.469
  weighted, assumed eta 0.5 |         0.471 |        0.715 |  0.498
  weighted, assumed eta 1.0 |         0.132 |        0.745 |  0.534
  weighted, assumed eta 2.0 |        -0.346 |        0.668 |  0.447
  true labels (oracle)      |        -0.022 |        0.749 |  0.546
```

**Position drives the clicks.** The click rate falls from 0.398 at rank 1 to 0.007 at rank 20, and so does the hire rate at those ranks (0.379 and 0.134), because the legacy order partly follows merit.

**A ranker trained on raw clicks learns the legacy order.** Its weight on `Age` is 0.884, where the label-trained oracle gives -0.022. It falls to an AUC of 0.681 against the hire label and an nDCG@5 of 0.469, against 0.749 and 0.546 for the oracle. It is worse than the oracle because it learned that old candidates were shown first.

**Weighting by the exposure curve recovers the oracle.** With the true curve (`eta = 1.0`) the weight on `Age` falls to 0.132, the AUC is 0.745 and nDCG@5 is 0.534, close to the oracle's.

**It depends on the propensities being right.** Assuming a flatter curve (`eta = 0.5`) removes only part of the bias (weight on `Age` 0.471, AUC 0.715). Assuming a steeper one (`eta = 2.0`) overcorrects: the weight on `Age` goes to -0.346, and the ranker is worse than with no correction at all, in AUC (0.668 against 0.681) and in nDCG@5 (0.447 against 0.469). A wrong propensity is not a small error, it can make the ranker worse than doing nothing.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/position-bias-and-click-models.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit

data = load_audit("AI Fair Recruitment")
cols = ["Experience_Years", "Technical_Test_Score", "Age"]            # core features plus the proxy feature Age
X_all = data.X_all[cols].iloc[data.idx_test].to_numpy(float)
X_all = (X_all - X_all.mean(0)) / X_all.std(0)
hired = data.y.iloc[data.idx_test].to_numpy()                        # true relevance, known here because we simulate the clicks
rng = np.random.default_rng(data.random_state)
LIST_SIZE, ETA = 20, 1.0

# 1. Split the candidates into lists of 20, the way a recruiter sees a page of results.
order = rng.permutation(len(hired))
lists = order[: len(order) // LIST_SIZE * LIST_SIZE].reshape(-1, LIST_SIZE)
train_lists, test_lists = lists[: len(lists) * 7 // 10], lists[len(lists) * 7 // 10:]

# 2. The logging policy: a legacy ranker that sorts mostly by Age, with some merit in it.
def log_rank(idx):
    key = X_all[idx, 2] + 0.5 * (X_all[idx, 0] + X_all[idx, 1]) + rng.normal(0, 0.5, len(idx))
    return np.argsort(np.argsort(-key)) + 1

# 3. Position-based click model: P(click) = P(examined | rank) * P(relevant), examined = rank^-eta.
rows, ranks, clicks = [], [], []
for idx in train_lists:
    rank = log_rank(idx)
    examined = rank ** -ETA
    clicks.append(rng.random(len(idx)) < examined * (0.05 + 0.9 * hired[idx]))
    rows.append(idx); ranks.append(rank)
rows, ranks, clicks = map(np.concatenate, (rows, ranks, clicks))
print(f"{len(train_lists)} training lists, click rate by position: " +
      ", ".join(f"rank {r}: {clicks[ranks == r].mean():.3f}" for r in (1, 2, 5, 10, 20)) +
      f" | hire rate at rank 1: {hired[rows][ranks == 1].mean():.3f}, at rank 20: {hired[rows][ranks == 20].mean():.3f}")

# 4. Rankers: trained on raw clicks, on clicks weighted by 1 / P(examined) with the right and wrong exposure curves, and on true labels.
def fit(target, weight=None):
    return LogisticRegression(max_iter=1000).fit(X_all[rows], target, sample_weight=weight)

def ipw(eta):
    return np.where(clicks, ranks ** eta, 1.0)                       # inverse propensity weights on the clicked rows

models = {"raw clicks": fit(clicks),
          "weighted, assumed eta 0.5": fit(clicks, ipw(0.5)),
          "weighted, assumed eta 1.0": fit(clicks, ipw(1.0)),
          "weighted, assumed eta 2.0": fit(clicks, ipw(2.0)),
          "true labels (oracle)": fit(hired[rows])}

def ndcg_at_5(model):
    total = 0
    discount = 1 / np.log2(np.arange(2, 7))
    for idx in test_lists:
        gains = hired[idx][np.argsort(-model.decision_function(X_all[idx]))][:5]
        ideal = np.sort(hired[idx])[::-1][:5]
        total += (gains * discount).sum() / max((ideal * discount).sum(), 1e-9)
    return total / len(test_lists)

test_rows = test_lists.ravel()
print(f"true eta is {ETA}")
print("  ranker                    | weight on Age | AUC vs hired | nDCG@5")
for name, m in models.items():
    print(f"  {name:25s} | {m.coef_[0][2]:13.3f} | {roc_auc_score(hired[test_rows], m.decision_function(X_all[test_rows])):12.3f} | {ndcg_at_5(m):6.3f}")
```

## Limitations

1. **The propensities must be known or estimated, and the result is only as good as they are.** The `eta = 2.0` row is a ranker that is worse than the one with no correction. Estimate propensities from randomized traffic or an intervention, not from a guess.
2. **The position-based model is a simplification.** Real users also look at what is around a result (context effects), stop after a good result (cascade behaviour), and click differently on mobile. Other click models handle some of that.
3. **Inverse weights can have large variance.** A click at rank 20 here gets a weight of 20. With real, smaller samples, clip the weights and report intervals.
4. **The click model only helps if clicks are caused by position and relevance.** Clicks driven by a thumbnail, a price or a name carry a bias that a position model does not remove.
5. **The simulation knows the true relevance.** In a real log you only see clicks, so the comparison with an oracle above cannot be repeated; use a randomized holdout to measure what you cannot see.

## Related Concepts

* [Popularity Bias](popularity-bias.md) - the accumulation of position effects over rounds
* [Fairness in Ranking](fairness-in-ranking.md) - the exposure measure these clicks are weighted by
* [Two-Sided Fairness in a Marketplace](two-sided-fairness.md) - weighing exposure for the ranked against utility for the searcher
* [Label Bias](label-bias.md) - clicks as a biased label
* [Sampling Bias](sampling-bias.md) - the log only holds what was shown
* [Proxy Variables](proxy-variables.md) - the feature a legacy order leaks into

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above

## Further Reading

* [Joachims, T., Swaminathan, A., Schnabel, T. (2017): Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468) - WSDM 2017; inverse propensity weighting for clicks.
* [Craswell, N., Zoeter, O., Taylor, M., Ramsey, B. (2008): An Experimental Comparison of Click Position-Bias Models](https://doi.org/10.1145/1341531.1341545) - WSDM 2008; the position-based and cascade models compared.
* [Wang, X., Golbandi, N., Bendersky, M., Metzler, D., Najork, M. (2018): Position Bias Estimation for Unbiased Learning to Rank in Personal Search](https://doi.org/10.1145/3159652.3159732) - WSDM 2018; estimating propensities from a log.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
