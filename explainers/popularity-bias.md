> *A ranking that learns from what people clicked will keep showing what it already showed. The items it never showed cannot earn the clicks that would move them up.*

## The One-Sentence Definition

**Popularity bias** is a feedback loop in which items that are already popular are shown more, so they collect more clicks, so they look even more popular, while items that were never shown cannot collect any.

## Why It Matters

[Fairness in Ranking](fairness-in-ranking.md) measures exposure for one list. A recommender or a search engine produces a new list every day, and today's clicks are tomorrow's training data. Abdollahpouri et al. (2019) showed that recommendation algorithms give popular items far more exposure than their share of interest justifies, and that users with niche tastes are served worst. For a marketplace or a hiring platform the same loop applies to people: whoever is ranked low early never gets the attention that would show whether they deserved a higher rank.

This is the ranking cousin of [Feedback Loop Bias](feedback-loop-bias.md). That explainer retrains a classifier on labels its own decisions shaped. Here the object is the *order of a list* and the quantity that compounds is *exposure*.

## Core Concept: Exposure Chooses the Data

Each round:

1. Rank the items by a score.
2. The position gives each item an exposure, here `1 / log2(rank + 1)` for the top `K` and zero below.
3. An item is clicked with probability exposure times how relevant it is.
4. The next score mixes the item's merit with its popularity so far, the share of rounds it was clicked: `(1 - w) * merit + w * popularity`.

Three things follow from this loop, and none of them depends on what the merit scores are:

* **The shown set freezes.** An item below `K` has no clicks, so its popularity stays at zero and it cannot rise.
* **Inside the shown set, luck replaces merit.** Early clicks are partly chance, and the loop turns them into rank.
* **Whatever group imbalance the first ranking had is locked in**, because the unseen items are exactly the ones that would change it.

## Concrete Example: AI Fair Recruitment

The [AI Fair Recruitment](../AI%20Fair%20Recruitment/) audit: the 23,062 test candidates are the items, `K` is 1,000, clicks are simulated from the hire label with probability `exposure * (0.2 + 0.6 * hired)`, and the simulation runs 30 rounds. Merit is the logistic regression's predicted hire probability. `w` is the weight of popularity in the ranker. The same loop runs twice: once with a ranker that does not use `Gender` (the benchmark's S2 feature set) and once with one that does (the baseline).

```
Gender not used by the ranker: female share of the pool 0.365, of the first top 1000 0.356
  w 0.00 | gini of exposure 0.962 | rank correlation of exposure with merit 1.000 | female share of exposure 0.368 | ever shown 1000 | hired never shown 5389
  w 0.10 | gini of exposure 0.961 | rank correlation of exposure with merit 0.855 | female share of exposure 0.369 | ever shown 1000 | hired never shown 5389
  w 0.25 | gini of exposure 0.961 | rank correlation of exposure with merit 0.614 | female share of exposure 0.368 | ever shown 1000 | hired never shown 5389
  w 0.50 | gini of exposure 0.961 | rank correlation of exposure with merit 0.500 | female share of exposure 0.367 | ever shown 1000 | hired never shown 5389
Gender used by the ranker: female share of the pool 0.365, of the first top 1000 0.070
  w 0.00 | gini of exposure 0.962 | rank correlation of exposure with merit 1.000 | female share of exposure 0.058 | ever shown 1000 | hired never shown 5395
  w 0.10 | gini of exposure 0.961 | rank correlation of exposure with merit 0.908 | female share of exposure 0.059 | ever shown 1000 | hired never shown 5395
  w 0.25 | gini of exposure 0.961 | rank correlation of exposure with merit 0.675 | female share of exposure 0.061 | ever shown 1000 | hired never shown 5395
  w 0.50 | gini of exposure 0.961 | rank correlation of exposure with merit 0.495 | female share of exposure 0.064 | ever shown 1000 | hired never shown 5395
```

**The shown set freezes.** Under every setting only the first 1,000 candidates are ever shown, and 5,389 candidates who were hired are never shown at all (5,395 for the gender-aware ranker). No amount of clicking among the first 1,000 changes that.

**Popularity replaces merit inside the shown set.** The rank correlation between a candidate's total exposure and their merit falls from 1.000 with no popularity to 0.855 at `w = 0.10`, 0.614 at `w = 0.25` and 0.500 at `w = 0.50`. Candidates are being ordered partly by which of them happened to be clicked first.

**The Gini coefficient of exposure does not show it.** It is 0.962 at `w = 0` and 0.961 at every other setting: exposure per position is fixed by the discount and the cutoff `K`, so concentration measured this way is set by the design of the list, not by the loop. A concentration metric alone would have called this system stable.

**A group imbalance is frozen, not fixed.** When the ranker does not use `Gender`, women are 36.8% of the exposure and 36.5% of the pool, and the loop leaves that where it is. When it does use `Gender`, women are 7.0% of the first top 1,000 against 36.5% of the pool, and the share of exposure they receive over 30 rounds is 5.8% with no popularity and 6.4% at `w = 0.50`. The small rise is reshuffling inside the shown set. Nothing enters from below.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/popularity-bias.py`; that file is generated from the block below).

```python
import numpy as np
from scipy.stats import spearmanr
from faircode.benchmark import load_audit
from faircode.models import build_model

K, ROUNDS = 1000, 30

def rank_of(score):
    rank = np.empty(len(score), dtype=int)
    rank[np.argsort(-score, kind="stable")] = np.arange(1, len(score) + 1)
    return rank

def gini(x):
    x = np.sort(x.astype(float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())

def simulate(merit, hired, female, w, seed):
    """The ranker scores (1 - w) * merit + w * popularity, popularity being the share of past rounds an item was clicked."""
    rng = np.random.default_rng(seed)
    clicks, exposure_total = np.zeros(len(merit)), np.zeros(len(merit))
    for r in range(1, ROUNDS + 1):
        rank = rank_of((1 - w) * merit + w * clicks / max(r - 1, 1))
        exposure = np.where(rank <= K, 1 / np.log2(rank + 1), 0.0)
        exposure_total += exposure
        # a click needs the item to be looked at (position) and, less strongly, to be relevant
        clicks += rng.random(len(merit)) < exposure * (0.2 + 0.6 * hired)
    shown = exposure_total > 0
    return (gini(exposure_total), spearmanr(exposure_total[shown], merit[shown])[0],
            (exposure_total * female).sum() / exposure_total.sum(), shown.sum(), (hired & ~shown).sum())

for strategy, title in (("unawareness_proxy_removal", "Gender not used by the ranker"), ("baseline", "Gender used by the ranker")):
    data = load_audit("AI Fair Recruitment", strategy)
    tr, te = data.idx_train, data.idx_test
    female, hired = data.group("gender")[te], data.y.iloc[te].to_numpy()
    model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
    merit = model.predict_proba(data.X.iloc[te])[:, 1]
    print(f"{title}: female share of the pool {female.mean():.3f}, of the first top {K} {female[rank_of(merit) <= K].mean():.3f}")
    for w in (0.0, 0.1, 0.25, 0.5):
        g, rho, share, n_shown, missed = simulate(merit, hired, female, w, data.random_state)
        print(f"  w {w:4.2f} | gini of exposure {g:.3f} | rank correlation of exposure with merit {rho:.3f} | female share of exposure {share:.3f} | ever shown {n_shown} | hired never shown {missed}")
```

## Limitations

1. **It is a simulation, and its constants are assumptions.** The click probability, the popularity weight `w`, the cutoff `K` and the number of rounds are chosen, not measured. The qualitative result (a frozen shown set and a merit correlation that decays) does not depend on them, but the sizes do.
2. **There is no exploration.** Real systems show a few random or new items to learn about them. That is the standard mitigation and it breaks the freeze, so the frozen set here is the worst case. See the mitigation below.
3. **Relevance comes from the hire label.** If historical decisions were biased, "relevant" inherits that bias. See [Label Bias](label-bias.md).
4. **Candidates are not independent items.** A recruiter who has looked at 1,000 people stops; a streaming service does not. The cutoff models a hard attention limit.
5. **Clicks are simulated, so the loop is closed by construction.** On real data you would estimate how much of an item's popularity is a result of its earlier exposure, for example with [Position Bias and Click Models](position-bias-and-click-models.md).

## Mitigations

* Reserve a share of positions for items below the cutoff, chosen at random or by uncertainty, so that they can earn clicks.
* Normalize clicks by exposure before they become popularity, so a click at rank 1 counts for less than a click at rank 500.
* Rank on merit estimated from outcomes instead of clicks where outcomes exist.
* Monitor the share of the catalog and of each group that was ever shown.

## Related Concepts

* [Fairness in Ranking](fairness-in-ranking.md) - exposure for one list, the quantity that accumulates here
* [Feedback Loop Bias](feedback-loop-bias.md) - the classifier version of the same loop
* [Two-Sided Fairness in a Marketplace](two-sided-fairness.md) - exposure for the ranked and utility for the searcher
* [Position Bias and Click Models](position-bias-and-click-models.md) - separating position from relevance in click data
* [Sampling Bias](sampling-bias.md) - the data you collect depends on what you showed
* [Delayed Feedback](delayed-feedback.md) - outcomes that arrive after the ranking was used

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above

## Further Reading

* [Abdollahpouri, H., Mansoury, M., Burke, R., Mobasher, B. (2019): The Unfairness of Popularity Bias in Recommendation](https://arxiv.org/abs/1907.13286) - RMSE workshop at RecSys 2019; popular items are over-recommended and niche users are served worst.
* [Singh, A., Joachims, T. (2018): Fairness of Exposure in Rankings](https://arxiv.org/abs/1802.07281) - KDD 2018; the exposure framework that popularity accumulates in.
* [Chaney, A. J. B., Stewart, B. M., Engelhardt, B. E. (2018): How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility](https://arxiv.org/abs/1710.11214) - RecSys 2018; simulations of feedback in recommenders.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
