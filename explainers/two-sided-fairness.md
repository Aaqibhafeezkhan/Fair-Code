> *A marketplace has two sets of people to be fair to: the ones being ranked and the ones doing the searching. Helping one can cost the other, but not always, and not by much until it overshoots.*

## The One-Sentence Definition

**Two-sided fairness** asks a ranking to be fair to the people or items being ranked (they get a fair share of exposure) and to the people who use it (they get a good list), and recognizes that the two can pull in different directions.

## Why It Matters

Burke (2017) pointed out that recommenders and marketplaces are multisided: providers want exposure, consumers want relevance, and the platform has its own goals. A fairness fix aimed at one side is easy to describe as a cost to the other: "if we show more of the under-exposed group, the list gets worse for the searcher". Whether that is true is an empirical question, and it should be answered with both numbers on the same sweep, not asserted.

[Fairness in Ranking](fairness-in-ranking.md) defines the ranked side's metric, exposure. This explainer adds the other side's metric, a ranking utility such as nDCG, and puts both on one curve.

## Core Concept: Two Numbers on One Sweep

Take a ranking and a re-ranking that has one knob. Here the knob is a constant added to the scores of the under-exposed group before sorting. For each setting of the knob, report:

* **The ranked side:** the share of the top `K` that belongs to the group, and the exposure each group receives per unit of hire rate (1.0 is exposure in proportion to merit, as in Singh and Joachims 2018).
* **The searching side:** nDCG at `K` against the true outcome, the position-discounted share of the best possible list that the re-ranking achieves. Here relevance is whether the candidate was hired.

The sweep shows how far one side can be helped before the other starts to pay.

## Concrete Example: AI Fair Recruitment

The [AI Fair Recruitment](../AI%20Fair%20Recruitment/) audit with the baseline logistic regression, which uses `Gender`. The 23,062 test candidates are ranked, the top 1,000 are shown, and 36.5% of the pool is female. The shift column is the constant added to the female candidates' predicted hire probability before sorting.

```
AI Fair Recruitment, Gender used by the ranker: 23062 candidates, top 1000 shown, 0.365 of the pool female
  shift | female share of top-1000 | exposure per unit of hire rate, female/male | nDCG@1000 | hired in top-1000
   0.00 |                  0.070 |                                      0.116 |   0.562 | 568
   0.01 |                  0.129 |                                      0.229 |   0.566 | 572
   0.02 |                  0.183 |                                      0.350 |   0.570 | 577
   0.03 |                  0.247 |                                      0.522 |   0.569 | 576
   0.04 |                  0.303 |                                      0.727 |   0.573 | 581
   0.05 |                  0.361 |                                      1.208 |   0.572 | 571
   0.06 |                  0.423 |                                      1.691 |   0.571 | 570
   0.08 |                  0.532 |                                      2.712 |   0.564 | 562
   0.10 |                  0.662 |                                      4.632 |   0.560 | 558
   0.15 |                  0.972 |                                     80.519 |   0.545 | 540
```

**With no shift, the ranked side is badly served.** Women are 7.0% of the top 1,000 against 36.5% of the pool, and receive 0.116 of men's exposure per unit of hire rate.

**Both sides improve at first.** At a shift of 0.05 women are 36.1% of the top 1,000 and the exposure ratio is 1.208, close to parity. The searching side does not pay: nDCG@1000 rises from 0.562 to 0.572, and the number of hired candidates in the top 1,000 goes from 568 to 571. The best nDCG is 0.573 at a shift of 0.04, where the female share is 30.3%. The `Gender` term was not tracking hiring outcomes: the hire rates differ by under 3 points in the data (24.0% for women, 26.9% for men), far less than the scores did.

**Then they conflict.** Past about 0.05 the ranked side is over-served and the searching side pays: at 0.10 women are 66.2% of the shortlist and nDCG falls to 0.560, and at 0.15 they are 97.2% and nDCG is 0.545 with 540 hired candidates. The exposure ratio per unit of hire rate runs from 1.208 at 0.05 to 80.519 at 0.15, so a target of parity (1.0) falls between 0.04 and 0.05.

The lesson is the shape, not the numbers: a stretch where the sides do not conflict (the baseline ranker was unfair *and* worse for the searcher), then a point past which they do. A platform needs both curves to see where it is.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/two-sided-fairness.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit
from faircode.models import build_model

data = load_audit("AI Fair Recruitment", "baseline")          # the ranker uses Gender, as in the baseline
tr, te = data.idx_train, data.idx_test
female = data.group("gender")[te]                             # 1 = female candidate (the people being ranked)
hired = data.y.iloc[te].to_numpy()                            # relevance for the people searching: was this candidate hired
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
score = model.predict_proba(data.X.iloc[te])[:, 1]
K = 1000

def rank_of(s):
    rank = np.empty(len(s), dtype=int)
    rank[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
    return rank

def dcg(relevance_in_order):
    return (relevance_in_order / np.log2(np.arange(2, len(relevance_in_order) + 2))).sum()

ideal = dcg(np.sort(hired)[::-1][:K])                          # the best possible top-K list

print(f"AI Fair Recruitment, Gender used by the ranker: {len(score)} candidates, top {K} shown, {female.mean():.3f} of the pool female")
print("  shift | female share of top-%d | exposure per unit of hire rate, female/male | nDCG@%d | hired in top-%d" % (K, K, K))
for shift in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15):
    rank = rank_of(score + shift * female)                    # re-rank: lift the under-exposed group's scores by a constant
    exposure = np.where(rank <= K, 1 / np.log2(rank + 1), 0.0)
    ratio = (exposure[female == 1].mean() / hired[female == 1].mean()) / (exposure[female == 0].mean() / hired[female == 0].mean())
    top = np.argsort(rank)[:K]
    print(f"  {shift:5.2f} | {female[top].mean():22.3f} | {ratio:42.3f} | {dcg(hired[top]) / ideal:7.3f} | {hired[top].sum():d}")
```

## Limitations

1. **The shift is chosen by looking at the test set.** The sweep describes the trade-off. To deploy a setting, choose it on held-out validation data, then report on the test set.
2. **The searching side's relevance is the hire label.** If past hiring was biased, nDCG rewards reproducing that bias and understates what the searcher gains from a fairer list. See [Label Bias](label-bias.md).
3. **Utility here is one measure.** nDCG@1000 against hire labels says nothing about diversity, the searcher's own preferences or long-run effects. Choose the measure the searcher actually cares about.
4. **It uses the group at ranking time.** A constant added for one group is explicit use of a protected attribute, which can be unlawful in hiring and lending. See [Disparate Treatment](disparate-treatment.md) and [Group-Specific Thresholds](group-specific-thresholds.md).
5. **The knob is crude.** A constant shift treats every candidate in the group alike; the exposure ratio swings from 0.727 to 1.208 across 0.01 of shift because the scores are packed together near the top. Methods such as fair top-k re-ranking and amortized exposure are finer.

## Related Concepts

* [Fairness in Ranking](fairness-in-ranking.md) - exposure for the ranked side
* [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md) - the classification version of the same question
* [Popularity Bias](popularity-bias.md) - exposure that compounds over rounds
* [Position Bias and Click Models](position-bias-and-click-models.md) - how clicks credit position
* [Demographic Parity](demographic-parity.md) - the classification counterpart of equal exposure
* [Label Bias](label-bias.md) - when the relevance label is itself biased

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above

## Further Reading

* [Burke, R. (2017): Multisided Fairness for Recommendation](https://arxiv.org/abs/1707.00093) - FAT/ML 2017; the case for fairness to more than one party.
* [Singh, A., Joachims, T. (2018): Fairness of Exposure in Rankings](https://arxiv.org/abs/1802.07281) - KDD 2018; exposure proportional to merit.
* [Zehlike, M. et al. (2017): FA*IR: A Fair Top-k Ranking Algorithm](https://arxiv.org/abs/1706.06368) - CIKM 2017; a re-ranking that guarantees a group's share of a top-k list.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
