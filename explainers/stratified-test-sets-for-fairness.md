> *If the group you most need to measure is a quarter of your test set, you are measuring it with a quarter of your data.*

## The One-Sentence Definition

A **stratified test set for fairness** is an evaluation set built so that each group (and each group-by-label cell) has enough rows to be measured, usually by sampling small groups at a higher rate and reweighting to recover population rates when an overall number is needed.

## Why It Matters

A random test set contains each group in proportion to the population, so the smallest groups get the fewest rows and the widest intervals. The group a fairness audit cares most about is often exactly that one. If you can choose or collect the evaluation data, oversampling the small group buys narrower intervals for the same labelling budget.

It is also a trap, because an oversampled set no longer reflects the population. An overall accuracy computed on it is biased towards the oversampled group's behaviour, and has to be corrected by weighting each row by its group's population share divided by its share in the sample.

## Core Concept: Choose the Mix, Then Weight Back

1. **Choose the target size** for each group (or cell) from how narrow an interval you need, see [Minimum Detectable Gap](minimum-detectable-gap.md).
2. **Sample the groups at different rates** to hit those sizes.
3. **Report per-group metrics unweighted:** they are valid inside each group.
4. **Report any overall metric with weights** `population share / sample share` per group.

## Concrete Example: German Credit Lending

[German Credit Lending](../German%20Credit%20Lending/): the model is logistic regression on the core columns, with out-of-fold predictions for all 1,000 applicants so each has an honest prediction. The disadvantaged group is the 371 applicants under 30. Evaluation sets of 200 rows are drawn 100 times each way: a plain random sample, and a stratified sample that is half under-30s.

```
all 1,000 rows: accuracy 0.709 overall, 0.663 for under-30s (371 people)

evaluation sets of 200 rows, 100 draws each
design                            under-30s in it   width of the under-30 interval    overall accuracy: unweighted / reweighted
random sample                                  74                            0.213                  0.708 / 0.709
stratified: half under 30                     100                            0.183                  0.694 / 0.703
```

- **Narrower interval for the group that matters.** The random sample holds 74 under-30s on average and gives a bootstrap interval for their accuracy 0.213 wide. The stratified one holds 100 and gives 0.183, about 14% narrower for the same 200 labels. The gain is modest because 100 is not much more than 74; a more extreme mix would narrow it further at the cost of the other group's interval.
- **The unweighted overall number is biased.** The stratified sample's plain accuracy is 0.694 against a true 0.709 for all 1,000 rows, because under-30s are less accurately predicted (0.663) and are over-represented. Weighting each row by its group's population share over its sample share gives 0.703, within sampling noise of the truth.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/stratified-test-sets-for-fairness.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.model_selection import cross_val_predict
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("German Credit Lending")       # 1,000 applicants; the disadvantaged group is those under 30
X, y = data.X, data.y.to_numpy()
young = data.group("age")                         # 1 = under 30, as a plain 0/1 array
# Out-of-fold predictions for everyone, so every row has an honest prediction to evaluate.
pred = cross_val_predict(build_model("logistic_regression", data.random_state), X, y, cv=10)
correct = (pred == y).astype(float)
truth = correct[young == 1].mean()
print(f"all 1,000 rows: accuracy {correct.mean():.3f} overall, {truth:.3f} for under-30s ({(young == 1).sum()} people)")

rng = np.random.default_rng(1)
def interval_width(values):
    boots = [rng.choice(values, len(values)).mean() for _ in range(500)]
    return np.subtract(*np.percentile(boots, [97.5, 2.5]))

def draw(size, share_young):
    n_young = int(round(size * share_young))
    rows = np.concatenate([rng.choice(np.flatnonzero(young == 1), n_young, replace=False),
                           rng.choice(np.flatnonzero(young == 0), size - n_young, replace=False)])
    return rows

print("\nevaluation sets of 200 rows, 100 draws each")
print(f"{'design':32s} {'under-30s in it':>16} {'width of the under-30 interval':>32} {'overall accuracy: unweighted / reweighted':>44}")
for label, share in (("random sample", None), ("stratified: half under 30", 0.5)):
    widths, plain, weighted, counts = [], [], [], []
    for _ in range(100):
        rows = rng.choice(len(y), 200, replace=False) if share is None else draw(200, share)
        group = young[rows]
        counts.append(group.sum())
        widths.append(interval_width(correct[rows][group == 1]))
        plain.append(correct[rows].mean())
        # reweight each row by population share of its group divided by its share in the sample
        w = np.where(group == 1, (young == 1).mean() / max(group.mean(), 1e-9), (young == 0).mean() / max(1 - group.mean(), 1e-9))
        weighted.append(np.average(correct[rows], weights=w))
    print(f"{label:32s} {np.mean(counts):16.0f} {np.mean(widths):32.3f} {np.mean(plain):22.3f} / {np.mean(weighted):.3f}")
```

## Limitations

1. **It needs the group labels before sampling.** Stratifying on the protected attribute presupposes you know it for the pool you sample from.
2. **Weights add variance.** A reweighted overall estimate is unbiased but noisier than an unweighted one from a random sample, and extreme weights make it unstable.
3. **Stratifying does not fix a biased pool.** If the pool under-represents a subgroup, no sampling rate recovers it, see [Sampling Bias](sampling-bias.md).
4. **Per-group metrics are valid only inside the group.** Combining groups with unequal rates needs the weights; do not average per-group numbers without them.
5. **More strata means smaller cells.** Stratifying on three attributes at once leaves few rows per cell, see [Disaggregated Evaluation](disaggregated-evaluation.md).

## Related Concepts

* [Minimum Detectable Gap](minimum-detectable-gap.md) - how many rows a gap of a given size needs
* [Cross-Validation for Fairness](cross-validation-for-fairness.md) - using all rows as test data
* [Disaggregated Evaluation](disaggregated-evaluation.md) - the per-group reporting this makes possible
* [Sampling Bias](sampling-bias.md) - when the pool itself is not representative
* [Class Imbalance](class-imbalance.md) - the same rebalancing idea for labels

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the audit used for the example above

## Further Reading

* Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley - stratified sampling and the weights that undo it.
* [Barocas, S. et al. (2021): Designing Disaggregated Evaluations of AI Systems](https://arxiv.org/abs/2103.06076) - AIES 2021; choosing groups and sample sizes for per-group evaluation.
* Lohr, S. L. (2021). Sampling: Design and Analysis (3rd ed.). Chapman and Hall - a modern treatment of weighting and stratification.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
