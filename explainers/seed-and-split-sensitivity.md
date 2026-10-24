> *Every fairness number in this repo comes from one split with one seed. How much of it is the seed?*

## The One-Sentence Definition

**Seed and split sensitivity** is how much a reported fairness metric changes when the only thing that changes is the random seed that decides which rows land in the training set and which in the test set.

## Why It Matters

The benchmark reports each metric from a single 80/20 split with `random_state = 42`, with a bootstrap interval and a permutation p-value around it ([Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md)). Those quantify the sampling noise of the test set for that split. They do not say what a different split would have given, because the split itself is a random draw.

On a large dataset the two coincide and the question is academic. On a small one it is not: the test set is a few hundred rows, a handful of people can move a gap by points, and the training set changes as well. Anyone quoting a single number from a small audit should know how much it would move.

## Core Concept: Repeat the Split, Report the Spread

```
for seed in 30 different seeds:
    split the audit's rows, train the same model, compute the gap, accuracy and p-value
report the mean, standard deviation and range, and how often the verdict changes
```

Compare the spread with the single value in `results/`. If the standard deviation is a large share of the gap, the single number is one draw from a wide distribution.

## Concrete Example: A Small Audit and a Large One

Two audits with the S2 logistic regression and the manifest's split, repeated with 30 seeds each. The first row of each block is the benchmark's own seed, which matches `results/` (the current results).

```
German Credit Lending - age, 30 different seeds for the split
  parity gap: seed 42 gives -0.0914; the 30 seeds give mean -0.0637, sd 0.0498, range -0.2029 to +0.0337
  accuracy:   seed 42 gives 0.700; mean 0.719, sd 0.028
  the gap is significant (p < 0.05) in 13% of the splits; its sign differs from the seed-42 sign in 7%
Benefits Denial - sex, 30 different seeds for the split
  parity gap: seed 42 gives -0.0509; the 30 seeds give mean -0.0542, sd 0.0063, range -0.0686 to -0.0432
  accuracy:   seed 42 gives 0.811; mean 0.811, sd 0.003
  the gap is significant (p < 0.05) in 100% of the splits; its sign differs from the seed-42 sign in 0%
```

**German Credit Lending (1,000 rows, 200 in the test set)** shows the problem:

- The benchmark seed gives a parity gap of -0.0914. Across 30 seeds the mean is -0.0637 and the standard deviation 0.0498, almost as large as the mean. The range runs from -0.2029 to +0.0337, so one seed in the 30 gives a gap of the opposite sign.
- The gap is significant at 0.05 in only 13% of the splits. The `results/` row has a p-value of 0.17 and is one of the 87% that are not.
- Accuracy moves less (0.719 mean, 0.028 standard deviation) but the benchmark seed is 0.700.

**Benefits Denial (32,561 rows, 6,513 in the test set)** shows the reverse. The gap is -0.0509 for the benchmark seed and -0.0542 on average with a standard deviation of 0.0063, it is significant in all 30 splits, and its sign never changes.

The takeaway is not that small audits are useless. It is that the size of the sampling noise is itself a result: with 200 test rows the gap's seed-to-seed spread is as large as the effect.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/seed-and-split-sensitivity.py`; that file is generated from the block below). `load_audit(folder, random_state=seed)` keeps the manifest's own split recipe and changes only the seed.

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model
from faircode.significance import permutation_test

def one_split(folder, attribute, seed):
    data = load_audit(folder, random_state=seed)          # the manifest's own split, but with another seed
    train, test = data.idx_train, data.idx_test
    model = build_model("logistic_regression", seed).fit(data.X.iloc[train], data.y.iloc[train])
    pred = model.predict(data.X.iloc[test])
    group = data.masks[attribute].to_numpy()[test]
    a, b = pred[group == 1].astype(float), pred[group == 0].astype(float)
    p = permutation_test(a, b, n_permutations=500, random_state=seed)
    return a.mean() - b.mean(), (pred == data.y.iloc[test].to_numpy()).mean(), p

SEEDS = range(30)
for folder, attribute, benchmark_seed in (("German Credit Lending", "age", 42), ("Benefits Denial", "sex", 42)):
    rows = np.array([one_split(folder, attribute, s) for s in SEEDS])
    gap, acc, p = rows[:, 0], rows[:, 1], rows[:, 2]
    base = one_split(folder, attribute, benchmark_seed)
    print(f"{folder} - {attribute}, 30 different seeds for the split")
    print(f"  parity gap: seed {benchmark_seed} gives {base[0]:+.4f}; the 30 seeds give mean {gap.mean():+.4f}, sd {gap.std(ddof=1):.4f}, "
          f"range {gap.min():+.4f} to {gap.max():+.4f}")
    print(f"  accuracy:   seed {benchmark_seed} gives {base[1]:.3f}; mean {acc.mean():.3f}, sd {acc.std(ddof=1):.3f}")
    print(f"  the gap is significant (p < 0.05) in {np.mean(p < 0.05):.0%} of the splits; its sign differs from the seed-{benchmark_seed} sign in "
          f"{np.mean(np.sign(gap) != np.sign(base[0])):.0%}")
```

## Limitations

1. **Thirty seeds estimate the spread roughly.** The standard deviation itself has noise, and the range depends on how many draws you take.
2. **The splits overlap.** Each split reuses the same 1,000 rows, so the thirty results are correlated, and the spread is a lower bound on the uncertainty about the population.
3. **It measures split noise, not model noise.** The seed also changes the model's initialisation here, and a non-linear model has more of it.
4. **It does not say which seed is right.** Do not pick the seed that gives the gap you prefer. Report the spread, or use cross-validation, see [Cross-Validation for Fairness](cross-validation-for-fairness.md).
5. **It costs training time.** Thirty fits is cheap for a logistic regression and not for a large model.

## Related Concepts

* [Cross-Validation for Fairness](cross-validation-for-fairness.md) - using every row as test data exactly once
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the uncertainty inside one split
* [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md) - the p-value that changes verdict across splits here
* [Disaggregated Evaluation](disaggregated-evaluation.md) - why small groups make every number noisier
* [Sampling Bias](sampling-bias.md) - the population the rows are a sample of

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the small audit
* [`Benefits Denial/`](../Benefits%20Denial/) - the large audit
* [`results/`](../results/) - the single-seed rows these are compared with

## Further Reading

* [Friedler, S. A. et al. (2019): A Comparative Study of Fairness-Enhancing Interventions in Machine Learning](https://arxiv.org/abs/1802.04422) - FAT* 2019; reports how results vary across random train-test splits.
* [Bouthillier, X. et al. (2021): Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) - MLSys 2021; the sources of variance in a benchmark and how to account for them.
* Efron, B., Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall - resampling to estimate the spread of a statistic.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
