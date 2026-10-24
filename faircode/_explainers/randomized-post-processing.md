> *When no single cut-off makes two groups' error rates equal, mix two decisions with a weighted coin. It can reach the target exactly, and it decides some people by chance.*

## The One-Sentence Definition

**Randomized post-processing** is a post-processing rule that, for each group and score range, predicts "positive" with a chosen probability instead of always or never, so that the groups' true and false positive rates can be made equal exactly.

## Why It Matters

[Group-Specific Thresholds](group-specific-thresholds.md) equalize one error rate by moving a cut-off. For equalized odds you need two rates to match at once, a true positive rate and a false positive rate, and a pair of thresholds usually cannot reach both. Hardt, Price and Srebro (2016) showed that the set of achievable pairs is the region under each group's ROC curve, and that the best equal-odds point can be reached by mixing two thresholds: use one with probability `p` and the other with probability `1 - p`.

That gives an exact answer and an uncomfortable one. Two applicants with the same score and group can get different decisions because of a random draw.

## Core Concept: A Linear Programme Over Score Bins

Divide scores into bins. For each group and bin, the variable `q` is the probability of a positive decision. The group's true positive rate is the sum of `q` weighted by how many true positives fall in each bin, and likewise for false positives. Choose all `q` to maximize accuracy subject to the two groups having the same true positive rate and the same false positive rate. It is a small linear programme.

```
maximize   accuracy(q)
subject to TPR(group 0; q) = TPR(group 1; q)
           FPR(group 0; q) = FPR(group 1; q)
           0 <= q <= 1
```

Most of the `q` come out as 0 or 1 (an ordinary threshold). Only the bin where the cut-off falls needs a fractional value, and the coarser the score, the more it matters.

## Concrete Example: Tenant Screening

[Tenant Screening](../Tenant%20Screening/): logistic regression scores, the disadvantaged group is Black applicants, the rules are fitted on the benchmark's training rows and evaluated on its test rows. The score is cut into 4, 6 and 20 bins to show what coarseness does.

```
one threshold at 0.5                     accuracy 0.639 | TPR white 0.725 black 0.809 | FPR white 0.460 black 0.609 | equalized-odds gap 0.1486

4 score bins: 2 of 8 probabilities are strictly between 0 and 1
  white: 0.00 0.48 1.00 1.00
  black: 0.00 0.00 1.00 1.00
randomized post-processing (4 bins), one draw accuracy 0.633 | TPR white 0.653 black 0.668 | FPR white 0.415 black 0.400 | equalized-odds gap 0.0153
  over 20 random draws: accuracy 0.633 +/- 0.003; the same person gets a different decision in 5.9% of rows between two draws

6 score bins: 2 of 12 probabilities are strictly between 0 and 1
  white: 0.03 0.58 1.00 1.00 1.00 1.00
  black: 0.00 0.00 1.00 1.00 1.00 1.00
randomized post-processing (6 bins), one draw accuracy 0.636 | TPR white 0.817 black 0.808 | FPR white 0.601 black 0.606 | equalized-odds gap 0.0088
  over 20 random draws: accuracy 0.637 +/- 0.002; the same person gets a different decision in 4.1% of rows between two draws

20 score bins: 2 of 40 probabilities are strictly between 0 and 1
  white: 0.25 0.00 0.00 0.00 0.00 0.01 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
  black: 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
randomized post-processing (20 bins), one draw accuracy 0.638 | TPR white 0.765 black 0.754 | FPR white 0.515 black 0.535 | equalized-odds gap 0.0202
  over 20 random draws: accuracy 0.636 +/- 0.002; the same person gets a different decision in 1.2% of rows between two draws

deterministic group thresholds (match TPR) accuracy 0.640 | TPR white 0.725 black 0.712 | FPR white 0.460 black 0.472 | equalized-odds gap 0.0133
```

- **A single threshold of 0.5 is far from equal odds:** true positive rate 0.725 for White and 0.809 for Black applicants, false positive rate 0.460 against 0.609, an equalized-odds gap of 0.149.
- **The programme closes it almost entirely:** gaps of 0.015, 0.009 and 0.020 at 4, 6 and 20 bins, at an accuracy of 0.633 to 0.638 against 0.639. The cost is under a point.
- **Few probabilities are fractional.** Two of 8, 12 or 40 are strictly between 0 and 1; the rest are plain thresholds, so the randomness is concentrated in one score range for one group.
- **People get different decisions on different draws:** between two draws 5.9% (4 bins), 4.1% (6) and 1.2% (20) of rows change. A finer score confines the randomness to fewer people.
- **A deterministic pair of thresholds does about as well here:** equalized-odds gap 0.0133 at accuracy 0.640, because the group ROC curves are close enough that the threshold pair nearly reaches the target. Randomization matters when no pair gets there.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/randomized-post-processing.py`; that file is generated from the block below).

```python
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no race, no proxies
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
group = data.masks["race"].to_numpy().astype(int)     # 1 = Black applicants
score = LogisticRegression(max_iter=1000).fit(X[train], y[train]).predict_proba(X)[:, 1]
def rates(pred, rows):
    out = {}
    for g in (0, 1):
        r = rows[group[rows] == g]
        out[g] = (pred[r][y[r] == 1].mean(), pred[r][y[r] == 0].mean())   # (TPR, FPR)
    return out

def report(name, pred):
    r = rates(pred, test)
    acc = (pred[test] == y[test]).mean()
    print(f"{name:40s} accuracy {acc:.3f} | TPR white {r[0][0]:.3f} black {r[1][0]:.3f} | FPR white {r[0][1]:.3f} black {r[1][1]:.3f} "
          f"| equalized-odds gap {max(abs(r[1][0] - r[0][0]), abs(r[1][1] - r[0][1])):.4f}")

report("one threshold at 0.5", (score >= 0.5).astype(int))

def solve(bins):
    """Hardt, Price, Srebro: for each group and score bin, the probability q of predicting positive so that both groups
    have the same TPR and FPR, maximizing accuracy. A linear programme in 2 x bins probabilities."""
    edges = np.quantile(score[train], np.linspace(0, 1, bins + 1)[1:-1])
    bin_of = np.digitize(score, edges)
    P = {(g, b, label): np.mean((group[train] == g) & (bin_of[train] == b) & (y[train] == label))
         for g in (0, 1) for b in range(bins) for label in (0, 1)}
    Pg = {(g, label): np.mean((group[train] == g) & (y[train] == label)) for g in (0, 1) for label in (0, 1)}
    index = lambda g, b: g * bins + b
    c = np.zeros(2 * bins)                            # maximize accuracy = minimize -(P(y=1, q) + P(y=0, 1-q))
    A_eq = np.zeros((2, 2 * bins))
    for g, sign in ((0, 1), (1, -1)):
        for b in range(bins):
            c[index(g, b)] = -(P[(g, b, 1)] - P[(g, b, 0)])
            A_eq[0, index(g, b)] = sign * P[(g, b, 1)] / Pg[(g, 1)]    # TPR(group 0) - TPR(group 1) = 0
            A_eq[1, index(g, b)] = sign * P[(g, b, 0)] / Pg[(g, 0)]    # FPR(group 0) - FPR(group 1) = 0
    q = linprog(c, A_eq=A_eq, b_eq=[0, 0], bounds=[(0, 1)] * (2 * bins)).x
    return q, bin_of

for bins in (4, 6, 20):
    q, bin_of = solve(bins)
    fractional = int(((q > 1e-6) & (q < 1 - 1e-6)).sum())
    draws = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pred = (rng.random(len(y)) < q[group * bins + bin_of]).astype(int)
        draws.append(pred)
    accs = np.array([(d[test] == y[test]).mean() for d in draws])
    print(f"\n{bins} score bins: {fractional} of {2 * bins} probabilities are strictly between 0 and 1")
    for g, name in ((0, "white"), (1, "black")):
        print(f"  {name}: " + " ".join(f"{q[g * bins + b]:.2f}" for b in range(bins)))
    report(f"randomized post-processing ({bins} bins), one draw", draws[0])
    print(f"  over 20 random draws: accuracy {accs.mean():.3f} +/- {accs.std(ddof=1):.3f}; the same person gets a different decision in "
          f"{np.mean([(draws[0][test] != d[test]).mean() for d in draws[1:]]):.1%} of rows between two draws")

# Deterministic alternative for comparison: one threshold per group, chosen on the training rows to match TPR.
t_white = 0.5
target = (score[train][(group[train] == 0) & (y[train] == 1)] >= t_white).mean()
t_black = np.quantile(score[train][(group[train] == 1) & (y[train] == 1)], 1 - target)
print()
report("deterministic group thresholds (match TPR)", np.where(group == 1, score >= t_black, score >= t_white).astype(int))
```

## Limitations

1. **It decides some people by chance.** Two identical applicants can be treated differently, which is hard to defend in lending and hiring and may be unlawful. See the legal discussion in [Group-Specific Thresholds](group-specific-thresholds.md).
2. **It needs the group at prediction time,** as group thresholds do.
3. **It equalizes the rates it is told to.** Here that is equalized odds. Calibration and predictive parity are not preserved, see [Fairness Metric Conflicts](fairness-metric-conflicts.md).
4. **The result depends on the draw.** Fix the random seed for any decision that must be reproducible, and say so.
5. **It is fitted on training scores.** The guarantee is on the training distribution, and the test gaps (0.009 to 0.020) show some slippage.

## Related Concepts

* [Group-Specific Thresholds](group-specific-thresholds.md) - the deterministic version
* [Equalized Odds](equalized-odds.md) - the criterion this achieves
* [Reject Option Classification](reject-option-classification.md) - another post-processing rule, deterministic and local
* [Mitigation Strategies](mitigation-strategies.md) - where post-processing sits
* [Fairness-Accuracy Pareto Frontier](fairness-accuracy-pareto-frontier.md) - the trade-off these rules traverse

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit used for the example above

## Further Reading

* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - NeurIPS 2016; the derived predictor and the linear programme.
* [Woodworth, B., Gunasekar, S., Ohannessian, M. I., Srebro, N. (2017): Learning Non-Discriminatory Predictors](https://arxiv.org/abs/1702.06081) - COLT 2017; limits of post-processing for equalized odds.
* [Corbett-Davies, S., Goel, S. (2018): The Measure and Mismeasure of Fairness](https://arxiv.org/abs/1808.00023) - a critique of fairness definitions, including randomized rules.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
