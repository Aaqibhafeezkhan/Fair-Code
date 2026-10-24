> *A single test split uses a fifth of the data once. Cross-validation tests every row once, and tells you how far the answer depends on which fifth you held out.*

## The One-Sentence Definition

**Cross-validation for fairness** splits the data into folds, trains on all but one, predicts the held-out fold, and pools the held-out predictions so that every row is evaluated exactly once, with the folds stratified by group and label so that no fold is missing the small cells.

## Why It Matters

[Seed and Split Sensitivity](seed-and-split-sensitivity.md) shows that on a small audit the gap from one 80/20 split can be a long way from the gap from another. Cross-validation is the standard answer for accuracy and works for fairness metrics with one change: the fold assignment has to protect the small groups. Plain random folds can put a handful of people from the smallest group-by-label cell into one test fold and almost none into another, which makes the per-fold gaps meaningless.

## Core Concept: Stratify on the Group-by-Label Cell

1. **Define a cell** for each combination of group and label (here: under 30 or not, good or bad credit, four cells).
2. **Split with stratification on the cell,** so each fold holds about a fifth of every cell.
3. **Pool the out-of-fold predictions** and compute the gap on all rows.
4. **Repeat with different seeds** and report the spread of the pooled gap, and the spread of the per-fold gaps as a measure of how unstable a single fold is.

## Concrete Example: German Credit Lending

[German Credit Lending](../German%20Credit%20Lending/): 1,000 applicants, the disadvantaged group is under 30, logistic regression on the core columns. Plain 5-fold is compared with 5-fold stratified by group and label, each repeated 20 times with different shuffles.

```
plain 5-fold                       pooled gap -0.0593 (sd across repeats 0.0108) | sd of the 5 fold gaps 0.0564 | smallest group-by-label cell in any test fold: 18 rows
5-fold stratified by group and label pooled gap -0.0645 (sd across repeats 0.0094) | sd of the 5 fold gaps 0.0552 | smallest group-by-label cell in any test fold: 27 rows
the benchmark split (200 test rows) gap -0.0914
```

- **The pooled gap is more stable and less extreme than the single split.** The benchmark split gives -0.0914 on its 200 test rows. Pooling all 1,000 held-out predictions gives about -0.059 (plain) or -0.065 (stratified), with a standard deviation across repeats of about 0.01. The single split is one of the unlucky draws from the previous explainer's 30-seed spread.
- **A single fold is very noisy.** The standard deviation of the five per-fold gaps is 0.056 under both schemes. Any one fold's gap tells you little.
- **Stratifying protects the small cells.** With plain folds the smallest group-by-label cell in any test fold dropped to 18 rows; with stratified folds it never went below 27. Both schemes give similar pooled gaps here, but the stratified one cannot produce a fold in which a cell is almost empty.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/cross-validation-for-fairness.py`; that file is generated from the block below). `StratifiedKFold` takes the cell code as its label.

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("German Credit Lending")       # 1,000 rows; the disadvantaged group is applicants under 30
X, y = data.X, data.y.to_numpy()
young = data.masks["age"].to_numpy().astype(int)
cell = young * 2 + y                              # group-by-label cells: 4 of them
rng = np.random.default_rng(0)

def gap_of(pred, rows):
    return pred[rows][young[rows] == 1].mean() - pred[rows][young[rows] == 0].mean()

def cross_validate(splitter, label, repeats=20):
    pooled, spread, smallest_cells = [], [], []
    for r in range(repeats):
        folds = list(splitter(r).split(X, cell))
        out = np.empty(len(y), dtype=int)
        fold_gaps, tiny = [], []
        for train, test in folds:
            model = build_model("logistic_regression", data.random_state).fit(X.iloc[train], y[train])
            out[test] = model.predict(X.iloc[test])
            fold_gaps.append(gap_of(out, test))
            tiny.append(np.bincount(cell[test], minlength=4).min())
        pooled.append(gap_of(out, np.arange(len(y))))
        spread.append(np.std(fold_gaps, ddof=1))
        smallest_cells.append(min(tiny))
    print(f"{label:34s} pooled gap {np.mean(pooled):+.4f} (sd across repeats {np.std(pooled, ddof=1):.4f}) | "
          f"sd of the 5 fold gaps {np.mean(spread):.4f} | smallest group-by-label cell in any test fold: {min(smallest_cells)} rows")

cross_validate(lambda r: KFold(5, shuffle=True, random_state=r), "plain 5-fold")
cross_validate(lambda r: StratifiedKFold(5, shuffle=True, random_state=r), "5-fold stratified by group and label")

# Compare with the single 80/20 split the benchmark uses.
model = build_model("logistic_regression", data.random_state).fit(X.iloc[data.idx_train], y[data.idx_train])
pred = np.zeros(len(y), dtype=int)
pred[data.idx_test] = model.predict(X.iloc[data.idx_test])
print(f"{'the benchmark split (200 test rows)':34s} gap {gap_of(pred, data.idx_test):+.4f}")
```

## Limitations

1. **Folds are not independent.** Each model trains on 80% of the same rows, so the five fold results are correlated and a naive standard error from them is too small.
2. **The pooled gap mixes five models.** It is a property of the procedure, not of one trained model. The model you deploy is a sixth one, fitted on everything.
3. **Stratifying on the cell assumes you know the groups.** It needs the protected attribute for every row, and for intersections the cells multiply.
4. **It does not give a p-value on its own.** Bootstrap the pooled predictions, with the caveat that the models behind them overlap. See [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md).
5. **It is more work for a large dataset.** The single-split value is already stable on a large audit, see the Benefits Denial result in [Seed and Split Sensitivity](seed-and-split-sensitivity.md).

## Related Concepts

* [Seed and Split Sensitivity](seed-and-split-sensitivity.md) - why one split is not enough on a small audit
* [Stratified Test Sets for Fairness](stratified-test-sets-for-fairness.md) - sizing the evaluation data to detect a gap
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - an interval on the pooled gap
* [Class Imbalance](class-imbalance.md) - the same stratification idea for rare labels
* [Subgroup Fairness](subgroup-fairness.md) - why small cells matter

## Related Projects in This Repo

* [`German Credit Lending/`](../German%20Credit%20Lending/) - the audit used for the example above

## Further Reading

* [Hastie, T., Tibshirani, R., Friedman, J. (2009): The Elements of Statistical Learning, chapter 7](https://hastie.su.domains/ElemStatLearn/) - Springer; model assessment and selection by cross-validation.
* Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. IJCAI - the case for stratified cross-validation.
* [Friedler, S. A. et al. (2019): A Comparative Study of Fairness-Enhancing Interventions in Machine Learning](https://arxiv.org/abs/1802.04422) - FAT* 2019; fairness results across many random splits.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
