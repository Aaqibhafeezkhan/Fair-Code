# Generated from explainers/cross-validation-for-fairness.md by scripts/build_explainers.py. Edit the markdown, not this file.
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
