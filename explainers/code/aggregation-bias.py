# Generated from explainers/aggregation-bias.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("COMPAS", strategy="unawareness")   # core + proxy columns, no race column
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()                             # 1 = scored High or Medium
black = data.masks["race"].to_numpy()             # 1 = Black defendants
columns = list(data.X.columns)
names = {0: "White defendants", 1: "Black defendants"}

def fit(rows):
    return LogisticRegression(max_iter=1000).fit(X[rows], y[rows])

pooled = fit(train)
separate = {g: fit(train[black[train] == g]) for g in (0, 1)}

print("accuracy on the test rows of each group")
for g in (0, 1):
    rows = test[black[test] == g]
    print(f"  {names[g]:17s} pooled model {pooled.score(X[rows], y[rows]):.3f} | model fitted on that group only {separate[g].score(X[rows], y[rows]):.3f}")

print("\ncoefficient per standard deviation: pooled | White defendants only | Black defendants only")
for i, col in enumerate(columns):
    print(f"  {col:15s} {pooled.coef_[0][i]:+.2f} | {separate[0].coef_[0][i]:+.2f} | {separate[1].coef_[0][i]:+.2f}")
