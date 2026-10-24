# Generated from explainers/fair-representation-learning.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial", strategy="unawareness")   # core + proxy columns, no sex column
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
s = data.masks["sex"].to_numpy()                  # 1 = women
print("columns:", list(data.X.columns))

def remove_directions(k):
    """Iterative nullspace projection: repeatedly fit a linear reader of sex on the training rows
    and project its direction out of every row."""
    P = np.eye(X.shape[1])
    for _ in range(k):
        reader = LogisticRegression(max_iter=1000).fit(X[train] @ P, s[train])
        w = reader.coef_[0] @ P
        w = w / np.linalg.norm(w)
        P = (np.eye(len(w)) - np.outer(w, w)) @ P
    return X @ P

def report(k):
    Z = remove_directions(k)
    task = LogisticRegression(max_iter=1000).fit(Z[train], y[train])
    pred = task.predict(Z[test])
    gap = pred[s[test] == 1].mean() - pred[s[test] == 0].mean()
    linear = LogisticRegression(max_iter=1000).fit(Z[train], s[train])
    forest = RandomForestClassifier(n_estimators=100, random_state=data.random_state, n_jobs=-1).fit(Z[train], s[train])
    print(f"{k} directions removed: accuracy {(pred == y[test]).mean():.3f} | parity gap {gap:+.4f} | "
          f"sex recoverable, AUC: linear reader {roc_auc_score(s[test], linear.predict_proba(Z[test])[:, 1]):.3f}, "
          f"random forest {roc_auc_score(s[test], forest.predict_proba(Z[test])[:, 1]):.3f}")

for k in (0, 1, 2, 4):
    report(k)
