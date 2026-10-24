# Generated from explainers/fairness-regularization.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy(dtype=float)
s = data.masks["sex"].to_numpy(dtype=float)      # 1 = women, the disadvantaged group
sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def fit(lam, steps=3000, lr=0.5):
    """Minimise  log-loss + lam * (soft parity gap)^2, where the soft gap is the difference in the
    group means of the predicted PROBABILITY (differentiable), not of the thresholded decision."""
    Xtr, ytr, str_ = X[train], y[train], s[train]
    w, b = np.zeros(X.shape[1]), 0.0
    g1, g0 = str_ == 1, str_ == 0
    for _ in range(steps):
        p = sigmoid(Xtr @ w + b)
        soft_gap = p[g1].mean() - p[g0].mean()
        dp = (p - ytr) / len(ytr)                                  # d log-loss / d z
        dgap = np.where(g1, 1 / g1.sum(), -1 / g0.sum()) * p * (1 - p)   # d soft_gap / d z
        dz = dp + lam * 2 * soft_gap * dgap
        w -= lr * Xtr.T @ dz
        b -= lr * dz.sum()
    return w, b

print(f"{'lambda':>8} {'accuracy':>9} {'soft gap':>9} {'thresholded gap':>16}")
for lam in (0, 1, 10, 100, 1000, 10000):
    w, b = fit(lam)
    p = sigmoid(X[test] @ w + b)
    pred = (p >= 0.5).astype(int)
    s_te = s[test]
    print(f"{lam:8d} {(pred == y[test]).mean():9.3f} {p[s_te == 1].mean() - p[s_te == 0].mean():+9.4f} "
          f"{pred[s_te == 1].mean() - pred[s_te == 0].mean():+16.4f}")
