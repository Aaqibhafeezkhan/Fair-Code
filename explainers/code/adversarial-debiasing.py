# Generated from explainers/adversarial-debiasing.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-9
X = (X - mean) / std
y = data.y.to_numpy(dtype=float)
s = data.masks["race"].to_numpy(dtype=float)      # 1 = Black applicants, the disadvantaged group
sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def train_predictor(alpha, steps=1500, lr=0.5):
    """Predictor p = sigmoid(X w + b). Adversary a = sigmoid(u * z + c) tries to read race from the
    predictor's own logit z. The predictor minimises task loss - alpha * adversary loss."""
    Xtr, ytr, str_ = X[train], y[train], s[train]
    w, b, u, c = np.zeros(X.shape[1]), 0.0, 0.1, 0.0
    for _ in range(steps):
        z = Xtr @ w + b
        p, a = sigmoid(z), sigmoid(u * z + c)
        # adversary step: get better at predicting race from z
        u -= lr * np.mean((a - str_) * z)
        c -= lr * np.mean(a - str_)
        # predictor step: fit the label, and push the adversary's loss UP
        dz = (p - ytr) - alpha * (a - str_) * u
        w -= lr * Xtr.T @ dz / len(ytr)
        b -= lr * dz.mean()
    return w, b

def report(alpha):
    w, b = train_predictor(alpha)
    z = X[test] @ w + b
    pred = (sigmoid(z) >= 0.5).astype(int)
    s_te, y_te = s[test], y[test]
    gap = pred[s_te == 1].mean() - pred[s_te == 0].mean()
    leak = roc_auc_score(s_te, z)                 # how well a threshold-free reader recovers race from the score
    print(f"alpha {alpha:4.1f}: accuracy {(pred == y_te).mean():.3f} | parity gap {gap:+.4f} | race recoverable from the score, AUC {leak:.3f}")

for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
    report(alpha)
