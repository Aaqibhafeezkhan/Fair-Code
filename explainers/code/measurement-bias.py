# Generated from explainers/measurement-bias.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
y_tr, y_te = data.y.iloc[train].to_numpy(), data.y.iloc[test].to_numpy()
women = data.masks["sex"].to_numpy()
rng = np.random.default_rng(0)

def measured(column, scale=1.0, noise=0.0):
    """Return the feature matrix as if `column` were mis-measured for women only: multiplied by
    `scale` (systematic under-recording) and/or with Gaussian noise (sd = noise * the column's sd)."""
    X = data.X.copy()
    col = X[column].to_numpy(dtype=float)
    bad = col * scale + rng.normal(0, noise * col.std(), len(col))
    X[column] = np.where(women == 1, bad, col)
    return X

def report(label, X):
    model = build_model("logistic_regression", data.random_state).fit(X.iloc[train], y_tr)
    pred = model.predict(X.iloc[test])
    w = women[test]
    print(f"{label:46s} accuracy {(pred == y_te).mean():.3f} | positive rate women {pred[w == 1].mean():.3f} "
          f"men {pred[w == 0].mean():.3f} | parity gap {pred[w == 1].mean() - pred[w == 0].mean():+.4f}")

report("true values", data.X)
report("education.num credited 15% low for women", measured("education.num", scale=0.85))
report("education.num noisy for women (sd 1.0x)", measured("education.num", noise=1.0))
report("capital.gain recorded 50% low for women", measured("capital.gain", scale=0.5))
