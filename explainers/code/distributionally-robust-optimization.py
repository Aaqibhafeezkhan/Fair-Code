# Generated from explainers/distributionally-robust-optimization.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.model_selection import train_test_split
from faircode.manifest import load_manifest
from faircode.benchmark import _load_dataset      # the loader the benchmark harness uses
from faircode.strategies import encode_features
from faircode.models import build_model

def load_audit(folder):
    manifest = load_manifest(f"{folder}/audit.yaml")
    df, y, masks = _load_dataset(manifest)        # masks[name] is 1 for the disadvantaged group
    columns = list(dict.fromkeys(manifest.core_features))   # S2 feature set: no protected columns, no proxies
    X = encode_features(df, columns)
    train, test = train_test_split(np.arange(len(df)), test_size=manifest.test_size,
                                   random_state=manifest.random_state, stratify=y)
    return manifest, X, y, {k: v.to_numpy() for k, v in masks.items()}, train, test
from sklearn.metrics import log_loss

manifest, X, y, masks, train, test = load_audit("Benefits Denial")
group = masks["sex"] * 2 + masks["age"]      # 0 male under 55, 1 male 55+, 2 female under 55, 3 female 55+
labels = ["male <55", "male 55+", "female <55", "female 55+"]
X_tr, X_te = X.iloc[train], X.iloc[test]
y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
g_tr, g_te = group[train], group[test]

def fit(sample_weight=None):
    model = build_model("logistic_regression", manifest.random_state)
    return model.fit(X_tr, y_tr, sample_weight=sample_weight)

def group_loss(model, X_, y_, g_):
    p = model.predict_proba(X_)[:, 1]
    return np.array([log_loss(y_[g_ == k], p[g_ == k], labels=[0, 1]) for k in range(4)])

def group_dro(rounds=25, eta=1.0):
    n = np.bincount(g_tr, minlength=4)
    q = np.ones(4) / 4                                  # weight on each group: the adversary's choice
    model = fit()
    for _ in range(rounds):
        q = q * np.exp(eta * group_loss(model, X_tr, y_tr, g_tr))   # raise weight where loss is high
        q /= q.sum()
        w = (q / n)[g_tr]                               # per-row weight = group weight / group size
        model = fit(w / w.mean())
    return model, q

def summarise(model, name):
    pred = model.predict(X_te)
    loss = group_loss(model, X_te, y_te, g_te)
    parity_gap = pred[masks["sex"][test] == 1].mean() - pred[masks["sex"][test] == 0].mean()
    print(f"{name:12s} accuracy {(pred == y_te).mean():.3f} | per-group log-loss " +
          " ".join(f"{lab} {v:.3f}" for lab, v in zip(labels, loss)) +
          f" | worst {loss.max():.3f} | sex parity gap {parity_gap:+.4f}")

summarise(fit(), "baseline")
for eta in (0.5, 1.0, 2.0):
    model, q = group_dro(eta=eta)
    summarise(model, f"DRO eta={eta}")
    print("             adversary's group weights", np.round(q, 3))
print("test group sizes", np.bincount(g_te, minlength=4))
