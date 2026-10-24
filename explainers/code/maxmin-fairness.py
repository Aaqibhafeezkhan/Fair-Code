# Generated from explainers/maxmin-fairness.md by scripts/build_explainers.py. Edit the markdown, not this file.
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

manifest, X, y, masks, train, test = load_audit("Benefits Denial")
group = masks["sex"] * 2 + masks["race"]     # 0 male White/Asian, 1 male other race, 2 female White/Asian, 3 female other race
labels = ["male, White/Asian", "male, other race", "female, White/Asian", "female, other race"]
X_tr, X_te = X.iloc[train], X.iloc[test]
y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
g_tr, g_te = group[train], group[test]

def fit(sample_weight=None):
    model = build_model("logistic_regression", manifest.random_state)
    return model.fit(X_tr, y_tr, sample_weight=sample_weight)

def group_error(model, floor=30):
    wrong = model.predict(X_tr) != y_tr
    # groups under `floor` training rows are never "the worst off": a tiny group's error is noise
    return np.array([wrong[g_tr == k].mean() if (g_tr == k).sum() >= floor else -1.0 for k in range(4)])

def maxmin_fit(rounds=6, step=1.5):
    weight = np.ones(4)
    model = fit()
    for _ in range(rounds):
        weight[group_error(model).argmax()] *= step      # push weight onto whoever is worst off right now
        w = weight[g_tr]
        model = fit(w / w.mean())
    return model, weight

def summarise(model, name):
    pred = model.predict(X_te)
    correct = pred == y_te
    by_group = [correct[g_te == k].mean() for k in range(4)]
    parity_gap = pred[masks["sex"][test] == 1].mean() - pred[masks["sex"][test] == 0].mean()
    print(f"{name:9s} overall {correct.mean():.3f} | " +
          " | ".join(f"{lab} {acc:.3f}" for lab, acc in zip(labels, by_group)) +
          f" | worst group {min(by_group):.3f} | sex parity gap {parity_gap:+.4f}")

summarise(fit(), "baseline")
model, weight = maxmin_fit()
summarise(model, "max-min")
print("group weights", np.round(weight, 2))
