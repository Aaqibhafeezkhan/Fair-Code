# Generated from explainers/fairness-in-ranking.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.model_selection import train_test_split
from faircode.manifest import load_manifest
from faircode.benchmark import _load_dataset      # the loader the benchmark harness uses
from faircode.strategies import encode_features
from faircode.models import build_model

def load_audit(folder, with_gender=False):
    manifest = load_manifest(f"{folder}/audit.yaml")
    df, y, masks = _load_dataset(manifest)        # masks[name] is 1 for the disadvantaged group
    columns = list(dict.fromkeys(manifest.core_features + manifest.proxy_features + (['Gender'] if with_gender else [])))
    X = encode_features(df, columns)
    train, test = train_test_split(np.arange(len(df)), test_size=manifest.test_size,
                                   random_state=manifest.random_state, stratify=y)
    return manifest, X, y, {k: v.to_numpy() for k, v in masks.items()}, train, test

manifest, X_biased, y, masks, train, test = load_audit("AI Fair Recruitment", with_gender=True)
_, X_fair, *_ = load_audit("AI Fair Recruitment", with_gender=False)
female = masks["gender"][test]                      # 1 = female candidate (the disadvantaged group)
hired = y.iloc[test].to_numpy()
K = 1000                                            # only the top 1,000 of 23,062 candidates are ever looked at

def scores(X):
    model = build_model("logistic_regression", manifest.random_state)
    return model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])[:, 1]

def exposure(score, k=None):
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(1, len(score) + 1)
    e = 1 / np.log2(rank + 1)                       # position discount: rank 1 -> 1.0, rank 3 -> 0.5, rank 7 -> 0.33
    return e if k is None else np.where(rank <= k, e, 0.0)

def report(score, name):
    top = np.argsort(-score, kind="stable")[:K]
    whole, cut = exposure(score), exposure(score, K)
    print(f"{name}\n  female share of the top {K}: {female[top].mean():.3f} (pool {female.mean():.3f})"
          f"\n  exposure ratio female/male, whole list: {whole[female == 1].mean() / whole[female == 0].mean():.3f}"
          f"\n  exposure ratio female/male, top {K} only: {cut[female == 1].mean() / cut[female == 0].mean():.3f}"
          f"\n  exposure per unit of hire rate, female/male: "
          f"{(cut[female == 1].mean() / hired[female == 1].mean()) / (cut[female == 0].mean() / hired[female == 0].mean()):.3f}")

report(scores(X_biased), "baseline (Gender is a feature)")
report(scores(X_fair), "mitigated (Gender removed)")
