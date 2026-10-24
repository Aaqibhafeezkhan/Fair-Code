# Generated from explainers/group-specific-thresholds.md by scripts/build_explainers.py. Edit the markdown, not this file.
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

manifest, X, y, masks, train, test = load_audit("Tenant Screening")
s_tr, s_te = masks["race"][train], masks["race"][test]
y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
model = build_model("logistic_regression", manifest.random_state).fit(X.iloc[train], y_tr)
p_tr, p_te = model.predict_proba(X.iloc[train])[:, 1], model.predict_proba(X.iloc[test])[:, 1]

def report(pred, name):
    tpr = [pred[(s_te == g) & (y_te == 1)].mean() for g in (0, 1)]
    print(f"{name:34s} accuracy {(pred == y_te).mean():.3f} | positive rate advantaged {pred[s_te == 0].mean():.3f} "
          f"disadvantaged {pred[s_te == 1].mean():.3f} | parity gap {pred[s_te == 1].mean() - pred[s_te == 0].mean():+.4f} "
          f"| TPR gap {tpr[1] - tpr[0]:+.4f}")

report((p_te >= 0.5).astype(int), "one threshold (0.5)")

# Demographic parity: give the disadvantaged group the threshold that matches the advantaged group's positive rate.
# Fit the threshold on the TRAINING scores only, then apply it to the test set.
target_rate = (p_tr[s_tr == 0] >= 0.5).mean()
t_dp = np.quantile(p_tr[s_tr == 1], 1 - target_rate)
report(np.where(s_te == 1, p_te >= t_dp, p_te >= 0.5).astype(int), "group thresholds, parity")
print(f"  disadvantaged-group threshold: {t_dp:.3f} (advantaged stays 0.500)")

# Equal opportunity: match the true-positive rate instead.
target_tpr = (p_tr[(s_tr == 0) & (y_tr == 1)] >= 0.5).mean()
t_eo = np.quantile(p_tr[(s_tr == 1) & (y_tr == 1)], 1 - target_tpr)
report(np.where(s_te == 1, p_te >= t_eo, p_te >= 0.5).astype(int), "group thresholds, equal opportunity")
print(f"  disadvantaged-group threshold: {t_eo:.3f}")
