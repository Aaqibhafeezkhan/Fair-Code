# Generated from explainers/inferring-protected-attributes.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial", strategy="unawareness")   # core + proxy columns, no sex column
train, test = data.idx_train, data.idx_test
women = data.masks["sex"].to_numpy()                          # the true group, which we pretend is missing
y = data.y.to_numpy()
rng = np.random.default_rng(0)

# 1. The audited model (the same S1 model whose gap we want to measure).
audited = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], y[train])
pred = audited.predict(data.X.iloc[test])
truth = women[test]
true_gap = pred[truth == 1].mean() - pred[truth == 0].mean()
print(f"true sex available: parity gap {true_gap:+.4f}  (women {pred[truth == 1].mean():.3f}, men {pred[truth == 0].mean():.3f})")

# 2. Infer sex from the other columns, using sex labels for only 5% of the TRAINING rows.
labelled = rng.choice(train, size=int(0.05 * len(train)), replace=False)
reader = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=data.random_state, n_jobs=-1)
reader.fit(data.X.iloc[labelled], women[labelled])
p_woman = reader.predict_proba(data.X.iloc[test])[:, 1]
print(f"inferred from {len(labelled)} labelled rows: accuracy {((p_woman >= 0.5) == truth).mean():.3f}, AUC {roc_auc_score(truth, p_woman):.3f}")

# 3. Two ways to use the inferred group to estimate the gap.
hard = p_woman >= 0.5
hard_gap = pred[hard].mean() - pred[~hard].mean()
soft_gap = (p_woman * pred).sum() / p_woman.sum() - ((1 - p_woman) * pred).sum() / (1 - p_woman).sum()
print(f"most likely group (hard label): parity gap {hard_gap:+.4f}")
print(f"probability-weighted:           parity gap {soft_gap:+.4f}")
