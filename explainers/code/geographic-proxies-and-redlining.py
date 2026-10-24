# Generated from explainers/geographic-proxies-and-redlining.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no race, no proxies
train, test = data.idx_train, data.idx_test
black = data.masks["race"].to_numpy().astype(int)
area = data.df["Residence_PUMA"].to_numpy()      # a Census public-use microdata area: a neighbourhood-sized region
y = data.y.to_numpy()

# 1. How much of race does the area alone reveal? Use each area's Black share in the training rows as the score.
share = pd.Series(black[train]).groupby(area[train]).mean()
score = pd.Series(area[test]).map(share).fillna(black[train].mean()).to_numpy()
print(f"{len(share)} areas, share of Black residents by area runs from {share.min():.2f} to {share.max():.2f}")
print(f"race recovered from the area alone: AUC {roc_auc_score(black[test], score):.3f} (0.5 is chance)")
core = data.X.to_numpy(dtype=float)
core = (core - core[train].mean(axis=0)) / (core[train].std(axis=0) + 1e-9)
reader = LogisticRegression(max_iter=1000).fit(core[train], black[train])
print(f"race recovered from the model's own features: AUC {roc_auc_score(black[test], reader.predict_proba(core[test])[:, 1]):.3f}")

# 2. What does adding the area do to a recidivism model's parity gap?
dummies = pd.get_dummies(pd.Series(area), prefix="area").to_numpy(dtype=float)
def report(name, features):
    model = LogisticRegression(max_iter=2000).fit(features[train], y[train])
    pred = model.predict(features[test])
    gap = pred[black[test] == 1].mean() - pred[black[test] == 0].mean()
    print(f"{name:34s} accuracy {(pred == y[test]).mean():.3f} | parity gap {gap:+.4f}")
report("core features", core)
report("core features + area", np.hstack([core, dummies]))
