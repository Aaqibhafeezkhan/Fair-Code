# Generated from explainers/race-in-clinical-models.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

with_race = load_audit("Healthcare Readmission", strategy="baseline")      # core + proxy + race, gender and age
without = load_audit("Healthcare Readmission", strategy="unawareness")     # the same without the protected columns
train, test = with_race.idx_train, with_race.idx_test
y = with_race.y.to_numpy()
black = (with_race.df["race"] == "AfricanAmerican").to_numpy().astype(int)[test]

def fit(data):
    X = data.X.to_numpy(dtype=float)
    Z = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
    model = LogisticRegression(max_iter=2000).fit(Z[train], y[train])
    return model.predict_proba(Z[test])[:, 1]

def report(name, p):
    y_te = y[test]
    # Use the score to pick the 10% of patients for a follow-up programme with a fixed number of places.
    chosen = p >= np.quantile(p, 0.90)
    print(f"{name}")
    print(f"  AUC {roc_auc_score(y_te, p):.3f}")
    for g, label in ((1, "African American"), (0, "other patients ")):
        rows = black == g
        print(f"  {label}: mean predicted risk {p[rows].mean():.3f} vs observed {y_te[rows].mean():.3f} "
              f"| given a place {chosen[rows].mean():.3f} | of those who were readmitted, given a place {chosen[rows & (y_te == 1)].mean():.3f}")

report("with race in the model", fit(with_race))
report("without race (and without gender and age)", fit(without))
