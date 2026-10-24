# Generated from explainers/discharge-leakage-in-readmission-models.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Healthcare Readmission", strategy="unawareness")   # core + proxy columns, no race, gender or age
train, test = data.idx_train, data.idx_test
y = data.y.to_numpy()
black = (data.df["race"] == "AfricanAmerican").to_numpy().astype(int)
disposition = data.df["discharge_disposition_id"].to_numpy()
END_OF_LIFE = {11, 13, 14, 19, 20, 21}          # expired, or discharged to hospice: readmission within 30 days is very unlikely

dead = np.isin(disposition, list(END_OF_LIFE))
print(f"rows {len(y):,}; died or went to hospice {dead.sum():,} ({dead.mean():.1%}); readmitted within 30 days: "
      f"{y[dead].mean():.1%} of them, {y[~dead].mean():.1%} of everyone else")
print(f"share of African American patients among them: {black[dead].mean():.1%} (all rows: {black.mean():.1%})")

def report(name, columns, rows_train, rows_test):
    X = data.X[columns]
    model = HistGradientBoostingClassifier(random_state=data.random_state).fit(X.iloc[rows_train], y[rows_train])
    p = model.predict_proba(X.iloc[rows_test])[:, 1]
    flagged = p >= np.quantile(p, 0.89)              # flag the top 11%, about the base rate
    b = black[rows_test]
    print(f"{name:52s} AUC {roc_auc_score(y[rows_test], p):.3f} | flagged rate: African American {flagged[b == 1].mean():.3f}, "
          f"others {flagged[b == 0].mean():.3f}, gap {flagged[b == 1].mean() - flagged[b == 0].mean():+.4f}")

everything = list(data.X.columns)
without = [c for c in everything if c != "discharge_disposition_id"]
alive_train, alive_test = train[~dead[train]], test[~dead[test]]
print()
report("all rows, with discharge disposition", everything, train, test)
report("all rows, without discharge disposition", without, train, test)
report("patients who left alive only, with disposition", everything, alive_train, alive_test)
report("patients who left alive only, without disposition", without, alive_train, alive_test)
