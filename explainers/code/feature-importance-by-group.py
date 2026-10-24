# Generated from explainers/feature-importance-by-group.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.inspection import permutation_importance
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X, y = data.X, data.y.to_numpy()
women = data.masks["sex"].to_numpy().astype(int)
model = build_model("gradient_boosting", data.random_state).fit(X.iloc[train], y[train])

def importance(rows):
    """Drop in accuracy when one column is shuffled, on `rows` only; mean and standard deviation over 20 shuffles."""
    result = permutation_importance(model, X.iloc[rows], y[rows], scoring="accuracy", n_repeats=20, random_state=0)
    return result.importances_mean, result.importances_std

everyone = importance(test)
by_group = {name: importance(test[women[test] == g]) for g, name in ((0, "men"), (1, "women"))}
print(f"gradient boosting, accuracy {model.score(X.iloc[test], y[test]):.3f}; test rows: {(women[test] == 0).sum()} men, {(women[test] == 1).sum()} women")
print(f"{'feature':15s} {'everyone':>10} {'men':>18} {'women':>18}")
for i, column in enumerate(X.columns):
    men, women_ = by_group["men"], by_group["women"]
    print(f"{column:15s} {everyone[0][i]:+10.4f} {men[0][i]:+10.4f} +/-{men[1][i]:.4f} {women_[0][i]:+10.4f} +/-{women_[1][i]:.4f}")
