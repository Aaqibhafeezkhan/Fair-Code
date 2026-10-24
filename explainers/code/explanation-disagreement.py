# Generated from explainers/explanation-disagreement.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial", strategy="unawareness")   # core + proxy columns, no protected columns
train, test = data.idx_train, data.idx_test
X = data.X.astype(float)
y = data.y.to_numpy()
women = data.masks["sex"].to_numpy().astype(int)
columns = list(X.columns)
model = GradientBoostingClassifier(random_state=data.random_state).fit(X.iloc[train], y[train])

def permutation(rows):
    return permutation_importance(model, X.iloc[rows], y[rows], scoring="accuracy", n_repeats=10, random_state=0).importances_mean

def shap_mean_abs(rows):
    return np.abs(shap.TreeExplainer(model).shap_values(X.iloc[rows])).mean(axis=0)

methods_by_group = {}
for name, rows in (("everyone", test), ("men", test[women[test] == 0]), ("women", test[women[test] == 1])):
    methods_by_group[name] = {"permutation": permutation(rows), "SHAP": shap_mean_abs(rows)}

# A third, non-tree view: standardized logistic regression coefficients (the model is different, so this is a different question).
Z = (X - X.iloc[train].mean()) / (X.iloc[train].std() + 1e-9)
logistic = np.abs(LogisticRegression(max_iter=1000).fit(Z.iloc[train], y[train]).coef_[0])

def rank(values):
    return len(values) - np.argsort(np.argsort(values))     # 1 = most important

print(f"{'feature':16s} {'permutation':>12} {'SHAP':>6} {'|logistic coef|':>16}   (rank of 10, everyone)")
perm, sh = rank(methods_by_group["everyone"]["permutation"]), rank(methods_by_group["everyone"]["SHAP"])
lg = rank(logistic)
for i in np.argsort(sh):
    print(f"{columns[i]:16s} {perm[i]:12d} {sh[i]:6d} {lg[i]:16d}")

print("\nSpearman correlation between the rankings")
everyone = methods_by_group["everyone"]
print(f"  permutation vs SHAP, everyone: {spearmanr(everyone['permutation'], everyone['SHAP'])[0]:.2f}")
print(f"  permutation vs logistic coefficients: {spearmanr(everyone['permutation'], logistic)[0]:.2f}; SHAP vs logistic: {spearmanr(everyone['SHAP'], logistic)[0]:.2f}")
for name in ("men", "women"):
    m = methods_by_group[name]
    print(f"  permutation vs SHAP, {name}: {spearmanr(m['permutation'], m['SHAP'])[0]:.2f}")
