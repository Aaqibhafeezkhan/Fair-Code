# Generated from explainers/partial-dependence-by-group.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.inspection import partial_dependence
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X, y = data.X.astype(float), data.y.to_numpy()   # float columns: partial dependence refuses integer ones
women = data.masks["sex"].to_numpy().astype(int)
model = build_model("gradient_boosting", data.random_state).fit(X.iloc[train], y[train])
# method="brute" averages over the rows you pass (here, one group). The default "recursion" for trees ignores them and
# uses the training rows, which would draw the same curve for every group.

grid = [6, 9, 10, 12, 13, 14, 16]                # years of schooling (education.num)
print("average predicted probability of 'income above $50K' as education.num varies")
print(f"{'education.num':>14} " + " ".join(f"{g:>7}" for g in grid))
rows = {}
for g, name in ((0, "men"), (1, "women")):
    subset = X.iloc[test[women[test] == g]]
    curve = partial_dependence(model, subset, ["education.num"], custom_values={"education.num": np.array(grid)},
                               kind="average", method="brute", response_method="predict_proba")["average"][0]
    rows[name] = curve
    print(f"{name:>14} " + " ".join(f"{v:7.3f}" for v in curve))
print(f"{'gap (women-men)':>14} " + " ".join(f"{w - m:+7.3f}" for w, m in zip(rows["women"], rows["men"])))

print("\ntest rows actually observed at each value (the curve is only supported where there is data)")
for g, name in ((0, "men"), (1, "women")):
    values = X["education.num"].to_numpy()[test[women[test] == g]]
    print(f"{name:>14} " + " ".join(f"{(values == v).sum():7d}" for v in grid))
