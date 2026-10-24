# Generated from explainers/disaggregated-evaluation.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Insurance Denial")            # S2 feature set; groups are age x gender
train, test = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], data.y.iloc[train])
pred = model.predict(data.X.iloc[test])
y_te = data.y.iloc[test].to_numpy()
older = data.masks["age"].to_numpy()[test]       # 1 = the disadvantaged age group
female = data.masks["gender"].to_numpy()[test]   # 1 = the disadvantaged gender group
rng = np.random.default_rng(0)

def row(label, mask):
    correct = (pred == y_te)[mask]
    boots = [rng.choice(correct, len(correct)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label:28s} n={mask.sum():4d}  accuracy {correct.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}")

print(f"overall accuracy {(pred == y_te).mean():.3f}\n")
row("everyone", np.ones(len(y_te), dtype=bool))
for a, an in ((0, "age: advantaged"), (1, "age: disadvantaged")):
    row(an, older == a)
for g, gn in ((0, "gender: advantaged"), (1, "gender: disadvantaged")):
    row(gn, female == g)
for a in (0, 1):
    for g in (0, 1):
        row(f"age {a} x gender {g}", (older == a) & (female == g))
