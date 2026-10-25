# Generated from explainers/decision-curve-analysis-by-group.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit
from faircode.models import build_model

def net_benefit(y, p, t):
    """TP/n - FP/n * t/(1-t): true positives, minus false positives weighted by the odds of the threshold."""
    pred = p >= t
    return (pred & (y == 1)).mean() - (pred & (y == 0)).mean() * t / (1 - t)

def treat_all(y, t):
    return y.mean() - (1 - y.mean()) * t / (1 - t)

def useful_ranges(y, p, grid):
    """Threshold ranges where the model beats both treating everyone and treating no one."""
    ok = [net_benefit(y, p, t) > max(treat_all(y, t), 0) for t in grid]
    ranges, start, last = [], None, None
    for t, flag in zip(grid, ok):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            ranges.append((start, last)); start = None
        last = t
    if start is not None:
        ranges.append((start, grid[-1]))
    return ranges

data = load_audit("COMPAS")
tr, te = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
p = model.predict_proba(data.X.iloc[te])[:, 1]
y = data.y.iloc[te].to_numpy()
s = data.group("race")[te]                           # 1 = disadvantaged group
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

print("COMPAS - race: net benefit per 100 people at each threshold probability")
print("  threshold | advantaged: model  treat-all | disadvantaged: model  treat-all")
for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    row = [f"{100 * f(y[s == g], p[s == g], t) if f is net_benefit else 100 * f(y[s == g], t):7.2f}"
           for g in (0, 1) for f in (net_benefit, lambda yy, *a: treat_all(yy, a[-1]))]
    print(f"  {t:9.2f} | {row[0]}  {row[1]}      | {row[2]}  {row[3]}")
for g, name in ((0, "advantaged"), (1, "disadvantaged")):
    spans = ", ".join(f"{a:.2f} to {b:.2f}" for a, b in useful_ranges(y[s == g], p[s == g], grid))
    print(f"{name:13s} beats treat-all and treat-none at thresholds: {spans}")
