# Generated from explainers/cost-sensitive-evaluation.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit
from faircode.models import build_model

data = load_audit("Tenant Screening")
tr, te = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
p_tr, p_te = (model.predict_proba(data.X.iloc[i])[:, 1] for i in (tr, te))
y_tr, y_te = data.y.iloc[tr].to_numpy(), data.y.iloc[te].to_numpy()
s = data.group("race")
s_tr, s_te = s[tr], s[te]                            # 1 = disadvantaged group (Black applicants)

def cost(y, p, t, c_fp, c_fn):
    """Expected cost per person: a false positive costs c_fp, a false negative costs c_fn."""
    pred = p >= t
    return ((pred & (y == 0)).sum() * c_fp + (~pred & (y == 1)).sum() * c_fn) / len(y)

grid = np.round(np.arange(0.05, 1.001, 0.005), 3)    # 1.0 flags nobody
def best_threshold(g, c_fp, c_fn):
    """The cutoff that minimizes cost on the TRAINING scores of group g."""
    costs = [cost(y_tr[s_tr == g], p_tr[s_tr == g], t, c_fp, c_fn) for t in grid]
    return grid[int(np.argmin(costs))]

print("Tenant Screening - race: expected cost per 100 applicants on the test set (a positive is a flagged applicant)")
scenarios = [("a false positive costs 1, a false negative costs 1", {0: (1, 1), 1: (1, 1)}),
             ("a false positive costs 5, a false negative costs 1", {0: (5, 1), 1: (5, 1)}),
             ("a false positive costs 5 for the advantaged group and 8 for the disadvantaged group, a false negative 1",
              {0: (5, 1), 1: (8, 1)})]
for name, costs in scenarios:
    print(name)
    rules = {"threshold 0.5": {g: 0.5 for g in (0, 1)},
             "Elkan c_fp/(c_fp+c_fn)": {g: costs[g][0] / sum(costs[g]) for g in (0, 1)},
             "cost-optimal per group": {g: best_threshold(g, *costs[g]) for g in (0, 1)},
             "flag nobody": {g: 1.01 for g in (0, 1)}}
    for label, t in rules.items():
        per = [100 * cost(y_te[s_te == g], p_te[s_te == g], t[g], *costs[g]) for g in (0, 1)]
        flagged = [(p_te[s_te == g] >= t[g]).mean() for g in (0, 1)]
        print(f"  {label:24s} thresholds {t[0]:.3f} / {t[1]:.3f} | flagged {flagged[0]:.3f} / {flagged[1]:.3f} | cost {per[0]:6.1f} / {per[1]:6.1f}")
