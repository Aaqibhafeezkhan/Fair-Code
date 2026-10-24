# Generated from explainers/fairness-accuracy-pareto-frontier.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no race, no proxies
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
black = data.masks["race"].to_numpy().astype(int)
score = LogisticRegression(max_iter=1000).fit(X[train], y[train]).predict_proba(X)[:, 1]
s_te, y_te, b_te = score[test], y[test], black[test]

# Sweep one threshold per group and keep (accuracy, |parity gap|) for each pair: 41 x 41 candidate rules.
grid = np.linspace(0.2, 0.8, 41)
points = []
for t0 in grid:
    for t1 in grid:
        pred = np.where(b_te == 1, s_te >= t1, s_te >= t0)
        points.append((float((pred == y_te).mean()), abs(float(pred[b_te == 1].mean() - pred[b_te == 0].mean())), t0, t1))
points = np.array(points)

def non_dominated(p):
    """A point is on the frontier if no other point has accuracy at least as high and a gap at most as large (and one is strictly better)."""
    keep = []
    for i, (acc, gap, *_ ) in enumerate(p):
        dominated = np.any((p[:, 0] >= acc) & (p[:, 1] <= gap) & ((p[:, 0] > acc) | (p[:, 1] < gap)))
        keep.append(not dominated)
    return p[np.array(keep)]
frontier = non_dominated(points)
frontier = frontier[np.argsort(frontier[:, 1])]
print(f"{len(points)} threshold pairs, {len(frontier)} on the accuracy-gap frontier")
print(f"single threshold 0.5:  accuracy {points[np.argmin(np.abs(points[:, 2] - 0.5) + np.abs(points[:, 3] - 0.5)), 0]:.3f}, "
      f"gap {points[np.argmin(np.abs(points[:, 2] - 0.5) + np.abs(points[:, 3] - 0.5)), 1]:.3f}")
print("\nthe frontier, smallest gap first")
print(f"  {'|parity gap|':>12} {'accuracy':>9} {'threshold, white':>17} {'threshold, black':>17}")
for acc, gap, t0, t1 in frontier:
    print(f"  {gap:12.4f} {acc:9.3f} {t0:17.2f} {t1:17.2f}")

# Where do the benchmark's five strategies sit? Logistic regression on this audit, from results/ (the current results).
fair = pd.read_csv("results/results_fairness.csv"); perf = pd.read_csv("results/results_performance.csv")
f = fair[(fair.audit == "tenant_screening") & (fair.model == "logistic_regression") & (fair.protected_attribute == "race")
         & (fair.metric == "demographic_parity_diff")].set_index("strategy")
p = perf[(perf.audit == "tenant_screening") & (perf.model == "logistic_regression") & (perf.metric == "accuracy")].set_index("strategy")
print("\nthe benchmark's strategies on the same audit (benchmark split, so the numbers can differ slightly from the sweep):")
for strategy in f.index:
    acc, gap = p.loc[strategy, "value"], abs(f.loc[strategy, "value"])
    better = frontier[frontier[:, 1] <= gap]
    best = better[:, 0].max() if len(better) else float("nan")
    print(f"  {strategy:28s} accuracy {acc:.3f}, |gap| {gap:.3f}; best sweep accuracy at that gap {best:.3f} ({acc - best:+.3f})")
