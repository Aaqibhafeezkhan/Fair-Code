# Generated from explainers/randomized-post-processing.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Tenant Screening")            # S2 feature set: core features, no race, no proxies
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
group = data.masks["race"].to_numpy().astype(int)     # 1 = Black applicants
score = LogisticRegression(max_iter=1000).fit(X[train], y[train]).predict_proba(X)[:, 1]
def rates(pred, rows):
    out = {}
    for g in (0, 1):
        r = rows[group[rows] == g]
        out[g] = (pred[r][y[r] == 1].mean(), pred[r][y[r] == 0].mean())   # (TPR, FPR)
    return out

def report(name, pred):
    r = rates(pred, test)
    acc = (pred[test] == y[test]).mean()
    print(f"{name:40s} accuracy {acc:.3f} | TPR white {r[0][0]:.3f} black {r[1][0]:.3f} | FPR white {r[0][1]:.3f} black {r[1][1]:.3f} "
          f"| equalized-odds gap {max(abs(r[1][0] - r[0][0]), abs(r[1][1] - r[0][1])):.4f}")

report("one threshold at 0.5", (score >= 0.5).astype(int))

def solve(bins):
    """Hardt, Price, Srebro: for each group and score bin, the probability q of predicting positive so that both groups
    have the same TPR and FPR, maximizing accuracy. A linear programme in 2 x bins probabilities."""
    edges = np.quantile(score[train], np.linspace(0, 1, bins + 1)[1:-1])
    bin_of = np.digitize(score, edges)
    P = {(g, b, label): np.mean((group[train] == g) & (bin_of[train] == b) & (y[train] == label))
         for g in (0, 1) for b in range(bins) for label in (0, 1)}
    Pg = {(g, label): np.mean((group[train] == g) & (y[train] == label)) for g in (0, 1) for label in (0, 1)}
    index = lambda g, b: g * bins + b
    c = np.zeros(2 * bins)                            # maximize accuracy = minimize -(P(y=1, q) + P(y=0, 1-q))
    A_eq = np.zeros((2, 2 * bins))
    for g, sign in ((0, 1), (1, -1)):
        for b in range(bins):
            c[index(g, b)] = -(P[(g, b, 1)] - P[(g, b, 0)])
            A_eq[0, index(g, b)] = sign * P[(g, b, 1)] / Pg[(g, 1)]    # TPR(group 0) - TPR(group 1) = 0
            A_eq[1, index(g, b)] = sign * P[(g, b, 0)] / Pg[(g, 0)]    # FPR(group 0) - FPR(group 1) = 0
    q = linprog(c, A_eq=A_eq, b_eq=[0, 0], bounds=[(0, 1)] * (2 * bins)).x
    return q, bin_of

for bins in (4, 6, 20):
    q, bin_of = solve(bins)
    fractional = int(((q > 1e-6) & (q < 1 - 1e-6)).sum())
    draws = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pred = (rng.random(len(y)) < q[group * bins + bin_of]).astype(int)
        draws.append(pred)
    accs = np.array([(d[test] == y[test]).mean() for d in draws])
    print(f"\n{bins} score bins: {fractional} of {2 * bins} probabilities are strictly between 0 and 1")
    for g, name in ((0, "white"), (1, "black")):
        print(f"  {name}: " + " ".join(f"{q[g * bins + b]:.2f}" for b in range(bins)))
    report(f"randomized post-processing ({bins} bins), one draw", draws[0])
    print(f"  over 20 random draws: accuracy {accs.mean():.3f} +/- {accs.std(ddof=1):.3f}; the same person gets a different decision in "
          f"{np.mean([(draws[0][test] != d[test]).mean() for d in draws[1:]]):.1%} of rows between two draws")

# Deterministic alternative for comparison: one threshold per group, chosen on the training rows to match TPR.
t_white = 0.5
target = (score[train][(group[train] == 0) & (y[train] == 1)] >= t_white).mean()
t_black = np.quantile(score[train][(group[train] == 1) & (y[train] == 1)], 1 - target)
print()
report("deterministic group thresholds (match TPR)", np.where(group == 1, score >= t_black, score >= t_white).astype(int))
