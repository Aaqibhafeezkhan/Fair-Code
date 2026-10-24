# Generated from explainers/differential-validity.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("AI Fair Recruitment", strategy="baseline")      # every column of the audit, Gender included
df = data.df
women = data.masks["gender"].to_numpy().astype(int)
hired = data.y.to_numpy()
rng = np.random.default_rng(0)

def interval(values, group, fn, n=300):
    a = np.flatnonzero(group)
    boots = [fn(values, rng.choice(a, len(a))) for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])

print("does each score predict being hired equally well for women and for men? (AUC of the score alone)")
for column in ("Technical_Test_Score", "Interview_Score", "Skill_Score", "Aptitude_Test_Score"):
    score = df[column].to_numpy(dtype=float)
    f = lambda v, rows: roc_auc_score(hired[rows], v[rows])
    aucs = {g: f(score, np.flatnonzero(women == g)) for g in (0, 1)}
    lo, hi = interval(score, women == 1, f)
    print(f"  {column:22s} men {aucs[0]:.3f} | women {aucs[1]:.3f} (95% interval {lo:.3f}-{hi:.3f})")

print("\ndifferential prediction: hire rate at the same technical test score (bands of 20 points)")
band = (df["Technical_Test_Score"].to_numpy() // 20).astype(int)
for b in range(5):
    rows = band == b
    m, w = hired[rows & (women == 0)].mean(), hired[rows & (women == 1)].mean()
    print(f"  {b * 20:2d}-{b * 20 + 19:2d}: men {m:.3f} | women {w:.3f} | women minus men {w - m:+.3f} (n = {(rows & (women == 0)).sum()}, {(rows & (women == 1)).sum()})")

# One regression for everyone, then look at the gap that remains at the same predicted score.
features = ["Technical_Test_Score", "Interview_Score", "Skill_Score", "Aptitude_Test_Score", "Experience_Years"]
X = df[features].to_numpy(dtype=float)
pooled = LogisticRegression(max_iter=2000).fit(X[data.idx_train], hired[data.idx_train])
p = pooled.predict_proba(X[data.idx_test])[:, 1]
w_test, y_test = women[data.idx_test], hired[data.idx_test]
print(f"\none model for everyone (no Gender input), test rows: predicted hire probability women {p[w_test == 1].mean():.3f}, men {p[w_test == 0].mean():.3f}; "
      f"observed {y_test[w_test == 1].mean():.3f} and {y_test[w_test == 0].mean():.3f}")
