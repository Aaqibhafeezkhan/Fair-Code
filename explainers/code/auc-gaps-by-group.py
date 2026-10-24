# Generated from explainers/auc-gaps-by-group.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from math import erf, sqrt
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit
from faircode.models import build_model

def delong_auc_var(y, score):
    """AUC and its DeLong variance (DeLong, DeLong, Clarke-Pearson 1988) for one sample."""
    pos, neg = score[y == 1], score[y == 0]
    # placement values: share of the other class each score beats (ties count half)
    v10 = np.array([(neg < p).mean() + 0.5 * (neg == p).mean() for p in pos])
    v01 = np.array([(pos > n).mean() + 0.5 * (pos == n).mean() for n in neg])
    return v10.mean(), v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)

def audit_auc(folder, attribute, n_boot=2000):
    """Per-group AUC, the gap with its two intervals, and the error rates at the default 0.5 cutoff."""
    data = load_audit(folder)
    tr, te = data.idx_train, data.idx_test
    model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
    p = model.predict_proba(data.X.iloc[te])[:, 1]
    y = data.y.iloc[te].to_numpy()
    s = data.group(attribute)[te]                       # 1 = disadvantaged group
    print(f"{folder} - {attribute}  (test rows: advantaged {(s == 0).sum()}, disadvantaged {(s == 1).sum()})")
    auc, var = {}, {}
    for g, name in ((0, "advantaged"), (1, "disadvantaged")):
        auc[g], var[g] = delong_auc_var(y[s == g], p[s == g])
        pred = p[s == g] >= 0.5
        yg = y[s == g]
        print(f"  {name:13s} AUC {auc[g]:.3f} | FPR at 0.5 {pred[yg == 0].mean():.3f} | TPR at 0.5 {pred[yg == 1].mean():.3f}")
    gap = auc[1] - auc[0]
    z = gap / sqrt(var[0] + var[1])
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    rng = np.random.default_rng(data.random_state)
    idx = {g: np.flatnonzero(s == g) for g in (0, 1)}
    boots = []
    for _ in range(n_boot):
        a = []
        for g in (0, 1):
            pick = rng.choice(idx[g], size=len(idx[g]), replace=True)
            a.append(roc_auc_score(y[pick], p[pick]))
        boots.append(a[1] - a[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  AUC gap {gap:+.3f} | bootstrap 95% interval [{lo:+.3f}, {hi:+.3f}] | DeLong z {z:+.2f}, p {p_value:.3f}")

audit_auc("Tenant Screening", "race")
audit_auc("Healthcare Readmission", "age")
audit_auc("Insurance Denial", "gender")
