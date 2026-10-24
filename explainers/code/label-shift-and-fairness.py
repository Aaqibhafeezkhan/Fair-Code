# Generated from explainers/label-shift-and-fairness.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], data.y.iloc[train])
pred = model.predict(data.X.iloc[test])
y = data.y.to_numpy()[test]
women = data.masks["sex"].to_numpy()[test].astype(bool)
observed = y[women].mean()
print(f"women's base rate in the test rows {observed:.3f}; men's {y[~women].mean():.3f}")

def reweighted(target_rate):
    """Label shift for women only: change the share of positives among women to `target_rate`, keeping the
    features-given-label relationship fixed, by weighting positives and negatives."""
    w = np.ones(len(y))
    w[women & (y == 1)] = target_rate / observed
    w[women & (y == 0)] = (1 - target_rate) / (1 - observed)
    return w

def metrics(w):
    def rate(mask):
        return np.average(pred[mask], weights=w[mask])
    sel_w, sel_m = rate(women), rate(~women)
    tpr_w, tpr_m = rate(women & (y == 1)), rate(~women & (y == 1))
    fpr_w, fpr_m = rate(women & (y == 0)), rate(~women & (y == 0))
    ppv_w = np.average(y[women & (pred == 1)], weights=w[women & (pred == 1)])
    ppv_m = y[~women & (pred == 1)].mean()
    return sel_w - sel_m, tpr_w - tpr_m, fpr_w - fpr_m, ppv_w - ppv_m

print(f"\n{'women base rate':>16} {'parity gap':>11} {'TPR gap':>9} {'FPR gap':>9} {'precision gap':>14}")
for target in (0.05, observed, 0.20, 0.30, 0.40):
    sel, tpr, fpr, ppv = metrics(reweighted(target))
    tag = "  <- as measured" if abs(target - observed) < 1e-9 else ""
    print(f"{target:16.3f} {sel:+11.4f} {tpr:+9.4f} {fpr:+9.4f} {ppv:+14.4f}{tag}")
