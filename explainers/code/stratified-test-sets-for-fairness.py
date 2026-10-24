# Generated from explainers/stratified-test-sets-for-fairness.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.model_selection import cross_val_predict
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("German Credit Lending")       # 1,000 applicants; the disadvantaged group is those under 30
X, y = data.X, data.y.to_numpy()
young = data.group("age")                         # 1 = under 30, as a plain 0/1 array
# Out-of-fold predictions for everyone, so every row has an honest prediction to evaluate.
pred = cross_val_predict(build_model("logistic_regression", data.random_state), X, y, cv=10)
correct = (pred == y).astype(float)
truth = correct[young == 1].mean()
print(f"all 1,000 rows: accuracy {correct.mean():.3f} overall, {truth:.3f} for under-30s ({(young == 1).sum()} people)")

rng = np.random.default_rng(1)
def interval_width(values):
    boots = [rng.choice(values, len(values)).mean() for _ in range(500)]
    return np.subtract(*np.percentile(boots, [97.5, 2.5]))

def draw(size, share_young):
    n_young = int(round(size * share_young))
    rows = np.concatenate([rng.choice(np.flatnonzero(young == 1), n_young, replace=False),
                           rng.choice(np.flatnonzero(young == 0), size - n_young, replace=False)])
    return rows

print("\nevaluation sets of 200 rows, 100 draws each")
print(f"{'design':32s} {'under-30s in it':>16} {'width of the under-30 interval':>32} {'overall accuracy: unweighted / reweighted':>44}")
for label, share in (("random sample", None), ("stratified: half under 30", 0.5)):
    widths, plain, weighted, counts = [], [], [], []
    for _ in range(100):
        rows = rng.choice(len(y), 200, replace=False) if share is None else draw(200, share)
        group = young[rows]
        counts.append(group.sum())
        widths.append(interval_width(correct[rows][group == 1]))
        plain.append(correct[rows].mean())
        # reweight each row by population share of its group divided by its share in the sample
        w = np.where(group == 1, (young == 1).mean() / max(group.mean(), 1e-9), (young == 0).mean() / max(1 - group.mean(), 1e-9))
        weighted.append(np.average(correct[rows], weights=w))
    print(f"{label:32s} {np.mean(counts):16.0f} {np.mean(widths):32.3f} {np.mean(plain):22.3f} / {np.mean(weighted):.3f}")
