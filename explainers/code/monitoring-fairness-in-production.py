# Generated from explainers/monitoring-fairness-in-production.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], data.y.iloc[train])
pred = model.predict(data.X.iloc[test])
women = data.masks["sex"].to_numpy()[test].astype(bool)
launch_gap = pred[women].mean() - pred[~women].mean()
print(f"gap measured on the test rows at launch: {launch_gap:+.4f}")

rng = np.random.default_rng(0)

def stream_batches(batch, n_batches, drift_at, drift):
    """Draw each batch from the real prediction pool. From batch `drift_at` on, women's positive predictions are turned off
    with probability `drift` / (women's selection rate), which moves the true gap by about -drift."""
    rows = []
    for b in range(n_batches):
        idx = rng.choice(len(pred), batch)
        p, w = pred[idx].copy(), women[idx]
        if b >= drift_at and drift:
            p[w & (p == 1) & (rng.random(len(p)) < drift / pred[women].mean())] = 0
        rows.append((p, w))
    return rows

def gap(rows):
    p = np.concatenate([r[0] for r in rows]); w = np.concatenate([r[1] for r in rows])
    return p[w].mean() - p[~w].mean(), p[w].mean() * (1 - p[w].mean()) / w.sum() + p[~w].mean() * (1 - p[~w].mean()) / (~w).sum()

def batches_until_alert(batch, window, drift, drift_at=20, n_batches=60, z=3):
    rows = stream_batches(batch, n_batches, drift_at, drift)
    for b in range(window - 1, n_batches):
        g, var = gap(rows[b - window + 1:b + 1])
        if abs(g - launch_gap) > z * np.sqrt(var):
            return b - drift_at + 1 if b >= drift_at else -1       # -1 = a false alarm before the drift
    return None

print(f"\nalert rule: the gap over the last W batches is more than 3 standard errors from the launch value")
print(f"{'batch size':>10} {'window':>7} {'drift':>6} {'median batches to alert':>25} {'never alerted':>14} {'false alarms before the drift':>31}")
for batch, window in ((200, 3), (500, 3), (1000, 3), (2000, 3), (1000, 1), (1000, 6)):
    for drift in (0.03, 0.06):
        results = [batches_until_alert(batch, window, drift) for _ in range(200)]
        false = np.mean([r == -1 for r in results])
        after = [r for r in results if r not in (None, -1)]
        print(f"{batch:10d} {window:7d} {drift:6.2f} {np.median(after) if after else float('nan'):25.1f} "
              f"{np.mean([r is None for r in results]):14.0%} {false:31.0%}")
