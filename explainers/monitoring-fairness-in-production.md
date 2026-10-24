> *An audit is a photograph. Monitoring is the film. The hard part is not computing the gap each week; it is deciding when a change in it is a change.*

## The One-Sentence Definition

**Fairness monitoring** is tracking a fairness metric over time on live data, with a rule for when its movement is large enough, compared with its noise, to trigger a review.

## Why It Matters

[Model Drift](model-drift.md) explains why a gap measured at launch need not hold later: the population shifts, the inputs change, the feedback loop turns. Knowing that is not a monitoring plan. A plan needs three decisions: how much data each reading uses (the batch and window size), what counts as a change (the alert rule), and what happens when it fires.

Two things pull in opposite directions. Small batches react fast and fire false alarms, because a gap computed on 200 rows is noisy. Large batches are steady and slow, because a drift in one group is diluted by everything that did not change. The simulation below puts numbers on that trade.

## Core Concept: A Control Chart on the Gap

1. **Fix a reference:** the gap and its noise at launch.
2. **Compute the gap on each window** (the last `W` batches pooled) with its standard error.
3. **Alert when the window's gap is more than `z` standard errors from the reference** (3 is common).
4. **Choose `W` and batch size by simulation:** how quickly does a real drift of a given size alert, and how often does the rule fire with no drift?

This is the control chart of statistical process control, applied to a group gap. Variants (CUSUM, exponentially weighted averages) detect small steady shifts faster at the same false-alarm rate.

## Concrete Example: Benefits Denial

[Benefits Denial](../Benefits%20Denial/): the S2 logistic regression is fixed at launch and its gap on the test rows is -0.0509 (women are selected less often than men). We then simulate a stream by drawing batches from the real prediction pool, and from batch 20 onwards turn off some of women's positive predictions to move the true gap by a known amount (-0.03 or -0.06). Each setting is repeated 200 times; the table reports the median number of batches after the drift begins until the alert fires.

```
gap measured on the test rows at launch: -0.0509

alert rule: the gap over the last W batches is more than 3 standard errors from the launch value
batch size  window  drift   median batches to alert  never alerted   false alarms before the drift
       200       3   0.03                      11.0             9%                              3%
       200       3   0.06                       3.0             0%                              4%
       500       3   0.03                       5.0             0%                              9%
       500       3   0.06                       2.0             0%                              4%
      1000       3   0.03                       3.0             0%                              4%
      1000       3   0.06                       2.0             0%                              6%
      2000       3   0.03                       3.0             0%                              4%
      2000       3   0.06                       2.0             0%                              4%
      1000       1   0.03                       4.0             0%                              7%
      1000       1   0.06                       1.0             0%                              5%
      1000       6   0.03                       4.0             0%                              3%
      1000       6   0.06                       2.0             0%                              2%
```

- **Bigger drifts alert faster:** with batches of 1,000 and a window of 3, a 0.06 drift alerts after 2 batches and a 0.03 drift after 3.
- **Small batches are slow for small drifts:** with 200 rows per batch a 0.03 drift takes a median of 11 batches and 9% of runs never alert within the 40 batches afterwards.
- **The rule raises false alarms:** between 2% and 9% of runs fire during the 20 batches before the drift, which is what a 3 standard error rule gives when it is checked at every batch. A rule that is checked weekly for a year will fire by chance.
- **A longer window is steadier, not always better:** at 1,000 rows a window of 6 gives fewer false alarms (3% and 2%) than a window of 1 (7% and 5%) and alerts about as fast for the larger drift.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/monitoring-fairness-in-production.py`; that file is generated from the block below).

```python
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
```

## Limitations

1. **The reference has its own noise.** The launch gap is estimated from the same kind of sample, so it is uncertain too. The code above ignores that, which makes the rule fire a little more often than its nominal rate.
2. **Repeated looks inflate false alarms.** Checking every batch gives many chances to fire by chance. Raise `z`, use a sequential method, or correct for the number of looks, see [Multiple Comparisons](multiple-comparisons.md).
3. **The simulated drift is simple.** Real drift moves features, base rates and labels at once, and may hit one intersection and not the whole group.
4. **Labels may arrive late.** A monitor that needs outcomes has to wait for them, see [Delayed Feedback](delayed-feedback.md). Selection rates need no labels and can be watched immediately.
5. **An alert is not a verdict.** It says the gap moved. Whether to retrain, roll back or investigate is a decision, and it should be written down before the alert fires.

## Related Concepts

* [Model Drift](model-drift.md) - why the gap moves
* [Distribution Shift](distribution-shift.md) - the change in inputs behind many drifts
* [Label Shift and Fairness](label-shift-and-fairness.md) - a change in base rates and which metrics it moves
* [Delayed Feedback](delayed-feedback.md) - when outcomes arrive late
* [Minimum Detectable Gap](minimum-detectable-gap.md) - how large a change a batch size can see

## Related Projects in This Repo

* [`Benefits Denial/`](../Benefits%20Denial/) - the audit used for the example above

## Further Reading

* [Gama, J. et al. (2014): A Survey on Concept Drift Adaptation](https://doi.org/10.1145/2523813) - ACM Computing Surveys 46(4); detecting and responding to drift.
* Montgomery, D. C. (2019). *Introduction to Statistical Quality Control* (8th ed.). Wiley - control charts, CUSUM and EWMA.
* [Ginart, T., Zhang, M. J., Zou, J. (2022): MLDemon: Deployment Monitoring for Machine Learning Systems](https://arxiv.org/abs/2104.13621) - AISTATS 2022; monitoring when labels are scarce.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
