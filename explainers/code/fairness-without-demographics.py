# Generated from explainers/fairness-without-demographics.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial")             # S2 feature set: core features, no protected columns
train, test = data.idx_train, data.idx_test
X, y = data.X, data.y.to_numpy()
sex, aged = data.masks["sex"].to_numpy(), data.masks["age"].to_numpy()
group = sex * 2 + aged                           # used ONLY to evaluate, never to train
labels = ["male <55", "male 55+", "female <55", "female 55+"]

def fit(weights=None):
    return build_model("logistic_regression", data.random_state).fit(X.iloc[train], y[train], sample_weight=weights)

def report(name, model):
    correct = model.predict(X.iloc[test]) == y[test]
    per_group = [correct[group[test] == k].mean() for k in range(4)]
    print(f"{name:34s} overall {correct.mean():.3f} | " + " ".join(f"{l} {a:.3f}" for l, a in zip(labels, per_group))
          + f" | worst group {min(per_group):.3f}")

baseline = fit()
report("baseline", baseline)

# Just Train Twice: upweight the training rows the first model got wrong. No group labels are used.
wrong = baseline.predict(X.iloc[train]) != y[train]
for upweight in (1.5, 2, 3, 5):
    report(f"upweight the errors x{upweight}", fit(np.where(wrong, upweight, 1.0)))
