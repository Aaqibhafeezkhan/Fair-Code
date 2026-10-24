# Generated from explainers/reweighing.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

def reweighing_weights(s, y):
    """Kamiran and Calders: weight = P(group) * P(label) / P(group, label), per (group, label) cell."""
    w = np.empty(len(y))
    for g in (0, 1):
        for label in (0, 1):
            cell = (s == g) & (y == label)
            w[cell] = (s == g).mean() * (y == label).mean() / cell.mean()
    return w

def run(folder, attribute):
    data = load_audit(folder)                     # S2 feature set: core features, no protected columns
    train, test = data.idx_train, data.idx_test
    s = data.group(attribute)                     # 1 = disadvantaged group, as a plain 0/1 array
    s_tr, s_te = s[train], s[test]
    y_tr, y_te = data.y.iloc[train].to_numpy(), data.y.iloc[test].to_numpy()
    w = reweighing_weights(s_tr, y_tr)
    print(folder, "-", attribute)
    print("  weights per (group, label):", {(g, l): round(float(w[(s_tr == g) & (y_tr == l)][0]), 3) for g in (0, 1) for l in (0, 1)})
    for name, weight in (("baseline", None), ("reweighing", w)):
        model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[train], y_tr, sample_weight=weight)
        pred = model.predict(data.X.iloc[test])
        print(f"  {name:10s} accuracy {(pred == y_te).mean():.3f} | positive rate disadvantaged {pred[s_te == 1].mean():.3f} "
              f"advantaged {pred[s_te == 0].mean():.3f} | parity gap {pred[s_te == 1].mean() - pred[s_te == 0].mean():+.4f}")

run("German Credit Lending", "age")
run("Benefits Denial", "sex")
