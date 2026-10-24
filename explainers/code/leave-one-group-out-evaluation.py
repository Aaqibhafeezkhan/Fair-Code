# Generated from explainers/leave-one-group-out-evaluation.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

data = load_audit("Benefits Denial", strategy="baseline")     # baseline: every feature, race and native.country included
train, test = data.idx_train, data.idx_test
X, y = data.X, data.y.to_numpy()

def leave_one_group_out(column, labels, model_name="random_forest", min_rows=30):
    """For each group: accuracy on its test rows from a model trained with everyone, and from a model
    trained on every group EXCEPT that one."""
    codes = data.X_all[column].to_numpy()
    everyone = build_model(model_name, data.random_state).fit(X.iloc[train], y[train])
    print(f"{column} ({model_name})")
    print(f"  {'held-out group':20s} {'test rows':>9} {'with it':>8} {'without it':>11} {'change':>8}")
    for code, label in labels.items():
        held = test[codes[test] == code]
        if len(held) < min_rows:
            continue
        kept = train[codes[train] != code]
        without = build_model(model_name, data.random_state).fit(X.iloc[kept], y[kept])
        a, b = everyone.score(X.iloc[held], y[held]), without.score(X.iloc[held], y[held])
        print(f"  {label:20s} {len(held):9d} {a:8.3f} {b:11.3f} {b - a:+8.3f}")

raw = pd.read_csv("Benefits Denial/adult.csv", usecols=["race", "native.country"])
race_names = dict(enumerate(sorted(raw["race"].unique())))          # the harness codes text columns alphabetically
leave_one_group_out("race", race_names)
print()
country_names = dict(enumerate(sorted(raw["native.country"].unique())))
common = raw["native.country"].value_counts().index[:4]
leave_one_group_out("native.country", {c: n for c, n in country_names.items() if n in set(common)})
