# Generated from explainers/causal-graphs-for-fairness.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial")
df = data.df
woman = data.masks["sex"].to_numpy().astype(float)
high_income = data.y.to_numpy().astype(float)      # 1 = income above $50K

def sex_coefficient(adjust):
    """Linear-probability regression of high income on being a woman, adjusting for the columns in `adjust`
    (categorical columns are one-hot encoded). Returns the coefficient on sex: the gap that remains."""
    columns = [np.ones(len(df)), woman]
    for name in adjust:
        if not pd.api.types.is_numeric_dtype(df[name]):
            columns.extend(pd.get_dummies(df[name], drop_first=True).to_numpy(dtype=float).T)
        else:
            columns.append(df[name].to_numpy(dtype=float))
    X = np.column_stack(columns)
    return np.linalg.lstsq(X, high_income, rcond=None)[0][1]

# Assumed graph (an assumption, not a finding):
#   sex -> education.num, hours.per.week, occupation, marital.status -> income;  sex -> income (direct);  age -> everything
print(f"women {woman.mean():.1%} of the rows; income above $50K: women {high_income[woman == 1].mean():.3f}, men {high_income[woman == 0].mean():.3f}")
print(f"\n{'adjusting for':38s} {'sex coefficient':>16}   what the graph says it is")
for label, adjust, meaning in (
        ("nothing", [], "the total gap"),
        ("age", ["age"], "age is a common cause: legitimate to adjust"),
        ("education.num", ["education.num"], "a mediator: removes the part of the gap that runs through schooling"),
        ("hours.per.week", ["hours.per.week"], "a mediator: removes the part through hours worked"),
        ("occupation", ["occupation"], "a mediator, and a contested one"),
        ("marital.status", ["marital.status"], "a mediator or a proxy; the graph decides"),
        ("age, education, hours", ["age", "education.num", "hours.per.week"], "the 'qualifications' set"),
        ("all of them", ["age", "education.num", "hours.per.week", "occupation", "marital.status"], "everything we have")):
    print(f"{label:38s} {sex_coefficient(adjust):+16.4f}   {meaning}")
