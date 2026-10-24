# Generated from explainers/path-specific-effects.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Benefits Denial")
df = data.df
woman = data.masks["sex"].to_numpy().astype(float)
y = data.y.to_numpy().astype(float)                 # 1 = income above $50K
age = df["age"].to_numpy(dtype=float)
mediators = {"education.num": df["education.num"].to_numpy(dtype=float), "hours.per.week": df["hours.per.week"].to_numpy(dtype=float)}
names = list(mediators)
M = np.column_stack([mediators[n] for n in names])

def ols(columns, target):
    X = np.column_stack([np.ones(len(target))] + columns)
    return np.linalg.lstsq(X, target, rcond=None)[0]

def decompose(rows):
    w, a, m, out = woman[rows], age[rows], M[rows], y[rows]
    total = ols([w, a], out)[1]                                         # sex -> income, mediators not adjusted
    full = ols([w, a] + [m[:, i] for i in range(m.shape[1])], out)      # income on sex, age and the mediators
    direct, b = full[1], full[3:]
    a_paths = np.array([ols([w, a], m[:, i])[1] for i in range(m.shape[1])])   # sex -> each mediator
    return total, direct, a_paths * b

rng = np.random.default_rng(0)
everyone = np.arange(len(y))
total, direct, indirect = decompose(everyone)
boots = [decompose(rng.choice(everyone, len(everyone))) for _ in range(300)]
def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5]); return f"[{lo:+.4f}, {hi:+.4f}]"

print(f"effect of being a woman on P(income above $50K), adjusting for age; linear model; {len(y):,} rows")
print(f"  total effect:                      {total:+.4f}  {interval([b[0] for b in boots])}")
for i, name in enumerate(names):
    print(f"  indirect, through {name:15s}  {indirect[i]:+.4f}  {interval([b[2][i] for b in boots])}")
print(f"  direct effect (not through these): {direct:+.4f}  {interval([b[1] for b in boots])}")
print(f"  check: direct + indirect = {direct + indirect.sum():+.4f}")
print(f"  share of the total that runs through the two mediators: {indirect.sum() / total:.0%}")
