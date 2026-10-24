# Generated from explainers/survivorship-bias.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("AI Fair Recruitment", strategy="baseline")   # every feature, Gender included
train, test = data.idx_train, data.idx_test
X = data.X
X_std = ((X - X.iloc[train].mean()) / (X.iloc[train].std() + 1e-9)).to_numpy(dtype=float)
y = data.y.to_numpy()
women = data.masks["gender"].to_numpy()
screen = X["Technical_Test_Score"].to_numpy(dtype=float)
columns = list(X.columns)

# The earlier gate: only candidates with a technical test score of 70 or more reach the stage we have data on.
survivor = screen >= 70
print(f"candidates {len(y)}, passed the test gate {survivor.sum()} ({survivor.mean():.1%})")
print(f"hire rate, everyone {y.mean():.3f} | survivors only {y[survivor].mean():.3f}")
print(f"gender gap in hire rate (women minus men), everyone {y[women == 1].mean() - y[women == 0].mean():+.4f} | "
      f"survivors only {y[survivor & (women == 1)].mean() - y[survivor & (women == 0)].mean():+.4f}")

def coefficients(rows):
    return LogisticRegression(max_iter=1000).fit(X_std[rows], y[rows]).coef_[0]

everyone, kept = coefficients(train), coefficients(train[survivor[train]])
print("\ncoefficient (per standard deviation), fitted on everyone vs on survivors only")
for i, col in enumerate(columns):
    print(f"  {col:22s} {everyone[i]:+.2f}   {kept[i]:+.2f}")
