# Generated from explainers/risk-based-pricing-gaps.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("German Credit Lending")       # S2 feature set: core features, no protected column
train, test = data.idx_train, data.idx_test
young = data.masks["age"].to_numpy().astype(int)[test]     # 1 = under 30, the disadvantaged group
rng = np.random.default_rng(0)

# Everyone is "approved"; the model decides how much to lend. The quoted amount is predicted from the other
# columns (the one continuous amount a lender sets per applicant in this file).
X = data.X.drop(columns=["credit_amount"])
amount = data.df["credit_amount"].to_numpy(dtype=float)

def bootstrap_gap(values, group, n=2000):
    a, b = values[group == 1], values[group == 0]
    gaps = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(gaps, [2.5, 97.5])

print(f"real amounts: under 30 {amount[test][young == 1].mean():,.0f}, 30 and over {amount[test][young == 0].mean():,.0f} "
      f"(gap {amount[test][young == 1].mean() - amount[test][young == 0].mean():+,.0f})")
for name, model in (("linear regression", LinearRegression()),
                    ("gradient boosting", GradientBoostingRegressor(random_state=data.random_state))):
    quoted = model.fit(X.iloc[train], amount[train]).predict(X.iloc[test])
    gap = quoted[young == 1].mean() - quoted[young == 0].mean()
    lo, hi = bootstrap_gap(quoted, young)
    relative = gap / quoted[young == 0].mean()
    print(f"{name:18s} quoted amount, under 30 minus 30 and over: {gap:+,.0f} ({relative:+.0%}), 95% interval [{lo:+,.0f}, {hi:+,.0f}]")
