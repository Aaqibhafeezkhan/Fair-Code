# Generated from explainers/adverse-action-notices.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("German Credit Lending")       # S2 feature set: core features, no protected column
train = data.idx_train
X = data.X.to_numpy(dtype=float)
Z = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()                            # 1 = good credit
young = data.masks["age"].to_numpy().astype(int) # 1 = applicants under 30, the disadvantaged group
columns = np.array(data.X.columns)

model = LogisticRegression(max_iter=1000).fit(Z[train], y[train])
declined = np.flatnonzero(model.predict(Z) == 0)  # every applicant the model would decline (all 1,000 rows)

# Reason codes: the features that pushed each declined applicant's score down the most.
contribution = Z[declined] * model.coef_[0]       # per-feature contribution to the log-odds, relative to the average applicant
order = np.argsort(contribution, axis=1)          # most negative first
main_reason = columns[order[:, 0]]

print(f"the model declines {len(declined)} of {len(y)} applicants")
for g, name in ((0, "age 30 and over"), (1, "under 30")):
    rows = young[declined] == g
    shares = pd.Series(main_reason[rows]).value_counts(normalize=True).head(4)
    print(f"\n{name}: {rows.sum()} declined ({rows.sum() / (young == g).sum():.0%} of the group); main reason")
    for reason, share in shares.items():
        print(f"  {reason:24s} {share:.0%}")

top_two = pd.DataFrame({"reason": np.concatenate([columns[order[:, 0]], columns[order[:, 1]]]),
                        "young": np.concatenate([young[declined], young[declined]])})
share = top_two.groupby(["reason", "young"]).size().unstack(fill_value=0)
share = share / share.sum()
print("\nlargest differences in how often a reason is in an applicant's top two (under 30 minus 30 and over):")
print((share[1] - share[0]).sort_values(key=abs, ascending=False).head(3).round(2).to_string())
