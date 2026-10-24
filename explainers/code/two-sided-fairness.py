# Generated from explainers/two-sided-fairness.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit
from faircode.models import build_model

data = load_audit("AI Fair Recruitment", "baseline")          # the ranker uses Gender, as in the baseline
tr, te = data.idx_train, data.idx_test
female = data.group("gender")[te]                             # 1 = female candidate (the people being ranked)
hired = data.y.iloc[te].to_numpy()                            # relevance for the people searching: was this candidate hired
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
score = model.predict_proba(data.X.iloc[te])[:, 1]
K = 1000

def rank_of(s):
    rank = np.empty(len(s), dtype=int)
    rank[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
    return rank

def dcg(relevance_in_order):
    return (relevance_in_order / np.log2(np.arange(2, len(relevance_in_order) + 2))).sum()

ideal = dcg(np.sort(hired)[::-1][:K])                          # the best possible top-K list

print(f"AI Fair Recruitment, Gender used by the ranker: {len(score)} candidates, top {K} shown, {female.mean():.3f} of the pool female")
print("  shift | female share of top-%d | exposure per unit of hire rate, female/male | nDCG@%d | hired in top-%d" % (K, K, K))
for shift in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15):
    rank = rank_of(score + shift * female)                    # re-rank: lift the under-exposed group's scores by a constant
    exposure = np.where(rank <= K, 1 / np.log2(rank + 1), 0.0)
    ratio = (exposure[female == 1].mean() / hired[female == 1].mean()) / (exposure[female == 0].mean() / hired[female == 0].mean())
    top = np.argsort(rank)[:K]
    print(f"  {shift:5.2f} | {female[top].mean():22.3f} | {ratio:42.3f} | {dcg(hired[top]) / ideal:7.3f} | {hired[top].sum():d}")
