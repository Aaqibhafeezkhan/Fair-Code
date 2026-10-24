# Generated from explainers/popularity-bias.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from scipy.stats import spearmanr
from faircode.benchmark import load_audit
from faircode.models import build_model

K, ROUNDS = 1000, 30

def rank_of(score):
    rank = np.empty(len(score), dtype=int)
    rank[np.argsort(-score, kind="stable")] = np.arange(1, len(score) + 1)
    return rank

def gini(x):
    x = np.sort(x.astype(float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())

def simulate(merit, hired, female, w, seed):
    """The ranker scores (1 - w) * merit + w * popularity, popularity being the share of past rounds an item was clicked."""
    rng = np.random.default_rng(seed)
    clicks, exposure_total = np.zeros(len(merit)), np.zeros(len(merit))
    for r in range(1, ROUNDS + 1):
        rank = rank_of((1 - w) * merit + w * clicks / max(r - 1, 1))
        exposure = np.where(rank <= K, 1 / np.log2(rank + 1), 0.0)
        exposure_total += exposure
        # a click needs the item to be looked at (position) and, less strongly, to be relevant
        clicks += rng.random(len(merit)) < exposure * (0.2 + 0.6 * hired)
    shown = exposure_total > 0
    return (gini(exposure_total), spearmanr(exposure_total[shown], merit[shown])[0],
            (exposure_total * female).sum() / exposure_total.sum(), shown.sum(), (hired & ~shown).sum())

for strategy, title in (("unawareness_proxy_removal", "Gender not used by the ranker"), ("baseline", "Gender used by the ranker")):
    data = load_audit("AI Fair Recruitment", strategy)
    tr, te = data.idx_train, data.idx_test
    female, hired = data.group("gender")[te], data.y.iloc[te].to_numpy()
    model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[tr], data.y.iloc[tr])
    merit = model.predict_proba(data.X.iloc[te])[:, 1]
    print(f"{title}: female share of the pool {female.mean():.3f}, of the first top {K} {female[rank_of(merit) <= K].mean():.3f}")
    for w in (0.0, 0.1, 0.25, 0.5):
        g, rho, share, n_shown, missed = simulate(merit, hired, female, w, data.random_state)
        print(f"  w {w:4.2f} | gini of exposure {g:.3f} | rank correlation of exposure with merit {rho:.3f} | female share of exposure {share:.3f} | ever shown {n_shown} | hired never shown {missed}")
