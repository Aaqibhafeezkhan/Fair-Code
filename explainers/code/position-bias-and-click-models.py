# Generated from explainers/position-bias-and-click-models.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit

data = load_audit("AI Fair Recruitment")
cols = ["Experience_Years", "Technical_Test_Score", "Age"]            # core features plus the proxy feature Age
X_all = data.X_all[cols].iloc[data.idx_test].to_numpy(float)
X_all = (X_all - X_all.mean(0)) / X_all.std(0)
hired = data.y.iloc[data.idx_test].to_numpy()                        # true relevance, known here because we simulate the clicks
rng = np.random.default_rng(data.random_state)
LIST_SIZE, ETA = 20, 1.0

# 1. Split the candidates into lists of 20, the way a recruiter sees a page of results.
order = rng.permutation(len(hired))
lists = order[: len(order) // LIST_SIZE * LIST_SIZE].reshape(-1, LIST_SIZE)
train_lists, test_lists = lists[: len(lists) * 7 // 10], lists[len(lists) * 7 // 10:]

# 2. The logging policy: a legacy ranker that sorts mostly by Age, with some merit in it.
def log_rank(idx):
    key = X_all[idx, 2] + 0.5 * (X_all[idx, 0] + X_all[idx, 1]) + rng.normal(0, 0.5, len(idx))
    return np.argsort(np.argsort(-key)) + 1

# 3. Position-based click model: P(click) = P(examined | rank) * P(relevant), examined = rank^-eta.
rows, ranks, clicks = [], [], []
for idx in train_lists:
    rank = log_rank(idx)
    examined = rank ** -ETA
    clicks.append(rng.random(len(idx)) < examined * (0.05 + 0.9 * hired[idx]))
    rows.append(idx); ranks.append(rank)
rows, ranks, clicks = map(np.concatenate, (rows, ranks, clicks))
print(f"{len(train_lists)} training lists, click rate by position: " +
      ", ".join(f"rank {r}: {clicks[ranks == r].mean():.3f}" for r in (1, 2, 5, 10, 20)) +
      f" | hire rate at rank 1: {hired[rows][ranks == 1].mean():.3f}, at rank 20: {hired[rows][ranks == 20].mean():.3f}")

# 4. Rankers: trained on raw clicks, on clicks weighted by 1 / P(examined) with the right and wrong exposure curves, and on true labels.
def fit(target, weight=None):
    return LogisticRegression(max_iter=1000).fit(X_all[rows], target, sample_weight=weight)

def ipw(eta):
    return np.where(clicks, ranks ** eta, 1.0)                       # inverse propensity weights on the clicked rows

models = {"raw clicks": fit(clicks),
          "weighted, assumed eta 0.5": fit(clicks, ipw(0.5)),
          "weighted, assumed eta 1.0": fit(clicks, ipw(1.0)),
          "weighted, assumed eta 2.0": fit(clicks, ipw(2.0)),
          "true labels (oracle)": fit(hired[rows])}

def ndcg_at_5(model):
    total = 0
    discount = 1 / np.log2(np.arange(2, 7))
    for idx in test_lists:
        gains = hired[idx][np.argsort(-model.decision_function(X_all[idx]))][:5]
        ideal = np.sort(hired[idx])[::-1][:5]
        total += (gains * discount).sum() / max((ideal * discount).sum(), 1e-9)
    return total / len(test_lists)

test_rows = test_lists.ravel()
print(f"true eta is {ETA}")
print("  ranker                    | weight on Age | AUC vs hired | nDCG@5")
for name, m in models.items():
    print(f"  {name:25s} | {m.coef_[0][2]:13.3f} | {roc_auc_score(hired[test_rows], m.decision_function(X_all[test_rows])):12.3f} | {ndcg_at_5(m):6.3f}")
