# Generated from explainers/small-group-shrinkage.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.linear_model import LogisticRegression
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Healthcare Readmission", strategy="unawareness")
train, test = data.idx_train, data.idx_test
X = data.X.to_numpy(dtype=float)
X = (X - X[train].mean(axis=0)) / (X[train].std(axis=0) + 1e-9)
y = data.y.to_numpy()
model = LogisticRegression(max_iter=2000).fit(X[train], y[train])
correct = (model.predict(X[test]) == y[test]).astype(float)

# 50 groups: age band x race. The 20,000 test rows are the "population"; each cell's true accuracy is known.
age = data.df["age"].to_numpy()[test]
race = data.df["race"].to_numpy()[test]
cell = np.array([f"{a}|{r}" for a, r in zip(age, race)])
cells, inverse, sizes = np.unique(cell, return_inverse=True, return_counts=True)
keep = sizes >= 30                                # cells with at least 30 rows in the population have a trustworthy "true" accuracy
cells, sizes = cells[keep], sizes[keep]
remap = -np.ones(len(keep), dtype=int); remap[keep] = np.arange(keep.sum())
inverse = remap[inverse]
in_cells = inverse >= 0
correct, inverse = correct[in_cells], inverse[in_cells]
truth = np.array([correct[inverse == i].mean() for i in range(len(cells))])
print(f"{len(cells)} cells of at least 30 rows; sizes from {sizes.min()} to {sizes.max()}; true accuracy from {truth.min():.3f} to {truth.max():.3f}")

def shrink(successes, counts):
    """Empirical-Bayes estimate: pull each cell's rate toward the pooled rate by a weight n / (n + k), where k is the
    prior strength estimated from how much the cells' raw rates vary beyond what binomial noise explains."""
    pooled = successes.sum() / counts.sum()
    raw = successes / np.maximum(counts, 1)
    noise = np.average(pooled * (1 - pooled) / np.maximum(counts, 1), weights=counts)
    between = max(np.average((raw - pooled) ** 2, weights=counts) - noise, 1e-6)
    k = pooled * (1 - pooled) / between
    return (successes + k * pooled) / (counts + k), k

rng = np.random.default_rng(0)
raw_err, shrunk_err, small_raw, small_shrunk, estimates = [], [], [], [], []
small = sizes <= np.quantile(sizes, 0.25)
worst = int(np.argmin(truth))
for _ in range(300):
    rows = rng.choice(len(correct), 600, replace=False)          # a small audit: 600 labelled rows
    counts = np.bincount(inverse[rows], minlength=len(cells)).astype(float)
    successes = np.bincount(inverse[rows], weights=correct[rows], minlength=len(cells))
    raw = np.where(counts > 0, successes / np.maximum(counts, 1), successes.sum() / counts.sum())
    shrunk, k = shrink(successes, counts)
    raw_err.append(np.sqrt(np.mean((raw - truth) ** 2))); shrunk_err.append(np.sqrt(np.mean((shrunk - truth) ** 2)))
    small_raw.append(np.sqrt(np.mean((raw[small] - truth[small]) ** 2))); small_shrunk.append(np.sqrt(np.mean((shrunk[small] - truth[small]) ** 2)))
    estimates.append((raw[worst], shrunk[worst]))

print(f"\nerror (RMSE against the true cell accuracy) from a 600-row audit, averaged over 300 audits")
print(f"  all {len(cells)} cells:        raw {np.mean(raw_err):.3f} | shrunk {np.mean(shrunk_err):.3f}")
print(f"  the smallest {small.sum()} cells:   raw {np.mean(small_raw):.3f} | shrunk {np.mean(small_shrunk):.3f}")
mean_raw, mean_shrunk = np.mean(estimates, axis=0)
print(f"\nthe cell with the lowest true accuracy, {cells[worst]} ({sizes[worst]} rows): true {truth[worst]:.3f}; "
      f"average estimate raw {mean_raw:.3f}, shrunk {mean_shrunk:.3f} (pooled accuracy {correct.mean():.3f})")
