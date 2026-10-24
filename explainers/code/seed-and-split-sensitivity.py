# Generated from explainers/seed-and-split-sensitivity.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model
from faircode.significance import permutation_test

def one_split(folder, attribute, seed):
    data = load_audit(folder, random_state=seed)          # the manifest's own split, but with another seed
    train, test = data.idx_train, data.idx_test
    model = build_model("logistic_regression", seed).fit(data.X.iloc[train], data.y.iloc[train])
    pred = model.predict(data.X.iloc[test])
    group = data.masks[attribute].to_numpy()[test]
    a, b = pred[group == 1].astype(float), pred[group == 0].astype(float)
    p = permutation_test(a, b, n_permutations=500, random_state=seed)
    return a.mean() - b.mean(), (pred == data.y.iloc[test].to_numpy()).mean(), p

SEEDS = range(30)
for folder, attribute, benchmark_seed in (("German Credit Lending", "age", 42), ("Benefits Denial", "sex", 42)):
    rows = np.array([one_split(folder, attribute, s) for s in SEEDS])
    gap, acc, p = rows[:, 0], rows[:, 1], rows[:, 2]
    base = one_split(folder, attribute, benchmark_seed)
    print(f"{folder} - {attribute}, 30 different seeds for the split")
    print(f"  parity gap: seed {benchmark_seed} gives {base[0]:+.4f}; the 30 seeds give mean {gap.mean():+.4f}, sd {gap.std(ddof=1):.4f}, "
          f"range {gap.min():+.4f} to {gap.max():+.4f}")
    print(f"  accuracy:   seed {benchmark_seed} gives {base[1]:.3f}; mean {acc.mean():.3f}, sd {acc.std(ddof=1):.3f}")
    print(f"  the gap is significant (p < 0.05) in {np.mean(p < 0.05):.0%} of the splits; its sign differs from the seed-{benchmark_seed} sign in "
          f"{np.mean(np.sign(gap) != np.sign(base[0])):.0%}")
