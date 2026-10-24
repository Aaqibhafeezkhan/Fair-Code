# Generated from explainers/permutation-test-fairness-gaps.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from sklearn.model_selection import train_test_split
from faircode.manifest import load_manifest
from faircode.benchmark import _load_dataset      # the loader the benchmark harness uses
from faircode.strategies import encode_features
from faircode.models import build_model

def load_audit(folder):
    manifest = load_manifest(f"{folder}/audit.yaml")
    df, y, masks = _load_dataset(manifest)        # masks[name] is 1 for the disadvantaged group
    columns = list(dict.fromkeys(manifest.core_features))   # S2 feature set: no protected columns, no proxies
    X = encode_features(df, columns)
    train, test = train_test_split(np.arange(len(df)), test_size=manifest.test_size,
                                   random_state=manifest.random_state, stratify=y)
    return manifest, X, y, {k: v.to_numpy() for k, v in masks.items()}, train, test
from faircode.significance import permutation_test

manifest, X, y, masks, train, test = load_audit("Benefits Denial")
sex = masks["sex"][test]                            # 1 = female
model = build_model("logistic_regression", manifest.random_state).fit(X.iloc[train], y.iloc[train])
pred = model.predict(X.iloc[test]).astype(float)
women, men = pred[sex == 1], pred[sex == 0]
print(f"test rows {len(pred)}: {len(women)} women, {len(men)} men | positive rate {women.mean():.3f} vs {men.mean():.3f} | gap {women.mean() - men.mean():+.4f}")

def permutation_p(a, b, n_permutations=2000, seed=42):
    """Two-sided p: how often does a random relabelling give a gap at least this big?"""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)                          # the group label carries no information (the null)
        hits += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed
    return hits / n_permutations

print("hand-written p:", permutation_p(women, men), "| faircode.significance p:", permutation_test(women, men, n_permutations=2000))

# The same model and the same 5-point gap, audited on smaller and smaller samples.
rng = np.random.default_rng(0)
for n in (200, 1000):
    p_values = []
    for i in range(200):
        pick = rng.choice(len(pred), n, replace=False)
        a, b = pred[pick][sex[pick] == 1], pred[pick][sex[pick] == 0]
        if min(len(a), len(b)) >= 5:
            p_values.append(permutation_test(a, b, n_permutations=300, random_state=i))
    p_values = np.array(p_values)
    print(f"{n:5d} rows: p < 0.05 in {np.mean(p_values < 0.05):.0%} of {len(p_values)} random samples (median p {np.median(p_values):.3f})")
