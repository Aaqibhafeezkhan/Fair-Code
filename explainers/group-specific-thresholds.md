> *One cutoff for everyone looks neutral. When score distributions differ by group, it is not.*

## The One-Sentence Definition

**Group-specific thresholds** are a post-processing mitigation: keep the trained model exactly as it is, and choose a different score cutoff for each group so that a chosen fairness criterion holds on the final decisions.

## Why It Matters

Many fairness fixes require retraining. Post-processing does not. When you inherit a scoring model, cannot touch it, or do not own the training pipeline, the cutoff is the one dial left. It is also the cleanest place to see the trade-off in numbers, because only one parameter per group changes.

It matters in practice too: Hardt, Price and Srebro (2016) showed that for a given score, the cutoffs that achieve equalized odds or equal opportunity can be found from the data, and the method is what Fairlearn's `ThresholdOptimizer` implements for the benchmark's S4 strategy. Seeing the plain version first makes the library version legible.

## Core Concept: Move the Cutoff, Not the Model

A single threshold `t` gives each group a positive rate `P(score >= t | group)`. If one group's scores sit higher on average, the same `t` produces different rates. To equalize a criterion, pick `t_g` per group:

* **Demographic parity:** choose `t_g` so each group's positive rate matches a target (here, the advantaged group's rate).
* **Equal opportunity:** choose `t_g` so each group's true-positive rate matches.

Both are quantile lookups on the scores. The one honest rule: fit the thresholds on training scores, then apply them to the test set. Choosing a cutoff on the test set that you then report on is circular.

## Concrete Example: Tenant Screening

The [Tenant Screening](../Tenant%20Screening/) audit: positive means the model flags a person as likely to reoffend within three years, the disadvantaged group is Black applicants, logistic regression, core features only, benchmark split.

```
one threshold (0.5)                accuracy 0.639 | positive rate advantaged 0.612 disadvantaged 0.725 | parity gap +0.1139 | TPR gap +0.0837
group thresholds, parity           accuracy 0.639 | positive rate advantaged 0.612 disadvantaged 0.598 | parity gap -0.0138 | TPR gap -0.0256
  disadvantaged-group threshold: 0.565 (advantaged stays 0.500)
group thresholds, equal opportunity accuracy 0.640 | positive rate advantaged 0.612 disadvantaged 0.611 | parity gap -0.0002 | TPR gap -0.0133
  disadvantaged-group threshold: 0.558
```

One threshold flags Black applicants at 72.5% against 61.2% for White applicants: a parity gap of +0.1139 and a true-positive-rate gap of +0.0837. Giving the disadvantaged group a cutoff of 0.565 (the other stays at 0.500) brings the parity gap to -0.0138 at the same 0.639 accuracy. The equal-opportunity cutoff of 0.558 brings the true-positive-rate gap to -0.0133 and the parity gap to -0.0002.

The benchmark's S2 baseline in `results/` (the current results) is the same model: accuracy 0.639 and a gap of +0.1139. Its S4 `post_processing` row, Fairlearn's `ThresholdOptimizer`, has a gap of -0.0054 at 0.639 accuracy. The hand-written version lands in the same place.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/group-specific-thresholds.py`; that file is generated from the block below).

```python
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

manifest, X, y, masks, train, test = load_audit("Tenant Screening")
s_tr, s_te = masks["race"][train], masks["race"][test]
y_tr, y_te = y.iloc[train].to_numpy(), y.iloc[test].to_numpy()
model = build_model("logistic_regression", manifest.random_state).fit(X.iloc[train], y_tr)
p_tr, p_te = model.predict_proba(X.iloc[train])[:, 1], model.predict_proba(X.iloc[test])[:, 1]

def report(pred, name):
    tpr = [pred[(s_te == g) & (y_te == 1)].mean() for g in (0, 1)]
    print(f"{name:34s} accuracy {(pred == y_te).mean():.3f} | positive rate advantaged {pred[s_te == 0].mean():.3f} "
          f"disadvantaged {pred[s_te == 1].mean():.3f} | parity gap {pred[s_te == 1].mean() - pred[s_te == 0].mean():+.4f} "
          f"| TPR gap {tpr[1] - tpr[0]:+.4f}")

report((p_te >= 0.5).astype(int), "one threshold (0.5)")

# Demographic parity: give the disadvantaged group the threshold that matches the advantaged group's positive rate.
# Fit the threshold on the TRAINING scores only, then apply it to the test set.
target_rate = (p_tr[s_tr == 0] >= 0.5).mean()
t_dp = np.quantile(p_tr[s_tr == 1], 1 - target_rate)
report(np.where(s_te == 1, p_te >= t_dp, p_te >= 0.5).astype(int), "group thresholds, parity")
print(f"  disadvantaged-group threshold: {t_dp:.3f} (advantaged stays 0.500)")

# Equal opportunity: match the true-positive rate instead.
target_tpr = (p_tr[(s_tr == 0) & (y_tr == 1)] >= 0.5).mean()
t_eo = np.quantile(p_tr[(s_tr == 1) & (y_tr == 1)], 1 - target_tpr)
report(np.where(s_te == 1, p_te >= t_eo, p_te >= 0.5).astype(int), "group thresholds, equal opportunity")
print(f"  disadvantaged-group threshold: {t_eo:.3f}")
```

## Limitations

1. **It uses the group at decision time.** The final rule treats two applicants with the same score differently because of their group. In US employment, race-norming of test scores has been prohibited since the Civil Rights Act of 1991, and similar disparate-treatment rules apply in lending and housing. Check the law before using it.
2. **It needs the protected attribute at inference.** Many deployed systems do not have it, or are not allowed to use it.
3. **It breaks score calibration across groups.** A score of 0.55 now means different things in each group. See [Calibration](calibration.md).
4. **It hides the cause.** The gap came from the model and its features. Moving the cutoff reports a fair-looking outcome and leaves the cause untouched.
5. **One pair of thresholds satisfies one criterion.** The parity cutoff left a true-positive-rate gap of -0.0256, and the equal-opportunity cutoff left a gap of -0.0133. In general the criteria conflict, see [Fairness Metric Conflicts](fairness-metric-conflicts.md).

## Related Concepts

* [Reject Option Classification](reject-option-classification.md) - another post-processing method, applied only near the cutoff
* [Equal Opportunity](equal-opportunity.md) - the criterion behind the second set of thresholds
* [Demographic Parity](demographic-parity.md) - the criterion behind the first set
* [Calibration](calibration.md) - what group-specific cutoffs give up
* [Mitigation Strategies](mitigation-strategies.md) - S4 in the benchmark
* [How Does Unequal Cost Change a Fairness Evaluation?](cost-sensitive-evaluation.md) - expected cost per group when errors are not equally costly

## Related Projects in This Repo

* [`Tenant Screening/`](../Tenant%20Screening/) - the audit used for the example above
* [`results/`](../results/) - the benchmark rows quoted for S2 and S4

## Further Reading

* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - NeurIPS 2016; the post-processing method and its derivation.
* [Corbett-Davies, S., Goel, S. (2018): The Measure and Mismeasure of Fairness](https://arxiv.org/abs/1808.00023) - a critique of fairness definitions, including the cost of group-specific thresholds.
* [Agarwal, A. et al. (2018): A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453) - ICML 2018; the Fairlearn reductions that the benchmark's S3 strategy uses.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
