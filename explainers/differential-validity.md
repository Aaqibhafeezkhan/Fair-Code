> *A score can predict equally well for two groups and still be set too high for one of them. Validity and prediction are different questions.*

## The One-Sentence Definition

**Differential validity** is a difference between groups in how well a score predicts an outcome (for example, the strength of the link between a test score and a hiring or performance outcome), and **differential prediction** is a difference in the level of the outcome at the same score.

## Why It Matters

Employment testing has asked for sixty years whether a selection test is "fair" in a statistical sense, and the standard answers (Cleary, 1968) are about prediction, not outcome rates. A test is predictively fair if the same score means the same expected outcome for each group, so a single regression line fits both. If one group's line is lower, a single cut-off treats the groups differently at the same score, even if the test is equally good at ranking people within each.

Two separate checks follow. How well does the score rank people within each group (validity)? And at the same score, does one group have a lower expected outcome (prediction)? A test can pass the first and fail the second, which is what the example shows.

## Core Concept: Two Checks

1. **Validity by group:** the AUC (or correlation) between the score and the outcome, computed per group, with an interval.
2. **Prediction by group:** the expected outcome at each score band, per group; a gap at the same score is differential prediction. In regression form, a group term (intercept difference) or a group-by-score term (slope difference).

Here the outcome is the audit's hiring decision, which is a decision, not a measure of job performance. That makes the second check a measure of how the decision treats equal scores, not of whether the score is a good predictor of work.

## Concrete Example: AI Fair Recruitment

[AI Fair Recruitment](../AI%20Fair%20Recruitment/): 115,309 candidates, women against men, with the audit's hiring decision as the outcome and four score columns.

```
does each score predict being hired equally well for women and for men? (AUC of the score alone)
  Technical_Test_Score   men 0.743 | women 0.748 (95% interval 0.743-0.753)
  Interview_Score        men 0.746 | women 0.747 (95% interval 0.742-0.752)
  Skill_Score            men 0.744 | women 0.747 (95% interval 0.742-0.752)
  Aptitude_Test_Score    men 0.744 | women 0.750 (95% interval 0.744-0.756)

differential prediction: hire rate at the same technical test score (bands of 20 points)
   0-19: men 0.066 | women 0.057 | women minus men -0.009 (n = 14538, 8516)
  20-39: men 0.138 | women 0.107 | women minus men -0.031 (n = 14549, 8489)
  40-59: men 0.245 | women 0.214 | women minus men -0.030 (n = 14463, 8556)
  60-79: men 0.376 | women 0.340 | women minus men -0.036 (n = 14570, 8599)
  80-99: men 0.522 | women 0.482 | women minus men -0.040 (n = 14544, 8485)

one model for everyone (no Gender input), test rows: predicted hire probability women 0.265, men 0.254; observed 0.246 and 0.266
```

- **Validity is equal.** Each score has an AUC of about 0.74 to 0.75 for both groups: for example technical test 0.743 for men and 0.748 for women (interval 0.743 to 0.753). The scores rank candidates equally well within each group.
- **Prediction is not.** At the same technical score band, women are hired 0.9 to 4.0 points less often: 0.107 against 0.138 at 20 to 39, 0.340 against 0.376 at 60 to 79, 0.482 against 0.522 at 80 to 99, with 8,000 or more women in every band, so these are not noise.
- **A model that ignores gender inherits the mismatch.** Trained on all candidates without a gender input, it predicts a hire probability of 0.265 for women and 0.254 for men on the test rows, while the observed rates are 0.246 and 0.266: it over-predicts for women by about 2 points and under-predicts for men by about 1, the signature of one regression line that does not fit both groups.

So the scores are fine as predictors and the decision is not consistent with them across groups. The outcome here is the audit's historical hiring decision, so this is evidence about the decision, not about job performance.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/differential-validity.py`; that file is generated from the block below).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("AI Fair Recruitment", strategy="baseline")      # every column of the audit, Gender included
df = data.df
women = data.masks["gender"].to_numpy().astype(int)
hired = data.y.to_numpy()
rng = np.random.default_rng(0)

def interval(values, group, fn, n=300):
    a = np.flatnonzero(group)
    boots = [fn(values, rng.choice(a, len(a))) for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])

print("does each score predict being hired equally well for women and for men? (AUC of the score alone)")
for column in ("Technical_Test_Score", "Interview_Score", "Skill_Score", "Aptitude_Test_Score"):
    score = df[column].to_numpy(dtype=float)
    f = lambda v, rows: roc_auc_score(hired[rows], v[rows])
    aucs = {g: f(score, np.flatnonzero(women == g)) for g in (0, 1)}
    lo, hi = interval(score, women == 1, f)
    print(f"  {column:22s} men {aucs[0]:.3f} | women {aucs[1]:.3f} (95% interval {lo:.3f}-{hi:.3f})")

print("\ndifferential prediction: hire rate at the same technical test score (bands of 20 points)")
band = (df["Technical_Test_Score"].to_numpy() // 20).astype(int)
for b in range(5):
    rows = band == b
    m, w = hired[rows & (women == 0)].mean(), hired[rows & (women == 1)].mean()
    print(f"  {b * 20:2d}-{b * 20 + 19:2d}: men {m:.3f} | women {w:.3f} | women minus men {w - m:+.3f} (n = {(rows & (women == 0)).sum()}, {(rows & (women == 1)).sum()})")

# One regression for everyone, then look at the gap that remains at the same predicted score.
features = ["Technical_Test_Score", "Interview_Score", "Skill_Score", "Aptitude_Test_Score", "Experience_Years"]
X = df[features].to_numpy(dtype=float)
pooled = LogisticRegression(max_iter=2000).fit(X[data.idx_train], hired[data.idx_train])
p = pooled.predict_proba(X[data.idx_test])[:, 1]
w_test, y_test = women[data.idx_test], hired[data.idx_test]
print(f"\none model for everyone (no Gender input), test rows: predicted hire probability women {p[w_test == 1].mean():.3f}, men {p[w_test == 0].mean():.3f}; "
      f"observed {y_test[w_test == 1].mean():.3f} and {y_test[w_test == 0].mean():.3f}")
```

## Limitations

1. **The outcome matters.** With a hiring decision as the outcome, differential prediction says the decision treats equal scores differently. With a measure of job performance it would say the test over- or under-predicts for a group. Name which you have.
2. **A criterion can itself be biased.** If the outcome (performance ratings, promotions) carries bias, equal prediction against it is not evidence of fairness, see [Label Bias](label-bias.md).
3. **Range restriction hides validity.** If you only see people who were hired, the correlation is attenuated, see [Survivorship Bias](survivorship-bias.md).
4. **Equal validity does not mean equal selection.** Groups can have equal AUC and different pass rates at any single cut-off.
5. **Small subgroups have wide intervals.** The comparison needs enough rows in each band and group, see [Disaggregated Evaluation](disaggregated-evaluation.md).

## Related Concepts

* [Predictive Parity](predictive-parity.md) - the classification-metric cousin of equal prediction at the same score
* [Calibration](calibration.md) - the same idea for a probability score
* [Aggregation Bias](aggregation-bias.md) - one line for groups whose lines differ
* [Hiring Pipeline Adverse Impact](hiring-pipeline-adverse-impact.md) - the selection-rate view of the same data
* [Label Bias](label-bias.md) - when the outcome used for validation is biased

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above

## Further Reading

* Cleary, T. A. (1968). Test Bias: Prediction of Grades of Negro and White Students in Integrated Colleges. Journal of Educational Measurement 5(2) - the regression definition of a fair test.
* Society for Industrial and Organizational Psychology (2018). Principles for the Validation and Use of Personnel Selection Procedures (5th ed.) - the professional standard for validity and differential prediction in selection.
* [Hardt, M., Price, E., Srebro, N. (2016): Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) - NeurIPS 2016; fairness criteria defined on predictions and outcomes.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
