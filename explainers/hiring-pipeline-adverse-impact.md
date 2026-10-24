> *Adverse impact is usually computed on the last decision. A hiring process is four or five decisions in a row, and the ratio at each one tells you where the gap enters.*

## The One-Sentence Definition

**Pipeline adverse impact** is the selection-rate ratio (the four-fifths ratio) computed at each stage of a multi-stage hiring process, among the people who reached that stage, together with the ratio for the whole funnel.

## Why It Matters

The four-fifths rule in the US Uniform Guidelines on Employee Selection Procedures (29 CFR 1607) compares the selection rate of a group with the rate of the group with the highest rate. It is applied to a selection procedure, and employers often have several: a résumé screen, a test, an interview, an offer. A single end-to-end ratio tells you whether the whole process has an adverse impact but not which step produced it, and a step that does nothing wrong can sit next to one that does.

Stage by stage numbers show where to look. They also keep an audit honest about which step the model in question is. The [AI Fair Recruitment](../AI%20Fair%20Recruitment/) audit models the last decision, and the stages before it, as the next example shows, may be perfectly balanced.

## Core Concept: Conditional Ratios, One Per Stage

```
for each stage, among the people who reached it:
    rate_women = share of women who pass,  rate_others = share of others who pass
    ratio = rate_women / rate_others,  with a bootstrap interval
```

The ratio is conditional on reaching the stage, so a stage's number is not changed by what happened before it, except through who is left. The whole-funnel ratio is the product of the conditional ratios, which is why small stage gaps matter: see [Compounding Disparities](compounding-disparities.md). A statistical test or an interval belongs next to each ratio ([Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md)); a ratio computed on a few hundred people can move a lot.

## Concrete Example: AI Fair Recruitment

[AI Fair Recruitment](../AI%20Fair%20Recruitment/): 115,309 candidates (men and women). The funnel is illustrative: four stages built from the audit's own score columns with a cut-off of 30 at each (skill screen, aptitude test, technical test, interview), each passing about 70% of the people who reach it, then the audit's actual hiring decision among those who passed all four. Women are the disadvantaged group; "others" are men.

```
stage              reached  women pass  others pass     four-fifths ratio (95% interval)
skill screen        115309       0.700        0.701              0.999 [0.990, 1.007]
aptitude test        80767       0.699        0.696              1.004 [0.995, 1.013]
technical test       56325       0.697        0.699              0.997 [0.987, 1.009]
interview            39313       0.699        0.700              0.999 [0.984, 1.012]
hiring decision      27504       0.677        0.718              0.942 [0.927, 0.958]

whole funnel, everyone who applied: 19330 hired; women's rate over others' rate 0.941
hires in the data overall: 29816 (10486 of them would not have passed all four cut-offs above)
```

- **The four score stages are balanced.** Women pass at 0.700, 0.699, 0.697 and 0.699 against 0.701, 0.696, 0.699 and 0.700 for men; every ratio is within 0.4% of 1 and every interval includes 1. The score distributions do not differ by gender in this data.
- **The gap enters at the decision.** Among the 27,504 people who passed all four, women are hired at 0.677 and men at 0.718, a ratio of 0.942 with an interval of 0.927 to 0.958. At the same qualifications the hire rate differs by 4 points.
- **The funnel is a stand-in, not the data's rule.** 10,486 of the 29,816 hires in the data would not have passed all four cut-offs, so the recorded hiring decision does not follow these thresholds. The stage view is a way to look at the audit, not a description of how those hires were made.

The overall ratio for the audit's hire rate is lower than the decision-stage ratio (women 0.240, men 0.269 across all 115,309 candidates, a ratio of 0.891), so the final decision is where the audit's gap sits, and a check that only looked at tests would have found nothing.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/hiring-pipeline-adverse-impact.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("AI Fair Recruitment")
df = data.df
women = data.masks["gender"].to_numpy().astype(int)       # 1 = female candidates, the audit's disadvantaged group
hired = data.y.to_numpy()
rng = np.random.default_rng(0)

# An illustrative four-stage funnel built from the audit's own score columns. Each cut-off passes about 70%.
stages = [("skill screen", "Skill_Score", 30), ("aptitude test", "Aptitude_Test_Score", 30),
          ("technical test", "Technical_Test_Score", 30), ("interview", "Interview_Score", 30)]
alive = np.ones(len(df), dtype=bool)

def ratio_with_interval(passed, reached, n_boot=500):
    """Women's selection rate divided by everyone else's, among the people who reached the stage, with a bootstrap interval."""
    w, m = reached & (women == 1), reached & (women == 0)
    rate = lambda mask, rows: passed[rows][mask[rows]].mean()
    point = passed[w].mean() / passed[m].mean()
    idx_w, idx_m = np.flatnonzero(w), np.flatnonzero(m)
    boots = [passed[rng.choice(idx_w, len(idx_w))].mean() / passed[rng.choice(idx_m, len(idx_m))].mean() for _ in range(n_boot)]
    return point, np.percentile(boots, [2.5, 97.5])

print(f"{'stage':16s} {'reached':>9} {'women pass':>11} {'others pass':>12} {'four-fifths ratio (95% interval)':>36}")
cumulative = np.ones(len(df), dtype=bool)
for name, column, cut in stages:
    passed = (df[column].to_numpy() >= cut)
    reached = cumulative.copy()
    ratio, (lo, hi) = ratio_with_interval(passed.astype(float), reached)
    w, m = reached & (women == 1), reached & (women == 0)
    print(f"{name:16s} {reached.sum():9d} {passed[w].mean():11.3f} {passed[m].mean():12.3f} {ratio:18.3f} [{lo:.3f}, {hi:.3f}]")
    cumulative &= passed

reached = cumulative.copy()
ratio, (lo, hi) = ratio_with_interval(hired.astype(float), reached)
w, m = reached & (women == 1), reached & (women == 0)
print(f"{'hiring decision':16s} {reached.sum():9d} {hired[w].mean():11.3f} {hired[m].mean():12.3f} {ratio:18.3f} [{lo:.3f}, {hi:.3f}]")

everyone = np.ones(len(df), dtype=bool)
end_to_end, _ = ratio_with_interval((cumulative & (hired == 1)).astype(float), everyone, n_boot=10)
print(f"\nwhole funnel, everyone who applied: {(cumulative & (hired == 1)).sum()} hired; women's rate over others' rate {end_to_end:.3f}")
print(f"hires in the data overall: {hired.sum()} ({(hired == 1).sum() - (cumulative & (hired == 1)).sum()} of them would not have passed all four cut-offs above)")
```

## Limitations

1. **The stages here are invented.** Real pipelines have their own stages, cut-offs and orderings. The point is the method, not these thresholds.
2. **A stage ratio is conditional on who is left.** If an earlier stage removed more of one group, the people who reach the next stage are not comparable between groups.
3. **Four-fifths is a rule of thumb, not a test.** The ratio above 0.8 can still be significant, and below 0.8 can be noise. See [Permutation Tests for Fairness Gaps](permutation-test-fairness-gaps.md).
4. **Stage definitions are choices.** Whether "applied" means opened the application or completed it changes the first ratio.
5. **Legal tests need legal sources.** What counts as adverse impact, and what an employer must show in response, comes from the guidelines and case law for your jurisdiction.

## Related Concepts

* [Disparate Impact](disparate-impact.md) - the four-fifths ratio and its legal background
* [Compounding Disparities](compounding-disparities.md) - how stage ratios multiply
* [Differential Validity](differential-validity.md) - whether the scores used at each stage predict equally well
* [Bootstrap Confidence Intervals](bootstrap-confidence-intervals.md) - the interval beside each ratio
* [Intersectional Bias](intersectional-bias.md) - stages can differ for intersections the single-attribute view misses

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit used for the example above

## Further Reading

* The Uniform Guidelines on Employee Selection Procedures (1978), 29 CFR Part 1607 - the four-fifths rule and the definition of adverse impact.
* [Bogen, M., Rieke, A. (2018): Help Wanted: An Examination of Hiring Algorithms, Equity, and Bias](https://www.upturn.org/work/help-wanted/) - Upturn; how algorithms enter each stage of a hiring pipeline.
* [Raghavan, M., Barocas, S., Kleinberg, J., Levy, K. (2020): Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices](https://arxiv.org/abs/1906.09208) - FAT* 2020; what vendors claim about bias at each stage and how to evaluate it.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
