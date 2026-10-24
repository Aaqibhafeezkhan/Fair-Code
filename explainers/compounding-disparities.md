> *A ratio of 0.9 at each of four stages looks like four small problems. End to end it is 0.66, which fails the four-fifths rule that none of the stages did.*

## The One-Sentence Definition

**Compounding disparities** are the multiplication of stage-level selection ratios across a sequence of decisions, so that small gaps at each stage produce a much larger gap in the final outcome.

## Why It Matters

The arithmetic is simple and still surprising. If women's chance of passing a stage is 90% of men's, and the stages are independent, women are 90% as likely to pass the first, 81% as likely to pass the first two, 73% to pass three and 66% to pass four. Each stage on its own clears a 0.8 four-fifths check, and the funnel does not.

This is why [Hiring Pipeline Adverse Impact](hiring-pipeline-adverse-impact.md) reports every stage and the whole funnel. It is also why an audit of one stage can be accurate and misleading: it tells you the stage's own ratio and nothing about what the other stages add.

## Core Concept: The Funnel Ratio Is the Product of the Conditional Ratios

For stages `1..k`, where `r_i` is the ratio of women's pass rate to men's among those who reached stage `i`:

```
whole-funnel ratio  =  r_1 * r_2 * ... * r_k
```

This holds exactly for the conditional ratios, whatever the correlation between stages. The unconditional ratios (each computed over everyone, not just those who reached the stage) multiply only if the stages are independent, and the two can differ when they are not.

## Concrete Example: A Simulated Four-Stage Funnel

This is a simulation, because the benchmark's hiring audit has no gap at its score stages (see the previous explainer). We put a known ratio into each of four stages with a base pass rate of 70%, run 200,000 applicants through, and measure.

```
ratio put in at each of 4 stages           measured at each stage  product of the four  whole funnel
                              1.00        1.000 1.001 0.991 1.008                0.999         0.999
                              0.95        0.953 0.943 0.950 0.952                0.813         0.813
                              0.90        0.897 0.904 0.904 0.907                0.666         0.666
                              0.85        0.846 0.851 0.850 0.852                0.521         0.521

correlated stages, 0.95 at each: unconditional stage ratios 0.948 0.952 0.948 0.944, their product 0.808, whole funnel 0.811
```

- **The measured stage ratios match what was put in** (0.953, 0.943, 0.950, 0.952 for an input of 0.95) within sampling noise.
- **The product of the four equals the whole-funnel ratio.** 0.95 per stage gives 0.813; 0.90 gives 0.666; 0.85 gives 0.521. Any input of 0.95 or below keeps every stage above 0.9 and still leaves the funnel near or below the threshold.
- **Correlated stages behave the same here.** When the pass decision at each stage depends on a shared underlying ability, the unconditional stage ratios multiply to 0.808 and the funnel ratio is 0.811, close but not identical, which is the case where the product of unconditional ratios is only an approximation.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/compounding-disparities.py`; that file is generated from the block below). It needs only NumPy.

```python
import numpy as np

rng = np.random.default_rng(0)
N = 200_000                                      # applicants per run, half of them women
women = np.arange(N) < N // 2

def funnel(stage_ratios, base_pass=0.7):
    """Four independent stages. Everyone passes a stage with probability base_pass, except that women's chance is
    multiplied by that stage's ratio. Returns each stage's ratio among the people who reached it, and the whole-funnel ratio."""
    alive = np.ones(N, dtype=bool)
    seen = []
    for ratio in stage_ratios:
        p = np.where(women, base_pass * ratio, base_pass)
        passed = rng.random(N) < p
        reached_w, reached_m = alive & women, alive & ~women
        seen.append(passed[reached_w].mean() / passed[reached_m].mean())
        alive &= passed
    whole = alive[women].mean() / alive[~women].mean()
    return seen, whole

print(f"{'ratio put in at each of 4 stages':34s} {'measured at each stage':>30} {'product of the four':>20} {'whole funnel':>13}")
for r in (1.0, 0.95, 0.90, 0.85):
    seen, whole = funnel([r] * 4)
    print(f"{r:34.2f} {' '.join(f'{s:.3f}' for s in seen):>30} {np.prod(seen):20.3f} {whole:13.3f}")

# The stages are independent above. When they are correlated, the whole-funnel ratio is NOT the product of the unconditional
# stage ratios: someone who passed the first stage is more likely to pass the second.
skill = rng.normal(size=N)
def correlated(ratios, noise=0.5):
    alive = np.ones(N, dtype=bool); unconditional = []
    for ratio in ratios:
        score = skill + noise * rng.normal(size=N)
        cut = np.quantile(score, 0.3)
        passed = score >= cut
        passed = passed & ~(women & (rng.random(N) > ratio))
        unconditional.append(passed[women].mean() / passed[~women].mean())
        alive &= passed
    return unconditional, alive[women].mean() / alive[~women].mean()
unconditional, whole = correlated([0.95] * 4)
print(f"\ncorrelated stages, 0.95 at each: unconditional stage ratios {' '.join(f'{u:.3f}' for u in unconditional)}, "
      f"their product {np.prod(unconditional):.3f}, whole funnel {whole:.3f}")
```

## Limitations

1. **It is a simulation.** The real stage gaps, and whether they are independent, have to come from your own data.
2. **The multiplication assumes the same people are compared.** If one group is screened out more at stage one, those who reach stage two are a different mix, and a stage-two ratio is conditional on that.
3. **Equal stage ratios are a special case.** Real funnels have a larger gap at one stage, and the interesting question is which.
4. **A threshold at each stage is not a rule at the end.** Four stages above 0.8 and a funnel below it is a reason to look at the funnel, not a legal conclusion.
5. **Compounding works in both directions.** A mitigation that nudges each stage by a little closes the funnel gap by the same multiplication.

## Related Concepts

* [Hiring Pipeline Adverse Impact](hiring-pipeline-adverse-impact.md) - measuring the ratio at each stage of real data
* [Disparate Impact](disparate-impact.md) - the four-fifths rule
* [Intersectional Bias](intersectional-bias.md) - compounding across attributes instead of stages
* [Feedback Loop Bias](feedback-loop-bias.md) - compounding across rounds of retraining
* [Subgroup Fairness](subgroup-fairness.md) - small gaps that add up inside a subgroup

## Related Projects in This Repo

* [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - the audit whose stages are examined in the companion explainer

## Further Reading

* The Uniform Guidelines on Employee Selection Procedures (1978), 29 CFR Part 1607 - the four-fifths rule and how it is applied to a selection process.
* [Bohnet, I., van Geen, A., Bazerman, M. (2016): When Performance Trumps Gender Bias: Joint Versus Separate Evaluation](https://doi.org/10.1287/mnsc.2015.2186) - Management Science 62(5); how small evaluation effects shape who advances.
* [Kleinberg, J., Raghavan, M. (2021): Algorithmic Monoculture and Social Welfare](https://arxiv.org/abs/2101.05853) - PNAS 118; how repeating the same selection algorithm across stages and employers compounds outcomes.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
