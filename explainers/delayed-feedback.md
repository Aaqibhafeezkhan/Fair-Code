> *Outcomes that have not happened yet look like outcomes that did not happen. When one group's outcomes arrive later, the monitor reads that as a difference between groups.*

## The One-Sentence Definition

**Delayed feedback** is the situation in which an outcome (readmission, default, a complaint) is only known some time after the decision, so recent decisions look better than they will turn out to be, and groups whose outcomes arrive at different speeds look different when they are not.

## Why It Matters

A fairness monitor that tracks the outcome rate by group is reading incomplete data. If a readmission can happen up to 30 days after discharge, last week's discharges have mostly not had the chance yet, and every group's rate looks low. If one group's outcomes tend to arrive later (a different follow-up pattern, a slower reporting channel), its rate looks lower still, and the monitored gap moves with no change in the model or the people.

This is a measurement problem in the same family as [Survivorship Bias](survivorship-bias.md) and the selective labels problem: what you see depends on a process that is not the model.

## Core Concept: Count Only Mature Cases, or Model the Delay

1. **Naive:** divide the outcomes seen so far by all decisions so far. Biased low, and by different amounts for groups with different delays.
2. **Mature-only:** use only decisions old enough that the outcome window has closed (here, discharged more than 30 days ago). Unbiased, but it ignores the most recent data and needs the window to be known.
3. **Model the delay:** estimate the distribution of the delay and correct the recent counts, as in survival analysis. More complete, and more assumptions.

## Concrete Example: Healthcare Readmission

This is a simulation, because the dataset has no dates. [Healthcare Readmission](../Healthcare%20Readmission/): the real 30-day readmission outcome of every patient, with African American patients and everyone else as the groups. Each patient is given a random discharge day in the last 90, and each readmission a delay in days, drawn from an exponential distribution with a mean we choose per group and capped at 30. "Today" is day 90.

```
true 30-day readmission rate: African American patients 0.112, others 0.112, gap -0.0001

mean days to readmission (group, others)  naive gap      naive rates  share of positives seen  careful gap
                                  10, 10    -0.0004    0.101 / 0.101                    90.1%      +0.0032
                                  10, 20    +0.0081    0.101 / 0.093                    84.1%      +0.0032
                                  20, 10    -0.0057    0.095 / 0.100                    88.3%      +0.0032
                                  20, 20    +0.0009    0.094 / 0.093                    83.1%      +0.0032
```

- **The true gap is zero:** both groups are readmitted at 0.112.
- **With equal delays the naive monitor reads zero** (-0.0004), but it still sees only 83% to 90% of the positives, so every rate is biased low.
- **With unequal delays it reads a gap that is not there.** If the others' readmissions take 20 days on average and the group's take 10, the naive gap is +0.0081; reversed, -0.0057. These are the same size as many real gaps in this repo's audits, and they come entirely from timing.
- **The mature-only monitor is steady** at +0.0032 whatever the delays, because it looks only at patients whose window has closed. That value is also not exactly zero: it is the sampling noise of the mature third of the rows.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/delayed-feedback.py`; that file is generated from the block below).

```python
import numpy as np
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness

data = load_audit("Healthcare Readmission")
y = data.y.to_numpy()                             # readmitted within 30 days
group = (data.df["race"] == "AfricanAmerican").to_numpy()
n = len(y)
rng = np.random.default_rng(0)
print(f"true 30-day readmission rate: African American patients {y[group].mean():.3f}, others {y[~group].mean():.3f}, "
      f"gap {y[group].mean() - y[~group].mean():+.4f}")

# Each patient is discharged on a random day in the last 90 days. A readmission happens some days after discharge.
# We only know about it once it has happened, and a "no" is only known once the 30-day window has closed.
discharge = rng.integers(0, 90, n)
def monitored_gap(mean_delay_group, mean_delay_others, today=90):
    delay = np.where(group, rng.exponential(mean_delay_group, n), rng.exponential(mean_delay_others, n))
    delay = np.minimum(delay, 29.9)
    seen_positive = (y == 1) & (discharge + delay <= today)
    window_closed = discharge + 30 <= today
    known_negative = (y == 0) & window_closed
    # a naive monitor counts every discharged patient and treats every outcome it has not seen as "no"
    naive = {g: seen_positive[group == g].mean() for g in (True, False)}
    # a careful monitor only counts patients whose 30-day window is closed
    mature = window_closed
    careful = {g: ((y == 1) & mature)[mature & (group == g)].mean() for g in (True, False)}
    return naive[True] - naive[False], careful[True] - careful[False], naive[True], naive[False], seen_positive.sum() / (y == 1).sum()

print(f"\n{'mean days to readmission (group, others)':>40} {'naive gap':>10} {'naive rates':>16} {'share of positives seen':>24} {'careful gap':>12}")
for md_g, md_o in ((10, 10), (10, 20), (20, 10), (20, 20)):
    naive_gap, careful_gap, ng, no, seen = monitored_gap(md_g, md_o)
    print(f"{f'{md_g}, {md_o}':>40} {naive_gap:+10.4f} {f'{ng:.3f} / {no:.3f}':>16} {seen:24.1%} {careful_gap:+12.4f}")
```

## Limitations

1. **The delays are invented.** Exponential delays with chosen means are a modelling choice. Real delay distributions are measured from your own history.
2. **The dataset has no dates.** The discharge days are random, so only the mechanism is demonstrated, not any real timing.
3. **Mature-only is slow.** It cannot see anything newer than the outcome window, so a problem that started yesterday is invisible to it.
4. **Censoring can be informative.** If patients who never return are lost to follow-up in a way that depends on the group, even mature counts are biased.
5. **Negatives are also delayed.** A "no" is only known once the window closes, which is what the mature-only rule uses.

## Related Concepts

* [Monitoring Fairness in Production](monitoring-fairness-in-production.md) - the monitor this biases
* [Survivorship Bias](survivorship-bias.md) - when the data you see is the data that got through
* [Label Shift and Fairness](label-shift-and-fairness.md) - a change in base rates, which delay can mimic
* [Feedback Loop Bias](feedback-loop-bias.md) - decisions that shape the data seen next
* [Reject Inference](reject-inference.md) - outcomes missing for the people not selected
* [How Does Position Bias Distort What a Ranking Learns?](position-bias-and-click-models.md) - separating position from relevance in click data

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit used for the example above

## Further Reading

* [Chapelle, O. (2014): Modeling Delayed Feedback in Display Advertising](https://dl.acm.org/doi/10.1145/2623330.2623634) - KDD 2014; correcting conversion rates for conversions that have not happened yet.
* Lakkaraju, H. et al. (2017). The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables. KDD - outcomes observed only for the people a decision-maker chose.
* Kleinbaum, D. G., Klein, M. (2012). *Survival Analysis: A Self-Learning Text* (3rd ed.). Springer - censoring and delay models.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
