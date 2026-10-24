# Generated from explainers/delayed-feedback.md by scripts/build_explainers.py. Edit the markdown, not this file.
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
