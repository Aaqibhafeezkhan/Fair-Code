# Generated from explainers/compounding-disparities.md by scripts/build_explainers.py. Edit the markdown, not this file.
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
