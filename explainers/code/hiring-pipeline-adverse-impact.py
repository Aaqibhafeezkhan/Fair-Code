# Generated from explainers/hiring-pipeline-adverse-impact.md by scripts/build_explainers.py. Edit the markdown, not this file.
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
