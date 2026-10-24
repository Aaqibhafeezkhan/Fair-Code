# Generated from explainers/confidence-intervals-for-ratios.md by scripts/build_explainers.py. Edit the markdown, not this file.
import numpy as np
from scipy.stats import norm
from faircode.benchmark import load_audit        # same loader, encoding and split as the benchmark harness
from faircode.models import build_model

def intervals(selected_a, n_a, selected_b, n_b, level=0.95, n_boot=4000, seed=0):
    """Ratio of selection rates, group a over group b, with three 95% intervals."""
    rng = np.random.default_rng(seed)
    ratio = (selected_a / n_a) / (selected_b / n_b)
    z = norm.ppf(0.5 + level / 2)
    # 1. percentile bootstrap on the ratio itself
    boot_a = rng.binomial(n_a, selected_a / n_a, n_boot) / n_a
    boot_b = rng.binomial(n_b, selected_b / n_b, n_boot) / n_b
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = boot_a / boot_b
    ratios = ratios[np.isfinite(ratios)]
    percentile = np.percentile(ratios, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    # 2. delta method on the log ratio: se = sqrt(1/x_a - 1/n_a + 1/x_b - 1/n_b)  (Katz et al. 1978)
    se = np.sqrt(1 / selected_a - 1 / n_a + 1 / selected_b - 1 / n_b)
    log_interval = np.exp(np.log(ratio) + np.array([-z, z]) * se)
    return ratio, percentile, log_interval, se

def show(label, sa, na, sb, nb):
    ratio, pct, log, se = intervals(sa, na, sb, nb)
    print(f"{label:46s} {sa:5d}/{na:<5d} vs {sb:5d}/{nb:<5d} ratio {ratio:9.3f} | percentile bootstrap [{pct[0]:8.3f}, {pct[1]:10.3f}] "
          f"| log-ratio (delta) [{log[0]:8.3f}, {log[1]:10.3f}]")

# 1. A stable ratio: Benefits Denial, logistic regression on the core features, selection of women against men.
data = load_audit("Benefits Denial")
model = build_model("logistic_regression", data.random_state).fit(data.X.iloc[data.idx_train], data.y.iloc[data.idx_train])
pred = model.predict(data.X.iloc[data.idx_test])
women = data.masks["sex"].to_numpy()[data.idx_test] == 1
show("Benefits Denial, women vs men", int(pred[women].sum()), int(women.sum()), int(pred[~women].sum()), int((~women).sum()))

# 2. A ratio with a near-zero denominator: COMPAS, gradient boosting with every feature (race included), Black vs White defendants.
compas = load_audit("COMPAS", strategy="baseline")
gb = build_model("gradient_boosting", compas.random_state).fit(compas.X.iloc[compas.idx_train], compas.y.iloc[compas.idx_train])
pred = gb.predict(compas.X.iloc[compas.idx_test])
black = compas.masks["race"].to_numpy()[compas.idx_test] == 1
print(f"COMPAS gradient boosting: {int(pred[~black].sum())} of {int((~black).sum())} White defendants are selected")
if pred[~black].sum() > 0:
    show("COMPAS gradient boosting, Black vs White", int(pred[black].sum()), int(black.sum()), int(pred[~black].sum()), int((~black).sum()))

# 3. The same arithmetic with a denominator of 3, 10 and 30 selected rows out of 1,444.
for denominator in (3, 10, 30):
    show(f"{denominator} selected in the advantaged group", 800, 1804, denominator, 1444)
