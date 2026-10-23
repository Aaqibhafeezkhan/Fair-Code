# What Is Differential Privacy (and Its Tension With Fairness)?

> *Differential privacy adds calibrated noise during training so no single person's data changes the model much. That protection is not free, and it is not evenly distributed: the accuracy it costs falls hardest on exactly the underrepresented groups a fairness audit is trying to protect.*

## The One-Sentence Definition

**Differential privacy (DP)** is a formal guarantee that the output of an analysis (here, a trained model) is almost unchanged whether or not any one individual's record was included, achieved during training by clipping each example's gradient contribution and adding random noise, with the strength of the guarantee set by a privacy budget **epsilon** (smaller epsilon = more noise = stronger privacy).

## Why It Matters

A bias-mitigation pipeline and a privacy requirement are often imposed on the same model by different stakeholders - a fairness team and a data-protection officer - who assume the two goals compose for free. They do not.

The standard training algorithm for a DP model, **DP-SGD**, does two things to every gradient step: it **clips** each per-example gradient to a fixed norm, and it **adds Gaussian noise** proportional to that norm. Both operations interact badly with class imbalance:

- Underrepresented groups tend to be the ones a model has not yet fit well, so their examples produce **large gradients** - the ones clipping cuts down the most. The majority's already-small gradients pass through nearly untouched.
- The noise is added to the batch-averaged gradient regardless of composition. A signal that is a small fraction of the batch (the minority's contribution) is **drowned by the noise** at a lower epsilon than the majority's signal is.

The result, documented by Bagdasaryan, Poursaeed and Shmatikov (2019), is that the accuracy drop from adding DP is "not borne equally": it is substantially larger for underrepresented classes and subgroups, and **if the original model was already unfair, DP makes it more unfair**. This is the same shape as [the fairness-accuracy trade-off](fairness-accuracy-tradeoff.md) - closing one gap opens another - except the axis being traded against fairness is privacy, not overall accuracy.

## How It Works

DP-SGD replaces the plain gradient step with:

```
1. compute per-example gradients g_i for the minibatch
2. clip:   g_i  <-  g_i / max(1, ||g_i|| / C)          # cap the L2 norm at C
3. sum, then add noise:   g  <-  (sum_i g_i  +  N(0, sigma^2 C^2 I)) / B
4. step:   w  <-  w - lr * g
```

`C` is the clipping norm and `sigma` the noise multiplier; together with the number of steps and the sampling rate they determine epsilon (via a privacy accountant). Smaller epsilon needs a larger `sigma`.

Why this is disparate:

- **Clipping is a per-example ceiling.** An example the model already predicts well has a small gradient and is unaffected. An example the model gets wrong - disproportionately from a group with little training data - has a large gradient that gets scaled down, so its correction is weakened. Over many steps the majority keeps learning at full strength while the minority's updates are throttled.
- **Noise has a fixed scale but the useful signal does not.** In a batch that is 90% majority, the minority's contribution to the summed gradient is roughly a tenth of the total before noise. The noise term `N(0, sigma^2 C^2)` is the same size either way, so it overwhelms the minority direction while the majority direction still stands out. As `sigma` rises (epsilon falls), the minority signal disappears first.

A minimal way to see the mechanism: train a logistic model with DP-SGD on data where 10% of rows follow a different feature-to-label rule than the other 90%, and read out accuracy per group as `sigma` increases. The majority accuracy holds; the minority accuracy falls faster and further. The Detection Code section gives a runnable version.

## Concrete Illustration (from the literature, not a Fair Code run)

**This repo's benchmark harness does not train any DP model** - there is no `epsilon` in `faircode/strategies.py` and no DP row in `results/`. Every number in this section is from the published study below, quoted so the pattern is concrete, not reproduced here.

Bagdasaryan et al. (2019), *Differential Privacy Has Disparate Impact on Model Accuracy* (NeurIPS 2019), trained the same models with and without DP-SGD across several tasks:

- A **gender-classification model** on facial images: adding DP reduced accuracy far more for darker-skinned faces (an underrepresented group in the training set) than for lighter-skinned faces, widening an accuracy gap that already existed in the non-private model.
- A **sentiment-analysis model** on tweets: DP reduced accuracy more on text written in African-American English than on the majority dialect.
- Across tasks, the finding held: "the reduction in accuracy incurred by deep learning models with differential privacy disproportionately impacts underrepresented subgroups," and the effect grows as epsilon shrinks. Their summary of the mechanism is that **the gradient clipping and noise addition of DP-SGD "disproportionately affect" the subgroups whose gradients are largest and whose share of the data is smallest** - i.e. "the poor get poorer."

The relevance to this repo: every audit here is built around a protected group that is, in its dataset, the smaller or worse-served one (younger applicants, minority defendants, one insurance tier). Those are precisely the groups DP-SGD costs the most accuracy. "We added a privacy guarantee" is therefore not evidence of "we did not make the fairness gap worse" - it needs its own per-group check.

## Detection Code

A self-contained, dependency-light DP-SGD toy: per-example gradient clipping plus Gaussian noise on a logistic model, with a per-group accuracy readout, run across a sweep of noise multipliers. It is illustrative - it shows the *mechanism*, not a calibrated epsilon.

```python
import numpy as np


def make_imbalanced_data(n=6000, minority_frac=0.10, seed=0):
    """90% majority + 10% minority, where the minority's label depends on a
    different set of features - so the minority needs its own signal to be
    learned, and that signal is what clipping + noise erode first."""
    rng = np.random.default_rng(seed)
    n_min = int(n * minority_frac)
    group = np.array([1] * n_min + [0] * (n - n_min))   # 1 = minority
    rng.shuffle(group)
    x = rng.normal(size=(n, 6))
    w_majority = np.array([1.5, -1.0, 0.7, 0.0, 0.0, 0.0])
    w_minority = np.array([0.0, 0.0, 0.0, 1.5, -1.0, 0.7])
    logits = np.where(group == 1, x @ w_minority, x @ w_majority)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    return x, y, group


def dp_sgd_logreg(x, y, steps=4000, batch=64, lr=0.5,
                  clip_norm=1.0, noise_multiplier=0.0, seed=1):
    """DP-SGD for logistic regression. noise_multiplier is 'sigma': 0 means
    clip-only (no privacy), larger means smaller epsilon."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = np.zeros(d)
    for _ in range(steps):
        idx = rng.integers(0, n, batch)
        xb, yb = x[idx], y[idx]
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
        per_example = (p - yb)[:, None] * xb                     # B x d
        norms = np.linalg.norm(per_example, axis=1, keepdims=True)
        per_example = per_example / np.maximum(1.0, norms / clip_norm)   # clip
        grad = per_example.sum(axis=0)
        if noise_multiplier > 0:
            grad = grad + rng.normal(scale=noise_multiplier * clip_norm, size=d)
        w -= lr * grad / batch
    return w


def accuracy_by_group(w, x, y, group):
    pred = (x @ w > 0).astype(int)
    maj = ((pred == y) & (group == 0)).sum() / (group == 0).sum()
    minority = ((pred == y) & (group == 1)).sum() / (group == 1).sum()
    return maj, minority


def sweep(seeds=range(6)):
    x_tr, y_tr, g_tr = make_imbalanced_data(seed=0)
    x_te, y_te, g_te = make_imbalanced_data(n=4000, seed=99)

    def mean_acc(**kw):
        rows = [accuracy_by_group(dp_sgd_logreg(x_tr, y_tr, seed=s, **kw),
                                  x_te, y_te, g_te) for s in seeds]
        return np.mean(rows, axis=0)

    print(f"{'setting':<28}{'majority':>10}{'minority':>10}{'gap':>8}")
    maj, minr = mean_acc(noise_multiplier=0.0)
    print(f"{'clip only (no privacy)':<28}{maj:>10.3f}{minr:>10.3f}{maj - minr:>+8.3f}")
    for sigma in (1.0, 2.0, 4.0, 8.0):
        maj, minr = mean_acc(noise_multiplier=sigma)
        print(f"{f'DP-SGD sigma={sigma}':<28}{maj:>10.3f}{minr:>10.3f}{maj - minr:>+8.3f}")


# sweep()
# The majority column stays roughly flat as sigma grows; the minority column
# and the gap move against the minority. Swap in a real dataset + a privacy
# accountant (e.g. Opacus, tensorflow-privacy) to attach an epsilon to each row.
```

## Limitations and Trade-offs

### 1. Epsilon has no universally correct value

There is no threshold at which a model is "private enough". Deployed systems have used epsilon from below 1 (strong) to over 10 (weak); the choice trades re-identification risk against utility and against the fairness cost described here, and it is a policy decision, not a statistical one - the same shape as choosing a fairness metric.

### 2. DP protects against a specific threat, not against bias

The DP guarantee is about **membership inference and individual re-identification**: an attacker cannot tell whether a particular person was in the training set. It says nothing about whether the labels are biased, whether the sampling was biased, or whether the model discriminates. A perfectly private model can be perfectly unfair. DP and fairness are orthogonal guarantees that happen to interfere.

### 3. The disparate impact can sometimes be mitigated, at a cost

Per-group clipping norms, adaptive clipping, or fair-DP training objectives can reduce the gap DP-SGD opens, but they add hyperparameters, can weaken the privacy accounting, and are not standard in off-the-shelf DP training libraries. Assume the vanilla DP-SGD behavior unless a mitigation is explicitly in place and measured.

### 4. The toy here is a mechanism demo, not a calibrated result

The Detection Code has no privacy accountant, so its `sigma` values do not map to a real epsilon, and its synthetic data exaggerates the minority's distinctness for clarity. Treat it as an illustration of *why* the effect happens; for a real number, run DP-SGD with an accountant on a real dataset and measure per-group accuracy directly.

## Related Concepts

* [What Is the Fairness-Accuracy Trade-off?](fairness-accuracy-tradeoff.md) - the same "closing one gap opens another" structure, with accuracy rather than privacy on the other axis.
* [What Is Class Imbalance?](class-imbalance.md) - why an underrepresented group's signal is a small share of each batch, the property DP-SGD's noise exploits.
* [Mitigation Strategies](mitigation-strategies.md) - this repo's five bias-mitigation strategies, none of which is DP; adding DP on top would need its own per-group evaluation.
* Membership inference - the attack DP is designed to prevent: given a trained model, deciding whether a specific record was in its training set. DP bounds how well any such attack can do.
* [What Is Underdiagnosis Bias?](underdiagnosis-bias.md) - another case where a technique aimed at one problem quietly worsens per-group outcomes.
* [What Is the Bias-Variance Trade-off?](bias-variance-tradeoff.md) - noise added for privacy shifts the same trade-off for small groups

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - a clinical model where a real deployment would plausibly face a patient-privacy requirement; the smallest race and age subgroups are the ones DP-SGD would cost the most.
* [`German Credit Lending/`](../German%20Credit%20Lending/) - younger applicants are the protected, smaller group; a DP-trained credit model would degrade for them fastest as epsilon tightens.

## Further Reading

* [Bagdasaryan, E., Poursaeed, O., Shmatikov, V. (2019): Differential Privacy Has Disparate Impact on Model Accuracy, *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*](https://arxiv.org/abs/1905.12101) - the paper this explainer's central claim comes from; shows the DP accuracy cost is larger for underrepresented subgroups across image, text, and tabular tasks.
* [Abadi, M. et al. (2016): Deep Learning with Differential Privacy, *ACM CCS 2016*](https://arxiv.org/abs/1607.00133) - the original DP-SGD algorithm (per-example clipping + Gaussian noise + moments accountant).
* [Dwork, C., Roth, A. (2014): The Algorithmic Foundations of Differential Privacy, *Foundations and Trends in Theoretical Computer Science*, 9(3-4)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) - the standard reference for the epsilon definition and its properties.
* [Cummings, R., Gupta, V., Kimpara, D., Morgenstern, J. (2019): On the Compatibility of Privacy and Fairness, *FairUMAP 2019*](https://arxiv.org/abs/1907.00212) - a formal look at when the two guarantees can and cannot be satisfied together.

---

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
