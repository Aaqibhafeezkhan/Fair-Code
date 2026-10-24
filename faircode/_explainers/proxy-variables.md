# Explainer: What is a Proxy Variable?

> *The sneaky reason AI stays biased even after you remove race from the data.*

---

## The One-Sentence Definition

A **proxy variable** is a data feature that *correlates* with a protected attribute (like race, gender, or class) even though it doesn't mention it directly - so removing the protected attribute from a model doesn't remove the bias it encodes.

---

## Why This Matters

Most people assume that removing race from an AI model makes it race-neutral. This is wrong.

Machine learning models don't care what a feature is *called*. They care what it *predicts*. If a feature like `custody_status` or `zip_code` is correlated with race in the training data, the model will learn and use that correlation - whether or not the word "race" appears anywhere in the dataset.

This is called **proxy discrimination**, and it's one of the hardest problems in algorithmic fairness.

---

## Common Proxy Variables

| Feature | Protected Attribute It Encodes | Why |
|---|---|---|
| Zip code | Race | Historical redlining segregated neighborhoods by race. Zip codes still reflect this. |
| Custody status | Race | Over-policing of Black communities leads to disproportionate pretrial detention. |
| Name | Gender / Ethnicity | Names encode gender and ethnicity with high accuracy. |
| College attended | Socioeconomic background | Elite colleges remain heavily stratified by class and race. |
| Credit score | Race / Class | Credit histories reflect historical exclusion from banking. |
| Prior arrests | Race | Arrest records reflect over-policing, not actual crime rates. |

---

## Real-World Proof: COMPAS Analysis

We tested this directly using the [ProPublica COMPAS dataset](https://github.com/propublica/compas-analysis).

### What We Did

**Step 1 - Biased model (includes race + proxies):**
```python
# compas-scores-raw.csv has no raw `race`/`sex` columns - race_binary is
# derived from Ethnic_Code_Text, exactly as unfair.py does
df['race_binary'] = df['Ethnic_Code_Text'].map({'African-American': 1, 'Caucasian': 0})
X = pd.get_dummies(df[[
    'race_binary',
    'Sex_Code_Text',
    'CustodyStatus',        # proxy for race
    'MaritalStatus'
]])
```

**Results:**
| Group | High-Risk Flag Rate |
|---|---|
| Black defendants | 87.16% |
| White defendants | 0.40% |
| **Fairness gap** | **86.77%** |

---

**Step 2 - Remove race only (naive approach):**

Many developers stop here. They drop `race_binary` and assume the model is now fair. Dropping race alone already takes the gap from 86.77% down to 18.38% - most of the signal was riding on race directly. But it isn't fully fixed: `CustodyStatus` is still acting as a racial proxy, and its own marginal contribution beyond race is the remaining ~2.7 points down to 15.69% once it's removed too (see Step 3).

---

**Step 3 - Remove race AND known proxies (our fix):**
```python
X = pd.get_dummies(df[[
    'Sex_Code_Text',
    'MaritalStatus'
    # Race removed ✓
    # CustodyStatus removed ✓ (proxy for race)
]])
```

**Results:**
| Group | High-Risk Flag Rate |
|---|---|
| Black defendants | 84.71% |
| White defendants | 69.02% |
| **Fairness gap** | **15.69%** |

### Summary

| Approach | Fairness Gap | Reduction |
|---|---|---|
| Biased model | 86.77% | - |
| Remove race only | 18.38% | 79% |
| Remove race + proxy | 15.69% | **82%** |

**Removing the protected attribute alone is not enough. You must audit every feature for correlation with protected attributes.**

---

## How to Detect Proxy Variables

```python
import pandas as pd
from scipy.stats import chi2_contingency

def check_proxy(df, feature, protected_attr):
    """
    Check if a feature is a proxy for a protected attribute
    using a chi-squared test of independence.
    Returns p-value - if < 0.05, likely a proxy.
    """
    contingency = pd.crosstab(df[feature], df[protected_attr])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        'feature': feature,
        'protected_attr': protected_attr,
        'p_value': round(p, 4),
        'is_proxy': p < 0.05
    }

# Example usage
result = check_proxy(df, 'CustodyStatus', 'Ethnic_Code_Text')
print(result)
# {'feature': 'CustodyStatus', 'protected_attr': 'Ethnic_Code_Text', 'p_value': 0.0, 'is_proxy': True}
```

Run this on every feature in your dataset before training. Any feature with `is_proxy: True` needs careful consideration - either remove it or apply fairness-aware techniques.

---

## The Bigger Picture

Proxy variables exist because our social world is stratified. Zip codes encode race because of redlining. Custody records encode race because of over-policing. Credit scores encode class because of historical exclusion.

Data doesn't exist in a vacuum. It reflects the society that generated it. A model trained on that data will learn those reflections - unless you actively audit and intervene.

**This is why algorithmic auditing is not optional. It is a prerequisite for deployment.**

---

## Limitations

1. **A small p-value is not a large effect.** The chi-squared test above says a feature and the protected attribute are associated, not how much of the attribute the feature carries. With tens of thousands of rows almost every feature has a p-value near zero, so `is_proxy` is true for trivial associations. Pair it with an effect size such as Cramér's V, as the Open Dataset Profiler's proxy hints do.
2. **One feature at a time can miss a cluster.** Several weak features can together reconstruct an attribute that none of them reveals alone. See [Proxy Entanglement](proxy-entanglement.md).
3. **The test needs categories.** A numeric feature has to be binned first, and the bins change the result. Sparse tables with small expected counts make the p-value unreliable.
4. **Removing a proxy also removes real signal.** A feature can be a proxy and a legitimate predictor at the same time, so dropping it trades accuracy for fairness. See [Fairness-Accuracy Trade-off](fairness-accuracy-tradeoff.md).
5. **You can only test the attributes you have.** If the protected attribute is not in the data, you cannot run the check. See [Inferring Protected Attributes](inferring-protected-attributes.md).

---

## Related Concepts

* [The Obermeyer Case](obermeyer-cost-proxy.md) - cost as a proxy for need
* [How Does Position Bias Distort What a Ranking Learns?](position-bias-and-click-models.md) - separating position from relevance in click data

---

## Related Projects in This Repo

- [`COMPAS/`](../COMPAS/) - Full COMPAS analysis: biased model → fair model → 82% gap reduction
- [`AI Fair Recruitment/`](../AI%20Fair%20Recruitment/) - AI recruitment bias: 96.1% gap reduction after feature audit
- The other audits run the same proxy analysis: [`German Credit Lending/`](../German%20Credit%20Lending/), [`Insurance Denial/`](../Insurance%20Denial/), [`Benefits Denial/`](../Benefits%20Denial/), [`Healthcare Readmission/`](../Healthcare%20Readmission/) and [`Tenant Screening/`](../Tenant%20Screening/)
- Planned in the [ROADMAP](../ROADMAP.md): facial recognition accuracy gaps, HMDA mortgage lending bias and an LLM bias audit

---

## Further Reading

- [ProPublica: Machine Bias (2016)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [Barocas & Hardt: Fairness and Machine Learning](https://fairmlbook.org/)
- [Obermeyer et al.: Dissecting racial bias in an algorithm used to manage the health of populations (Science, 2019)](https://science.sciencemag.org/content/366/6464/447)

---

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
