> *Most dataset problems are not in the data. They are in the questions nobody asked about where it came from.*

## The One-Sentence Definition

A **datasheet for a dataset** is a short list of standard questions, covering why the data was made, what is in it, how it was collected and processed, and what it should and should not be used for, answered in writing so a user can judge whether it fits their purpose.

## Why It Matters

The audits in this repo each start with a CSV downloaded from somewhere, and almost every bias in the explainers can be traced to a gap in what is known about that file: who was sampled ([Sampling Bias](sampling-bias.md)), who is missing ([Missing Data as Bias in EHRs](missing-data-bias-ehr.md)), how the label was produced ([Label Bias](label-bias.md)), which people were filtered out first ([Survivorship Bias](survivorship-bias.md)).

Electronics ship with a datasheet describing what each component can and cannot do. Datasets usually ship with nothing. The proposal by Gebru and colleagues is to answer the same questions every time, so that an unknown is visible as an unknown.

## Core Concept: Five Question Groups

1. **Motivation:** why was it created, by whom, with whose money?
2. **Composition:** what is an instance, how many are there, what is the label, who is in it, what is missing, is anything sensitive?
3. **Collection:** how was it acquired, over what period, did people consent?
4. **Preprocessing:** what was cleaned, filtered or relabelled, and is the raw data kept?
5. **Uses:** what has it been used for, and what should it not be used for?

Half of the composition answers can be computed from the file. The other half cannot, and an honest datasheet says so instead of guessing.

## Concrete Example: Healthcare Readmission

This datasheet is generated for the [Healthcare Readmission](../Healthcare%20Readmission/) audit data (the 130-hospital diabetes dataset). Everything computable is read from the file and the audit manifest; the questions the repo cannot answer are printed as UNKNOWN.

```
Datasheet: Healthcare Readmission - Clinical Bias

MOTIVATION
  Why was the dataset created, by whom, who funded it?    UNKNOWN: not recorded in this repo

COMPOSITION
  Instances in the file                                   101,766 rows, 50 columns
  Instances after the audit's row filters                 99,492 (2 filters)
  Positive rate of the target (readmitted)                11.2%
  Largest groups of gender                                Female 53.8%, Male 46.2%, Unknown/Invalid 0.0%
  Largest groups of race                                  Caucasian 74.8%, AfricanAmerican 18.9%, ? 2.2%, Hispanic 2.0%
  Largest groups of age                                   [70-80) 25.6%, [60-70) 22.1%, [50-60) 17.0%, [80-90) 16.9%
  Missing values in weight                                96.9%
  Missing values in max_glu_serum                         94.7%
  Missing values in A1Cresult                             83.3%
  Does it contain sensitive attributes?                   yes: race, gender, age

COLLECTION
  How was the data acquired, over what period?            UNKNOWN: not recorded in this repo
  Did the people in it consent to this use?               UNKNOWN: not recorded in this repo

USES
  Used for                                                the Healthcare Readmission audit and the benchmark harness
  Should it be used for clinical decisions?               no: one health system, one coding scheme

3 of the questions above cannot be answered from the repo.
```

The computable answers already matter. After the audit's two row filters 99,492 of 101,766 rows remain and the positive rate is 11.2%. Race is 74.8% Caucasian and 18.9% African American, with 2.2% recorded as `?`. The missingness table is the sharper finding: `weight` is missing for 96.9% of rows, `max_glu_serum` for 94.7% and `A1Cresult` for 83.3%, so any model that uses them is learning from the few rows where a test happened to be ordered, which is exactly the pattern [Missing Data as Bias in EHRs](missing-data-bias-ehr.md) describes.

Three questions cannot be answered from the repo: why the data was created, how it was acquired, and whether the patients consented to this use. Those unknowns are part of the dataset's description.

## Detection/Implementation Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/datasheets-for-datasets.py`; that file is generated from the block below). Change the folder name to produce one for another audit.

```python
import pandas as pd
from faircode.benchmark import load_audit

data = load_audit("Healthcare Readmission")                  # applies the manifest's row filters and target
manifest = data.manifest
raw = pd.read_csv(manifest.dataset_path, low_memory=False)
missing = raw.replace("?", pd.NA).isna().mean().sort_values(ascending=False)
unknown = []

def ask(question, answer=None):
    if answer is None:
        unknown.append(question)
        answer = "UNKNOWN: not recorded in this repo"
    print(f"  {question:55s} {answer}")

print(f"Datasheet: {manifest.title}\n")
print("MOTIVATION")
ask("Why was the dataset created, by whom, who funded it?")
print("\nCOMPOSITION")
ask("Instances in the file", f"{len(raw):,} rows, {raw.shape[1]} columns")
ask("Instances after the audit's row filters", f"{len(data.y):,} ({len(manifest.row_filters)} filters)")
ask(f"Positive rate of the target ({manifest.target.column})", f"{data.y.mean():.1%}")
for pa in manifest.protected_attributes:
    shares = raw[pa.column].value_counts(normalize=True).head(4)
    ask(f"Largest groups of {pa.column}", ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
for col, share in missing.head(3).items():
    ask(f"Missing values in {col}", f"{share:.1%}")
ask("Does it contain sensitive attributes?", "yes: race, gender, age")
print("\nCOLLECTION")
ask("How was the data acquired, over what period?")
ask("Did the people in it consent to this use?")
print("\nUSES")
ask("Used for", "the Healthcare Readmission audit and the benchmark harness")
ask("Should it be used for clinical decisions?", "no: one health system, one coding scheme")
print(f"\n{len(unknown)} of the questions above cannot be answered from the repo.")
```

## Limitations

1. **It documents; it does not fix.** A datasheet makes a problem visible and leaves the problem in place.
2. **The unanswerable questions need a person.** Motivation, consent and collection come from the data's authors, not from the file.
3. **Computed answers can mislead.** A `?` counted as missing is a convention. The same file coded differently would give a different missingness figure.
4. **Datasheets go stale.** A dataset that is updated or re-filtered needs a new sheet.
5. **It is not a licence to use.** A complete datasheet can say the data is unsuitable, and that is its best outcome.

## Related Concepts

* [Model Cards](model-cards.md) - the matching document for the model trained on the data
* [Sampling Bias](sampling-bias.md) - the composition questions are how you spot it
* [Missing Data as Bias in EHRs](missing-data-bias-ehr.md) - what the missingness numbers above mean
* [Label Bias](label-bias.md) - the collection and preprocessing questions expose it
* [Protected Attribute](protected-attribute.md) - the sensitive columns the composition section must name
* [Algorithmic Impact Assessment](algorithmic-impact-assessment.md) - where a dataset's unknowns enter a deployment review

## Related Projects in This Repo

* [`Healthcare Readmission/`](../Healthcare%20Readmission/) - the audit data described above
* [Open Dataset Profiler](../profiler.html) - computes the composition answers for any file

## Further Reading

* [Gebru, T. et al. (2021): Datasheets for Datasets](https://arxiv.org/abs/1803.09010) - Communications of the ACM 64(12); the question list and the motivation.
* [Bender, E. M., Friedman, B. (2018): Data Statements for Natural Language Processing](https://aclanthology.org/Q18-1041/) - TACL 6; the same idea for language data.
* [Holland, S. et al. (2018): The Dataset Nutrition Label](https://arxiv.org/abs/1805.03677) - a related, more quantitative proposal for summarizing a dataset.

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
