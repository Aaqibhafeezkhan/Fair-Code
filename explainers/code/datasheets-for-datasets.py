# Generated from explainers/datasheets-for-datasets.md by scripts/build_explainers.py. Edit the markdown, not this file.
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
