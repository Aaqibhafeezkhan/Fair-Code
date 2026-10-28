"""scripts/check_dataset_sources.py compares each audit's dataset link in README.md and its scripts (#1208)."""

import importlib
from pathlib import Path

check = importlib.import_module("scripts.check_dataset_sources")

KAGGLE = "https://www.kaggle.com/datasets/ppb00x/credit-risk-customers"
NEW = "https://www.kaggle.com/datasets/someone/credit-risk-moved"


def _repo(root: Path, readme_url, unfair_url, fair_url=None) -> Path:
    link = f" · [Kaggle source]({readme_url})" if readme_url else ""
    (root / "README.md").write_text(
        "## Projects\n\n"
        "### 01 · German Credit Lending - Lending Bias\n\n"
        f"**Dataset:** `credit_customers.csv` - German Credit (1,000 records){link}\n\n"
        "### 02 · COMPAS - Criminal Justice Bias\n\n"
        "**Dataset:** `compas-scores-raw.csv` - ProPublica's public COMPAS dataset\n\n"
        "## Why This Matters\n\n[elsewhere](https://example.org/not-a-dataset)\n",
        encoding="utf-8",
    )
    for name, url in (("unfair.py", unfair_url), ("fair.py", fair_url)):
        folder = root / "German Credit Lending"
        folder.mkdir(exist_ok=True)
        header = f"# Source: German Credit\n# {url}\n" if url else "# Load dataset\n"
        (folder / name).write_text(header + "import pandas as pd\n", encoding="utf-8")
    (root / "COMPAS").mkdir()
    (root / "COMPAS" / "unfair.py").write_text("# 1. Load your uploaded file\n", encoding="utf-8")
    return root


def test_matching_links_pass(tmp_path):
    assert check.disagreements(_repo(tmp_path, KAGGLE, KAGGLE, KAGGLE)) == []


def test_script_without_a_link_passes(tmp_path):
    assert check.disagreements(_repo(tmp_path, KAGGLE, KAGGLE, None)) == []


def test_link_changed_in_readme_only_fails(tmp_path):
    problems = check.disagreements(_repo(tmp_path, NEW, KAGGLE, KAGGLE))
    assert [p.split(":")[0] for p in problems] == ["German Credit Lending/unfair.py", "German Credit Lending/fair.py"]
    assert all(KAGGLE in p and NEW in p for p in problems)


def test_link_changed_in_one_script_only_fails(tmp_path):
    problems = check.disagreements(_repo(tmp_path, KAGGLE, NEW, KAGGLE))
    assert len(problems) == 1
    assert problems[0].startswith(f"German Credit Lending/unfair.py: cites {NEW}")


def test_readme_without_a_link_fails_when_a_script_has_one(tmp_path):
    problems = check.disagreements(_repo(tmp_path, None, KAGGLE))
    assert problems == [
        f"German Credit Lending/unfair.py: cites {KAGGLE}, but the README's Dataset line for German Credit Lending has no link"
    ]


def test_trailing_slash_and_full_stop_are_ignored(tmp_path):
    assert check.disagreements(_repo(tmp_path, KAGGLE + "/", KAGGLE + ".")) == []


def test_project_block_without_a_folder_is_reported(tmp_path):
    root = _repo(tmp_path, KAGGLE, KAGGLE)
    (root / "COMPAS" / "unfair.py").unlink()
    (root / "COMPAS").rmdir()
    assert check.disagreements(root) == ["README.md: project block 'COMPAS' has no folder of that name"]


def test_readme_with_no_project_blocks_fails(tmp_path):
    (tmp_path / "README.md").write_text("# Fair-Code\n", encoding="utf-8")
    assert len(check.disagreements(tmp_path)) == 1


def test_repository_links_agree():
    assert check.disagreements() == []
