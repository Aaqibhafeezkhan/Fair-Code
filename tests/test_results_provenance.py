"""results/PROVENANCE.json ties the result files to the run that made them (#1129)."""

import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("yaml", reason="faircode.benchmark needs the optional benchmark extra")
pytest.importorskip("sklearn", reason="faircode.benchmark needs the optional benchmark extra")
pytest.importorskip("fairlearn", reason="faircode.benchmark needs the optional benchmark extra")

from faircode import benchmark  # noqa: E402

diff_results = importlib.import_module("scripts.diff_results")
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_collect_provenance_records_the_run():
    record = benchmark.collect_provenance(["b", "a"], 200, 100, description="test run")
    assert record["description"] == "test run" and record["audits"] == ["a", "b"]
    assert record["n_resamples"] == 200 and record["n_permutations"] == 100
    assert set(record["libraries"]) == set(benchmark.PROVENANCE_LIBRARIES)
    assert record["libraries"]["pandas"] and record["python"].count(".") == 2
    assert record["dirty"] in (True, False, None)
    assert len(record["lockfile_sha256"]) == 64          # the repo has requirements-lock.txt


def test_write_report_with_provenance_hashes_each_result_file(tmp_path):
    fairness = pd.DataFrame({"audit": ["a"], "strategy": ["s"], "model": ["m"], "protected_attribute": ["x"],
                             "metric": ["demographic_parity_diff"], "value": [0.1], "ci_low": [0.0], "ci_high": [0.2],
                             "p_value": [0.5], "significant": [False], "n_disadvantaged": [10], "n_advantaged": [10],
                             "small_sample_warning": [True], "note": [None]})
    performance = pd.DataFrame({"audit": ["a"], "strategy": ["s"], "model": ["m"], "metric": ["accuracy"], "value": [0.7],
                                "ci_low": [0.6], "ci_high": [0.8], "n": [20]})
    run = benchmark.collect_provenance(["a"], 10, 10)
    benchmark.write_report(fairness, performance, tmp_path, make_plots=False, provenance=run)
    record = json.loads((tmp_path / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert record["schema"] == 1 and record["runs"] == [run]
    for name, digest in record["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert set(record["files"]) == {"results_fairness.csv", "results_performance.csv", "summary.csv"}


def test_write_report_without_provenance_writes_no_file(tmp_path):
    empty = pd.DataFrame()
    benchmark.write_report(empty, empty, tmp_path, make_plots=False)
    assert not (tmp_path / "PROVENANCE.json").exists()


def test_the_committed_provenance_matches_the_committed_result_files():
    record = json.loads((REPO_ROOT / "results" / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert record["runs"], "results/PROVENANCE.json lists no run"
    for name, digest in record["files"].items():
        actual = hashlib.sha256((REPO_ROOT / "results" / name).read_bytes()).hexdigest()
        assert actual == digest, (f"results/{name} changed without results/PROVENANCE.json: rerun `faircode benchmark` "
                                  "or update the record with faircode.benchmark.write_provenance")


def test_diff_results_reads_both_kinds_of_provenance_and_names_version_differences():
    current = diff_results.read_provenance(REPO_ROOT / "results")
    reference = diff_results.read_provenance(REPO_ROOT / "paper" / "results-frozen")
    assert current["source"] == "PROVENANCE.json" and reference["source"] == "MANIFEST.md"
    assert reference["libraries"]["numpy"] and reference["commit"]
    report = diff_results.provenance_report(REPO_ROOT / "results", REPO_ROOT / "paper" / "results-frozen")
    assert "current  : commit" in report and "reference: commit" in report and "versions that differ:" in report


def test_a_directory_with_no_provenance_is_reported_as_such(tmp_path):
    assert diff_results.read_provenance(tmp_path) is None
    assert "no provenance recorded" in diff_results.provenance_report(tmp_path, tmp_path)
