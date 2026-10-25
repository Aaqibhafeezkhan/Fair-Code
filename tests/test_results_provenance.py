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


def _result_frames(audits):
    fairness = pd.DataFrame({
        "audit": audits,
        "strategy": ["baseline"] * len(audits),
        "protected_attribute": ["age"] * len(audits),
        "metric": ["demographic_parity_diff"] * len(audits),
        "value": [0.1] * len(audits),
        "significant": [False] * len(audits),
        "model": ["logistic_regression"] * len(audits),
    })
    performance = pd.DataFrame({
        "audit": audits,
        "strategy": ["baseline"] * len(audits),
        "model": ["logistic_regression"] * len(audits),
        "metric": ["accuracy"] * len(audits),
        "value": [0.7] * len(audits),
    })
    return fairness, performance


def _write_result_set(out_dir, audits, *, provenance=True):
    fairness, performance = _result_frames(audits)
    run = {"audits": list(audits)} if provenance else None
    return benchmark.write_report(fairness, performance, out_dir, make_plots=False, provenance=run)


def _output_bytes(out_dir):
    names = (*benchmark.RESULT_FILES, benchmark.PROVENANCE_FILE)
    return {name: (out_dir / name).read_bytes() for name in names if (out_dir / name).exists()}


def test_subset_run_refuses_other_audits_without_changing_any_output(tmp_path):
    out_dir = tmp_path / "results"
    _write_result_set(out_dir, ["audit_a", "audit_b"])
    before = _output_bytes(out_dir)
    fairness, performance = _result_frames(["audit_b"])

    with pytest.raises(ValueError, match="contains results for other audits.*audit_a"):
        benchmark.write_report(
            fairness, performance, out_dir, make_plots=False, provenance={"audits": ["audit_b"]}
        )

    assert _output_bytes(out_dir) == before


def test_fresh_and_compatible_same_audit_outputs_can_be_written(tmp_path):
    out_dir = tmp_path / "results"
    _write_result_set(out_dir, ["audit_a"])
    before = _output_bytes(out_dir)
    fairness, performance = _result_frames(["audit_a"])
    fairness.loc[0, "value"] = 0.4

    benchmark.write_report(
        fairness, performance, out_dir, make_plots=False, provenance={"audits": ["audit_a"]}
    )

    after = _output_bytes(out_dir)
    assert after.keys() == before.keys()
    assert after["results_fairness.csv"] != before["results_fairness.csv"]
    assert json.loads(after["PROVENANCE.json"])["runs"] == [{"audits": ["audit_a"]}]


def test_complete_legacy_outputs_allow_a_compatible_rerun_without_provenance(tmp_path):
    out_dir = tmp_path / "results"
    _write_result_set(out_dir, ["audit_a"], provenance=False)
    fairness, performance = _result_frames(["audit_a"])

    benchmark.write_report(fairness, performance, out_dir, make_plots=False)

    assert set(_output_bytes(out_dir)) == set(benchmark.RESULT_FILES)


@pytest.mark.parametrize("broken", ["missing", "empty", "no-audit-column", "inconsistent-audits"])
def test_incomplete_or_malformed_outputs_are_not_overwritten(tmp_path, broken):
    out_dir = tmp_path / "results"
    _write_result_set(out_dir, ["audit_a", "audit_b"], provenance=False)
    if broken == "missing":
        (out_dir / "summary.csv").unlink()
    elif broken == "empty":
        (out_dir / "results_performance.csv").write_text("", encoding="utf-8")
    elif broken == "no-audit-column":
        (out_dir / "results_performance.csv").write_text("metric,value\naccuracy,0.7\n", encoding="utf-8")
    else:
        (out_dir / "summary.csv").write_text("audit\naudit_c\n", encoding="utf-8")
    before = _output_bytes(out_dir)
    fairness, performance = _result_frames(["audit_b"])

    with pytest.raises(ValueError):
        benchmark.write_report(fairness, performance, out_dir, make_plots=False)

    assert _output_bytes(out_dir) == before


def test_malformed_or_stale_provenance_is_not_overwritten(tmp_path):
    out_dir = tmp_path / "results"
    _write_result_set(out_dir, ["audit_a", "audit_b"])
    (out_dir / benchmark.PROVENANCE_FILE).write_text("{broken", encoding="utf-8")
    before = _output_bytes(out_dir)
    fairness, performance = _result_frames(["audit_b"])

    with pytest.raises(ValueError, match="provenance is malformed or inconsistent"):
        benchmark.write_report(
            fairness, performance, out_dir, make_plots=False, provenance={"audits": ["audit_b"]}
        )

    assert _output_bytes(out_dir) == before


def test_benchmark_cli_reports_a_refused_subset_run_cleanly(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "results"
    _write_result_set(out_dir, ["audit_a", "audit_b"])
    before = _output_bytes(out_dir)
    fairness, performance = _result_frames(["audit_b"])
    monkeypatch.setattr(benchmark, "run_benchmark", lambda **_: (fairness, performance))
    from faircode.cli import main

    status = main(["benchmark", "--no-plots", "--out", str(out_dir), "audit_b/audit.yaml"])

    assert status == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert _output_bytes(out_dir) == before


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
