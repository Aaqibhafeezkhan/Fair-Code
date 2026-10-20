"""scripts/diff_results.py compares results/ against the paper/results-frozen/ snapshot (#1065)."""

import importlib
import json

import pandas as pd
import pytest

dr = importlib.import_module("scripts.diff_results")

FAIR_COLS = ["audit", "strategy", "model", "protected_attribute", "metric", "value"]


def _fair(rows):
    return pd.DataFrame(rows, columns=FAIR_COLS)


def _write(directory, fairness, performance=None):
    directory.mkdir(parents=True, exist_ok=True)
    fairness.to_csv(directory / "results_fairness.csv", index=False)
    (performance if performance is not None else pd.DataFrame(
        [["a", "baseline", "lr", "accuracy", 0.7]], columns=["audit", "strategy", "model", "metric", "value"])
     ).to_csv(directory / "results_performance.csv", index=False)


def test_rows_are_classified_moved_unchanged_and_one_sided():
    ref = _fair([["a", "baseline", "lr", "sex", "dp", 0.10], ["a", "baseline", "lr", "sex", "eo", 0.20],
                 ["a", "baseline", "lr", "sex", "gone", 0.30]])
    cur = _fair([["a", "baseline", "lr", "sex", "dp", 0.15], ["a", "baseline", "lr", "sex", "eo", 0.205],
                 ["a", "baseline", "lr", "sex", "new", 0.40]])
    rows = {r["metric"]: r for r in dr.diff_table(cur, ref, dr.KINDS["fairness"][1], threshold=0.01)}
    assert rows["dp"]["status"] == "moved" and rows["dp"]["delta"] == pytest.approx(0.05)
    assert rows["eo"]["status"] == "unchanged"
    assert rows["new"]["status"] == "only_current" and rows["new"]["current"] == 0.40
    assert rows["gone"]["status"] == "only_reference" and rows["gone"]["reference"] == 0.30


def test_a_value_missing_on_one_side_only_counts_as_moved():
    ref = _fair([["a", "b", "lr", "sex", "dp", None]])
    cur = _fair([["a", "b", "lr", "sex", "dp", 0.1]])
    (row,) = dr.diff_table(cur, ref, dr.KINDS["fairness"][1])
    assert row["status"] == "moved" and row["delta"] is None
    (same,) = dr.diff_table(ref, ref, dr.KINDS["fairness"][1])
    assert same["status"] == "unchanged"


def test_a_delta_exactly_at_the_threshold_is_not_flagged():
    ref = _fair([["a", "b", "lr", "sex", "dp", 0.10]])
    cur = _fair([["a", "b", "lr", "sex", "dp", 0.11]])
    (row,) = dr.diff_table(cur, ref, dr.KINDS["fairness"][1], threshold=0.01)
    assert row["status"] == "unchanged"


def test_main_prints_flagged_rows_and_fails_on_change(tmp_path, capsys):
    _write(tmp_path / "ref", _fair([["a", "baseline", "lr", "sex", "dp", 0.10]]))
    _write(tmp_path / "cur", _fair([["a", "baseline", "lr", "sex", "dp", 0.30]]))
    args = ["--current", str(tmp_path / "cur"), "--reference", str(tmp_path / "ref")]
    assert dr.main(args) == 0
    out = capsys.readouterr().out
    assert "fairness: 1 rows compared, 1 flagged" in out
    assert "a / baseline / lr / sex / dp: 0.1000 -> 0.3000 (+0.2000)" in out
    assert dr.main(args + ["--fail-on-change"]) == 1


def test_main_json_and_all_and_clean_exit(tmp_path, capsys):
    same = _fair([["a", "baseline", "lr", "sex", "dp", 0.10]])
    _write(tmp_path / "ref", same)
    _write(tmp_path / "cur", same)
    args = ["--current", str(tmp_path / "cur"), "--reference", str(tmp_path / "ref")]
    assert dr.main(args + ["--fail-on-change", "--kind", "fairness"]) == 0
    assert "0 flagged" in capsys.readouterr().out
    assert dr.main(args + ["--all", "--kind", "fairness"]) == 0
    assert "a / baseline / lr / sex / dp: 0.1000 -> 0.1000" in capsys.readouterr().out
    assert dr.main(args + ["--json", "--kind", "fairness"]) == 0
    assert json.loads(capsys.readouterr().out)["fairness"][0]["status"] == "unchanged"


def test_missing_directory_is_a_clean_error(tmp_path, capsys):
    assert dr.main(["--current", str(tmp_path / "nope")]) == 2
    assert "not found" in capsys.readouterr().err


def test_the_real_snapshot_and_current_results_compare():
    diff = dr.diff_dirs(dr.CURRENT_DIR, dr.REFERENCE_DIR)
    assert set(diff) == {"fairness", "performance"}
    assert diff["fairness"] and diff["performance"]
