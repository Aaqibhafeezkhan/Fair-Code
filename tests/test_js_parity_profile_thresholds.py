"""Python/JS parity of threshold overrides, --cross, --reference and out-of-range options. (split out of test_js_parity_profile.py, #985)."""

import json
import subprocess

import pandas as pd
import pytest

from faircode import profile
from js_helpers import CSV_PATHS, REPO_ROOT


def test_python_js_public_params_parity_for_a_defaulted_run():
    """A web-profiler export with no threshold ever touched must still record
    the 7 resolved defaults in provenance.params, matching the CLI/MCP path -
    E.publicParams({}) mirrors provenance.public_params(_resolve_opts(None)) (#490)."""
    from faircode.profiler import _resolve_opts
    from faircode.provenance import public_params

    expected = public_params(_resolve_opts(None))

    script = (
        "require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(globalThis.FairCodeProfiler.publicParams({})));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert json.loads(completed.stdout) == expected
    assert set(expected) == {
        "cross", "imbalance_flag", "intersection_floor", "max_categorical_card",
        "age_reference_year", "keywords", "max_age", "max_dimension_groups", "min_group_size",
        "min_share", "missing_flag", "reference_flag",
    }
    assert "reference" not in expected


def test_python_js_profiler_parity_with_overrides_cross_and_thresholds(tmp_path):
    """Non-default options - --map/--cross/--reference/thresholds - only ever
    had cross-engine parity coverage for their default-off path (issue #376).
    A future change to either _resolve_opts (Python) or resolveOpts (JS), or
    to either engine's override-handling branch, could silently diverge here
    with nothing in this suite to catch it."""
    csv = CSV_PATHS["adult.csv"]
    overrides = {"education": "categorical"}
    reference = {"race": {"White": 0.7, "Black": 0.2, "Other": 0.1}}
    opts = {"cross": ["age", "race"], "min_group_size": 500, "reference": reference}

    python_result = profile(pd.read_csv(csv), overrides, opts)

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"overrides": overrides, "opts": opts}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    # Confirm the options actually took effect on both sides, not just that
    # both silently ignored them the same way.
    assert any(d["name"] == "education" and d["kind"] == "categorical"
               for d in python_result["dimensions"])
    assert python_result["intersections"][0]["dims"] == ["age", "race"]
    assert any("reference" in d for d in python_result["dimensions"])


def test_python_js_reject_out_of_range_min_share_parity(tmp_path):
    """Both engines reject an out-of-range tunable (min_share=1.5) rather
    than silently producing a self-contradictory report (#511)."""
    csv = CSV_PATHS["small.csv"]

    with pytest.raises(ValueError, match="min_share must be between 0 and 1"):
        profile(pd.read_csv(csv), opts={"min_share": 1.5})

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": {"min_share": 1.5}}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert completed.returncode != 0
    assert "min_share must be between 0 and 1" in completed.stderr


def test_python_js_cross_parity_on_unmatched_column(tmp_path):
    """An unmatched `cross` column raises the same error on both engines
    instead of the JS engine silently falling back to the first two detected
    dimensions with no error (#420)."""
    csv = CSV_PATHS["adult.csv"]

    with pytest.raises(ValueError, match="cross column\\(s\\) don't match any profiled dimension: nonexistent_col"):
        profile(pd.read_csv(csv), opts={"cross": ["age", "nonexistent_col"]})

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": {"cross": ["age", "nonexistent_col"]}}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode != 0
    assert "cross column(s) don't match any profiled dimension: nonexistent_col" in completed.stderr


def test_python_js_reference_parity_on_unmatched_column(tmp_path):
    """A reference baseline whose column(s) don't match any profiled
    dimension raises the same error on both engines instead of the JS
    engine silently applying nothing (#419)."""
    csv = CSV_PATHS["adult.csv"]
    reference = {"totally_wrong_col": {"a": 0.5, "b": 0.5}}

    with pytest.raises(ValueError, match="reference file's column\\(s\\) don't match any profiled dimension: totally_wrong_col"):
        profile(pd.read_csv(csv), opts={"reference": reference})

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": {"reference": reference}}), encoding="utf-8")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode != 0
    assert "reference file's column(s) don't match any profiled dimension: totally_wrong_col" in completed.stderr


def test_python_js_parse_reference_mixed_scale_parity(tmp_path):
    """parse_reference / parseReference decide percent-vs-fraction per column,
    not once across the whole table, so a reference file mixing conventions
    between columns parses identically on both engines (#513)."""
    from faircode.profiler import parse_reference

    ref_df = pd.DataFrame({
        "column": ["sex", "sex", "race", "race", "race"],
        "group": ["Female", "Male", "White", "Black", "Other"],
        "share": [0.6, 0.4, 70, 20, 10],
    })
    py_result = parse_reference(ref_df)
    assert py_result == {
        "sex": {"Female": 0.6, "Male": 0.4},
        "race": {"White": 0.7, "Black": 0.2, "Other": 0.1},
    }

    table_json = tmp_path / "table.json"
    table_json.write_text(json.dumps({
        "columns": list(ref_df.columns),
        "rows": ref_df.to_dict(orient="records"),
    }), encoding="utf-8")

    script = (
        "const fs=require('fs');"
        "require(process.argv[1]);"
        "const t=JSON.parse(fs.readFileSync(process.argv[2],'utf-8'));"
        "process.stdout.write(JSON.stringify(globalThis.FairCodeProfiler.parseReference(t)));"
    )
    completed = subprocess.run(
        ["node", "-e", script,
         str(REPO_ROOT / "assets" / "profiler-engine.js"), str(table_json)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert json.loads(completed.stdout) == py_result


def test_threshold_input_recovers_panel_after_invalid_value():
    """profiler-ui.js's threshold-input handler must recover when the engine
    rejects the typed value (e.g. min_share 1.5, see #602): reprofile() fails
    and showError() hides the whole #results panel, including the very input
    the user needs to correct. The handler must revert that opt to its
    last-known-good value and retry once, the same recovery the mapping-select
    handler got in #466. Source-level check (mirrors
    test_compare_card_renderers_special_case_kind_mismatch) - this handler is
    DOM-coupled and has no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")

    marker = "thresholdInputs.forEach(function (input) {\n    input.addEventListener('input'"
    handler = src[src.index(marker):]
    handler = handler[: handler.index("\n  });\n")]

    # it must notice that the re-profile failed ...
    assert "if (!reprofile(false))" in handler
    # ... revert the offending opt to the value it held before ...
    assert "currentOpts[opt] = previous;" in handler
    # ... and retry once so the #results panel (and this input) come back
    assert handler.count("reprofile(false)") >= 2


@pytest.mark.parametrize("opts", [{"max_categorical_card": 0}, {"max_categorical_card": 1},
                                  {"max_dimension_groups": 0}, {"max_dimension_groups": -1}])
def test_python_js_reject_out_of_range_group_limits_with_the_same_message(tmp_path, opts):
    """#974: both engines refuse the same values with the same text."""
    path = tmp_path / "d.csv"
    path.write_text("sex\n" + "M\nF\n" * 10)
    opts_file = tmp_path / "opts.json"
    opts_file.write_text(json.dumps({"overrides": {}, "opts": opts}))
    with pytest.raises(ValueError) as py_err:
        profile(pd.read_csv(path), None, opts)
    done = subprocess.run(["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode != 0 and str(py_err.value) in done.stderr


def test_python_js_parse_reference_skips_the_same_unreadable_shares(tmp_path):
    """#1002: n/a, empty, ?, nan, inf, 1_0 and 0x10 are not shares in either engine; plain decimals,
    exponents and a percent suffix are, and the skipped rows leave the others untouched."""
    from faircode.profiler import parse_reference
    values = ["0.4", "n/a", "", "?", "nan", "inf", "-inf", "Infinity", "1_0", "0x10", "1e-1", " 30% ", ".5"]
    df = pd.DataFrame({"column": ["c"] * len(values), "group": [f"g{i}" for i in range(len(values))], "share": values})
    py = parse_reference(df)
    assert sorted(py["c"]) == ["g0", "g10", "g11", "g12"]
    table = {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    script = ("require(process.argv[1]);process.stdout.write(JSON.stringify("
              "globalThis.FairCodeProfiler.parseReference(JSON.parse(process.argv[2]))));")
    js = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(table)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js == py


def test_python_js_parse_reference_skip_null_and_nan_cells(tmp_path):
    """#1028: JSON null and NaN share cells are skipped in both engines, so a blank cell in a
    reference table never becomes a share."""
    from faircode.profiler import parse_reference
    rows = [{"column": "c", "group": "a", "share": 0.5}, {"column": "c", "group": "b", "share": None},
            {"column": "c", "group": "d", "share": float("nan")}, {"column": "c", "group": "e", "share": 0.5}]
    py = parse_reference(pd.DataFrame(rows))
    table = {"columns": ["column", "group", "share"], "rows": [dict(r, share=None if r["share"] != r["share"] else r["share"]) for r in rows]}
    script = ("require(process.argv[1]);process.stdout.write(JSON.stringify("
              "globalThis.FairCodeProfiler.parseReference(JSON.parse(process.argv[2]))));")
    js = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(table)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert py == js == {"c": {"a": 0.5, "e": 0.5}}
