"""Python/JS parity of profile(), compare(), parsing, thresholds and the dashboard card renderers. (split out of test_js_parity.py, #938)."""

import json
import re
import subprocess

import pandas as pd
import pytest

from faircode import compare, profile
from faircode.loaders import read_table
from js_helpers import CSV_PATHS, FIXTURES, REPO_ROOT, _extract, requires_openpyxl


def test_sheetjs_cdn_url_matches():
    engine = (REPO_ROOT / "assets" / "profiler-engine.js").read_text(encoding="utf-8")
    cli = (REPO_ROOT / "scripts" / "engine-js.js").read_text(encoding="utf-8")

    engine_url = _extract(r'script\.src\s*=\s*"([^"]+)"', engine)
    cli_url = _extract(r'XLSX_CDN_URL\s*=\s*"([^"]+)"', cli)

    assert engine_url == cli_url


def test_compare_card_renderers_special_case_kind_mismatch():
    """driftCard() and buildCompareHtmlReport()'s per-dimension section must
    both read cd.kind_mismatch, so a skipped comparison isn't drawn as a
    "none drift" badge next to a real score change (#519). Source-level check
    (mirrors test_sheetjs_cdn_url_matches) - these renderers are DOM-coupled
    and have no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")

    drift_card = src[src.index("function driftCard("):]
    drift_card = drift_card[: drift_card.index("\n  }\n")]
    assert "kind_mismatch" in drift_card
    assert "comparison skipped" in drift_card

    report = src[src.index("function buildCompareHtmlReport("):]
    assert "if (cd.kind_mismatch)" in report
    # the skipped badge is styled in both the live css and the report's own <style>
    assert ".drift-badge.skipped" in src
    assert ".drift-badge.skipped" in (REPO_ROOT / "assets" / "profiler.css").read_text(encoding="utf-8")


@pytest.mark.parametrize("csv_name", list(CSV_PATHS))
def test_python_js_profiler_parity(csv_name):
    """The Python and JavaScript profilers should produce equivalent structured JSON."""

    csv = CSV_PATHS[csv_name]

    python_result = profile(pd.read_csv(csv))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    javascript_result = json.loads(completed.stdout)

    # Flags are human-readable messages. They duplicate information already
    # present in the structured output and may differ because Python and
    # JavaScript format floating-point values differently (e.g. 6.25 -> 6.2
    # vs 6.3). Compare the structured data instead.
    python_result = dict(python_result)
    javascript_result = dict(javascript_result)

    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_profiler_parity_preserves_mid_field_quotes(tmp_path):
    """Quotes after field content are literal in pandas and the browser parser."""
    csv = tmp_path / "space_before_quote.csv"
    csv.write_text(
        'sex, race, age\n'
        'Male, "White", 25\n'
        'Female, "Black", 30\n'
        'Male, "White", 45\n'
        'Female, "Asian", 22\n',
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
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


def test_python_js_profiler_parity_sniffs_quoted_newlines(tmp_path):
    """Embedded newlines do not split logical rows during delimiter sniffing."""
    dataset = tmp_path / "sniff_quoted_newline.dat"
    dataset.write_text(
        'sex;race;age;notes\n'
        'M;White;25;"single line"\n'
        'F;Black;30;"multi\nline note"\n'
        'M;White;45;"ok"\n'
        'F;Asian;22;"fine"\n'
        'M;White;50;"good"\n',
        encoding="utf-8",
    )

    python_result = profile(read_table(str(dataset)))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(dataset)],
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


def test_python_js_profiler_parity_detects_dates_appended_after_numeric_ages(tmp_path):
    """A merged-in birthdate block cannot become a fabricated elderly group."""
    csv = tmp_path / "mixed-age-and-birthdate.csv"
    ages = [str(age) for age in range(20, 80)]
    dates = ["1985-03-21", "1990-07-14", "2001-11-02"] * 20
    csv.write_text("age\n" + "\n".join(ages + dates) + "\n", encoding="utf-8")

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
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
    assert all(dim["name"] != "age" for dim in python_result["dimensions"])


def test_python_js_profiler_parity_handles_scientific_and_nonfinite_ages(tmp_path):
    """Scientific notation follows pandas' numeric inference; inf/nan stay missing."""
    csv = tmp_path / "scientific-and-nonfinite-ages.csv"
    csv.write_text(
        "age,sex\n"
        "1e2,F\n"
        "25,M\n"
        "40,F\n"
        "60,M\n"
        "inf,F\n"
        "nan,M\n",
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
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
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    labels = {group["label"] for group in age["groups"]}
    assert "75+" in labels
    assert "inf" not in labels
    assert "nan" not in labels
    assert age["missing_pct"] == 0.3333


def test_python_js_profiler_handles_overflowing_scientific_age_as_missing(tmp_path):
    """A value such as 1e400 parses to Infinity in JS and must not become an age group."""
    csv = tmp_path / "overflow-age.csv"
    csv.write_text(
        "age\n"
        "25\n"
        "40\n"
        "60\n"
        "1e400\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)
    age = next(d for d in javascript_result["dimensions"] if d["name"] == "age")
    labels = {group["label"] for group in age["groups"]}

    assert "inf" not in labels
    assert "1e400" not in labels
    assert age["missing_pct"] == 0.25


def test_python_js_profiler_parity_rejects_negative_age_sentinels(tmp_path):
    """Signed sentinel ages stay missing in both profiler engines."""
    csv = tmp_path / "negative-age-sentinels.csv"
    csv.write_text(
        "age,sex\n25,F\n30,M\nage -5,F\n-1,M\n45,F\n",
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
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
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    assert "75+" not in {group["label"] for group in age["groups"]}
    assert age["missing_pct"] == 0.4


def test_python_js_na_token_parity_on_literal_na_and_none(tmp_path):
    """NA_TOKENS / isMissing() must match pandas' default STR_NA_VALUES
    exactly and case-sensitively: literal "None" is missing, bare lowercase
    "na" is a real category. The JS engine used to have both backwards and
    lower-cased the cell before comparing (#491)."""
    csv = tmp_path / "na_test.csv"
    csv.write_text(
        "status,x\n"
        "active,1\ninactive,2\nna,3\nna,4\nNone,5\nNone,6\n"
        "active,7\ninactive,8\nactive,9\ninactive,10\n",
        encoding="utf-8",
    )

    python_result = profile(read_table(str(csv)))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)
    assert javascript_result == python_result

    status = next(d for d in python_result["dimensions"] if d["name"] == "status")
    labels = {g["label"] for g in status["groups"]}
    assert "na" in labels          # bare lowercase "na" is NOT a pandas NA token
    assert "None" not in labels    # "None" IS a pandas NA token
    assert status["missing_pct"] == 0.2


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


def test_python_js_parity_for_implausible_ages(tmp_path):
    """#840: ages above max_age are flagged and left out of the bands identically in
    both engines, including in the intersection and with a custom --max-age."""
    path = tmp_path / "ages.csv"
    path.write_text("sex,age\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[25, 33, 47, 62, 80, 150, 200, 1985][i % 8]}" for i in range(64)) + "\n")
    for opts in ({}, {"max_age": 70}):
        opts_file = tmp_path / "opts.json"
        opts_file.write_text(json.dumps({"overrides": {}, "opts": opts}))
        python_result = dict(profile(pd.read_csv(path), None, opts))
        completed = subprocess.run(
            ["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
            capture_output=True, text=True, encoding="utf-8", check=True)
        javascript_result = json.loads(completed.stdout)
        py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
        assert javascript_result == python_result
        age = next(d for d in python_result["dimensions"] if d["name"] == "age")
        assert age["implausible_values"] == (24 if not opts else 32)
        implausible = [f for f in py_flags if "implausible age" in f]
        assert implausible and implausible == [f for f in js_flags if "implausible age" in f]


def test_python_js_parity_for_negative_sentinel_ages(tmp_path):
    """#863: negative sentinel ages are flagged identically in both engines."""
    path = tmp_path / "neg_ages.csv"
    path.write_text("sex,age\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[25, 33, -1, 62, -9, 150, 40, 200][i % 8]}" for i in range(64)) + "\n")
    opts_file = tmp_path / "opts.json"
    opts_file.write_text(json.dumps({"overrides": {}, "opts": {}}))
    python_result = dict(profile(pd.read_csv(path), None, {}))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    javascript_result = json.loads(completed.stdout)
    py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
    assert javascript_result == python_result
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    assert age["implausible_values"] == 32
    assert age["has_negative_ages"] is True
    sentinel_flags = [f for f in py_flags if "sentinel/implausible age" in f]
    assert sentinel_flags and sentinel_flags == [f for f in js_flags if "sentinel/implausible age" in f]


def test_python_js_parity_for_non_english_column_names(tmp_path):
    """#847: accent-stripped, multilingual keyword detection agrees between engines,
    including the no-kind-detected --map hint."""
    names = ["sexo", "Género", "Geschlecht", "raza", "Rasse", "Raça", "edad", "Alter", "Âge",
             "Fecha de nacimiento", "Bundesland", "país", "Código postal", "estado",
             "generosity", "alternative", "landing", "Straße", "etat_civil",
             "estado_civil", "estado civil", "EstadoCivil", "marital_status",
             "stato_civile", "état civil"]
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var out={};JSON.parse(process.argv[2]).forEach(function(n){"
        "out[n]=E.profile({columns:[n],rows:[{[n]:'a'},{[n]:'b'}]}).dimensions[0].kind});"
        "process.stdout.write(JSON.stringify(out));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(names)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    from faircode.detect import classify_name
    js = json.loads(completed.stdout)
    for name in names:
        py = classify_name(name) or "categorical"
        assert js[name] == py, name

    path = tmp_path / "plain.csv"
    path.write_text("colA,colB\n" + "\n".join(f"{'x' if i % 2 else 'y'},{'p' if i % 3 else 'q'}" for i in range(40)) + "\n")
    python_result = dict(profile(pd.read_csv(path)))
    completed = subprocess.run(["node", "scripts/engine-js.js", "profile", str(path)],
                               capture_output=True, text=True, encoding="utf-8", check=True)
    javascript_result = json.loads(completed.stdout)
    assert javascript_result["flags"][-1] == python_result["flags"][-1]
    assert "--map COL=KIND" in python_result["flags"][-1]
    python_result.pop("flags"); javascript_result.pop("flags")
    assert javascript_result == python_result


def test_python_js_parity_for_birth_year_conversion(tmp_path):
    """#862: age_reference_year turns birth years into ages identically in both engines,
    including the intersection, and a future year stays implausible."""
    path = tmp_path / "yob.csv"
    path.write_text("sex,yob\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[1985, 1990, 1972, 2001, 1950, 2010, 40, 2090][i % 8]}" for i in range(64)) + "\n")
    opts_file = tmp_path / "opts.json"
    opts_file.write_text(json.dumps({"overrides": {}, "opts": {"age_reference_year": 2026}}))
    python_result = dict(profile(pd.read_csv(path), None, {"age_reference_year": 2026}))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    javascript_result = json.loads(completed.stdout)
    py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
    assert javascript_result == python_result
    yob = next(d for d in python_result["dimensions"] if d["name"] == "yob")
    assert yob["implausible_values"] == 8 and yob["n_groups"] >= 4  # only 2090 is implausible
    assert [f for f in py_flags if "implausible" in f] == [f for f in js_flags if "implausible" in f]


def test_python_js_parity_for_extra_keywords_and_italian_dutch_terms(tmp_path):
    """#856: the user vocabulary (normalised, exact_only, validation errors) and the new
    Italian/Dutch built-ins behave identically in both engines."""
    from faircode.detect import classify_name, normalize_keywords

    extra = {"sex": ["GNDR", "Jenis"], "race": ["EtnGrp"], "age": ["Umur"], "exact_only": ["umur"]}
    names = ["sesso", "Età", "razza", "leeftijd", "geslacht", "geboortedatum", "regione", "paese",
             "gndr", "jenis", "umur", "umuraa", "jenis_x", "etngrp", "other"]
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;var out={};"
        "var names=JSON.parse(process.argv[2]),extra=JSON.parse(process.argv[3]);"
        "names.forEach(function(n){"
        "out[n]=[E.profile({columns:[n],rows:[{[n]:'a'},{[n]:'b'}]},{},{}).dimensions[0].kind,"
        "E.profile({columns:[n],rows:[{[n]:'a'},{[n]:'b'}]},{},{keywords:extra}).dimensions[0].kind]});"
        "out.__norm=E.normalizeKeywords(extra);"
        "var bad=[{x:[1]},{sex:'a'},{sex:['a b']},[1],{sex:['']}];out.__errors=bad.map(function(b){"
        "try{E.normalizeKeywords(b);return null}catch(e){return e.message}});"
        "process.stdout.write(JSON.stringify(out));"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
         json.dumps(names), json.dumps(extra)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(done.stdout)
    for n in names:
        assert js[n] == [classify_name(n) or "categorical", classify_name(n, extra) or "categorical"], n
    assert js["__norm"] == normalize_keywords(extra) == {
        "sex": ["gndr", "jenis"], "race": ["etngrp"], "age": ["umur"], "exact_only": ["umur"]}
    py_errors = []
    for b in ({"x": [1]}, {"sex": "a"}, {"sex": ["a b"]}, [1], {"sex": [""]}):
        try:
            normalize_keywords(b)
            py_errors.append(None)
        except ValueError as exc:
            py_errors.append(str(exc))
    assert js["__errors"] == py_errors and all(py_errors)
    assert js["umuraa"] == ["categorical", "categorical"]   # exact_only: no prefix match


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


def test_python_js_intersection_parity_keeps_non_numeric_age_sentinels(tmp_path):
    """labelize() gives a non-numeric age sentinel its own crosstab label on
    both engines, instead of mapping it to null and dropping the row (#524)."""
    csv = tmp_path / "age_sentinels.csv"
    ages = ["25", "30", "45", "unknown", "unknown", "unknown",
            "prefer not to say", "22", "33", "41"] * 3
    sexes = ["M", "F"] * 15
    csv.write_text(
        "age,sex\n" + "\n".join(a + "," + s for a, s in zip(ages, sexes)) + "\n",
        encoding="utf-8",
    )

    opts = {"cross": ["age", "sex"]}
    python_result = profile(pd.read_csv(csv, dtype={"age": str}), opts=opts)

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": opts}), encoding="utf-8")
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)
    assert javascript_result == python_result

    a_labels = {c["a"] for c in python_result["intersections"][0]["cells"]}
    assert "prefer not to say" in a_labels


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


def test_python_js_json_parity_inconsistent_keys():
    """Records-orient JSON where later records add columns the first one
    doesn't have (#144). The JS parseJSON() used to derive columns from only
    the first record, silently dropping any column that first appeared later
    - pandas' read_json unions keys across every record instead."""

    json_path = FIXTURES / "inconsistent_keys.json"

    python_result = profile(pd.read_json(json_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-json", str(json_path)],
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


def test_python_js_json_parity_columns_orientation():
    """Columns-orient JSON ({"col": {"0": v, ...}}, pandas' read_json default
    for a plain object) - #155 documented and tested this for the CLI, but
    the JS engine's parseJSON() only handled records/split and threw on it.
    Now handled the same way as the records branch (union of index keys)."""

    json_path = FIXTURES / "columns_orient.json"

    python_result = profile(pd.read_json(json_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-json", str(json_path)],
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


@requires_openpyxl
def test_python_js_xlsx_parity():
    """.xlsx support (#158) - the JS engine's parseXLSX() (via SheetJS,
    fetched from the same pinned CDN profiler.html loads) should agree with
    pandas.read_excel() on the same workbook. Skips if the CDN is
    unreachable rather than failing the suite - see scripts/engine-js.js.
    """
    xlsx_path = FIXTURES / "adult_sample.xlsx"

    python_result = profile(pd.read_excel(xlsx_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-xlsx", str(xlsx_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode == 3:
        pytest.skip("SheetJS CDN unreachable: " + completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr

    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_compare_parity_age_banding_mismatch(tmp_path):
    """The age-banding-mismatch guard (#318) agrees between engines too -
    `kind` alone can't detect it (it's set from the column name and is
    identical on both sides), so this exercises isAgeBandLabel()'s port of
    _is_age_band_label() directly, not just the already-covered common path.
    """
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    path_a.write_text("DOB\n" + "\n".join(["15/05/1980"] * 50 + ["20/06/1985"] * 50))
    path_b.write_text("DOB\n" + "\n".join(["18"] * 50 + ["35"] * 50))

    python_result = compare(
        profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv"
    )

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_b)],
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
    dim = python_result["dimensions"][0]
    assert dim["kind_mismatch"] is True
    assert dim["kind_a"] == dim["kind_b"] == "age"


@pytest.mark.parametrize("csv_name", list(CSV_PATHS))
def test_python_js_compare_parity(csv_name):
    """faircode.compare() and the JS engine's compare() should agree too (#111).

    Compares each fixture against itself - not meant to exercise every drift
    level, just to confirm the two independent compare()/compare_to_html()
    implementations (faircode/report.py and assets/profiler-compare.js) are
    working off identically-shaped, identically-valued structured data.
    """

    csv = CSV_PATHS[csv_name]
    df = pd.read_csv(csv)

    python_result = compare(profile(df), profile(df), "a.csv", "b.csv")

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(csv), str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    # As above: flags are human-readable and float-formatted differently
    # between Python and JS. Compare the structured data instead. Names also
    # differ (JS uses the file's basename via path.basename, Python uses
    # whatever the caller passed) - normalize both before comparing.
    python_result = dict(python_result)
    javascript_result = dict(javascript_result)

    for result in (python_result, javascript_result):
        result.pop("flags", None)
        for side in ("a", "b"):
            result[side] = dict(result[side])
            result[side].pop("name", None)

    assert javascript_result == python_result


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


def test_dim_card_renders_an_expand_control_past_display_groups():
    """profiler-ui.js's dimCard() must render extra groups past DISPLAY_GROUPS
    up front (hidden) with a toggle button, rather than the old static "...and
    N more groups" text with no way to actually see them (#740). Source-level
    check (mirrors test_threshold_input_recovers_panel_after_invalid_value) -
    this is DOM-coupled and has no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")

    marker = "  function dimCard(d) {"
    fn = src[src.index(marker):]
    fn = fn[: fn.index("\n  function renderIntersections")]

    assert "dim-extra-groups" in fn
    assert 'aria-expanded="false"' in fn
    assert "dim-more-btn" in fn
    # the click handler must flip the hidden attribute and the aria state together
    assert "extra.hidden = expanded" in fn
    assert "btn.setAttribute('aria-expanded'" in fn
    # the old dead-end text is gone
    assert "… and " not in fn


def test_drift_card_renders_an_expand_control_past_display_groups():
    """assets/profiler-compare.js's driftCard() gets the same #740 treatment,
    with a resultsEl-level delegated click handler (driftCard rebuilds via an
    innerHTML string, not a DOM node dimCard() can attach a listener to
    directly)."""
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")

    card_fn_marker = "  function driftCard(cd) {"
    card_fn = src[src.index(card_fn_marker):]
    card_fn = card_fn[: card_fn.index("\n\n  // ── Report export")]
    assert "dim-extra-groups" in card_fn
    assert "dim-more-btn" in card_fn
    assert "… and " not in card_fn

    delegated_marker = "resultsEl.addEventListener('click'"
    assert delegated_marker in src
    handler = src[src.index(delegated_marker):]
    handler = handler[: handler.index("\n  });")]
    assert "closest" in handler
    assert "extra.hidden = expanded" in handler


def test_python_js_sample_dataset_is_byte_identical():
    """faircode profile --sample (faircode/sample_data.py) and the web
    profiler's "Try it with a sample dataset" button
    (assets/profiler-ui.js's buildSampleCSV()) must produce the exact same
    CSV text, so a first-time user sees an identical demo either way (#741)."""
    from faircode.sample_data import build_sample_csv

    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    match = re.search(r"function buildSampleCSV\(\) \{[\s\S]*?\n  \}", src)
    assert match, "could not find buildSampleCSV() in profiler-ui.js"

    script = match.group(0) + ";process.stdout.write(buildSampleCSV());"
    completed = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert completed.stdout == build_sample_csv()


@pytest.mark.parametrize("name,text", [
    ("duplicate_headers", "sex,sex,race\nM,F,A\nM,F,B\nM,F,A\nF,M,B\n"),
    ("duplicate_headers_colliding_suffix", "a,a,a,a.1\n1,2,3,4\n"),
    ("quoted_empty_single_column_row", 'region\r\n""\r\nN\r\nS\r\n'),
    ("quoted_empty_last_line", 'region\r\nN\r\n""'),
    ("non_ascii_digit_ages", "age\n٣٠\n25\n40\n٣٠\n"),
])
def test_python_js_parity_on_parser_edge_cases(tmp_path, name, text):
    """#834 (pandas-style header de-duplication), #835 (a quoted-empty row is a
    missing value, not a blank line) and #836 (only ASCII digits are ages):
    the CSV parsers and the profile of the result agree on each."""
    path = tmp_path / f"{name}.csv"
    path.write_text(text, encoding="utf-8", newline="")
    df = pd.read_csv(path)
    script = (
        "require(process.argv[1]);var fs=require('fs');"
        "var t=globalThis.FairCodeProfiler.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));"
        "process.stdout.write(JSON.stringify({columns:t.columns,n:t.rows.length}));"
    )
    parsed = json.loads(subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert parsed["columns"] == list(df.columns)
    assert parsed["n"] == len(df)

    js = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    py = dict(profile(df))
    js.pop("flags", None)
    py.pop("flags", None)
    assert js == py
