"""Python/JS parity of profile() and compare() on the real audit datasets, plus column-name and keyword handling. The parsing, age, threshold and renderer tests live in test_js_parity_profile_*.py (split out of test_js_parity.py, #938; #985)."""

import json
import subprocess

import pandas as pd
import pytest

from faircode import compare, profile
from js_helpers import CSV_PATHS, REPO_ROOT


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
