"""The web CSV/HTML exports against the Python writers, including the compare sections. (split out of test_js_parity.py, #938)."""

import json
import subprocess

import pandas as pd
import pytest

from faircode import compare, profile
from js_helpers import REPO_ROOT, SHARED_PRELUDE, _run_ui_exports, _sex_race_csv, result_with_an_empty_subgroup


def test_web_csv_export_matches_python_to_csv(tmp_path):
    """The web "Download CSV" output (#755) must equal faircode's to_csv()
    for the same data, with and without proxy hints (#758, #760)."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints
    from faircode.report import to_csv

    path = _sex_race_csv(tmp_path)
    df = pd.read_csv(path)
    result = dict(profile(df))
    plain = _run_ui_exports(path, False)["csv"]
    assert plain.replace("\r\n", "\n") == to_csv(result).replace("\r\n", "\n")

    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    result["proxy_hints"] = proxy_hints(df, dims, alpha=0.9)
    with_hints = _run_ui_exports(path, True)["csv"].splitlines()
    py = to_csv(result).splitlines()
    i = with_hints.index("proxy_hint_a,proxy_hint_b,p_value,cramers_v,n_tests,low_expected")
    assert with_hints[:i] == py[:i]
    assert with_hints[i + 1].startswith("sex,race,") and py[i + 1].startswith("sex,race,")


def test_web_compare_csv_lists_one_sided_dimensions_like_python(tmp_path):
    """#842: a dimension present in only one dataset gets a dimension,present_in row
    in both writers, in the same position and order."""
    from faircode.report import compare_to_csv

    path_a = tmp_path / "a.csv"
    path_a.write_text("sex,race\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex,region\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'n' if i % 3 else 's'}" for i in range(60)) + "\n")

    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    csvs = src[src.index("var csvRow = E.csvRow"):src.index("async function downloadCompareCsvReport")]
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;" + csvs +
        "function load(p){return E.parseCSV(fs.readFileSync(p,'utf-8'));}"
        "var pa=E.profile(load(process.argv[2]),{},{}),pb=E.profile(load(process.argv[3]),{},{});"
        "process.stdout.write(buildCompareCsvReport(E.compare(pa,pb,'a.csv','b.csv')));"
    )
    web = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path_a), str(path_b)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    py = compare_to_csv(compare(profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv"))
    section = "\n\ndimension,present_in\nrace,a_only\nregion,b_only\n\nflag\n"
    assert section in web.replace("\r\n", "\n")
    assert section in py.replace("\r\n", "\n")


def test_web_compare_csv_lists_rename_suggestions_like_python(tmp_path):
    """#896: possible_renames gets a dimension_a,dimension_b,overlap section after the
    present_in one in both writers (and no section when there are none)."""
    from faircode.report import compare_to_csv

    path_a = tmp_path / "a.csv"
    path_a.write_text("sex,race\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex,ethnicity\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")

    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    csvs = src[src.index("var csvRow = E.csvRow"):src.index("async function downloadCompareCsvReport")]
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;" + csvs +
        "function load(p){return E.parseCSV(fs.readFileSync(p,'utf-8'));}"
        "var pa=E.profile(load(process.argv[2]),{},{}),pb=E.profile(load(process.argv[3]),{},{});"
        "process.stdout.write(buildCompareCsvReport(E.compare(pa,pb,'a.csv','b.csv')));"
    )
    web = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path_a), str(path_b)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    py = compare_to_csv(compare(profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv"))
    section = "\n\ndimension,present_in\nrace,a_only\nethnicity,b_only\n\ndimension_a,dimension_b,overlap,reason\nrace,ethnicity,1,labels\n\nflag\n"
    assert section in web.replace("\r\n", "\n")
    assert section in py.replace("\r\n", "\n")


def test_web_and_python_csv_exports_defuse_formula_labels_identically(tmp_path):
    """#791: a label like =1+1 is written with a leading ' by both engines, and
    both still agree on the whole file (#790: one shared helper)."""
    from faircode.report import to_csv

    rows = ["sex"] + ["=1+1"] * 30 + ["male"] * 30
    path = tmp_path / "inj.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    web = _run_ui_exports(path, False)["csv"]
    py = to_csv(dict(profile(pd.read_csv(path))))
    assert "'=1+1" in web and "'=1+1" in py
    assert web.replace("\r\n", "\n") == py.replace("\r\n", "\n")


def test_web_csv_includes_the_reference_section_like_python(tmp_path):
    """#805: after scoring against a reference baseline, the browser CSV carries
    the same expected/actual/delta/deviation section as faircode's to_csv()."""
    from faircode.report import to_csv

    rows = ["sex"] + ["male"] * 45 + ["female"] * 15
    path = tmp_path / "ref.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    reference = {"sex": {"male": 0.5, "female": 0.5}}
    web = _run_ui_exports(path, False, reference)["csv"]
    py = to_csv(dict(profile(pd.read_csv(path), None, {"reference": reference})))
    assert "reference_label" in web
    assert web.replace("\r\n", "\n") == py.replace("\r\n", "\n")


def test_web_compare_csv_lists_renamed_dimensions_like_python(tmp_path):
    """#920: dimensions paired by --rename get a renamed_dimension_a,renamed_dimension_b section
    before the flags in both writers (and none when nothing was renamed)."""
    from faircode.report import compare_to_csv

    path_a = tmp_path / "a.csv"
    path_a.write_text("sex,race\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex,ethnicity\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    csvs = src[src.index("var csvRow = E.csvRow"):src.index("async function downloadCompareCsvReport")]
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;" + csvs +
        "function load(p){return E.parseCSV(fs.readFileSync(p,'utf-8'));}"
        "var pa=E.profile(load(process.argv[2]),{},{}),pb=E.profile(load(process.argv[3]),{},{});"
        "process.stdout.write(buildCompareCsvReport(E.compare(pa,pb,'a.csv','b.csv',{race:'ethnicity'})));"
    )
    web = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path_a), str(path_b)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    pa, pb = profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b))
    py = compare_to_csv(compare(pa, pb, "a.csv", "b.csv", renames={"race": "ethnicity"}))
    section = "\n\nrenamed_dimension_a,renamed_dimension_b\nrace,ethnicity\n\nflag\n"
    assert section in web.replace("\r\n", "\n") and section in py.replace("\r\n", "\n")
    assert "renamed_dimension_a" not in compare_to_csv(compare(pa, pb, "a.csv", "b.csv"))


def test_web_html_report_says_imbalance_inf_for_an_empty_subgroup_like_python(tmp_path):
    """#975: a zero-row group gives imbalance_ratio None; the web
    report builder prints the same meta text as to_html()."""
    from faircode.report import to_html
    result = result_with_an_empty_subgroup()
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    helpers = src[src.index("var GRADE_COLOR"):src.index("function render(")]
    builders = src[src.index("function buildHtmlReport(r)"):src.index("async function downloadCsvReport")]
    script = ("var DISPLAY_GROUPS=12;var MAX_DISPLAYED_FLAGS=50;" + SHARED_PRELUDE + helpers +
              "require(process.argv[1]);var E=globalThis.FairCodeProfiler;" + builders +
              "process.stdout.write(buildHtmlReport(JSON.parse(process.argv[2])));")
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(result)],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    assert "imbalance inf (empty subgroup)" in done.stdout
    assert "imbalance inf (empty subgroup)" in to_html(result)
