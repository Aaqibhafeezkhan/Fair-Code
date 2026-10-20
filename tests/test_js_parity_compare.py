"""Rename suggestions and --rename: parity of compare() between the engines and the web wiring. (split out of test_js_parity.py, #938)."""

import json
import subprocess

import pandas as pd
import pytest

from faircode import compare, profile
from js_helpers import REPO_ROOT, SHARED_PRELUDE


def test_python_js_compare_parity_for_quality_flags_and_renames(tmp_path):
    """#868/#866: implausible-age carry-over, the no-kind-detected note and rename
    suggestions are produced identically by both engines."""
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    path_a.write_text("sex,age,race\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[25, 30, 41, 55][i % 4]},{['White', 'Black', 'Asian'][i % 3]}" for i in range(48)) + "\n")
    path_b.write_text("sex,age,ethnicity\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[25, 200, 41, 150][i % 4]},{['White', 'Black', 'Asian'][i % 3]}" for i in range(48)) + "\n")
    python_result = compare(profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv")
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_b)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    javascript_result = json.loads(completed.stdout)
    py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
    assert javascript_result == dict(python_result)
    for needle in ("implausible age values differ", "look like the same dimension"):
        assert [f for f in py_flags if needle in f] == [f for f in js_flags if needle in f] != []
    assert python_result["possible_renames"][0]["b"] == "ethnicity"

    # #887: recoded labels (M/F vs Male/Female, White/Black vs white/black) still pair up.
    path_c = tmp_path / "c.csv"
    path_d = tmp_path / "d.csv"
    path_c.write_text("sex,race\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{['White', 'Black'][i % 3 % 2]}" for i in range(48)) + "\n")
    path_d.write_text("gender,ethnicity\n" + "\n".join(
        f"{'Male' if i % 2 else 'Female'},{['white', 'black'][i % 3 % 2]}" for i in range(48)) + "\n")
    py_cd = compare(profile(pd.read_csv(path_c)), profile(pd.read_csv(path_d)), "c.csv", "d.csv")
    js_cd = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_c), str(path_d)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js_cd["possible_renames"] == py_cd["possible_renames"]
    assert [(r["a"], r["b"]) for r in py_cd["possible_renames"]] == [("sex", "gender"), ("race", "ethnicity")]

    # #902: recoded to codes AND renamed - paired on the shared name word and the group count.
    path_e = tmp_path / "e.csv"
    path_f = tmp_path / "f.csv"
    path_e.write_text("sex,race\n" + "\n".join(f"{'M' if i % 2 else 'F'},{['White', 'Black'][i % 3 % 2]}" for i in range(48)) + "\n")
    path_f.write_text("sex_code,region_code\n" + "\n".join(f"{1 + i % 2},{['n', 's'][i % 3 % 2]}" for i in range(48)) + "\n")
    py_ef = compare(profile(pd.read_csv(path_e)), profile(pd.read_csv(path_f)), "e.csv", "f.csv")
    js_ef = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_e), str(path_f)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js_ef["possible_renames"] == py_ef["possible_renames"]
    assert py_ef["possible_renames"] == [{"a": "sex", "b": "sex_code", "overlap": 0.0, "reason": "names"}]
    assert [f for f in py_ef["flags"] if "similar names" in f] == [f for f in js_ef["flags"] if "similar names" in f] != []

    # #917: tokenisation keeps unicode words whole, so accented and CJK names still
    # pair (or stay quiet) identically in both engines.
    path_g = tmp_path / "g.csv"
    path_h = tmp_path / "h.csv"
    path_g.write_text("género\n" + "\n".join(f"{['Blanco', 'Negro'][i % 2]}" for i in range(48)) + "\n")
    path_h.write_text("genero\n" + "\n".join(f"{['Casa', 'Luna'][i % 2]}" for i in range(48)) + "\n")
    py_gh = compare(profile(pd.read_csv(path_g)), profile(pd.read_csv(path_h)), "g.csv", "h.csv")
    js_gh = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_g), str(path_h)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js_gh["possible_renames"] == py_gh["possible_renames"]
    assert py_gh["possible_renames"] == [{"a": "género", "b": "genero", "overlap": 0.0, "reason": "names"}]
    path_i = tmp_path / "i.csv"
    path_j = tmp_path / "j.csv"
    path_i.write_text("性别\n" + "\n".join(f"{'M' if i % 2 else 'F'}" for i in range(48)) + "\n")
    path_j.write_text("性别代码\n" + "\n".join(f"{['Male', 'Female'][i % 2]}" for i in range(48)) + "\n")
    py_ij = compare(profile(pd.read_csv(path_i)), profile(pd.read_csv(path_j)), "i.csv", "j.csv")
    js_ij = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_i), str(path_j)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js_ij == dict(py_ij)


def test_python_js_compare_rename_option_parity(tmp_path):
    """#913: --rename compares a renamed dimension as one in both engines, with the same flag,
    renamed_dimensions and errors."""
    path_a, path_b = tmp_path / "a.csv", tmp_path / "b.csv"
    path_a.write_text("sex,race\n" + "\n".join(f"{'M' if i % 2 else 'F'},{['White', 'Black', 'Asian'][i % 3]}" for i in range(48)) + "\n")
    path_b.write_text("sex,ethnicity\n" + "\n".join(f"{'M' if i % 2 else 'F'},{['White', 'Black', 'White'][i % 3]}" for i in range(48)) + "\n")
    py = compare(profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv", renames={"race": "ethnicity"})
    js = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_b), json.dumps({"race": "ethnicity"})],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js == json.loads(json.dumps(py))
    assert py["renamed_dimensions"] == [{"a": "race", "b": "ethnicity"}] and "possible_renames" not in py
    bad = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_b), json.dumps({"race": "nope"})],
        capture_output=True, text=True, encoding="utf-8")
    assert "--rename: 'nope' is not a dimension of the second dataset" in bad.stderr
    with pytest.raises(ValueError):
        compare(profile(pd.read_csv(path_b)), profile(pd.read_csv(path_b)), "b.csv", "b.csv", renames={"sex": "sex"})
    js_identity = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_b), str(path_b), json.dumps({"sex": "sex"})],
        capture_output=True, text=True, encoding="utf-8")
    assert js_identity.returncode != 0 and "renames a dimension to itself" in js_identity.stderr
    # #968: two A names to one B dimension give the same message in both engines.
    path_c = tmp_path / "c.csv"
    path_c.write_text("gender,ethnicity\n" + "\n".join(f"{'M' if i % 2 else 'F'},{['White', 'Black', 'White'][i % 3]}" for i in range(48)) + "\n")
    twice = {"race": "ethnicity", "sex": "ethnicity"}
    with pytest.raises(ValueError, match="'ethnicity' is renamed more than once"):
        compare(profile(pd.read_csv(path_a)), profile(pd.read_csv(path_c)), "a.csv", "c.csv", renames=twice)
    js_twice = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_c), json.dumps(twice)],
        capture_output=True, text=True, encoding="utf-8")
    assert js_twice.returncode != 0 and "--rename: 'ethnicity' is renamed more than once" in js_twice.stderr


def test_compare_view_offers_compare_as_one_buttons_for_rename_suggestions():
    """#913: each possible rename gets a button that sets currentRenames and re-compares; a
    renamed pair gets an undo button; the pairing is recorded in the export's params."""
    js = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    assert "E.compare(profA, profB, slot.A.name, slot.B.name, currentRenames)" in js
    assert "currentRenames[btn.dataset.renameA] = btn.dataset.renameB;" in js
    assert "delete currentRenames[btn.dataset.renameA];" in js
    assert "provenance.params.renames = Object.assign({}, currentRenames)" in js
    assert js.index("currentRenames = {};") < js.index("resetThresholdInputs();", js.index("currentRenames = {};"))


def test_name_based_rename_tolerates_a_slightly_different_group_count_in_both_engines(tmp_path):
    """#918: sex (M/F/U) vs sex_code (1/2/9/x) is suggested with group_counts and a wording that
    says so; a gap too large for the column (2 vs 4 groups) still is not."""
    def run(name_a, rows_a, name_b, rows_b):
        pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
        pa.write_text(f"{name_a}\n" + "\n".join(rows_a) + "\n")
        pb.write_text(f"{name_b}\n" + "\n".join(rows_b) + "\n")
        py = compare(profile(pd.read_csv(pa)), profile(pd.read_csv(pb)), "a.csv", "b.csv")
        js = json.loads(subprocess.run(["node", "scripts/engine-js.js", "compare", str(pa), str(pb)],
                                       capture_output=True, text=True, encoding="utf-8", check=True).stdout)
        return py, js

    py, js = run("sex", ["MFU"[i % 3] for i in range(60)], "sex_code", ["129x"[i % 4] for i in range(60)])
    assert py["possible_renames"] == js["possible_renames"] == [
        {"a": "sex", "b": "sex_code", "overlap": 0.0, "reason": "names", "group_counts": [3, 4]}]
    flag = "similar names and a similar number of groups (3 vs 4)"
    assert [f for f in py["flags"] if flag in f] == [f for f in js["flags"] if flag in f] != []

    py, js = run("sex", ["MF"[i % 2] for i in range(60)], "sex_code", ["129x"[i % 4] for i in range(60)])
    assert "possible_renames" not in py and "possible_renames" not in js


def _name_rename_results(tmp_path, cols_a, cols_b):
    """compare() of two frames of `{column: group_count}` columns (all typed sex) in both engines."""
    def frame(cols, prefix):   # disjoint labels per side, so only the name signal can pair them
        return pd.DataFrame({name: [f"{prefix}{i % k}" for i in range(60)] for name, k in cols.items()})
    overrides = {name: "sex" for name in list(cols_a) + list(cols_b)}
    opts = {"max_categorical_card": 100}
    df_a, df_b = frame(cols_a, "a"), frame(cols_b, "b")
    path_a, path_b, cfg = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "opts.json"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    cfg.write_text(json.dumps({"overrides": overrides, "opts": opts}))
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "var o=JSON.parse(fs.readFileSync(process.argv[4],'utf-8'));"
        "var pa=E.profile(E.parseCSV(fs.readFileSync(process.argv[2],'utf-8')),o.overrides,o.opts);"
        "var pb=E.profile(E.parseCSV(fs.readFileSync(process.argv[3],'utf-8')),o.overrides,o.opts);"
        "process.stdout.write(JSON.stringify(E.compare(pa,pb,'a.csv','b.csv')));"
    )
    js = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
                                    str(path_a), str(path_b), str(cfg)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    py = compare(profile(pd.read_csv(path_a), overrides, opts), profile(pd.read_csv(path_b), overrides, opts), "a.csv", "b.csv")
    assert js.get("possible_renames") == py.get("possible_renames")
    return py.get("possible_renames", [])


@pytest.mark.parametrize("count_a,count_b,suggested", [
    (1, 2, True), (2, 3, True), (3, 4, True), (2, 4, False),        # small columns: a gap of one
    (7, 9, True), (7, 10, False), (8, 10, True), (8, 11, False),     # larger: up to a quarter of the bigger
    (12, 16, True), (12, 17, False), (20, 26, True), (20, 27, False),
])
def test_name_rename_group_count_tolerance_boundaries_match_in_both_engines(tmp_path, count_a, count_b, suggested):
    """#949: the tolerance is max(1, larger // 4); each side of every boundary, in Python and JS."""
    renames = _name_rename_results(tmp_path, {"sex": count_a}, {"sex_code": count_b})
    assert bool(renames) is suggested, renames


def test_name_rename_prefers_most_shared_words_then_the_closest_group_count(tmp_path):
    """#949: with two B candidates the one sharing more words wins; with equal words, the closer count."""
    renames = _name_rename_results(tmp_path, {"patient_sex": 4}, {"sex_code": 4, "patient_sex_code": 5})
    assert [(r["a"], r["b"]) for r in renames] == [("patient_sex", "patient_sex_code")]      # 2 shared words beat 1
    renames = _name_rename_results(tmp_path, {"patient_sex": 4}, {"sex_x": 5, "sex_y": 4})
    assert [(r["a"], r["b"]) for r in renames] == [("patient_sex", "sex_y")]                  # equal words: closer count


def test_compare_html_reports_list_renamed_and_possibly_renamed_dimensions_alike(tmp_path):
    """#953: both HTML writers show a 'Compared as one' list after --rename and a 'Possible renames'
    list for suggestions, with identical text (the evidence wording is shared)."""
    import re
    from faircode.report import compare_to_html
    path_a, path_b = tmp_path / "a.csv", tmp_path / "b.csv"
    path_a.write_text("sex,race\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")
    path_b.write_text("sex,ethnicity\n" + "\n".join(f"{'m' if i % 2 else 'f'},{'x' if i % 3 else 'y'}" for i in range(60)) + "\n")
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    helpers = (src[src.index("function stripNegZero"):src.index("function wireSlot")]
               + src[src.index("function kindMismatchReason"):src.index("function driftCard")])
    builders = src[src.index("function proxyHintItems"):src.index("function compareReportBaseName")]
    script = (
        "var DISPLAY_GROUPS=12;var MAX_DISPLAYED_FLAGS=50;" + SHARED_PRELUDE + helpers +
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;" + builders +
        "function load(p){return E.parseCSV(fs.readFileSync(p,'utf-8'));}"
        "var pa=E.profile(load(process.argv[2]),{},{}),pb=E.profile(load(process.argv[3]),{},{});"
        "process.stdout.write(JSON.stringify({suggested:buildCompareHtmlReport(E.compare(pa,pb,'a.csv','b.csv')),"
        "paired:buildCompareHtmlReport(E.compare(pa,pb,'a.csv','b.csv',{race:'ethnicity'}))}));"
    )
    js = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path_a), str(path_b)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    pa, pb = profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b))
    py = {"suggested": compare_to_html(compare(pa, pb, "a.csv", "b.csv")),
          "paired": compare_to_html(compare(pa, pb, "a.csv", "b.csv", renames={"race": "ethnicity"}))}

    def section(html, title):
        match = re.search(r'<section class="flags"><h2>%s</h2><ul>.*?</ul></section>' % title, html)
        return match.group(0) if match else None

    for kind, title in (("suggested", "Possible renames"), ("paired", "Compared as one")):
        assert section(py[kind], title) is not None and section(py[kind], title) == section(js[kind], title)
    assert "--rename race=ethnicity" in section(py["suggested"], "Possible renames")
    assert "100% of their group labels overlap" in section(py["suggested"], "Possible renames")
    assert section(py["paired"], "Possible renames") is None and section(py["suggested"], "Compared as one") is None
    assert section(js["paired"], "Possible renames") is None and section(js["suggested"], "Compared as one") is None


@pytest.mark.parametrize("name_a,name_b,suggested", [
    ("SexCode", "sex_x", True),            # camel-case hump: Sex + Code -> sex (code is generic)
    ("patientSex", "PatientGender", True),  # patient shared
    ("HTTPServer", "http_host", False),     # an acronym run stays one word (no lower-to-upper cut): httpserver
    ("g\u00e9nero", "GENERO", True),       # accent folded
    ("\u00c9tatCivil", "etat_x", True),    # a capital first letter after a lower one: Etat + Civil
    ("sexCode", "raceCode", False),         # only the generic word is shared
    ("a1B2", "c3", False),
])
def test_rename_name_tokens_split_without_lookbehind_identically(tmp_path, name_a, name_b, suggested):
    """#956: the scan-based hump splitter in JS equals Python's, on camel case, acronyms and accents."""
    renames = _name_rename_results(tmp_path, {name_a: 3}, {name_b: 3})
    assert bool(renames) is suggested, renames
