"""Python/JS parity of the proxy hints: chi-squared, correction, the exact test and its parameters. (split out of test_js_parity.py, #938)."""

import csv
import io
import json
import re
import subprocess

import pandas as pd
import pytest

from faircode import compare, profile
from js_helpers import REPO_ROOT, SHARED_JS, SHARED_PRELUDE, _run_js_proxy_hints, _run_ui_exports, _sex_race_csv, shared_prologue


def test_proxy_hint_results_are_live_regions():
    """Proxy-hint results should be announced to screen-reader users."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")

    assert 'id="proxyHintsResults" aria-live="polite"' in html
    assert 'id="compareProxyHintsResults" aria-live="polite"' in html


def test_python_js_proxy_hints_parity_on_a_correlated_pair(tmp_path):
    """The web engine's proxyHints() (#738) is an opt-in, JS-only port of
    faircode/proxy.py's proxy_hints() - not covered by profile()/compare()'s
    bit-for-bit parity contract, but it should still agree with scipy's
    chi2_contingency on the same data: this builds a sex/race pair with a
    strong, deliberate correlation and checks both engines flag it with
    matching chi2/Cramer's V and a p-value that agrees to several digits
    (the two chi-squared CDF implementations differ past float precision)."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints

    rows = ["sex,race"]
    for i in range(60):
        sex = "male" if i % 2 == 0 else "female"
        if sex == "male":
            race = "White" if i % 5 != 0 else "Black"
        else:
            race = "White" if i % 4 == 0 else "Black"
        rows.append(f"{sex},{race}")
    csv = tmp_path / "proxy_pair.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    df = pd.read_csv(csv)
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    python_hints = proxy_hints(df, dims, alpha=0.9)
    js_hints = _run_js_proxy_hints(csv, alpha=0.9)

    assert len(python_hints) == 1
    assert len(js_hints) == 1
    py_hint, js_hint = python_hints[0], js_hints[0]
    assert (py_hint["a"], py_hint["b"]) == (js_hint["a"], js_hint["b"]) == ("sex", "race")
    assert py_hint["chi2"] == js_hint["chi2"]
    assert py_hint["cramers_v"] == js_hint["cramers_v"]
    assert py_hint["p_value"] == pytest.approx(js_hint["p_value"], rel=1e-6)
    # #810: the small-cell diagnostics must agree too
    assert py_hint["low_expected_share"] == js_hint["low_expected_share"]
    assert py_hint["low_expected"] == js_hint["low_expected"]
    assert py_hint["n_tests"] == js_hint["n_tests"] == 1


def test_python_js_proxy_hints_parity_flags_small_expected_cells(tmp_path):
    """#810: a sparse high-cardinality pair is flagged low_expected by both engines."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints

    rows = ["sex,race"]
    races = ["White", "Black", "Asian", "Latino", "Other", "Native"]
    for i in range(24):
        rows.append(f"{'male' if i % 2 == 0 else 'female'},{races[(i // 2 + i) % 6]}")
    csv = tmp_path / "sparse.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df = pd.read_csv(csv)
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    py = proxy_hints(df, dims, alpha=1.0)
    js = _run_js_proxy_hints(csv, alpha=1.0)
    assert len(py) == len(js) == 1
    assert py[0]["low_expected"] is True and js[0]["low_expected"] is True
    assert py[0]["low_expected_share"] == js[0]["low_expected_share"] > 0.2


def test_python_js_proxy_hints_parity_finds_nothing_for_unrelated_columns(tmp_path):
    """A column pair with no real association should not be flagged by either
    engine - proxyHints()'s job is precision (avoid crying wolf), not recall."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints

    rows = ["sex,race"]
    sexes = ["male", "female"]
    races = ["White", "Black", "Asian"]
    for i in range(90):
        rows.append(f"{sexes[i % 2]},{races[i % 3]}")
    csv = tmp_path / "proxy_unrelated.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    df = pd.read_csv(csv)
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    assert proxy_hints(df, dims, alpha=0.05) == []
    assert _run_js_proxy_hints(csv) == []


def test_proxy_hints_ui_wiring_present_in_profiler_html_and_ui_js():
    """Source-level check (like #740's dim/drift-card tests): the web results
    view must expose the opt-in proxy-hints button/section wired up in
    profiler-ui.js, and the dropzone hint text should no longer tell web
    users the feature is CLI-only, now that it's available in-browser (#738)."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    assert 'id="proxyHintsBlock"' in html
    assert 'id="proxyHintsBtn"' in html
    assert 'id="proxyHintsResults"' in html
    assert "CLI.) Proxy-hint detection: use" not in html

    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    assert "proxyHintsBtn.addEventListener('click', renderProxyHints)" in ui
    assert "E.proxyHints(currentTable, currentResult.dimensions, alpha, heldOut," in ui
    assert 'id="proxyCorrectionInput"' in html


def test_web_html_report_has_core_sections_and_proxy_hints(tmp_path):
    """buildHtmlReport() (the file the web "Download report" button emits) had
    no test (#765) and silently omitted proxy hints (#758)."""
    path = _sex_race_csv(tmp_path)
    plain = _run_ui_exports(path, False)["html"]
    assert "Dataset Representation Profile" in plain
    assert "<h2>sex" in plain and "<h2>race" in plain
    assert "Proxy hints" not in plain
    hinted = _run_ui_exports(path, True)["html"]
    assert "Proxy hints" in hinted and "sex ↔ race" in hinted


def test_web_compare_csv_and_html_match_python_and_include_proxy_hints(tmp_path):
    """Executes the real buildCompareCsvReport/buildCompareHtmlReport (#756,
    #757): the CSV's group and summary sections must equal Python's
    compare_to_csv() (flags differ in float formatting, so they are excluded),
    and attached proxy hints must reach both the CSV and the HTML report."""
    from faircode.report import compare_to_csv

    path_a = _sex_race_csv(tmp_path)
    rows = ["sex,race"] + [
        f"{'male' if i % 3 else 'female'},{'White' if i % 3 else 'Asian'}" for i in range(80)
    ]
    path_b = tmp_path / "b.csv"
    path_b.write_text("\n".join(rows) + "\n", encoding="utf-8")

    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    helpers = (src[src.index("function stripNegZero"):src.index("function wireSlot")]
               + src[src.index("function kindMismatchReason"):src.index("function driftCard")])
    builders = src[src.index("function proxyHintItems"):src.index("function compareReportBaseName")]
    csvs = src[src.index("var csvRow = E.csvRow"):src.index("async function downloadCompareCsvReport")]
    script = (
        "var DISPLAY_GROUPS=12;" + SHARED_PRELUDE + helpers +
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        + builders + csvs +
        "function load(p){return E.parseCSV(fs.readFileSync(p,'utf-8'));}"
        "var ta=load(process.argv[2]),tb=load(process.argv[3]);"
        "var pa=E.profile(ta,{},{}),pb=E.profile(tb,{},{});var c=E.compare(pa,pb,'a.csv','b.csv');"
        "var plain=buildCompareCsvReport(c);"
        "c.proxy_hints_a=E.proxyHints(ta,pa.dimensions,1);c.proxy_hints_b=E.proxyHints(tb,pb.dimensions,1);"
        "process.stdout.write(JSON.stringify({plain:plain,csv:buildCompareCsvReport(c),html:buildCompareHtmlReport(c)}));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path_a), str(path_b)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    out = json.loads(completed.stdout)

    py = compare_to_csv(compare(profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv"))
    def head(text):
        # Python writes 0.0 where JS writes 0; compare numerically, not textually.
        def norm(cell):
            try:
                return float(cell)
            except ValueError:
                return cell
        body = text.replace("\r\n", "\n").split("\n\nflag\n")[0]
        return [[norm(c) for c in row] for row in csv.reader(io.StringIO(body))]

    assert head(out["plain"]) == head(py)
    assert "dataset,proxy_hint_a,proxy_hint_b,p_value,cramers_v,n_tests,low_expected" in out["csv"]
    assert "Proxy hints - A" in out["html"] and "Proxy hints - B" in out["html"]


def test_compare_view_download_csv_and_proxy_controls_are_wired():
    """#789: the compare view's new controls exist in profiler.html and are
    bound in profiler-compare.js; renaming an id in either file would
    otherwise break the buttons with the suite green. Also covers the
    --proxy-alpha inputs (#786) for both views."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    js = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    for element_id in ("compareDownloadCsvBtn", "compareProxyHintsBtn",
                       "compareProxyHintsBlock", "compareProxyHintsResults",
                       "compareProxyAlphaInput", "proxyAlphaInput"):
        assert f'id="{element_id}"' in html, element_id
        if element_id != "proxyAlphaInput":
            assert f"getElementById('{element_id}')" in js, element_id
    assert "downloadCsvBtn.addEventListener('click', downloadCompareCsvReport)" in js
    assert "proxyBtn.addEventListener('click', renderCompareProxyHints)" in js
    assert "E.proxyHints(slot.A.table, currentProfiles.A.dimensions, alpha, heldA, correction, currentOpts.max_age, currentOpts.age_reference_year, exact, exactSettings.permutations, exactSettings.maxRows)" in js
    assert 'id="compareProxyCorrectionInput"' in html


def test_python_js_proxy_correction_parity(tmp_path):
    """#806: the JS adjustPValues/proxyHints(correction) match faircode.proxy for
    both methods, and the HTML/CSV exports show the adjusted p-value."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import adjust_p_values, proxy_hints

    n = 120
    sex = ["m", "f"] * (n // 2)
    rows = ["sex,race,age"]
    for i in range(n):
        race = "a" if (i % 2 == 0) == (i % 10 != 0) else "b"
        age = 55 if (sex[i] == "m" and i % 3 == 0) or i % 11 == 0 else 25
        rows.append(f"{sex[i]},{race},{age}")
    path = tmp_path / "three.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df = pd.read_csv(path)
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]

    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "var t=E.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));var r=E.profile(t,{},{});"
        "var out={};['bonferroni','holm'].forEach(function(m){"
        "out[m]=E.proxyHints(t,r.dimensions,1,null,m);});"
        "try{E.proxyHints(t,r.dimensions,0.05,null,'fdr')}catch(e){out.bad=e.message}"
        "process.stdout.write(JSON.stringify(out));"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(done.stdout)
    for method in ("bonferroni", "holm"):
        py = proxy_hints(df, dims, alpha=1.0, correction=method)
        key = lambda h: (h["a"], h["b"])  # noqa: E731
        py_map = {key(h): h["p_adjusted"] for h in py}
        js_map = {key(h): h["p_adjusted"] for h in js[method]}
        assert py_map.keys() == js_map.keys() and py_map
        for k in py_map:
            assert py_map[k] == pytest.approx(js_map[k], rel=1e-6)
    assert "correction must be" in js["bad"]
    assert adjust_p_values([0.01, 0.04, 0.03], "holm") == pytest.approx([0.03, 0.06, 0.06])


def test_adjust_p_values_js_parity_ties_and_edge_cases():
    """#820: verify that JS adjustPValues exactly matches Python adjust_p_values
    for ties (breaking ties stably in input order), single tested pair (m=1), and empty input (m=0)."""
    from faircode.proxy import adjust_p_values

    cases = [
        ([], "bonferroni"),
        ([], "holm"),
        ([0.042], "bonferroni"),
        ([0.042], "holm"),
        ([0.8], "bonferroni"),
        ([0.8], "holm"),
        ([0.02, 0.02, 0.04], "holm"),
        ([0.05, 0.01, 0.01, 0.03], "holm"),
        ([0.03, 0.03], "bonferroni"),
        ([0.03, 0.03], "holm"),
        ([0.1, 0.1, 0.1], "holm"),
        ([0.01, 0.04, 0.03], "bonferroni"),
        ([0.01, 0.04, 0.03], "holm"),
    ]

    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var cases=JSON.parse(process.argv[2]);"
        "var results=cases.map(function(c){return E.adjustPValues(c[0],c[1]);});"
        "process.stdout.write(JSON.stringify(results));"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(cases)],
        capture_output=True, text=True, encoding="utf-8", check=True
    )
    js_results = json.loads(done.stdout)
    for (ps, method), js_res in zip(cases, js_results):
        py_res = adjust_p_values(ps, method)
        assert js_res == pytest.approx(py_res), f"Mismatch for {ps} with {method}: JS={js_res}, PY={py_res}"


def test_python_js_exact_p_values_agree_for_small_cell_tables(tmp_path):
    """#861: with exact on, a sparse 2x2 gets Fisher's exact p and a sparse 6x2 a seeded
    permutation p - identical PRNG stream, so both engines return the same numbers."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints

    rows = ["sex,race"]
    # 2x2, small: sex x group
    for i in range(14):
        rows.append(f"{'male' if i < 5 else 'female'},{'x' if (i % 7) < 3 else 'y'}")
    csv = tmp_path / "small2x2.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df = pd.read_csv(csv)
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    py = proxy_hints(df, dims, alpha=1.0, exact=True)
    js = _run_js_proxy_hints(csv, alpha=1.0, exact=True)
    assert len(py) == len(js) == 1 and py[0]["p_method"] == js[0]["p_method"] == "fisher"
    assert py[0]["p_value"] == pytest.approx(js[0]["p_value"], rel=1e-9)
    assert py[0]["p_chi2"] == pytest.approx(js[0]["p_chi2"], rel=1e-6)

    rows = ["sex,race"]
    races = ["White", "Black", "Asian", "Latino", "Other", "Native"]
    for i in range(24):
        rows.append(f"{'male' if i % 2 == 0 else 'female'},{races[(i // 2 + i) % 6]}")
    csv2 = tmp_path / "sparse6x2.csv"
    csv2.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df2 = pd.read_csv(csv2)
    py2 = proxy_hints(df2, dims, alpha=1.0, exact=True)
    js2 = _run_js_proxy_hints(csv2, alpha=1.0, exact=True)
    assert py2[0]["p_method"] == js2[0]["p_method"] == "permutation"
    assert py2[0]["p_value"] == js2[0]["p_value"]          # same seeded stream, same count

    # a larger sparse table: same seeded stream through the sparse statistic (#886)
    rows = ["sex,race"] + [f"s{(i * 7) % 40},r{(i * 13 + i // 5) % 60}" for i in range(400)]
    csv3 = tmp_path / "sparse_big.csv"
    csv3.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df3 = pd.read_csv(csv3)
    (py3,), (js3,) = proxy_hints(df3, dims, alpha=1.0, exact=True), _run_js_proxy_hints(csv3, alpha=1.0, exact=True)
    assert py3["p_method"] == js3["p_method"] == "permutation" and py3["p_value"] == js3["p_value"]

    # sparse but over the row cap: chi-squared p kept and labelled (#885). Too many categories for
    # detection, so the dimensions are handed to the engines directly, with a lowered cap (#906).
    rows = ["sex,race"] + [f"s{i // 2},r{i // 3}" for i in range(600)]
    csv4 = tmp_path / "capped.csv"
    csv4.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (py4,) = proxy_hints(pd.read_csv(csv4), dims, alpha=1.0, exact=True, max_rows=599)
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var t=E.parseCSV(require('fs').readFileSync(process.argv[2],'utf-8'));"
        "process.stdout.write(JSON.stringify(E.proxyHints(t,[{name:'sex',kind:'sex'},{name:'race',kind:'race'}],1,null,null,undefined,undefined,true,undefined,599)));"
    )
    (js4,) = json.loads(subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(csv4)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert py4["p_method"] == js4["p_method"] == "chi2-capped"
    assert "p_chi2" not in js4 and py4["low_expected"] and js4["low_expected"]
    assert "PERMUTATION_MAX_ROWS = 20000" in (REPO_ROOT / "assets" / "profiler-engine.js").read_text(encoding="utf-8")   # #909

    # off by default: no p_method key, p-values untouched
    assert "p_method" not in proxy_hints(df2, dims, alpha=1.0)[0]
    assert "p_method" not in _run_js_proxy_hints(csv2, alpha=1.0)[0]


def test_python_js_exact_permutations_and_row_cap_parameters_agree(tmp_path):
    """#906: a non-default shuffle count draws the same seeded stream in both engines; a lowered
    cap labels the pair chi2-capped in both; bad values are rejected by both."""
    from faircode.proxy import proxy_hints
    rows = ["sex,race"] + [f"{'male' if i % 2 == 0 else 'female'},{'abcdef'[(i // 2 + i) % 6]}" for i in range(24)]
    csv = tmp_path / "sparse.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    df = pd.read_csv(csv)
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var t=E.parseCSV(require('fs').readFileSync(process.argv[2],'utf-8'));"
        "var d=[{name:'sex',kind:'sex'},{name:'race',kind:'race'}];"
        "function run(p,m){try{return E.proxyHints(t,d,1,null,null,undefined,undefined,true,p,m)[0]}catch(e){return e.message}}"
        "process.stdout.write(JSON.stringify({p99:run(99,undefined),p3000:run(3000,undefined),capped:run(undefined,10),"
        "bad:run(0,undefined),badrows:run(undefined,1.5),toomany:run(1000001,undefined)}));"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(csv)],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    for js_key, kwargs in (("p99", {"permutations": 99}), ("p3000", {"permutations": 3000})):
        (py,) = proxy_hints(df, dims, alpha=1.0, exact=True, **kwargs)
        assert out[js_key]["p_method"] == py["p_method"] == "permutation"
        assert out[js_key]["p_value"] == py["p_value"]
    (py_capped,) = proxy_hints(df, dims, alpha=1.0, exact=True, max_rows=10)
    assert out["capped"]["p_method"] == py_capped["p_method"] == "chi2-capped"
    assert "positive integer" in out["bad"] and "positive integer" in out["badrows"]
    assert "at most 1,000,000" in out["toomany"]                                   # #926


def test_web_exact_test_inputs_are_parsed_and_wired_into_both_views():
    """#925: blank means the default, otherwise a whole number (shuffles up to 1,000,000), and the
    values reach proxyHints(), the provenance params and the copied CLI command."""
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "function t(a,b){try{return E.parseExactSettings(a,b)}catch(e){return e.message}}"
        "process.stdout.write(JSON.stringify({blank:t('',' '),ok:t('250','30000'),zero:t('0',''),frac:t('1.5',''),"
        "big:t('1000001',''),text:t('',  'abc'),pad:t(' 99 ',undefined)}));"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out["blank"] == {} and out["ok"] == {"permutations": 250, "maxRows": 30000} and out["pad"] == {"permutations": 99}
    for bad in ("zero", "frac", "big"):
        assert "Shuffles must be a whole number from 1 to 1,000,000" in out[bad]
    assert "Row cap must be a whole number" in out["text"]
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    for element_id in ("proxyExactPermutationsInput", "proxyExactMaxRowsInput",
                       "compareProxyExactPermutationsInput", "compareProxyExactMaxRowsInput"):
        assert f'id="{element_id}"' in html
    for name in ("profiler-ui.js", "profiler-compare.js"):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "E.parseExactSettings(" in js and "permutations: exactSettings.permutations, maxRows: exactSettings.maxRows" in js


def test_web_exact_inputs_follow_the_checkbox_and_report_errors_inline():
    """#962: the Shuffles / Row cap inputs are disabled until the exact box is ticked, and a bad value
    marks its own input aria-invalid with a role=alert message instead of a paragraph in the results."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    for pre in ("proxy", "compareProxy"):
        for field in ("Permutations", "MaxRows"):
            tag = re.search(r'<input[^>]*id="%sExact%sInput"[^>]*>' % (pre, field), html).group(0)
            assert " disabled" in tag and f'aria-describedby="{pre}ExactError"' in tag
        assert f'id="{pre}ExactError" role="alert" hidden' in html
    script = r"""
    var out={};
    ['proxy','compareProxy'].forEach(function(prefix){
      function el(){var e={attrs:{},hidden:true,disabled:true,checked:false,textContent:'',value:'',
        setAttribute:function(k,v){e.attrs[k]=v;},removeAttribute:function(k){delete e.attrs[k];}};return e;}
      var els={};['ExactInput','ExactPermutationsInput','ExactMaxRowsInput','ExactError'].forEach(function(s){els[prefix+s]=el();});
      var window={};var document={getElementById:function(id){return els[id];}};
      """ + shared_prologue(1) + r"""
      var api=shared.createExactInputs(prefix);
      var r={},box=els[prefix+'ExactInput'],perm=els[prefix+'ExactPermutationsInput'],rows=els[prefix+'ExactMaxRowsInput'],err=els[prefix+'ExactError'];
      api.sync();r.off=[perm.disabled,rows.disabled];
      box.checked=true;api.sync();r.on=[perm.disabled,rows.disabled];
      api.showError('Shuffles must be a whole number from 1 to 1,000,000.');
      r.shuffles=[perm.attrs['aria-invalid'],rows.attrs['aria-invalid']||null,err.hidden,err.textContent];
      api.showError('Row cap must be a whole number from 1.');
      r.rows=[perm.attrs['aria-invalid']||null,rows.attrs['aria-invalid'],err.hidden];
      box.checked=false;api.sync();
      r.cleared=[perm.disabled,rows.attrs['aria-invalid']||null,err.hidden,err.textContent];
      out[prefix]=r;
    });
    process.stdout.write(JSON.stringify(out));
    """
    done = subprocess.run(["node", "-e", script, str(SHARED_JS)], capture_output=True, text=True,
                          encoding="utf-8", check=True)
    for name, r in json.loads(done.stdout).items():
        assert r["off"] == [True, True] and r["on"] == [False, False], name
        assert r["shuffles"] == ["true", None, False, "Shuffles must be a whole number from 1 to 1,000,000."], name
        assert r["rows"] == [None, "true", False], name
        assert r["cleared"] == [True, None, True, ""], name
    for name in ("profiler-ui.js", "profiler-compare.js"):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "showExactError(err.message);" in js and "if (exactControls().box.checked) {" in js
        assert js.rstrip().endswith("wireExactInputs();\n})();")      # wired after the DOM is ready
