"""Shared helpers for the tests/test_js_parity_*.py files (split out of test_js_parity.py, #938)."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

FIXTURES = Path(__file__).resolve().parent / "fixtures"

requires_openpyxl = pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None,
    reason="optional 'excel' extra not installed",
)

def _extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    assert match, f"Could not find {pattern!r}"
    return match.group(1)

# Real audit datasets are already tracked in their own audit folders - reuse
# them instead of keeping a second multi-megabyte copy under tests/fixtures.
CSV_PATHS = {
    "small.csv": FIXTURES / "small.csv",
    "adult.csv": REPO_ROOT / "Benefits Denial" / "adult.csv",
    "compas-scores-raw.csv": REPO_ROOT / "COMPAS" / "compas-scores-raw.csv",
    "credit_customers.csv": REPO_ROOT / "German Credit Lending" / "credit_customers.csv",
    "AI_Fair_Recruitment_Dataset.csv": REPO_ROOT / "AI Fair Recruitment" / "AI_Fair_Recruitment_Dataset.csv",
}

def _run_js_proxy_hints(csv_path, alpha=0.05, exact=False):
    script = (
        "require(process.argv[1]);"
        "var fs=require('fs');"
        "var table=globalThis.FairCodeProfiler.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));"
        "var r=globalThis.FairCodeProfiler.profile(table,{},{});"
        "var hints=globalThis.FairCodeProfiler.proxyHints(table,r.dimensions,"
        + repr(alpha) + ",null,null,undefined,undefined," + ("true" if exact else "false") + ");"
        "process.stdout.write(JSON.stringify(hints));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(csv_path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(completed.stdout)

def _run_ui_exports(csv_path, with_hints, reference=None, provenance=None):
    """Execute profiler-ui.js's real buildHtmlReport/buildCsvReport (sliced out
    of the DOM-coupled file) against a real engine profile of `csv_path`."""
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    helpers = src[src.index("var GRADE_COLOR"):src.index("function render(")]
    builders = src[src.index("function buildHtmlReport(r)"):src.index("async function downloadCsvReport")]
    script = (
        "var DISPLAY_GROUPS=12;" + helpers +
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        + builders +
        "var t=E.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));var r=E.profile(t,{}," + json.dumps({"reference": reference} if reference else {}) + ");"
        + ("r.proxy_hints=E.proxyHints(t,r.dimensions,0.9);" if with_hints else "") +
        "process.stdout.write(JSON.stringify({html:buildHtmlReport(r),csv:buildCsvReport(r," + json.dumps(provenance) + ")}));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(csv_path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(completed.stdout)

def _sex_race_csv(tmp_path):
    rows = ["sex,race"]
    for i in range(60):
        sex = "male" if i % 2 == 0 else "female"
        race = ("White" if i % 5 != 0 else "Black") if sex == "male" else ("White" if i % 4 == 0 else "Black")
        rows.append(f"{sex},{race}")
    path = tmp_path / "sr.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


# Delimiter-sniffing samples and the delimiter both engines must pick (#964); used by test_loaders.py and
# test_js_parity_profile.py.
SNIFF_CASES = [
    ("", ","),                                             # empty sample: the default
    ("a\tb\tc\n1\t2\t3\n", "\t"),
    ("a;b\n1;2\n", ";"),
    ("a|b|c\n1|2|3\n", "|"),
    ("a,b;c\n1,2;3\n", ","),                               # a tie: the earlier candidate wins
    ('a,b\n"x,y,z",2\n', ","),                             # delimiters inside quotes are not counted
    ("a\n1\n2\n", ","),                                    # one column: nothing to find, the default
    ("a,b\n1,2,3\n", ","),                                 # inconsistent rows: no delimiter qualifies
    ('a;b\n"1;2"', ","),                                   # quote closing at the end of the sample
    ('a;b\n5" disk;2\n', ";"),                             # a quote that is not at a field start is literal
    ("a;b\r\n\r\n1;2\r\n", ";"),                           # CRLF and a blank line
    ('a;b\n"1;2\n3', ";"),                                 # unterminated quote at the end: that row is dropped
    ("a;b\n1;2\n1;2\n1;2\n1;2\n1;2;3\n", ";"),             # only the first five logical rows are checked
]
