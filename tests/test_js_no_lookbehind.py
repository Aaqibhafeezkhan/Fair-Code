"""Browser scripts must not contain regex lookbehind (#956).

A regex literal using lookbehind is an *early* syntax error in engines without support (Safari before
16.4), and a syntax error stops the whole script, so one rename hint or one markdown table would take
the profiler, the dashboard or the explainers page down with it.
"""

import json
import re
import subprocess

from js_helpers import REPO_ROOT

LOOKBEHIND = re.compile(r"\(\?<[=!]")


def test_no_browser_script_uses_regex_lookbehind():
    offenders = []
    for path in sorted((REPO_ROOT / "assets").glob("*.js")) + sorted(REPO_ROOT.glob("*.html")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if LOOKBEHIND.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders, f"regex lookbehind is unsupported before Safari 16.4: {offenders}"


def test_scanning_pipe_splitter_matches_the_python_regex_split():
    # explainers-ui.js splits table rows with a scan; scripts/build_explainers.py uses a lookbehind regex.
    src = (REPO_ROOT / "assets" / "explainers-ui.js").read_text(encoding="utf-8")
    start = src.index("const splitUnescaped")
    end = src.index("const splitRow")
    rows = ["| a | b |", "| a \\| b | c |", "|x|y|z|", "| trailing \\\\| two |", "| \\| |", "|||", "no pipes", "| a \\|"]
    script = (src[start:end] + "process.stdout.write(JSON.stringify(JSON.parse(process.argv[1]).map(function(r){return splitUnescaped(r);})));")
    out = json.loads(subprocess.run(["node", "-e", script, json.dumps(rows)], capture_output=True, text=True,
                                    encoding="utf-8", check=True).stdout)
    assert out == [re.split(r"(?<!\\)\|", row) for row in rows]
