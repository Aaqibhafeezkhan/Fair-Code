"""The web 'Copy as CLI command' builder: its lines are run through the real CLI. (split out of test_js_parity.py, #938)."""

import json
import re
import subprocess

from js_helpers import REPO_ROOT


def test_web_cli_command_builder_emits_flags_the_cli_parses(tmp_path, monkeypatch, capsys):
    """#915: E.buildCliCommand's output (heredoc, quoting, held-out syntax) is run through the real
    CLI and the options come back in the provenance, for a profile and a compare run."""
    import shlex
    from faircode.cli import main
    rows = ["id,sex,race,job"] + [f"{i},{'M' if i % 2 else 'F'},{'abc'[i % 3]},{'x' if i % 2 else 'y'}" for i in range(60)]
    (tmp_path / "my data.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "h.csv").write_text("id,eth\n" + "\n".join(f"{i},{'uv'[i % 2]}" for i in range(60)) + "\n")
    (tmp_path / "ref.csv").write_text("column,group,share\nsex,M,0.5\nsex,F,0.5\n")
    rows_a = ["id,sex,race"] + [f"{i},{'M' if i % 2 else 'F'},{'abc'[i % 3]}" for i in range(60)]
    (tmp_path / "a.csv").write_text("\n".join(rows_a) + "\n")
    (tmp_path / "b.csv").write_text("\n".join(r.replace("race", "ethnicity") for r in rows_a) + "\n")
    specs = {
        "profile": {"kind": "profile", "files": ["my data.csv"], "opts": {"max_age": 100, "min_group_size": 5, "keywords": {"sex": ["sesso"]}, "cross": ["sex", "race"]},
                    "overrides": {"job": "categorical"}, "encoding": "latin-1", "referenceName": "ref.csv",
                    "proxy": {"alpha": 0.2, "correction": "holm", "exact": True, "permutations": 99},
                    "held": {"main": [{"name": "h.csv", "column": "eth", "key": "id", "normalize": True, "encoding": "latin-1"}]}},
        "compare": {"kind": "compare", "files": ["a.csv", "b.csv"], "opts": {"max_age": 90}, "overrides": {}, "encoding": "auto",
                    "renames": {"race": "ethnicity"}, "proxy": {"alpha": 0.05, "correction": None, "exact": False},
                    "held": {"a": [{"name": "h.csv", "column": "eth", "key": "id"}], "b": []}},
    }
    script = ("require(process.argv[1]);var E=globalThis.FairCodeProfiler;var s=JSON.parse(process.argv[2]);"
              "var o={};Object.keys(s).forEach(function(k){o[k]=E.buildCliCommand(s[k]);});process.stdout.write(JSON.stringify(o));")
    cmds = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(specs)],
                                     capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    monkeypatch.chdir(tmp_path)

    def run(command):
        rest = command
        if command.startswith("cat >"):   # the keywords heredoc writes keywords-<hash>.json first
            (tmp_path / command.split()[2]).write_text(command.split("\n")[1])
            rest = command.split("EOF\n", 1)[1]
        tokens = shlex.split(rest)
        assert tokens[:1] == ["faircode"]
        assert main(tokens[1:] + ["--json"]) == 0
        return json.loads(capsys.readouterr().out)

    out = run(cmds["profile"])
    params = out["provenance"]["params"]
    assert (params["max_age"], params["min_group_size"], params["alpha"], params["correction"], params["exact"]) == (100, 5, 0.2, "holm", True)
    assert params["keywords"] == {"sex": ["sesso"]} and params["exact_permutations"] == 99
    held = out["provenance"]["proxy_hints_with"][0]
    assert (held["column"], held["key"], held["key_normalize"], held["encoding"]) == ("eth", "id", True, "latin-1")
    assert "'my data.csv'" in cmds["profile"] and "--proxy-hints-with 'h.csv=eth:id~@latin-1'" in cmds["profile"]
    out = run(cmds["compare"])
    assert out["renamed_dimensions"] == [{"a": "race", "b": "ethnicity"}]
    assert out["provenance"]["proxy_hints_with_a"][0]["column"] == "eth" and out["provenance"]["params"]["max_age"] == 90
    assert "--proxy-alpha" not in cmds["compare"] and "--keywords" not in cmds["compare"]
    for name, btn in (("profiler-ui.js", "copyCliBtn"), ("profiler-compare.js", "copyCliBtn")):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "E.buildCliCommand({" in js and f"flashButton({btn}, " in js
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    assert 'id="copyCliBtn"' in html and 'id="compareCopyCliBtn"' in html


def test_cli_command_builder_quotes_awkward_names_so_they_round_trip():
    """#934: file names, columns, keys and --map values with quotes, $, backticks, !, a leading ~,
    newlines, backslashes, non-ASCII and the empty string each come back from shlex as one
    argument, so a copied command never runs text from a name as shell code."""
    import shlex
    names = ["o'brien.csv", "a$b.csv", "`id`.csv", "wow!.csv", "~home.csv", "two\nlines.csv", "back\\slash.csv",
             "名前.csv", 'say "hi".csv', "a b  c.csv", "semi;colon&amp|pipe.csv", "$(touch x).csv", ""]
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;var names=JSON.parse(process.argv[2]);"
        "var out=names.map(function(n){return E.buildCliCommand({kind:'profile',files:[n],opts:{},overrides:{},encoding:'auto',"
        "proxy:{alpha:0.05,exact:false},held:{main:[{name:n,column:n,key:n,normalize:true}]}});});"
        "process.stdout.write(JSON.stringify(out));"
    )
    cmds = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(names)],
                                     capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    for name, cmd in zip(names, cmds):
        tokens = shlex.split(cmd)
        assert tokens[:2] == ["faircode", "profile"] and tokens[2] == name, (name, tokens)
        if name:   # an empty key means "no key", so the held-out spec of "" is just "="
            assert tokens[-2] == "--proxy-hints-with" and tokens[-1] == f"{name}={name}:{name}~", (name, tokens)


def test_cli_command_for_the_built_in_sample_uses_sample_flag(capsys):
    """#928: the profile demo copies `faircode profile --sample ...` (which runs), and the web-only
    compare sample pair says so in a comment instead of naming files that do not exist."""
    import shlex
    from faircode.cli import main
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "process.stdout.write(JSON.stringify({p:E.buildCliCommand({kind:'profile',sample:true,files:['data.csv'],opts:{max_age:100},encoding:'auto'}),"
        "c:E.buildCliCommand({kind:'compare',sample:true,files:['a.csv','b.csv'],opts:{},encoding:'auto'}),"
        "own:E.buildCliCommand({kind:'profile',files:['mine.csv'],opts:{}})}));"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out["p"] == "faircode profile --sample --max-age 100" and out["own"] == "faircode profile mine.csv"
    assert main(shlex.split(out["p"])[1:] + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_rows"] > 0
    first, _, rest = out["c"].partition("\n")
    assert first.startswith("# the sample pair is built into the web page") and rest == "faircode compare a.csv b.csv"
    for name, marker in (("profiler-ui.js", "currentName === 'sample-health-data.csv'"), ("profiler-compare.js", "slot.A.name === 'sample-baseline.csv'")):
        assert marker in (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")


def test_cli_command_includes_the_export_options_the_checkboxes_mirror(tmp_path, monkeypatch, capsys):
    """#929: outputs (csv / csvProvenance / csvBom / html / json) become flags the CLI accepts and
    writes; the views pass the include-provenance checkbox as a csv + --csv-provenance pair."""
    import shlex
    from faircode.cli import main
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "process.stdout.write(JSON.stringify(E.buildCliCommand({kind:'profile',sample:true,files:[],opts:{},"
        "outputs:{csv:'out file.csv',csvProvenance:true,csvBom:true,html:'r.html'}})));"
    )
    cmd = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert cmd == "faircode profile --sample --csv 'out file.csv' --csv-provenance --csv-bom --html r.html"
    monkeypatch.chdir(tmp_path)
    assert main(shlex.split(cmd)[1:]) == 0
    assert (tmp_path / "out file.csv").read_bytes().startswith(b"\xef\xbb\xbf") and (tmp_path / "r.html").is_file()
    assert "provenance" in (tmp_path / "out file.csv").read_text(encoding="utf-8-sig").lower()
    for name, marker in (("profiler-ui.js", "cliOutputs(document.getElementById('csvProvenanceInput').checked"),
                         ("profiler-compare.js", "document.getElementById('compareCsvProvenanceInput').checked")):
        assert marker in (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")


def test_keywords_heredoc_file_is_named_from_its_content():
    """#944: the same vocabulary always gives the same file name, a different one a different name,
    and the name never collides with an existing words.json."""
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "function c(k){return E.buildCliCommand({kind:'profile',files:['d.csv'],opts:{keywords:k}});}"
        "process.stdout.write(JSON.stringify([c({sex:['sesso']}),c({sex:['sesso']}),c({sex:['genero']}),c({age:['eta'],sex:['sesso']})]));"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    names = [c.split()[2] for c in out]
    assert names[0] == names[1] and len({names[0], names[2], names[3]}) == 3
    for name, cmd in zip(names, out):
        assert re.fullmatch(r"keywords-[0-9a-f]{8}\.json", name) and name != "words.json"
        assert f"--keywords {name}" in cmd
