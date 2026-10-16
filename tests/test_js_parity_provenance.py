"""Provenance and encoding parity between the Python exports and the web profiler. (split out of test_js_parity.py, #938)."""

import json
import subprocess

import pandas as pd

from faircode import profile
from js_helpers import REPO_ROOT, _run_ui_exports, _sex_race_csv


def test_web_and_python_csv_provenance_sections_are_identical(tmp_path):
    """#800: the same provenance dict renders to the same CSV section in both
    engines - dotted keys for nested objects, JSON text for lists, empty for null."""
    from faircode.report import to_csv

    path = _sex_race_csv(tmp_path)
    prov = {"faircode_version": "2.3.0", "dataset_hash": None, "dataset_hash_note": "from stdin",
            "params": {"min_share": 0.05, "cross": ["sex", "race"], "reference_flag": 0.05},
            "overrides": {"zip": "geography"}}
    web = _run_ui_exports(path, False, None, prov)["csv"]
    py = to_csv(dict(profile(pd.read_csv(path))), provenance=prov)
    assert "provenance_key,provenance_value" in web
    assert "params.cross,\"[\"\"sex\"\", \"\"race\"\"]\"" in web
    assert web.replace("\r\n", "\n") == py.replace("\r\n", "\n")
    assert "provenance_key" not in _run_ui_exports(path, False)["csv"]


def test_web_and_python_csv_provenance_render_held_out_entries_identically(tmp_path):
    """#811: a list of {path, column, sha256} objects renders the same text in both
    engines, including a non-ASCII path (Python json.dumps escapes it)."""
    from faircode.report import to_csv

    path = _sex_race_csv(tmp_path)
    prov = {"faircode_version": "2.3.0", "dataset_hash": "sha256:ab",
            "params": {"min_share": 0.05}, "overrides": {},
            "proxy_hints_with": [
                {"path": "données/race.csv", "column": "race", "sha256": "sha256:cd"},
                {"path": "-", "column": "age", "sha256": None, "sha256_note": "read from stdin"}]}
    web = _run_ui_exports(path, False, None, prov)["csv"]
    py = to_csv(dict(profile(pd.read_csv(path))), provenance=prov)
    assert "proxy_hints_with" in web
    assert web.replace("\r\n", "\n") == py.replace("\r\n", "\n")


def test_web_decode_text_matches_python_loaders_and_names_the_encoding(tmp_path):
    """#857: decodeText() reads latin-1/cp1252/UTF-16/UTF-32/BOM'd UTF-8 like faircode.loaders
    and reports the encoding provenance should record (None for plain UTF-8)."""
    from faircode.loaders import read_table

    text = "sex,race\nM,Café\nF,Señor\n"
    cases = {
        "latin.csv": (text.encode("latin-1"), "latin-1", "latin-1"),
        "cp.csv": (text.encode("cp1252"), "cp1252", "cp1252"),
        "u16.csv": (text.encode("utf-16"), "auto", "utf-16"),
        "u16be.csv": (b"\xfe\xff" + text.encode("utf-16-be"), "auto", "utf-16"),
        "u32.csv": (text.encode("utf-32"), "auto", "utf-32"),
        "bom8.csv": (text.encode("utf-8-sig"), "auto", "utf-8-sig"),
        "plain.csv": (text.encode("utf-8"), "auto", None),
        "explicit8.csv": (text.encode("utf-8"), "utf-8", "utf-8"),
    }
    paths = {}
    for name, (raw, _choice, _enc) in cases.items():
        paths[name] = tmp_path / name
        paths[name].write_bytes(raw)
    spec = {name: [str(paths[name]), choice] for name, (_r, choice, _e) in cases.items()}
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;var out={};"
        "var spec=JSON.parse(process.argv[2]);Object.keys(spec).forEach(function(n){"
        "var b=fs.readFileSync(spec[n][0]);var ab=b.buffer.slice(b.byteOffset,b.byteOffset+b.length);"
        "var d=E.decodeText(ab,spec[n][1]);out[n]={text:d.text,encoding:d.encoding}});"
        "process.stdout.write(JSON.stringify(out));"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(spec)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    for name, (_raw, choice, enc) in cases.items():
        assert out[name]["text"] == text, name
        assert out[name]["encoding"] == enc, name
        py = read_table(str(paths[name]), encoding=None if choice == "auto" else choice)
        assert list(py["race"]) == ["Café", "Señor"], name


def test_web_encoding_picker_is_wired_and_recorded_in_provenance():
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    for select_id in ("encodingSelect", "compareEncodingSelect"):
        assert f'id="{select_id}"' in html
    for value in ("auto", "utf-8", "utf-16", "cp1252", "latin-1"):
        assert html.count(f'<option value="{value}"') >= 2, value
    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    cmp_js = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    assert "E.decodeText(reader.result" in ui and "E.decodeText(reader.result" in cmp_js
    assert "readAsText(" not in ui.split("function readFile")[1].split("function runText")[0]
    assert "provenance.encoding = currentEncoding" in ui
    assert "provenance.encoding_a = slot.A.encoding" in cmp_js and "provenance.encoding_b = slot.B.encoding" in cmp_js
    held = (REPO_ROOT / "assets" / "profiler-heldout.js").read_text(encoding="utf-8")
    assert "decodeText" in held


def test_web_reference_file_honours_the_encoding_picker():
    """#890: the reference baseline is read as bytes and decoded with the picker's choice
    like the dataset, so a latin-1 census label matches the dataset's group."""
    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    assert "readAsText(" not in ui
    handler = ui.split("referenceInput.addEventListener('change'")[1].split("referenceClearBtn.addEventListener")[0]
    assert "E.decodeText(reader.result, encodingSelect ? encodingSelect.value : 'auto')" in handler
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var bytes=Buffer.from('column,group,share\\nrace,Caf\\u00e9,0.5\\nrace,Other,0.5\\n','latin1');"
        "var buf=bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.length);"
        "var viaUtf8=E.parseReference(E.parseCSV(new TextDecoder('utf-8').decode(buf)));"
        "var viaPicker=E.parseReference(E.parseCSV(E.decodeText(buf,'latin-1').text));"
        "process.stdout.write(JSON.stringify({utf8:Object.keys(viaUtf8.race),picker:Object.keys(viaPicker.race)}));"
    )
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert "Café" in out["picker"] and "Café" not in out["utf8"]


def test_web_held_out_provenance_records_the_encoding():
    """#891: each held-out spec carries the decoded encoding and both views copy it into
    the entry after the digest, like provenance.held_out_entries()."""
    held = (REPO_ROOT / "assets" / "profiler-heldout.js").read_text(encoding="utf-8")
    assert "encoding = decoded.encoding || undefined" in held and "encoding: encoding" in held
    shared = (REPO_ROOT / "assets" / "profiler-shared.js").read_text(encoding="utf-8")   # one copy for both views (#983)
    body = shared.split("async function heldOutEntries")[1].split("// Skipped")[0]
    assert body.index("sha256_note") < body.index("entry.encoding = specs[i].encoding")


def test_web_provenance_params_carry_the_proxy_settings():
    """#898: publicParams(opts, proxy) adds alpha/correction/exact like provenance.proxy_params,
    and both views pass the settings behind the on-screen hints."""
    from faircode.profiler import _resolve_opts
    from faircode.provenance import proxy_params, public_params
    script = ("require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
              "process.stdout.write(JSON.stringify({plain:E.publicParams({}),"
              "withp:E.publicParams({},{alpha:0.01,correction:'holm',exact:true}),"
              "dflt:E.publicParams({},{alpha:0.05,correction:'',exact:false})}));")
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    base = _resolve_opts({})
    assert out["plain"] == public_params(base)
    assert out["withp"] == public_params({**base, **proxy_params(0.01, "holm", True)})
    assert out["dflt"] == public_params({**base, **proxy_params(0.05, None, False)})
    for name in ("profiler-ui.js", "profiler-compare.js"):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "E.publicParams(currentOpts, " in js and "currentProxySettings = {" in js


def test_web_provenance_records_the_xlsx_sheet():
    """#899: the sheet read and the number ignored reach provenance for the dataset, both
    compare slots and held-out workbooks, after the digest like provenance.sheet_fields()."""
    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    cmp_js = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    eng = (REPO_ROOT / "assets" / "profiler-engine.js").read_text(encoding="utf-8")
    assert "Object.assign(provenance, currentSheet)" in ui and "sheets_ignored: result.ignoredSheets.length" in ui
    assert "provenance.sheet_a = " in cmp_js and "provenance.sheets_ignored_b = " in cmp_js
    assert "spec.sheet = sheets.sheetName" in eng
    assert "entry.sheet = specs[i].sheet" in (REPO_ROOT / "assets" / "profiler-shared.js").read_text(encoding="utf-8")


def test_web_reference_baseline_is_re_applied_when_the_encoding_picker_changes():
    """#900: the picker hands the reference File over to be re-decoded after the dataset is
    re-read (which resets it), or re-reads it directly when the dataset is not re-read."""
    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    picker = ui.split("encodingSelect.addEventListener('change'")[1].split("function runText")[0]
    assert "pendingReferenceFile = referenceFile;" in picker and "loadReferenceFile(referenceFile)" in picker
    run_table = ui.split("function runTable(")[1].split("function reprofile")[0]
    assert "var reapply = pendingReferenceFile;" in run_table and "if (reapply) loadReferenceFile(reapply);" in run_table
    assert "function loadReferenceFile(f)" in ui and "referenceFile = null;" in ui.split("function resetReference")[1][:80]
    assert "pendingReferenceFile = null;" in ui.split("function showError")[1][:80]   # a failed load must not leak it


def test_web_held_out_provenance_marks_skipped_near_unique_columns_like_python():
    """#951: buildHeldOut() flags the skipped spec, heldOutEntries() writes tested: false + skip_reason
    in the same position and wording as provenance.held_out_entries, and the views export the notes."""
    from faircode.provenance import held_out_entries
    script = r"""
    require(process.argv[1]);var E=globalThis.FairCodeProfiler;var fs=require('fs');var out={};
    var rows=[],held=[];for(var i=0;i<100;i++){rows.push({sex:i%2?'m':'f'});held.push({ident:'id'+i,race:'ab'[i%2]});}
    var t={columns:['sex'],rows:rows};
    var window={};var document={};
    var shared=new Function('window','document',fs.readFileSync(process.argv[2],'utf-8')+';return window.FairCodeShared;')(window,document);
    (async function(){
      var specs=[{name:'h.json',column:'ident',data:JSON.stringify(held)},{name:'h.json',column:'race',data:JSON.stringify(held)}];
      var notes=[];await E.buildHeldOut(specs,t,notes,20);
      var res={flags:specs.map(function(s){return [s.skipped,s.skipNote||null];}),notes:notes};
      res.entries=await shared.heldOutEntries(specs);
      process.stdout.write(JSON.stringify(res));
    })();
    """
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
                           str(REPO_ROOT / "assets" / "profiler-shared.js")],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    res = json.loads(done.stdout)
    assert res["flags"][0][0] is True and "identifier" in res["flags"][0][1] and res["flags"][1] == [False, None]
    assert res["notes"] == [res["flags"][0][1]]
    py = held_out_entries(["h.json=ident", "h.json=race"], None, skipped=["ident"])
    for entries in (res["entries"],):
        assert entries[0]["tested"] is False and entries[0]["skip_reason"] == py[0]["skip_reason"]
        assert list(entries[0])[-2:] == ["tested", "skip_reason"] and "tested" not in entries[1]
    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    cmp_js = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    assert "setSkipNotes(currentResult, specs);" in ui and "setSkipNotes(currentCmp, specsA.concat(specsB));" in cmp_js


def test_web_version_constant_matches_the_package_version():
    """#996: one VERSION in profiler-shared.js, equal to faircode.__version__ and pyproject.toml, and
    both views read it from there instead of keeping a literal."""
    import re
    from faircode import __version__
    shared = (REPO_ROOT / "assets" / "profiler-shared.js").read_text(encoding="utf-8")
    assert re.search(r"var VERSION = '([^']+)';", shared).group(1) == __version__
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    for name in ("profiler-ui.js", "profiler-compare.js"):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "var FAIRCODE_VERSION = S.VERSION;" in js and "'2.3.0'" not in js
