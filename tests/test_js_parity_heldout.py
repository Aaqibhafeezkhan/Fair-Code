"""Python/JS parity of held-out columns, join keys, key normalisation and the held-out web control. (split out of test_js_parity.py, #938)."""

import json
import subprocess

import pandas as pd
import pytest

from js_helpers import REPO_ROOT


def test_python_js_proxy_hints_held_out_column_parity(tmp_path):
    """#781: the web proxyHints(..., heldOut) flags a dropped column (race)
    against a surviving proxy (zip_code) like proxy_hints(held_out=...), and
    parseHeldOut enforces the same row-count/collision rules."""
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints

    zip_code = ["111"] * 40 + ["222"] * 40
    race = ["A"] * 40 + ["B"] * 40
    kept = tmp_path / "kept.csv"
    kept.write_text("zip_code,sex\n" + "\n".join(
        f"{z},{'m' if i % 2 else 'f'}" for i, z in enumerate(zip_code)) + "\n", encoding="utf-8")
    full = tmp_path / "full.csv"
    full.write_text("race\n" + "\n".join(race) + "\n", encoding="utf-8")
    short = tmp_path / "short.csv"
    short.write_text("race\nA\nB\n", encoding="utf-8")

    df = pd.read_csv(kept)
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    py = proxy_hints(df, dims, held_out={"race": pd.read_csv(full)["race"]})

    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "function load(p){return E.parseCSV(fs.readFileSync(p,'utf-8'));}"
        "var t=load(process.argv[2]),h=load(process.argv[3]),s=load(process.argv[4]);"
        "var r=E.profile(t,{},{});var out={};"
        "out.hints=E.proxyHints(t,r.dimensions,0.05,{race:E.parseHeldOut(h,'race',t)});"
        "try{E.parseHeldOut(s,'race',t)}catch(e){out.short=e.message}"
        "try{E.parseHeldOut(h,'zip_code',t)}catch(e){out.missing=e.message}"
        "try{E.proxyHints(t,r.dimensions,0)}catch(e){out.alpha=e.message}"
        "process.stdout.write(JSON.stringify(out));"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
         str(kept), str(full), str(short)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(done.stdout)

    py_pair = next(h for h in py if {h["a"], h["b"]} == {"zip_code", "race"})
    js_pair = next(h for h in js["hints"] if {h["a"], h["b"]} == {"zip_code", "race"})
    assert py_pair["chi2"] == js_pair["chi2"] and py_pair["cramers_v"] == js_pair["cramers_v"]
    assert "rows must align 1:1" in js["short"]
    assert "not found" in js["missing"]
    assert "alpha must be in (0, 1]" in js["alpha"]


def test_held_out_rows_control_is_wired_into_profile_and_compare_views():
    """#801-#803: rows (any number, per dataset in compare) replace the old
    single file+column inputs; profiler.html loads the shared control first."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    for element_id in ("heldOutRows", "heldOutAddBtn", "compareHeldOutRowsA",
                       "compareHeldOutAddA", "compareHeldOutRowsB", "compareHeldOutAddB"):
        assert f'id="{element_id}"' in html, element_id
    assert 'id="heldOutFileInput"' not in html
    assert html.index("profiler-heldout.js") < html.index("profiler-ui.js")
    assert html.index("profiler-heldout.js") < html.index("profiler-compare.js")
    ui = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    cmp_js = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")
    assert "E.buildHeldOut(specs, currentTable, heldNotes, currentOpts.max_categorical_card)" in ui
    assert "E.buildHeldOut(specsA, slot.A.table, heldNotes, currentOpts.max_categorical_card)" in cmp_js
    assert "E.buildHeldOut(specsB, slot.B.table, heldNotes, currentOpts.max_categorical_card)" in cmp_js
    control = (REPO_ROOT / "assets" / "profiler-heldout.js").read_text(encoding="utf-8")
    assert ".xlsx" in control and "file.arrayBuffer()" in control


def test_build_held_out_handles_several_columns_formats_and_collisions(tmp_path):
    """#801/#802: buildHeldOut() merges several files (csv, json, and .xlsx via the
    first-sheet reader - stubbed here so no CDN is needed) into one map, and
    rejects a column named twice like the CLI's repeated --proxy-hints-with."""
    main_csv = tmp_path / "main.csv"
    main_csv.write_text("zip\n111\n111\n222\n222\n", encoding="utf-8")
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "globalThis.XLSX={read:function(){return {SheetNames:['S'],Sheets:{S:{}}}},"
        "utils:{sheet_to_json:function(){return [{age:'old'},{age:'old'},{age:'young'},{age:'young'}]}}};"
        "var t=E.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));var out={};"
        "(async function(){"
        "out.map=await E.buildHeldOut(["
        "{name:'a.csv',column:'race',data:'race\\nA\\nA\\nB\\nB\\n'},"
        "{name:'b.json',column:'sex',data:JSON.stringify([{sex:'m'},{sex:'f'},{sex:'m'},{sex:'f'}])},"
        "{name:'c.xlsx',column:'age',data:new ArrayBuffer(1)}],t);"
        "try{await E.buildHeldOut([{name:'a.csv',column:'race',data:'race\\nA\\nA\\nB\\nB\\n'},"
        "{name:'z.csv',column:'race',data:'race\\nA\\nA\\nB\\nB\\n'}],t)}catch(e){out.dup=e.message}"
        "process.stdout.write(JSON.stringify(out));})();"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(main_csv)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert out["map"] == {"race": ["A", "A", "B", "B"], "sex": ["m", "f", "m", "f"],
                          "age": ["old", "old", "young", "young"]}
    assert "already supplied" in out["dup"]


def test_build_held_out_reports_ignored_xlsx_sheets(tmp_path):
    """#816: only the first sheet of a held-out workbook is read; the notes array gets
    an ignored-sheets line, and a column that lives on a later sheet gets an error
    that names the skipped sheets instead of a bare "not found"."""
    main_csv = tmp_path / "main.csv"
    main_csv.write_text("zip\n111\n111\n222\n222\n", encoding="utf-8")
    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "globalThis.XLSX={read:function(){return {SheetNames:['First','Second','Third'],Sheets:{First:{},Second:{},Third:{}}}},"
        "utils:{sheet_to_json:function(){return [{age:'old'},{age:'old'},{age:'young'},{age:'young'}]}}};"
        "var t=E.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));var out={notes:[]};"
        "(async function(){"
        "out.map=await E.buildHeldOut([{name:'c.xlsx',column:'age',data:new ArrayBuffer(1)}],t,out.notes);"
        "try{await E.buildHeldOut([{name:'c.xlsx',column:'race',data:new ArrayBuffer(1)}],t,[])}"
        "catch(e){out.err=e.message}"
        "out.plain=await E.buildHeldOut([{name:'a.csv',column:'race',data:'race\\nA\\nA\\nB\\nB\\n'}],t);"
        "process.stdout.write(JSON.stringify(out));})();"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(main_csv)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert out["map"] == {"age": ["old", "old", "young", "young"]}
    assert out["notes"] == ["c.xlsx: only the first sheet 'First' was read; 'Second', 'Third' ignored"]
    assert "not found" in out["err"] and "'Second', 'Third' ignored" in out["err"]
    assert out["plain"] == {"race": ["A", "A", "B", "B"]}


def test_build_held_out_joins_on_a_key_like_python(tmp_path):
    """#822: a PATH=COLUMN:KEY held-out file is matched on the key, not row order, and
    both engines reject the same bad inputs."""
    pytest.importorskip("pandas")
    from faircode.proxy import parse_held_out_specs

    main_csv = tmp_path / "main.csv"
    main_csv.write_text("id,zip\n1,111\n2,111\n3,222\n4,222\n", encoding="utf-8")
    held_csv = tmp_path / "held.csv"
    held_csv.write_text("id,race\n4,B\n2,A\n3,B\n1,A\n9,Z\n", encoding="utf-8")
    df = pd.read_csv(main_csv)
    py = parse_held_out_specs([f"{held_csv}=race:id"], df, pd.read_csv)
    assert list(py["race"]) == ["A", "A", "B", "B"]

    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "var t=E.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));var out={};"
        "var held=fs.readFileSync(process.argv[3],'utf-8');"
        "(async function(){"
        "out.ok=await E.buildHeldOut([{name:'h.csv',column:'race',key:'id',data:held}],t);"
        "async function err(name,spec,data){try{await E.buildHeldOut([spec],t)}catch(e){out[name]=e.message}}"
        "await err('missing_key',{name:'h.csv',column:'race',key:'nope',data:held});"
        "await err('dup',{name:'h.csv',column:'race',key:'id',data:'id,race\\n1,A\\n1,B\\n2,A\\n3,A\\n4,A\\n'});"
        "await err('unmatched',{name:'h.csv',column:'race',key:'id',data:'id,race\\n1,A\\n2,A\\n'});"
        "await err('positional',{name:'h.csv',column:'race',data:held});"
        "process.stdout.write(JSON.stringify(out));})();"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
         str(main_csv), str(held_csv)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert out["ok"] == {"race": ["A", "A", "B", "B"]}
    assert "not found in the profiled dataset" in out["missing_key"]
    assert "duplicate" in out["dup"]
    assert "no row for 2 key(s)" in out["unmatched"]
    assert "rows must align 1:1" in out["positional"]


def test_build_held_out_composite_and_normalised_keys_match_python(tmp_path):
    """#859: PATH=COLUMN:id+visit joins on two columns, and opt-in normalisation
    (trim, case, leading zeros) matches in both engines; exact matching stays the default."""
    from faircode.proxy import parse_held_out_specs

    main_csv = tmp_path / "main.csv"
    main_csv.write_text("id,visit,zip\n007,v1,111\n007,v2,111\nB2,v1,222\nB2,v2,222\n", encoding="utf-8")
    held_csv = tmp_path / "held.csv"
    held_csv.write_text("id,visit,race\n 7,V2,y\n7,V1,x\nb2,v1,z\nB2 ,v2,w\n", encoding="utf-8")
    df = pd.read_csv(main_csv, dtype=str)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    py = parse_held_out_specs([f"{held_csv}=race:id+visit"], df, read, normalize_keys=True)
    assert list(py["race"]) == ["x", "y", "z", "w"]
    import pytest as _pt
    with _pt.raises(ValueError, match="no row for"):
        parse_held_out_specs([f"{held_csv}=race:id+visit"], df, read)

    script = (
        "require(process.argv[1]);var fs=require('fs');var E=globalThis.FairCodeProfiler;"
        "var t=E.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));var held=fs.readFileSync(process.argv[3],'utf-8');"
        "var out={};(async function(){"
        "out.ok=await E.buildHeldOut([{name:'h.csv',column:'race',key:'id+visit',normalize:true,data:held}],t);"
        "try{await E.buildHeldOut([{name:'h.csv',column:'race',key:'id+visit',data:held}],t)}catch(e){out.exact=e.message}"
        "out.norm=['  A01 ','a01','0042','000','x'].map(E.normalizeKeyText);"
        "out.ws=['\\u001fa\\u001f','\\ufeffa\\ufeff','\\u0085a\\u0085','\\u00a0a\\u3000','\\u180ea'].map(E.normalizeKeyText);"
        "out.num=['42.0','007.00','4.5','42.','0.0','1.05','\\uff14\\uff12','\\uff10\\uff10\\uff17','\\uff21\\uff11','\\u2461','\\u00b2','\\uff14\\uff12\\uff0e\\uff10'].map(E.normalizeKeyText);"
        "out.uni=['e\\u0301','\\u00e9','Stra\\u00dfe','STRASSE','\\u03a3\\u03a3','\\u03c3\\u03c2'].map(E.normalizeKeyText);"
        "process.stdout.write(JSON.stringify(out));})();"
    )
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(main_csv), str(held_csv)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert out["ok"] == {"race": ["x", "y", "z", "w"]}
    assert "no row for" in out["exact"]
    from faircode.proxy import normalize_key_text
    assert out["norm"] == [normalize_key_text(v) for v in ["  A01 ", "a01", "0042", "000", "x"]] == ["a01", "a01", "42", "0", "x"]
    numbers = ["42.0", "007.00", "4.5", "42.", "0.0", "1.05",
               "\uff14\uff12", "\uff10\uff10\uff17", "\uff21\uff11", "\u2461", "\u00b2", "\uff14\uff12\uff0e\uff10"]   # #903, #935
    assert out["num"] == [normalize_key_text(v) for v in numbers] == [
        "42", "7", "4.5", "42.", "0", "1.05", "42", "7", "a1", "2", "2", "42"]
    spaced = ["\x1fa\x1f", "\ufeffa\ufeff", "\x85a\x85", "\xa0a\u3000", "\u180ea"]   # #897
    assert out["ws"] == [normalize_key_text(v) for v in spaced] == ["\x1fa\x1f", "a", "a", "a", "\u180ea"]
    accented = ["e\u0301", "\u00e9", "Stra\u00dfe", "STRASSE", "\u03a3\u03a3", "\u03c3\u03c2"]
    assert out["uni"] == [normalize_key_text(v) for v in accented]
    assert out["uni"][0] == out["uni"][1] and out["uni"][2] == out["uni"][3]


def test_held_out_control_adds_rows_collects_specs_and_validates():
    """Drives the real assets/profiler-heldout.js through a minimal DOM stub:
    rows can be added/removed, filled rows become specs ({name, column, data},
    xlsx as an ArrayBuffer), empty rows are skipped, half-filled rows throw."""
    script = r"""
    function El(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.files=[];
      this.attrs={};}
    El.prototype.appendChild=function(c){this.children.push(c);c.parent=this;return c;};
    El.prototype.removeChild=function(c){this.children.splice(this.children.indexOf(c),1);};
    El.prototype.setAttribute=function(k,v){this.attrs[k]=v;};
    El.prototype.addEventListener=function(t,f){(this.listeners[t]=this.listeners[t]||[]).push(f);};
    El.prototype.click=function(){(this.listeners.click||[]).forEach(function(f){f();});};
    El.prototype.querySelectorAll=function(sel){var out=[];(function walk(n){n.children.forEach(function(c){
      if(c.tag===sel)out.push(c);walk(c);});})(this);return out;};
    global.document={createElement:function(t){return new El(t);}};
    global.window=global;
    require(process.argv[1]);
    var container=new El('div'),addBtn=new El('button');
    var ctl=FairCodeHeldOut.init(container,addBtn,'held-out');
    (async function(){
      var out={rows_initial:container.children.length};
      addBtn.click();addBtn.click();out.rows_after_add=container.children.length;
      function inputs(i){return container.children[i].querySelectorAll('input');}
      function fileOf(name,text){return {name:name,text:async function(){return text;},
        arrayBuffer:async function(){return 'AB:'+name;}};}
      inputs(0)[0].files=[fileOf('a.csv','race\nA\n')];inputs(0)[1].value=' race ';
      inputs(2)[0].files=[fileOf('b.xlsx','')];inputs(2)[1].value='age';
      out.specs=await ctl.collect();   // middle row empty -> skipped
      inputs(1)[1].value='orphan';
      try{await ctl.collect();}catch(e){out.half=e.message;}
      inputs(1)[1].value='';
      container.children[2].children[5].click();       // remove row 3
      out.rows_after_remove=container.children.length;
      ctl.reset();
      out.rows_after_reset=container.children.length;
      out.specs_after_reset=await ctl.collect();
      process.stdout.write(JSON.stringify(out));
    })();
    """
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-heldout.js")],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert out["rows_initial"] == 1 and out["rows_after_add"] == 3
    assert out["specs"] == [
        {"name": "a.csv", "column": "race", "data": "race\nA\n", "file": {"name": "a.csv"}},
        {"name": "b.xlsx", "column": "age", "data": "AB:b.xlsx", "file": {"name": "b.xlsx"}}]
    assert "needs both a file and a column name" in out["half"]
    assert out["rows_after_remove"] == 2
    assert out["rows_after_reset"] == 1 and out["specs_after_reset"] == []


def test_held_out_control_has_a_per_row_encoding_override():
    """#914: each held-out row can override the dataset encoding without changing siblings."""
    script = r"""
    function El(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.files=[];
      this.attrs={};}
    El.prototype.appendChild=function(c){this.children.push(c);c.parent=this;return c;};
    El.prototype.removeChild=function(c){this.children.splice(this.children.indexOf(c),1);};
    El.prototype.setAttribute=function(k,v){this.attrs[k]=v;};
    El.prototype.addEventListener=function(t,f){(this.listeners[t]=this.listeners[t]||[]).push(f);};
    El.prototype.querySelectorAll=function(sel){var out=[];(function walk(n){n.children.forEach(function(c){
      if(c.tag===sel)out.push(c);walk(c);});})(this);return out;};
    global.document={createElement:function(t){return new El(t);}};
    global.window=global;
    var seen=[];
    global.FairCodeProfiler={decodeText:function(_data,encoding){seen.push(encoding);
      return {text:'race\nCafé\n',encoding:encoding};}};
    require(process.argv[1]);
    var container=new El('div'),addBtn=new El('button');
    var ctl=FairCodeHeldOut.init(container,addBtn,'held-out',function(){return 'latin-1';});
    var inputs=container.children[0].querySelectorAll('input');
    var select=container.children[0].querySelectorAll('select')[0];
    inputs[0].files=[{name:'held.csv',arrayBuffer:async function(){return 'bytes';}}];
    inputs[1].value='race';select.value='utf-8';
    ctl.collect().then(function(specs){process.stdout.write(JSON.stringify({seen:seen,spec:specs[0]}));});
    """
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-heldout.js")],
        capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    assert out["seen"] == ["utf-8"]
    assert out["spec"]["encoding"] == "utf-8"
    assert out["spec"]["data"] == "race\nCafé\n"


def test_js_key_normalisation_matches_python_for_every_code_point():
    """#897/#911: normalizeKeyText agrees with normalize_key_text on `c + "a" + c` and on `c`
    for every Unicode code point, so trim and full case folding cannot drift apart. A code point
    that either runtime has not assigned yet (the two ship different Unicode versions) is
    skipped, since only the newer one would fold it (#935, #939)."""
    import unicodedata
    from faircode.proxy import normalize_key_text
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;var out=[];var un=/^\\p{Cn}$/u;"
        "for(var cp=0;cp<=0x10FFFF;cp++){if(cp>=0xD800&&cp<=0xDFFF)continue;var c=String.fromCodePoint(cp);"
        "if(un.test(c)){out.push(null,null);continue;}"
        "out.push(E.normalizeKeyText(c),E.normalizeKeyText(c+'a'+c));}"
        "process.stdout.write(JSON.stringify(out));"
    )
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(done.stdout)
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    bad = []
    for i, cp in enumerate(cps):
        c = chr(cp)
        if js[2 * i] is None or unicodedata.category(c) == "Cn":
            continue
        for got, want in ((js[2 * i], normalize_key_text(c)), (js[2 * i + 1], normalize_key_text(c + "a" + c))):
            if got != want:
                bad.append((hex(cp), got, want))
    assert not bad, bad[:5]


def test_web_held_out_key_accepts_the_cli_tilde_and_bang_markers():
    """#912/#907: `id~` normalises, `id!` stays exact, in parseHeldOut() like
    proxy.resolve_key_marker(); the key box strips them too."""
    from faircode.proxy import parse_held_out_specs
    df = pd.DataFrame({"id": ["A1", "B2"], "x": [1, 2]})
    held = pd.DataFrame({"id": ["a1", "b2"], "race": ["u", "v"]})
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var t={columns:['id','x'],rows:[{id:'A1',x:'1'},{id:'B2',x:'2'}]},"
        "h={columns:['id','race'],rows:[{id:'a1',race:'u'},{id:'b2',race:'v'}]};"
        "function run(key,norm){try{return E.parseHeldOut(h,'race',t,{},key,norm)}catch(e){return e.message}}"
        "process.stdout.write(JSON.stringify({tilde:run('id~',false),bang:run('id!',true),plain:run('id',false),"
        "global:run('id',true),nokey:run('nokey~',false)}));"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out["tilde"] == out["global"] == ["u", "v"]
    assert "no row for" in out["bang"] and "no row for" in out["plain"]
    assert "not found" in out["nokey"]
    py = parse_held_out_specs(["h.csv=race:id~"], df, lambda p: held)
    assert list(py["race"]) == out["tilde"]
    js = (REPO_ROOT / "assets" / "profiler-heldout.js").read_text(encoding="utf-8")
    assert "/[~!]$/.exec(key)" in js


def test_web_held_out_skips_near_unique_columns_like_python():
    """#908: buildHeldOut() drops an ID-like column with the same note proxy.py gives, keeps
    a low-cardinality one, and honours a wider max_categorical_card."""
    from faircode.proxy import near_unique_note
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var rows=[],held=[];for(var i=0;i<100;i++){rows.push({sex:i%2?'m':'f'});held.push({ident:'id'+i,race:'abcd'[i%4]});}"
        "var t={columns:['sex'],rows:rows},h={columns:['ident','race'],rows:held};"
        "var csv=function(){return 'x'};"
        "async function run(card){var notes=[];var out=await E.buildHeldOut(["
        "{name:'h.json',column:'ident',data:JSON.stringify(held)},{name:'h.json',column:'race',data:JSON.stringify(held)}],"
        "t,notes,card);return {cols:Object.keys(out),notes:notes};}"
        "(async function(){process.stdout.write(JSON.stringify({d:await run(),w:await run(200)}));})();"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out["d"]["cols"] == ["race"] and out["d"]["notes"] == [near_unique_note("ident", 100, 20)]
    assert out["w"]["cols"] == ["ident", "race"] and out["w"]["notes"] == []

    # #937: same rule as proxy.py: an all-numeric held-out column (a raw age on a small
    # table) is tested; a text ID column is still skipped with the note.
    script_n = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "var rows=[],held=[];for(var i=0;i<40;i++){rows.push({sex:i%2?'m':'f'});held.push({ident:'id'+i,age:20+(i%30)+Math.floor(i/30)});}"
        "var t={columns:['sex'],rows:rows};"
        "var notes=[];(async function(){var out=await E.buildHeldOut(["
        "{name:'h.json',column:'age',data:JSON.stringify(held)},{name:'h.json',column:'ident',data:JSON.stringify(held)}],"
        "t,notes,20);process.stdout.write(JSON.stringify({cols:Object.keys(out),notes:notes}));})();"
    )
    out = json.loads(subprocess.run(["node", "-e", script_n, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out["cols"] == ["age"] and out["notes"] == [near_unique_note("ident", 40, 20)]
    for name in ("profiler-ui.js", "profiler-compare.js"):
        assert "currentOpts.max_categorical_card)" in (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")


def test_held_out_column_box_accepts_the_cli_encoding_suffix():
    """#924: `race@latin-1` in the column box picks the row's encoding and is cleaned; an
    unknown suffix (a real column containing @) is left alone."""
    script = r"""
    function El(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.files=[];
      this.attrs={};}
    El.prototype.appendChild=function(c){this.children.push(c);c.parent=this;return c;};
    El.prototype.removeChild=function(c){this.children.splice(this.children.indexOf(c),1);};
    El.prototype.setAttribute=function(k,v){this.attrs[k]=v;};
    El.prototype.addEventListener=function(t,f){(this.listeners[t]=this.listeners[t]||[]).push(f);};
    El.prototype.querySelectorAll=function(sel){var out=[];(function walk(n){n.children.forEach(function(c){
      if(c.tag===sel)out.push(c);walk(c);});})(this);return out;};
    global.document={createElement:function(t){return new El(t);}};
    global.window=global;
    var seen=[];
    global.FairCodeProfiler={decodeText:function(_d,encoding){seen.push(encoding);return {text:'x',encoding:encoding};}};
    require(process.argv[1]);
    var container=new El('div'),addBtn=new El('button');
    var ctl=FairCodeHeldOut.init(container,addBtn,'held-out',function(){return 'auto';});
    var inputs=container.children[0].querySelectorAll('input');
    var select=container.children[0].querySelectorAll('select')[0];
    (async function(){
      var out={};
      inputs[0].files=[{name:'h.csv',arrayBuffer:async function(){return 'b';}}];
      inputs[1].value='race@ISO-8859-1';
      var a=await ctl.collect();
      out.a={column:a[0].column,box:inputs[1].value,select:select.value};
      inputs[1].value='email@home';select.value='inherit';
      var b=await ctl.collect();
      out.b={column:b[0].column,select:select.value};
      process.stdout.write(JSON.stringify({out:out,seen:seen}));
    })();
    """
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-heldout.js")],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    res = json.loads(done.stdout)
    assert res["out"]["a"] == {"column": "race", "box": "race", "select": "latin-1"}
    assert res["out"]["b"] == {"column": "email@home", "select": "inherit"}
    assert res["seen"] == ["latin-1", "auto"]


def test_near_unique_share_boundary_and_constant_agree_between_engines():
    """#957: exactly half the rows distinct is skipped, one fewer is kept, in Python and in JS,
    and both engines name the same NEAR_UNIQUE_SHARE."""
    from faircode.proxy import NEAR_UNIQUE_SHARE, parse_held_out_specs
    assert NEAR_UNIQUE_SHARE == 0.5
    js_src = (REPO_ROOT / "assets" / "profiler-engine.js").read_text(encoding="utf-8")
    assert f"var NEAR_UNIQUE_SHARE = {NEAR_UNIQUE_SHARE};" in js_src

    n = 100
    df = pd.DataFrame({"sex": ["m", "f"] * (n // 2)})

    def held(distinct):
        return pd.DataFrame({"tag": [f"t{i % distinct}" for i in range(n)]})

    py = {}
    for distinct in (49, 50, 51):
        notes = []
        kept = parse_held_out_specs(["h.csv=tag"], df, lambda p, d=distinct: held(d), notes=notes, max_categorical_card=20)
        py[distinct] = ("tag" in kept, bool(notes))
    assert py == {49: (True, False), 50: (False, True), 51: (False, True)}

    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;var rows=[];"
        "for(var i=0;i<100;i++)rows.push({sex:i%2?'m':'f'});var t={columns:['sex'],rows:rows};"
        "async function run(d){var held=[];for(var i=0;i<100;i++)held.push({tag:'t'+(i%d)});"
        "var notes=[];var out=await E.buildHeldOut([{name:'h.json',column:'tag',data:JSON.stringify(held)}],t,notes,20);"
        "return [Object.keys(out).indexOf('tag')!==-1,notes.length>0];}"
        "(async function(){var r={};for(var d of [49,50,51])r[d]=await run(d);process.stdout.write(JSON.stringify(r));})();"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert {int(k): tuple(v) for k, v in out.items()} == py


def test_web_held_out_workbook_row_with_an_encoding_gets_the_ignored_notice():
    """#952: an explicit encoding on an .xlsx row (or a non-Auto dataset picker) is flagged by the
    control and buildHeldOut() adds the same note the CLI prints; Auto and inherit-Auto stay quiet."""
    from faircode.loaders_extra import ENCODING_IGNORED_NOTE
    script = r"""
    function El(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.files=[];this.attrs={};}
    El.prototype.appendChild=function(c){this.children.push(c);c.parent=this;return c;};
    El.prototype.removeChild=function(c){this.children.splice(this.children.indexOf(c),1);};
    El.prototype.setAttribute=function(k,v){this.attrs[k]=v;};
    El.prototype.addEventListener=function(t,f){(this.listeners[t]=this.listeners[t]||[]).push(f);};
    El.prototype.querySelectorAll=function(sel){var out=[];(function walk(n){n.children.forEach(function(c){
      if(c.tag===sel)out.push(c);walk(c);});})(this);return out;};
    global.document={createElement:function(t){return new El(t);}};
    global.window=global;
    global.XLSX={read:function(){return {SheetNames:['S'],Sheets:{S:{}}}},
      utils:{sheet_to_json:function(){return [{race:'a'},{race:'b'}]}}};
    require(process.argv[1]);require(process.argv[2]);
    var E=globalThis.FairCodeProfiler;
    var t={columns:['id'],rows:[{id:'1'},{id:'2'}]};
    var dataset='auto';
    var container=new El('div'),addBtn=new El('button');
    var ctl=FairCodeHeldOut.init(container,addBtn,'held-out',function(){return dataset;});
    var inputs=container.children[0].querySelectorAll('input');
    var select=container.children[0].querySelectorAll('select')[0];
    async function run(rowValue,datasetValue){
      inputs[0].files=[{name:'h.xlsx',arrayBuffer:async function(){return new ArrayBuffer(1);}}];
      inputs[1].value='race';select.value=rowValue;dataset=datasetValue;
      var specs=await ctl.collect(),notes=[];await E.buildHeldOut(specs,t,notes);
      return {ignored:specs[0].encodingIgnored,notes:notes};
    }
    (async function(){
      var out={explicit:await run('latin-1','auto'),inherit_picker:await run('inherit','cp1252'),
               quiet:await run('inherit','auto'),row_auto:await run('auto','cp1252')};
      process.stdout.write(JSON.stringify(out));
    })();
    """
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
                           str(REPO_ROOT / "assets" / "profiler-heldout.js")],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(done.stdout)
    note = f"{ENCODING_IGNORED_NOTE} (held-out file h.xlsx)"
    assert out["explicit"] == {"ignored": True, "notes": [note]}
    assert out["inherit_picker"] == {"ignored": True, "notes": [note]}
    assert out["quiet"] == {"notes": []} and out["row_auto"] == {"notes": []}      # the flag is absent unless true


def test_non_ascii_digit_columns_are_screened_the_same_in_both_engines():
    """#967: a held-out column written in Arabic-Indic, full-width or Devanagari digits is not
    numeric, so a near-unique one is skipped as an identifier by the CLI and by the web."""
    from faircode.proxy import parse_held_out_specs
    n = 40
    df = pd.DataFrame({"sex": ["m", "f"] * (n // 2)})
    shifts = {"arabic": 0x0660, "fullwidth": 0xFF10, "devanagari": 0x0966}
    held = {name: [str(i).translate({48 + d: base + d for d in range(10)}) for i in range(n)]
            for name, base in shifts.items()}
    py = {}
    for name in held:
        notes = []
        kept = parse_held_out_specs([f"h.csv={name}"], df, lambda p: pd.DataFrame(held), notes=notes,
                                    max_categorical_card=20)
        py[name] = (name in kept, bool(notes))
    assert py == {name: (False, True) for name in held}
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;var rows=[],held=[];"
        "var shifts={arabic:0x660,fullwidth:0xFF10,devanagari:0x966};"
        "for(var i=0;i<40;i++){rows.push({sex:i%2?'m':'f'});var r={};"
        "Object.keys(shifts).forEach(function(k){r[k]=String(i).replace(/[0-9]/g,function(d){return String.fromCharCode(shifts[k]+Number(d));});});held.push(r);}"
        "var t={columns:['sex'],rows:rows};var res={};"
        "(async function(){for(var k of Object.keys(shifts)){var notes=[];"
        "var out=await E.buildHeldOut([{name:'h.json',column:k,data:JSON.stringify(held)}],t,notes,20);"
        "res[k]=[Object.keys(out).indexOf(k)!==-1,notes.length>0];}"
        "process.stdout.write(JSON.stringify(res));})();"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert {k: tuple(v) for k, v in out.items()} == py


def test_web_held_out_rejects_a_repeated_column_even_when_the_first_was_skipped():
    """#980: like proxy.py, a column given twice is an error whether the first was kept or
    skipped as near-unique."""
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;var rows=[],held=[];"
        "for(var i=0;i<100;i++){rows.push({sex:i%2?'m':'f'});held.push({ident:'id'+i,race:'ab'[i%2]});}"
        "var t={columns:['sex'],rows:rows};var res={};"
        "(async function(){for(var c of ['ident','race']){try{await E.buildHeldOut(["
        "{name:'h.json',column:c,data:JSON.stringify(held)},{name:'h.json',column:c,data:JSON.stringify(held)}],t,[],20);"
        "res[c]='no error';}catch(e){res[c]=e.message;}}"
        "process.stdout.write(JSON.stringify(res));})();"
    )
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js")],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out == {c: f"held-out column '{c}' was already supplied" for c in ("ident", "race")}
