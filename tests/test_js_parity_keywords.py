"""The extra column-name keywords box in the web views: wiring, storage, sync and the restored note. (split out of test_js_parity.py, #938)."""

import json
import subprocess

from js_helpers import REPO_ROOT


def test_keywords_textarea_is_wired_into_both_web_views():
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    assert 'id="keywordsInput"' in html and 'id="compareKeywordsInput"' in html
    assert html.count('data-opt-json="keywords"') == 2
    for name in ("profiler-ui.js", "profiler-compare.js"):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "[data-opt-json]" in js and "JSON.parse(raw)" in js
        # #888: remembered across loads, restored on page load, thresholds still reset.
        assert "faircode.keywords.v1" in js and "readStoredKeywords()" in js and "writeStoredKeywords(" in js
        assert js.count("resetThresholdInputs();") >= 2
    # #889: a role=alert message linked to each textarea, filled by setKeywordsError.
    for box, note in (("keywordsInput", "keywordsError"), ("compareKeywordsInput", "compareKeywordsError")):
        assert f'aria-describedby="{note}"' in html and f'id="{note}" role="alert"' in html


def test_keywords_helpers_persist_validate_and_explain_errors():
    """#888/#889: the real storage and validation helpers of both views, run in node
    against a stub localStorage/DOM."""
    script = r"""
    var fs=require('fs');
    require(process.argv[1]);var E=globalThis.FairCodeProfiler;
    var out={};
    process.argv.slice(2).forEach(function(file){
      var src=fs.readFileSync(file,'utf-8');
      var helpers=src.slice(src.indexOf('var KEYWORDS_KEY'),src.indexOf('function resetThresholdInputs'));
      var store={};
      var window={localStorage:{getItem:function(k){return k in store?store[k]:null;},
        setItem:function(k,v){store[k]=String(v);},removeItem:function(k){delete store[k];}}};
      var note={hidden:true,textContent:''};
      var document={getElementById:function(id){return id==='n'?note:null;}};
      var attrs={};
      var area={getAttribute:function(k){return k==='aria-describedby'?'n':attrs[k]||null;},
        setAttribute:function(k,v){attrs[k]=v;},removeAttribute:function(k){delete attrs[k];}};
      var api=new Function('E','window','document','area',helpers+
        ';return {read:readStoredKeywords,write:writeStoredKeywords,check:checkKeywordsText,err:setKeywordsError};');
      var h=api(E,window,document,area),r={};
      h.write('{"sex":["sesso"]}'); r.stored=h.read();
      h.write(''); r.cleared=h.read();
      r.ok=h.check('{"sex":["sesso"]}'); r.empty=h.check('');
      r.badJson=h.check('{"sex": [').error;
      r.unknown=h.check('{"colour":["x"]}').error;
      r.twoWords=h.check('{"sex":["two words"]}').error;
      h.err(area,'boom'); r.shown=[attrs['aria-invalid'],note.hidden,note.textContent];
      h.err(area,''); r.hidden=[attrs['aria-invalid']||null,note.hidden,note.textContent];
      window.localStorage.getItem=function(){throw new Error('blocked');};
      r.blocked=h.read();
      out[file.split('/').pop()]=r;
    });
    process.stdout.write(JSON.stringify(out));
    """
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"),
         str(REPO_ROOT / "assets" / "profiler-ui.js"), str(REPO_ROOT / "assets" / "profiler-compare.js")],
        capture_output=True, text=True, encoding="utf-8", check=True)
    results = json.loads(done.stdout)
    assert set(results) == {"profiler-ui.js", "profiler-compare.js"}
    for r in results.values():
        assert r["stored"] == '{"sex":["sesso"]}' and r["cleared"] == ""
        assert r["ok"] == {"value": {"sex": ["sesso"]}} and r["empty"] == {"empty": True}
        assert r["badJson"].startswith("Not valid JSON")
        assert "unknown key" in r["unknown"] and "single word" in r["twoWords"]
        assert r["shown"] == ["true", False, "boom"] and r["hidden"] == [None, True, ""]
        assert r["blocked"] == ""


def test_keywords_boxes_of_both_views_stay_in_sync():
    """#901: an edit in one view announces itself and the other view refills its box from the
    shared stored text and re-applies it; the editing box ignores its own event."""
    for name, rerun in (("profiler-ui.js", "reprofile(false);"), ("profiler-compare.js", "maybeCompare(false);")):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "announceKeywords(area);" in js and "new CustomEvent(KEYWORDS_EVENT" in js
        listener = js.split("document.addEventListener(KEYWORDS_EVENT")[1].split("// Page load")[0]
        assert "ev.detail.source === area.id" in listener and "readStoredKeywords()" in listener and rerun in listener


def test_keywords_restored_note_and_forget_button_are_wired_in_both_views():
    """#930: a restored vocabulary shows a note with a Forget button; editing, syncing or forgetting
    hides it; Forget clears the box, the stored text and the applied options and re-runs."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    for pre in ("keywords", "compareKeywords"):
        assert f'id="{pre}Restored" hidden' in html and f'id="{pre}Forget"' in html
    script = r"""
    var fs=require('fs');
    var out={};
    process.argv.slice(1).forEach(function(file){
      var src=fs.readFileSync(file,'utf-8');
      var helper=src.slice(src.indexOf('function showRestoredNote'),src.indexOf('function resetThresholdInputs'));
      var note={hidden:true};
      var document={getElementById:function(id){return id==='keywordsRestored'?note:null;}};
      var show=new Function('document',helper+';return showRestoredNote;')(document);
      show({id:'keywordsInput'},true);var on=note.hidden;
      show({id:'keywordsInput'},false);out[file.split('/').pop()]=[on,note.hidden];
    });
    process.stdout.write(JSON.stringify(out));
    """
    done = subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-ui.js"),
                           str(REPO_ROOT / "assets" / "profiler-compare.js")],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(done.stdout) == {"profiler-ui.js": [False, True], "profiler-compare.js": [False, True]}
    for name, rerun in (("profiler-ui.js", "reprofile(false);"), ("profiler-compare.js", "maybeCompare(false);")):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "showRestoredNote(area, checked.value !== undefined);" in js
        assert "showRestoredNote(area, false);   // an edit makes it the person's own text" in js
        forget = js.split("area.id.replace('Input', 'Forget')")[1].split("// Page load")[0]
        for needle in ("area.value = '';", "delete currentOpts[area.dataset.optJson];", "writeStoredKeywords('');",
                       "announceKeywords(area);", rerun):
            assert needle in forget, (name, needle)
