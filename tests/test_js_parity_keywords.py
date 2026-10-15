"""The extra column-name keywords box in the web views: wiring, storage, sync and the restored note. (split out of test_js_parity.py, #938)."""

import json
import subprocess

from js_helpers import REPO_ROOT, SHARED_JS, shared_prologue


def test_keywords_textarea_is_wired_into_both_web_views():
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    assert 'id="keywordsInput"' in html and 'id="compareKeywordsInput"' in html
    assert html.count('data-opt-json="keywords"') == 2
    for name in ("profiler-ui.js", "profiler-compare.js"):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "[data-opt-json]" in js and "checkKeywordsText(" in js
        # #888: remembered across loads, restored on page load, thresholds still reset.
        assert "readStoredKeywords()" in js and "writeStoredKeywords(" in js
        assert js.count("resetThresholdInputs();") >= 2
    shared = SHARED_JS.read_text(encoding="utf-8")   # one stored text and one parse for both views (#983)
    assert "faircode.keywords.v1" in shared and "JSON.parse(raw)" in shared
    # #889: a role=alert message linked to each textarea, filled by setKeywordsError.
    for box, note in (("keywordsInput", "keywordsError"), ("compareKeywordsInput", "compareKeywordsError")):
        assert f'aria-describedby="{note}"' in html and f'id="{note}" role="alert"' in html


def test_keywords_helpers_persist_validate_and_explain_errors():
    """#888/#889: the real storage and validation helpers of both views, run in node
    against a stub localStorage/DOM."""
    script = r"""
    require(process.argv[1]);
    var store={};
    var window={FairCodeProfiler:globalThis.FairCodeProfiler,localStorage:{getItem:function(k){return k in store?store[k]:null;},
      setItem:function(k,v){store[k]=String(v);},removeItem:function(k){delete store[k];}}};
    var note={hidden:true,textContent:''};
    var document={getElementById:function(id){return id==='n'?note:null;}};
    var attrs={};
    var area={getAttribute:function(k){return k==='aria-describedby'?'n':attrs[k]||null;},
      setAttribute:function(k,v){attrs[k]=v;},removeAttribute:function(k){delete attrs[k];}};
    """ + shared_prologue(2) + r"""
    var r={};
    shared.writeStoredKeywords('{"sex":["sesso"]}'); r.stored=shared.readStoredKeywords();
    shared.writeStoredKeywords(''); r.cleared=shared.readStoredKeywords();
    r.ok=shared.checkKeywordsText('{"sex":["sesso"]}'); r.empty=shared.checkKeywordsText('');
    r.badJson=shared.checkKeywordsText('{"sex": [').error;
    r.unknown=shared.checkKeywordsText('{"colour":["x"]}').error;
    r.twoWords=shared.checkKeywordsText('{"sex":["two words"]}').error;
    shared.setKeywordsError(area,'boom'); r.shown=[attrs['aria-invalid'],note.hidden,note.textContent];
    shared.setKeywordsError(area,''); r.hidden=[attrs['aria-invalid']||null,note.hidden,note.textContent];
    window.localStorage.getItem=function(){throw new Error('blocked');};
    r.blocked=shared.readStoredKeywords();
    process.stdout.write(JSON.stringify(r));
    """
    done = subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(SHARED_JS)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    r = json.loads(done.stdout)
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
        assert "announceKeywords(area);" in js and "new CustomEvent(KEYWORDS_EVENT" in SHARED_JS.read_text(encoding="utf-8")
        listener = js.split("document.addEventListener(KEYWORDS_EVENT")[1].split("// Page load")[0]
        assert "ev.detail.source === area.id" in listener and "readStoredKeywords()" in listener and rerun in listener


def test_keywords_restored_note_and_forget_button_are_wired_in_both_views():
    """#930: a restored vocabulary shows a note with a Forget button; editing, syncing or forgetting
    hides it; Forget clears the box, the stored text and the applied options and re-runs."""
    html = (REPO_ROOT / "profiler.html").read_text(encoding="utf-8")
    for pre in ("keywords", "compareKeywords"):
        assert f'id="{pre}Restored" hidden' in html and f'id="{pre}Forget"' in html
    script = r"""
    var note={hidden:true};
    var window={};
    var document={getElementById:function(id){return id==='keywordsRestored'?note:null;}};
    """ + shared_prologue(1) + r"""
    shared.showRestoredNote({id:'keywordsInput'},true);var on=note.hidden;
    shared.showRestoredNote({id:'keywordsInput'},false);
    process.stdout.write(JSON.stringify([on,note.hidden]));
    """
    done = subprocess.run(["node", "-e", script, str(SHARED_JS)],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(done.stdout) == [False, True]
    for name, rerun in (("profiler-ui.js", "reprofile(false);"), ("profiler-compare.js", "maybeCompare(false);")):
        js = (REPO_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "showRestoredNote(area, checked.value !== undefined);" in js
        assert "showRestoredNote(area, false);   // an edit makes it the person's own text" in js
        forget = js.split("area.id.replace('Input', 'Forget')")[1].split("// Page load")[0]
        for needle in ("area.value = '';", "delete currentOpts[area.dataset.optJson];", "writeStoredKeywords('');",
                       "announceKeywords(area);", rerun):
            assert needle in forget, (name, needle)
