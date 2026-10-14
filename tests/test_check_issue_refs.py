"""scripts/check_issue_refs.py compares roadmap/changelog issue numbers with GitHub (#963)."""

import importlib
import json

script = importlib.import_module("scripts.check_issue_refs")

STATES = {1: "closed", 2: "closed", 3: "open", 4: "open"}


def test_checked_roadmap_items_may_cite_only_closed_issues():
    ok = "- [x] Done work (#1, #2)\n"
    bad = "- [x] Done work (#1, #3)\n"
    assert script.check_roadmap(ok, STATES) == []
    assert script.check_roadmap(bad, STATES) == ["ROADMAP.md:1: checked item cites open issue(s) #3"]


def test_unchecked_roadmap_items_need_an_open_issue_unless_they_cite_none():
    assert script.check_roadmap("- [ ] Todo (#1, #3)\n", STATES) == []          # one open is enough
    assert script.check_roadmap("- [ ] Todo (#1, #2)\n", STATES) == [
        "ROADMAP.md:1: unchecked item cites only closed issue(s) #1, #2"]
    assert script.check_roadmap("- [ ] Submit the paper next year\n", STATES) == []   # no issue cited
    assert script.check_roadmap("- [x] Shipped in PR #999\n", STATES) == []     # not an issue: ignored
    assert script.check_roadmap("Prose mentioning #3 is not a checklist item\n", STATES) == []


def test_changelog_closes_must_name_closed_issues():
    text = "- fixed (closes #1)\n- fixed (closes #1, #3)\n- hmm closes #4 and #2\n- mentions #3 without closing\n"
    assert script.check_changelog(text, STATES) == [
        "CHANGELOG.md:2: 'closes #3' but #3 is still open",
        "CHANGELOG.md:3: 'closes #4' but #4 is still open",
    ]


def test_main_reads_an_issues_file_and_reports(tmp_path, capsys, monkeypatch):
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"number": n, "state": s.upper()} for n, s in STATES.items()]))
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("- [x] a (#3)\n")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("- b (closes #1)\n")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    assert script.main(["--issues", str(issues)]) == 1
    out = capsys.readouterr().out
    assert "checked item cites open issue(s) #3" in out and "1 issue reference(s) disagree" in out
    roadmap.write_text("- [x] a (#1)\n")
    assert script.main(["--issues", str(issues)]) == 0
    assert "agree with GitHub (4 issues checked)" in capsys.readouterr().out


def test_without_gh_the_check_is_skipped_unless_required(capsys, monkeypatch):
    monkeypatch.setattr(script, "load_states", lambda issues_file: None)
    assert script.main([]) == 0 and "skipping the issue-reference check" in capsys.readouterr().out
    assert script.main(["--require"]) == 2
    capsys.readouterr()
