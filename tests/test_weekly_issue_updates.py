import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

script = importlib.import_module("scripts.update_weekly_issue")


@pytest.mark.parametrize(
    ("report_state", "existing_issue", "expected"),
    [
        (True, None, "create"),
        (True, 42, "edit"),
        (False, None, "none"),
        (False, 42, "close"),
        (None, None, "none"),
        (None, 42, "none"),
    ],
)
def test_report_state_selects_create_edit_close_or_noop(report_state, existing_issue, expected):
    assert script.choose_action(report_state, existing_issue) == expected


def test_traction_issue_body_preserves_actionable_differences_and_instructions():
    body = script.traction_issue_body(
        "Traction figures are out of step:\n"
        "  - issues_closed: METRICS.md (latest weekly row, 2026-W43) says 28, GitHub says 29\n"
        "  - stars: README=100, GitHub=101\n"
    )

    assert "- issues_closed: METRICS.md (latest weekly row, 2026-W43) says 28, GitHub says 29" in body
    assert "- stars: README=100, GitHub=101" in body
    assert "Update the figures in README.md, ROADMAP.md and METRICS.md together" in body


def test_external_links_issue_body_reuses_the_report_renderer(monkeypatch):
    report_module = importlib.import_module("scripts.check_external_links")
    monkeypatch.setattr(report_module, "render_issue_body", lambda report: f"stale={len(report['stale_ignored'])}", raising=False)
    monkeypatch.setattr(report_module, "report_needs_issue", lambda report: bool(report["stale_ignored"]), raising=False)

    assert script.external_links_issue_body({"stale_ignored": ["one"]}) == "stale=1"
    assert script.external_links_report_needs_issue({"stale_ignored": ["one"]}) is True


def test_external_links_integration_keeps_issue_open_for_stale_ignores():
    report = {
        "checked": 1,
        "ok": [],
        "broken": [],
        "unknown": [],
        "ignored": [],
        "stale_ignored": [{"url": "https://example.org/retired", "reason": "no longer referenced"}],
    }

    body = script.external_links_issue_body(report)

    assert "1 stale ignore-list entry" in body
    assert "https://example.org/retired" not in body
    assert script.external_links_report_needs_issue(report) is True
    assert script.external_links_report_needs_issue({**report, "stale_ignored": []}) is False


@pytest.mark.parametrize(
    ("state", "issues", "expected_action", "expected_command"),
    [
        (True, [], "create", "create"),
        (True, [{"number": 42, "title": "Traction figures out of date (weekly check)"}], "edit", "edit"),
        (False, [], "none", None),
        (False, [{"number": 42, "title": "Traction figures out of date (weekly check)"}], "close", "close"),
        (None, [], "none", None),
    ],
)
def test_update_issue_runs_only_the_selected_github_operation(
    state, issues, expected_action, expected_command
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[:3] == ["gh", "issue", "list"]:
            return SimpleNamespace(stdout=json.dumps(issues))
        if "--body-file" in command:
            body_path = command[command.index("--body-file") + 1]
            assert Path(body_path).read_text(encoding="utf-8") == "report body"
        return SimpleNamespace(stdout="")

    action = script.update_issue("owner/repo", "traction", state, "report body", runner=runner)

    assert action == expected_action
    if expected_command:
        assert calls[-1][2] == expected_command
    assert len(calls) == (0 if state is None else 1 + (expected_command is not None))
