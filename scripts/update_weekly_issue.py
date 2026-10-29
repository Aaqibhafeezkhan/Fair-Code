"""Update the tracking issue for a weekly Fair Code report."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPORTS = {
    "external-links": {
        "title": "Dead external links (weekly check)",
        "close_comment": "The weekly check found no broken links or stale ignore entries.",
    },
    "traction": {
        "title": "Traction figures out of date (weekly check)",
        "close_comment": "The weekly check found every traction figure matching GitHub.",
    },
}


def choose_action(report_state: bool | None, existing_issue: int | None) -> str:
    """Return create/edit/close/none for a report state and matching open issue."""
    if report_state is None:
        return "none"
    if report_state:
        return "edit" if existing_issue is not None else "create"
    return "close" if existing_issue is not None else "none"


def traction_issue_body(stderr: str) -> str:
    """Build the existing traction report body from check_traction stderr."""
    lines = [
        "The weekly `check_traction.py --live` run found figures that differ from GitHub or from each other:",
        "",
    ]
    for line in stderr.splitlines()[1:]:
        lines.append(f"- {line[4:]}" if line.startswith("  - ") else line)
    lines += [
        "",
        "Update the figures in README.md, ROADMAP.md and METRICS.md together, then run `python3 scripts/check_traction.py --live`. "
        "This issue closes itself on the next weekly run once they match.",
    ]
    return "\n".join(lines) + "\n"


def external_links_issue_body(report: dict) -> str:
    """Use the external-link report's tested renderer, including its stale count."""
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    report_module = importlib.import_module("scripts.check_external_links")
    return report_module.render_issue_body(report)


def external_links_report_needs_issue(report: dict) -> bool:
    """Use the external-link report's tested issue-state rule, including stale ignores."""
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    report_module = importlib.import_module("scripts.check_external_links")
    return report_module.report_needs_issue(report)


def update_issue(
    repo: str,
    kind: str,
    report_state: bool | None,
    body: str = "",
    runner=subprocess.run,
) -> str:
    """Apply the report's open/edit/close decision through the GitHub CLI."""
    if report_state is None:
        return "none"

    config = REPORTS[kind]
    existing = runner(
        ["gh", "issue", "list", "--repo", repo, "--state", "open", "--search",
         f'"{config["title"]}" in:title', "--json", "number,title"],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = [issue["number"] for issue in json.loads(existing.stdout)
               if issue["title"] == config["title"]]
    issue_number = matches[0] if matches else None
    action = choose_action(report_state, issue_number)

    if action in {"create", "edit"}:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as file:
            file.write(body)
            body_path = Path(file.name)
        try:
            if action == "create":
                command = ["gh", "issue", "create", "--repo", repo, "--title", config["title"],
                           "--label", "documentation", "--body-file", str(body_path)]
            else:
                command = ["gh", "issue", "edit", str(issue_number), "--repo", repo,
                           "--body-file", str(body_path)]
            runner(command, check=True)
        finally:
            body_path.unlink(missing_ok=True)
    elif action == "close":
        runner(
            ["gh", "issue", "close", str(issue_number), "--repo", repo,
             "--reason", "completed", "--comment", config["close_comment"]],
            check=True,
        )
    return action


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=REPORTS)
    parser.add_argument("report", type=Path, help="external-links JSON report or traction status file")
    parser.add_argument("details", type=Path, nargs="?", help="traction stderr file")
    args = parser.parse_args(argv)
    repo = os.environ["GITHUB_REPOSITORY"]

    if args.kind == "external-links":
        report = json.loads(args.report.read_text(encoding="utf-8"))
        state = external_links_report_needs_issue(report)
        body = external_links_issue_body(report)
    else:
        if args.details is None:
            parser.error("traction requires a status file and stderr file")
        status = args.report.read_text(encoding="utf-8").strip()
        state = True if status == "1" else False if status == "0" else None
        body = traction_issue_body(args.details.read_text(encoding="utf-8")) if state else ""

    update_issue(repo, args.kind, state, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
