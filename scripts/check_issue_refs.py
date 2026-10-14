#!/usr/bin/env python3
"""Fail when issue numbers cited in docs disagree with GitHub (#963, #981).

The roadmap, changelog, contributors and metrics cite issue numbers by hand and have been wrong
before (two sweeps listed the slow-test marker and the held-out encodings under each other's
numbers). Rules:

  * ROADMAP.md: a checked item (`- [x]`) may cite only closed issues, and an unchecked item
    (`- [ ]`) must cite at least one open issue (unless it cites none at all);
  * CHANGELOG.md: every `closes #N` / `(closes #N, #M)` must name a closed issue;
  * CONTRIBUTORS.md: every `closing issue #N` / `closing #N` must name a closed issue;
  * METRICS.md: every `#N` in a sweep note (`> ...`) must be an existing issue or pull request.

    python3 scripts/check_issue_refs.py                 # asks `gh` for the issue and PR states
    python3 scripts/check_issue_refs.py --issues FILE   # FILE: JSON [{"number": N, "state": "OPEN"|"CLOSED"|"MERGED"}]

Without `gh` (or a network) it prints a note and exits 0, so `make lint` works offline; pass
`--require` to make that an error (CI).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = re.compile(r"#(\d+)")
CLOSES = re.compile(r"closes? ((?:#\d+(?:,? (?:and )?)?)+)", re.IGNORECASE)
CLOSING = re.compile(r"closing\s+(?:issues?\s+)?((?:#\d+(?:(?:,| and|-)\s*)?)+)", re.IGNORECASE)


def load_states(issues_file: str | None) -> dict[int, str] | None:
    """{issue/PR number: "open" | "closed" | "merged"}, or None when GitHub cannot be asked."""
    if issues_file:
        data = json.loads(Path(issues_file).read_text(encoding="utf-8"))
    else:
        if shutil.which("gh") is None:
            return None
        done = subprocess.run(["gh", "issue", "list", "--state", "all", "--limit", "5000", "--json", "number,state"],
                              capture_output=True, text=True, cwd=ROOT)
        if done.returncode != 0:
            return None
        data = json.loads(done.stdout)
        prs = subprocess.run(["gh", "pr", "list", "--state", "all", "--limit", "5000", "--json", "number,state"],
                             capture_output=True, text=True, cwd=ROOT)
        if prs.returncode == 0:
            data.extend(json.loads(prs.stdout))
    return {int(i["number"]): str(i["state"]).lower() for i in data}


def check_roadmap(text: str, states: dict[int, str]) -> list[str]:
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"\s*- \[([ xX])\]", line)
        if not match:
            continue
        cited = [int(n) for n in REF.findall(line) if int(n) in states]
        if not cited:
            continue
        done = match.group(1) != " "
        if done:
            open_ones = [n for n in cited if states[n] == "open"]
            if open_ones:
                problems.append(f"ROADMAP.md:{number}: checked item cites open issue(s) "
                                + ", ".join(f"#{n}" for n in open_ones))
        elif all(states[n] in ("closed", "merged") for n in cited):
            problems.append(f"ROADMAP.md:{number}: unchecked item cites only closed issue(s) "
                            + ", ".join(f"#{n}" for n in cited))
    return problems


def check_changelog(text: str, states: dict[int, str]) -> list[str]:
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        for group in CLOSES.findall(line):
            for ref in REF.findall(group):
                n = int(ref)
                if n in states and states[n] == "open":
                    problems.append(f"CHANGELOG.md:{number}: 'closes #{n}' but #{n} is still open")
    return problems


def check_contributors(text: str, states: dict[int, str]) -> list[str]:
    problems = []
    for match in CLOSING.finditer(text):
        number = text[:match.start()].count("\n") + 1
        for ref in REF.findall(match.group(1)):
            n = int(ref)
            if n in states and states[n] == "open":
                problems.append(f"CONTRIBUTORS.md:{number}: 'closing #{n}' but #{n} is still open")
    return problems


def check_metrics(text: str, states: dict[int, str]) -> list[str]:
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith(">"):
            continue
        for ref in REF.findall(line):
            n = int(ref)
            if n not in states:
                problems.append(f"METRICS.md:{number}: '#{n}' is not an existing issue or pull request")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--issues", metavar="FILE", help="issue states as JSON instead of asking gh")
    parser.add_argument("--require", action="store_true", help="fail when the issue states cannot be fetched")
    args = parser.parse_args(argv)
    states = load_states(args.issues)
    if states is None:
        print("note: could not fetch issue states from GitHub (gh missing or offline); skipping the issue-reference check")
        return 2 if args.require else 0
    problems = []
    for filename, checker in (
        ("ROADMAP.md", check_roadmap),
        ("CHANGELOG.md", check_changelog),
        ("CONTRIBUTORS.md", check_contributors),
        ("METRICS.md", check_metrics),
    ):
        path = ROOT / filename
        if path.is_file():
            problems.extend(checker(path.read_text(encoding="utf-8"), states))
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} issue reference(s) disagree with GitHub")
        return 1
    print(f"OK: issue references in ROADMAP.md, CHANGELOG.md, CONTRIBUTORS.md and METRICS.md agree with GitHub ({len(states):,} issues checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
