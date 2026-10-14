#!/usr/bin/env python3
"""Fail when the issue numbers cited in the roadmap and changelog disagree with GitHub (#963).

The roadmap, changelog and metrics cite issue numbers by hand and have been wrong before (two
sweeps listed the slow-test marker and the held-out encodings under each other's numbers). Two
rules, applied to issues only (a number that is not an issue, such as a pull request, is ignored):

  * ROADMAP.md: a checked item (`- [x]`) may cite only closed issues, and an unchecked item
    (`- [ ]`) must cite at least one open issue (unless it cites none at all);
  * CHANGELOG.md: every `closes #N` / `(closes #N, #M)` must name a closed issue.

    python3 scripts/check_issue_refs.py                 # asks `gh` for the issue states
    python3 scripts/check_issue_refs.py --issues FILE   # FILE: JSON [{"number": N, "state": "OPEN"|"CLOSED"}]

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


def load_states(issues_file: str | None) -> dict[int, str] | None:
    """{issue number: "open" | "closed"}, or None when GitHub cannot be asked."""
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
        elif all(states[n] == "closed" for n in cited):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--issues", metavar="FILE", help="issue states as JSON instead of asking gh")
    parser.add_argument("--require", action="store_true", help="fail when the issue states cannot be fetched")
    args = parser.parse_args(argv)
    states = load_states(args.issues)
    if states is None:
        print("note: could not fetch issue states from GitHub (gh missing or offline); skipping the issue-reference check")
        return 2 if args.require else 0
    problems = (check_roadmap((ROOT / "ROADMAP.md").read_text(encoding="utf-8"), states)
                + check_changelog((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), states))
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} issue reference(s) disagree with GitHub")
        return 1
    print(f"OK: issue references in ROADMAP.md and CHANGELOG.md agree with GitHub ({len(states):,} issues checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
