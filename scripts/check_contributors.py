#!/usr/bin/env python3
"""Fail when CONTRIBUTORS.md disagrees with the merged pull requests on GitHub (#1149).

The file's rule is "ordered by merged PR count, most first, ties broken by commit count, then by earliest
first-merged date", and its counts are kept by hand each sweep. This parses every entry header and its
`**N merged PRs · M commits · first merged YYYY-MM-DD**` line and checks, per section (Core contributors,
Contributors):

  * the merged PR count and the first-merged date against `gh pr list --state merged`, counting only pull
    requests up to the PR the snapshot line names ("covering everything merged through **PR #N**"), since
    anything merged later is expected to be missing until the next sweep;
  * the order of the entries against the rule above;
  * that every merged-PR author up to the snapshot (bots and the maintainer aside) has an entry.

Commit counts come from the contributors API, which counts differently, so they are only used to break ties
in the order, never compared with GitHub. An entry whose header says "account deleted" has no login left on
GitHub (its pull requests show a ghost author), so its counts are listed as skipped, not failed.

    python3 scripts/check_contributors.py                 # asks `gh` for the merged pull requests
    python3 scripts/check_contributors.py --prs FILE      # FILE: gh pr list --state merged --limit 5000 --json number,author,mergedAt
    python3 scripts/check_contributors.py --fix            # refresh known counts/dates and sort entries in place

Without `gh` (or a network) it prints a note and exits 0, so `make lint` works offline; pass `--require` to make
that an error. `--fix` requires GitHub data or `--prs FILE` and preserves entry prose and commit counts. Exit code:
0 = consistent (or skipped), 1 = a stale count, date, order or missing entry.
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
MAINTAINER = "yakew7"
SECTIONS = ("Core contributors", "Contributors")

ENTRY = re.compile(
    r"^### (?P<header>.+)\n\n\*\*(?P<merged>\d+) merged PRs? · (?P<commits>\d+) commits? · "
    r"first merged (?P<first>\d{4}-\d{2}-\d{2})\*\*",
    re.MULTILINE,
)
SNAPSHOT = re.compile(r"covering everything merged through \*\*PR #(\d+)\*\*")
HANDLE = re.compile(r"\[@([^\]]+)\]\(https://github\.com/[^)]+\)|`([^`]+)`")


def parse(text: str) -> tuple[list[dict], int | None]:
    """([entry, ...], snapshot PR number). An entry has section, handle, merged, commits, first, deleted."""
    snapshot = SNAPSHOT.search(text)
    sections = [(m.start(), m.group(1)) for m in re.finditer(r"^## (.+)$", text, re.MULTILINE)]
    entries = []
    for match in ENTRY.finditer(text):
        section = next((name for start, name in reversed(sections) if start < match.start()), "")
        if section not in SECTIONS:
            continue
        header = match.group("header")
        handle = HANDLE.search(header)
        entries.append({
            "section": section,
            "handle": (handle.group(1) or handle.group(2)) if handle else header,
            "merged": int(match.group("merged")),
            "commits": int(match.group("commits")),
            "first": match.group("first"),
            "deleted": "account deleted" in header,
        })
    return entries, int(snapshot.group(1)) if snapshot else None


def merged_by_author(prs: list[dict], through: int | None) -> dict[str, dict]:
    """{login: {"count": N, "first": earliest merge date}} for the human authors of merged PRs up to `through`."""
    authors: dict[str, dict] = {}
    for pr in prs:
        login = (pr.get("author") or {}).get("login") or ""
        if (pr.get("author") or {}).get("is_bot") or login.startswith("app/") or login == "":
            continue
        if through is not None and pr["number"] > through:
            continue
        info = authors.setdefault(login, {"count": 0, "first": "9999-99-99"})
        info["count"] += 1
        info["first"] = min(info["first"], (pr.get("mergedAt") or "")[:10] or info["first"])
    return authors


def check(entries: list[dict], prs: list[dict], through: int | None) -> tuple[list[str], list[str]]:
    """(problems, skipped notes)."""
    problems: list[str] = []
    skipped: list[str] = []
    authors = merged_by_author(prs, through)
    listed = set()
    for entry in entries:
        handle = entry["handle"]
        listed.add(handle.lower())
        if entry["deleted"]:
            skipped.append(f"{handle}: account deleted, counts not comparable")
            continue
        actual = authors.get(next((k for k in authors if k.lower() == handle.lower()), ""), None)
        if actual is None:
            problems.append(f"{handle}: listed with {entry['merged']} merged PRs but GitHub shows none")
            continue
        if actual["count"] != entry["merged"]:
            problems.append(f"{handle}: file says {entry['merged']} merged PRs, GitHub says {actual['count']}")
        if actual["first"] != entry["first"]:
            problems.append(f"{handle}: file says first merged {entry['first']}, GitHub says {actual['first']}")
    for login in sorted(authors, key=str.lower):
        if login.lower() not in listed and login.lower() != MAINTAINER:
            problems.append(f"{login}: {authors[login]['count']} merged PRs on GitHub but no entry in the file")
    for section in SECTIONS:
        rows = [e for e in entries if e["section"] == section]
        want = sorted(rows, key=lambda e: (-e["merged"], -e["commits"], e["first"]))
        for position, (have, expected) in enumerate(zip(rows, want), 1):
            if have is not expected:
                problems.append(f"{section}: entry {position} is {have['handle']} "
                                f"({have['merged']} PRs, {have['commits']} commits, {have['first']}) "
                                f"but the rule puts {expected['handle']} "
                                f"({expected['merged']} PRs, {expected['commits']} commits, {expected['first']}) there")
                break
    return problems, skipped


def fix_document(text: str, prs: list[dict], through: int | None) -> str:
    """Refresh known PR counts/dates and order entries without editing their prose."""
    entries, _ = parse(text)
    authors = merged_by_author(prs, through)
    by_login = {login.lower(): info for login, info in authors.items()}
    refreshed_entries = []
    for entry in entries:
        refreshed = dict(entry)
        actual = by_login.get(entry["handle"].lower())
        if actual is not None and not entry["deleted"]:
            refreshed["merged"] = actual["count"]
            refreshed["first"] = actual["first"]
        refreshed_entries.append(refreshed)

    section_heads = [(m.start(), m.group(1)) for m in re.finditer(r"^## (.+)$", text, re.MULTILINE)]
    entry_matches = list(ENTRY.finditer(text))
    replacements = []
    stats_pattern = re.compile(r"\*\*\d+ merged PRs? · \d+ commits? · first merged \d{4}-\d{2}-\d{2}\*\*")
    for section_index, (section_start, section) in enumerate(section_heads):
        if section not in SECTIONS:
            continue
        section_end = section_heads[section_index + 1][0] if section_index + 1 < len(section_heads) else len(text)
        matches = [m for m in entry_matches if section_start < m.start() < section_end]
        if not matches:
            continue
        start = matches[0].start()
        section_entries = [entry for entry in refreshed_entries if entry["section"] == section]
        blocks = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else section_end
            block = text[match.start():end]
            entry = section_entries[index]
            stats = (f"**{entry['merged']} merged PR{'s' if entry['merged'] != 1 else ''} · "
                     f"{entry['commits']} commit{'s' if entry['commits'] != 1 else ''} · "
                     f"first merged {entry['first']}**")
            block, count = stats_pattern.subn(stats, block, count=1)
            if count != 1:
                raise ValueError(f"could not locate statistics line for {entry['handle']}")
            blocks.append((entry, block))
        blocks.sort(key=lambda item: (-item[0]["merged"], -item[0]["commits"], item[0]["first"]))
        replacements.append((start, section_end, "".join(block for _, block in blocks)))

    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def fetch_prs() -> list[dict] | None:
    if shutil.which("gh") is None:
        return None
    done = subprocess.run(["gh", "pr", "list", "--state", "merged", "--limit", "5000",
                           "--json", "number,author,mergedAt"], capture_output=True, text=True)
    return json.loads(done.stdout) if done.returncode == 0 else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file", type=Path, default=ROOT / "CONTRIBUTORS.md")
    parser.add_argument("--prs", metavar="FILE", help="merged pull requests as JSON instead of asking gh")
    parser.add_argument("--require", action="store_true", help="fail instead of skipping when GitHub cannot be reached")
    parser.add_argument("--fix", action="store_true", help="rewrite known PR counts/dates and sort contributor entries")
    args = parser.parse_args(argv)

    text = args.file.read_text(encoding="utf-8")
    entries, through = parse(text)
    if not entries:
        print(f"error: no contributor entries found in {args.file}")
        return 1
    prs = json.loads(Path(args.prs).read_text(encoding="utf-8")) if args.prs else fetch_prs()
    if prs is None:
        if args.fix:
            print("error: --fix needs merged pull request data from GitHub or --prs FILE")
            return 2
        print("note: could not fetch merged pull requests from GitHub (gh missing or offline); "
              "skipping the contributors check")
        return 1 if args.require else 0
    if args.fix:
        fixed = fix_document(text, prs, through)
        if fixed != text:
            args.file.write_text(fixed, encoding="utf-8")
            print(f"updated: {args.file}")
        text = fixed
        entries, through = parse(text)
    if through is None:
        print("note: no 'covering everything merged through **PR #N**' snapshot line found; counting every merged PR")

    problems, skipped = check(entries, prs, through)
    for note in skipped:
        print(f"skipped: {note}")
    if problems:
        print(f"CONTRIBUTORS.md disagrees with GitHub ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {len(entries)} entries match the merged pull requests through #{through} and are in rule order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
