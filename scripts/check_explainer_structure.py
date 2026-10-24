#!/usr/bin/env python3
"""Checks that every explainers/*.md has the structure CONTRIBUTING.md section 8 asks for (#1118).

A good explainer has, in this order: a definition, why it matters, a concrete example, detection
code, limitations, related concepts and further reading. It also has at least one `python` fenced
block, at least two Further Reading entries, and the project footer line. Headings are worded
differently from page to page ("Why It Matters" / "Why This Matters", "Limitations" /
"Limitations and Trade-offs"), so each required section lists the heading patterns it accepts.

    python3 scripts/check_explainer_structure.py

Exit code: 0 = every explainer passes, 1 = at least one problem (each is printed with the file and
what is missing).

KNOWN_EXCEPTIONS lists the gaps that already exist, per page, so the check can land before the
pages are fixed. An exception is removed by fixing the page; the check also fails when an exception
is no longer needed, so the list can only shrink.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPLAINERS_DIR = ROOT / "explainers"

# (name, regex matched against the lower-cased "## " heading text), in the required order.
SECTIONS = [
    ("definition", r"definition|core concept"),
    ("why it matters", r"why (it|this) matters"),
    ("concrete example", r"concrete example|real-world (proof|case)|real implementation|tabular proof|the benchmark"),
    ("detection code", r"detect|how to measure"),
    ("limitations", r"limitations"),
    ("related concepts", r"related (concepts|projects)"),
    ("further reading", r"further reading"),
]
SECTION_NAMES = [name for name, _ in SECTIONS]

MIN_FURTHER_READING_ENTRIES = 2
FOOTER_PREFIX = "*Part of [The Fair Code Project]"

# slug -> what that page is allowed to lack for now. "order" excuses a section order that differs
# from CONTRIBUTING.md. Remove an entry when the page is fixed (the check fails on a stale one).
KNOWN_EXCEPTIONS: dict[str, set[str]] = {}

FENCE_RE = re.compile(r"^\s*(```|~~~)")
H2_RE = re.compile(r"^##\s+(.*?)\s*#*\s*$")


def _headings_and_flags(text: str):
    """The (line_index, lower-cased text) of every top-level "## " heading outside code fences,
    plus whether a python fence exists."""
    headings, in_fence, has_python = [], False, False
    for i, line in enumerate(text.splitlines()):
        fence = FENCE_RE.match(line)
        if fence:
            if not in_fence and line.strip().lower().startswith(("```python", "```py", "~~~python")):
                has_python = True
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = H2_RE.match(line)
        if m:
            headings.append((i, m.group(1).strip().lower()))
    return headings, has_python


def _further_reading_entries(text: str) -> int:
    """List items under the last "Further Reading" heading, up to the next heading or rule."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = H2_RE.match(line)
        if m and "further reading" in m.group(1).lower():
            start = i
    if start is None:
        return 0
    count = 0
    for line in lines[start + 1:]:
        if H2_RE.match(line):
            break
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+\S", line):
            count += 1
    return count


def problems_in(text: str) -> set[str]:
    """What is wrong with one explainer, as a set of short names (a section name, "order",
    "python block", "further reading entries" or "footer")."""
    headings, has_python = _headings_and_flags(text)
    found: dict[str, int] = {}
    for name, pattern in SECTIONS:
        for line_no, heading in headings:
            if re.search(pattern, heading):
                found[name] = line_no
                break
    problems = {name for name in SECTION_NAMES if name not in found}
    in_order = [found[name] for name in SECTION_NAMES if name in found]
    if in_order != sorted(in_order):
        problems.add("order")
    if not has_python:
        problems.add("python block")
    if "further reading" in found and _further_reading_entries(text) < MIN_FURTHER_READING_ENTRIES:
        problems.add("further reading entries")
    last = next((line for line in reversed(text.splitlines()) if line.strip()), "")
    if not last.startswith(FOOTER_PREFIX):
        problems.add("footer")
    return problems


def check(directory: Path | None = None, exceptions: dict[str, set[str]] | None = None) -> list[str]:
    """One message per problem across every `*.md` in `directory` (empty when all is well)."""
    directory = EXPLAINERS_DIR if directory is None else directory
    exceptions = KNOWN_EXCEPTIONS if exceptions is None else exceptions
    messages = []
    seen = set()
    for path in sorted(Path(directory).glob("*.md")):
        slug = path.stem
        seen.add(slug)
        found = problems_in(path.read_text(encoding="utf-8"))
        allowed = exceptions.get(slug, set())
        new = found - allowed
        if new:
            messages.append(f"{path.name}: missing or wrong: {', '.join(sorted(new))}")
        stale = allowed - found
        if stale:
            messages.append(
                f"{path.name}: no longer needs the exception for {', '.join(sorted(stale))}; "
                "remove it from KNOWN_EXCEPTIONS in scripts/check_explainer_structure.py"
            )
    for slug in sorted(set(exceptions) - seen):
        messages.append(f"KNOWN_EXCEPTIONS names {slug!r}, but explainers/{slug}.md does not exist")
    return messages


def main() -> int:
    messages = check()
    if messages:
        print("Explainer structure problems (see CONTRIBUTING.md, 'A good explainer should include'):",
              file=sys.stderr)
        for message in messages:
            print(f"  - {message}", file=sys.stderr)
        return 1
    count = len(list(EXPLAINERS_DIR.glob("*.md")))
    print(f"OK: {count} explainers have the required sections, a python block, further reading and the footer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
