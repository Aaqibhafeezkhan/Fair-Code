#!/usr/bin/env python3
"""Checks that the traction figures agree across README.md, ROADMAP.md and METRICS.md (#1097).

The same numbers (stars, forks, contributors, watching, social reach, countries, audits) are typed by
hand in the README's Traction table, the ROADMAP's "Current traction" row, METRICS.md's badges and the
newest row of its weekly table. When one is edited without the others they drift. This reads each
figure from each place and fails, naming the files and the values, when two of them disagree. It also
checks that the README's "Show N completed items" count equals the number of `- [x]` lines inside that
section. It compares the files with each other, never with GitHub, so it works offline. The explainer
count has its own checker, scripts/check_explainer_count.py.

    python3 scripts/check_traction.py

Exit code: 0 = everything agrees, 1 = at least one disagreement or a figure that could not be found.

The README table calls its contributor figure "External Contributors", which excludes the repository
owner, so it is expected to be exactly one lower than the total used in ROADMAP.md and METRICS.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIGURES = ("stars", "contributors", "forks", "watching", "social", "countries", "audits")

# README row label -> figure; the contributors row is offset by the owner (see the module docstring).
README_ROWS = {
    "GitHub Stars": "stars",
    "External Contributors": "contributors",
    "Forks": "forks",
    "Watching": "watching",
    "Combined Social Reach (Instagram + LinkedIn)": "social",
    "Countries Reached (Website Visitors)": "countries",
    "Code Audits Published": "audits",
}
README_OFFSETS = {"contributors": -1}

ROADMAP_HEADER = "| Stars | Contributors | Forks | Watching | Social Reach | Countries | Audits |"
ROADMAP_COLUMNS = ("stars", "contributors", "forks", "watching", "social", "countries", "audits")

METRICS_BADGES = {"Stars": "stars", "Contributors": "contributors", "Forks": "forks",
                  "Watching": "watching", "Countries": "countries"}
# | Week | Stars | Forks | Watching | Contributors | Social Views | Countries | Issues Closed | Code Audits |
METRICS_ROW_COLUMNS = {1: "stars", 2: "forks", 3: "watching", 4: "contributors", 5: "social",
                       6: "countries", 8: "audits"}

# Observations are (figure, where, value), value being the number as written, e.g. "30K+" or "51".
Observation = tuple[str, str, str]


def _clean(cell: str) -> str:
    """A table cell without markdown noise or a trailing " total"."""
    return re.sub(r"\s+total$", "", cell.replace("*", "").replace("`", "").strip())


def _number(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def readme_observations(text: str) -> tuple[list[Observation], list[str]]:
    found, problems = [], []
    for label, figure in README_ROWS.items():
        m = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([^|\n]+?)\s*\|", text, re.M)
        if not m:
            problems.append(f"README.md: no Traction table row for {label!r}")
            continue
        value = _clean(m.group(1))
        offset = README_OFFSETS.get(figure)
        number = _number(value)
        if offset and number is not None:
            value = str(number - offset)
        found.append((figure, f"README.md ({label})", value))
    return found, problems


def roadmap_observations(text: str) -> tuple[list[Observation], list[str]]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(ROADMAP_HEADER):
            for row in lines[i + 1:i + 4]:
                if row.startswith("|") and re.search(r"\d", row):
                    cells = [_clean(c) for c in row.strip().strip("|").split("|")]
                    return [(fig, f"ROADMAP.md (Current traction, {fig})", cells[idx])
                            for idx, fig in enumerate(ROADMAP_COLUMNS) if idx < len(cells)], []
            break
    return [], ["ROADMAP.md: no 'Current traction' table found"]


def metrics_observations(text: str) -> tuple[list[Observation], list[str]]:
    found, problems = [], []
    for name, figure in METRICS_BADGES.items():
        m = re.search(r"img\.shields\.io/badge/" + name + r"-([^-?]+)-", text)
        if m:
            found.append((figure, f"METRICS.md (badge {name})", _clean(m.group(1).replace("%2B", "+"))))
        else:
            problems.append(f"METRICS.md: no {name} badge")
    rows = [ln for ln in text.splitlines() if re.match(r"\|\s*\d{4}-W\d+", ln)]
    if not rows:
        problems.append("METRICS.md: no weekly row found")
    else:
        row = rows[-1]
        cells = [_clean(c) for c in row.strip().strip("|").split("|")]
        week = cells[0]
        for idx, figure in METRICS_ROW_COLUMNS.items():
            if idx < len(cells) and cells[idx] not in ("-", ""):
                found.append((figure, f"METRICS.md (latest weekly row, {week})", cells[idx]))
    return found, problems


def completed_items(text: str) -> tuple[int | None, int | None]:
    """(claimed, counted) for the README's "Show N completed items" section; None if not found."""
    m = re.search(r"Show (\d+) completed items", text)
    if not m:
        return None, None
    end = text.find("</details>", m.end())
    section = text[m.end():end if end != -1 else len(text)]
    return int(m.group(1)), len(re.findall(r"^- \[x\]", section, re.M))


def check(root: Path = ROOT) -> list[str]:
    """Every disagreement or missing figure, as one message each (empty when all agree)."""
    texts = {}
    for name in ("README.md", "ROADMAP.md", "METRICS.md"):
        path = root / name
        texts[name] = path.read_text(encoding="utf-8") if path.is_file() else ""
    observations: list[Observation] = []
    problems: list[str] = []
    for found, missing in (readme_observations(texts["README.md"]), roadmap_observations(texts["ROADMAP.md"]),
                           metrics_observations(texts["METRICS.md"])):
        observations += found
        problems += missing
    for figure in FIGURES:
        seen = [(where, value) for fig, where, value in observations if fig == figure]
        if len({value for _, value in seen}) > 1:
            detail = "; ".join(f"{where} says {value}" for where, value in seen)
            problems.append(f"{figure} disagree: {detail}")
    claimed, counted = completed_items(texts["README.md"])
    if claimed is None:
        problems.append("README.md: no 'Show N completed items' summary found")
    elif claimed != counted:
        problems.append(f"README.md: 'Show {claimed} completed items' but the section has {counted} '- [x]' lines")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("Traction figures are out of step:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("Traction figures agree across README.md, ROADMAP.md and METRICS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
