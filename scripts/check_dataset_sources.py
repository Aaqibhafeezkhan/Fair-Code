#!/usr/bin/env python3
"""Checks that each audit's dataset source link agrees between README.md and the audit's scripts (#1208).

The source address of a dataset is written in the README's project block (the `**Dataset:**` line under
`### NN · <Audit folder> - ...`) and again in a comment in the audit's `unfair.py` and `fair.py`. When one
copy is edited and the other is not (the NIJ link in #1175 went dead in one place only), they drift. This
reads every URL in a full-line `#` comment of each script and fails, naming the file, when it is not one of
the URLs on that audit's README `**Dataset:**` line. A script with no URL in its comments is not an error.
It reads files only, so it works offline.

    python3 scripts/check_dataset_sources.py

Exit code: 0 = every script's source link matches the README, 1 = at least one disagreement.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ("unfair.py", "fair.py")

HEADING = re.compile(r"^### \d+ · (.+?) - ", re.MULTILINE)
DATASET_LINE = re.compile(r"^\*\*Dataset:\*\*.*$", re.MULTILINE)
MARKDOWN_URL = re.compile(r"\]\((https?://[^)\s]+)\)")
COMMENT_URL = re.compile(r"https?://[^\s)>\"']+")


def _clean(url: str) -> str:
    return url.rstrip(".,;").rstrip("/")


def readme_sources(readme_text: str) -> dict[str, set[str]]:
    """{audit folder: URLs on its README `**Dataset:**` line} for each numbered project block."""
    sources: dict[str, set[str]] = {}
    headings = list(HEADING.finditer(readme_text))
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(readme_text)
        block = readme_text[heading.end():end]
        next_section = re.search(r"^## ", block, re.MULTILINE)
        if next_section:
            block = block[:next_section.start()]
        line = DATASET_LINE.search(block)
        urls = MARKDOWN_URL.findall(line.group(0)) if line else []
        sources[heading.group(1).strip()] = {_clean(url) for url in urls}
    return sources


def script_sources(script_text: str) -> set[str]:
    """URLs that appear in the script's full-line `#` comments."""
    urls: set[str] = set()
    for line in script_text.splitlines():
        if line.lstrip().startswith("#"):
            urls.update(_clean(url) for url in COMMENT_URL.findall(line))
    return urls


def disagreements(root: Path | None = None) -> list[str]:
    """One message per script comment URL that the README's dataset line for that audit does not have."""
    root = ROOT if root is None else root
    sources = readme_sources((root / "README.md").read_text(encoding="utf-8"))
    if not sources:
        return ["README.md: no '### NN · <Audit> - ...' project blocks found, so nothing was compared"]
    problems = []
    for audit, readme_urls in sources.items():
        folder = root / audit
        if not folder.is_dir():
            problems.append(f"README.md: project block '{audit}' has no folder of that name")
            continue
        for name in SCRIPTS:
            path = folder / name
            if not path.is_file():
                continue
            for url in sorted(script_sources(path.read_text(encoding="utf-8")) - readme_urls):
                listed = ", ".join(sorted(readme_urls)) or "no link"
                problems.append(f"{audit}/{name}: cites {url}, but the README's Dataset line for {audit} has {listed}")
    return problems


def main() -> int:
    problems = disagreements()
    if problems:
        print("Dataset source links disagree between README.md and the audit scripts:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("Dataset source links agree between README.md and the audit scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
