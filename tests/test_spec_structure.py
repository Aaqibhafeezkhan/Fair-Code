"""faircode/SPEC.md keeps one section per topic and a complete table of contents (#955)."""

import re
from pathlib import Path

SPEC = (Path(__file__).resolve().parent.parent / "faircode" / "SPEC.md").read_text(encoding="utf-8")


def _sections():
    parts = re.split(r"^## (\d+)\. ", SPEC, flags=re.M)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def test_sections_are_consecutive_and_all_in_the_table_of_contents():
    sections = _sections()
    assert sorted(sections) == list(range(1, max(sections) + 1))
    toc = next(line for line in SPEC.splitlines() if "[1. Detection]" in line)   # the one-line contents
    for number, body in sections.items():
        title = body.split("\n", 1)[0].strip()
        anchor = re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")
        assert f"[{number}. " in toc and f"](#{number}-{anchor})" in toc, (number, title)


def test_proxy_hint_specification_lives_in_its_own_section_not_under_metrics():
    sections = _sections()
    metrics, proxy = sections[3], sections[13]
    assert proxy.startswith("Proxy hints")
    for needle in ("PATH=COLUMN", "--proxy-exact", "NEAR_UNIQUE_SHARE", "held-out", "Copy as CLI command"):
        assert needle in proxy, needle
        assert needle not in metrics, f"{needle!r} belongs in section 13, not under per-dimension metrics"
    assert "section 13" in metrics                # a pointer is left behind
    assert metrics.count("\n") < 60               # the metrics section is back to its own topic
