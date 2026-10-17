"""One release number everywhere it is written (#1047)."""

import re
from pathlib import Path

from faircode import __version__

ROOT = Path(__file__).resolve().parent.parent


def test_the_version_is_the_same_in_every_file_that_states_it():
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    citation = re.search(r'^version: "([^"]+)"', (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M).group(1)
    web = re.search(r"var VERSION = '([^']+)'", (ROOT / "assets" / "profiler-shared.js").read_text(encoding="utf-8")).group(1)
    assert __version__ == pyproject == citation == web


def test_the_changelog_names_the_current_version_and_keeps_an_unreleased_section():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}] - " in changelog and "## [Unreleased]" in changelog
    assert f"Latest-v{__version__}-" in changelog and f"SemVer-{__version__}-" in changelog
    assert changelog.index("## [Unreleased]") < changelog.index(f"## [{__version__}]")
