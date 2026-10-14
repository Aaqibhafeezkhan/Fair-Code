"""brag-output/ASSETS.md lists every bundled third-party asset (#936)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "brag-output" / "composition" / "assets"
ASSETS_MD = ROOT / "brag-output" / "ASSETS.md"


def _documented_names():
    text = ASSETS_MD.read_text(encoding="utf-8")
    return set(re.findall(r"`([^`]+)`", text))


def test_every_bundled_asset_has_a_row_in_assets_md():
    # A row may name a file, or a group such as `fonts/ibmplexmono-400/500/600/700.woff2`.
    flat = " ".join(_documented_names())
    missing = []
    for path in sorted(p for p in ASSET_DIR.rglob("*") if p.is_file()):
        rel = path.relative_to(ASSET_DIR).as_posix()
        if "/cues/" in rel:
            continue   # analysis output of the music track, covered by the music row
        names = {rel, path.name}
        if path.name.startswith("ibmplexmono-"):   # one row covers the four weights
            names.add("ibmplexmono-400/500/600/700.woff2")
        if not any(name in flat for name in names):
            missing.append(rel)
    assert not missing, f"add these to brag-output/ASSETS.md: {missing}"


def test_assets_md_records_credits_licences_and_open_points():
    text = ASSETS_MD.read_text(encoding="utf-8")
    for needle in ("## Credits", "ende.app", "CC BY 4.0", "Kenney", "CC0", "SIL OFL 1.1", "## Open points"):
        assert needle in text, needle
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert rows and all(line.rstrip().endswith(("| known |", "| check (see open points) |")) for line in rows)
