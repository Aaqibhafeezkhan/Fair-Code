import importlib
import tempfile
from pathlib import Path

import pytest
from PIL import Image
import json
import shutil

_VALID_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="32" y="18" width="13" height="64" fill="#14171A"/>
  <rect x="32" y="18" width="42" height="13" fill="#14171A"/>
  <rect x="32" y="44" width="36" height="13" fill="#4F7A5B"/>
</svg>
"""


def test_parse_mark_accepts_the_real_logo_svg():
    script = importlib.import_module("scripts.generate_favicons")

    spec = script._parse_mark(script.ROOT / "assets" / "icons" / "logo.svg")

    assert spec["size"] == 100.0
    assert spec["mark_fill"] == "#14171A"
    assert spec["accent_fill"] == "#4F7A5B"


def test_parse_mark_rejects_non_square_viewbox(tmp_path):
    script = importlib.import_module("scripts.generate_favicons")
    bad_svg = tmp_path / "bad.svg"
    bad_svg.write_text(
        _VALID_SVG.replace('viewBox="0 0 100 100"', 'viewBox="0 0 100 80"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected a square viewBox starting at 0,0"):
        script._parse_mark(bad_svg)


def test_parse_mark_rejects_viewbox_not_at_origin(tmp_path):
    script = importlib.import_module("scripts.generate_favicons")
    bad_svg = tmp_path / "bad.svg"
    bad_svg.write_text(
        _VALID_SVG.replace('viewBox="0 0 100 100"', 'viewBox="10 0 100 100"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected a square viewBox starting at 0,0"):
        script._parse_mark(bad_svg)


def test_parse_mark_rejects_a_fourth_rect(tmp_path):
    script = importlib.import_module("scripts.generate_favicons")
    bad_svg = tmp_path / "bad.svg"
    bad_svg.write_text(
        _VALID_SVG.replace(
            "</svg>",
            '  <rect x="0" y="0" width="10" height="10" fill="#000000"/>\n</svg>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer exactly three rects"):
        script._parse_mark(bad_svg)


def test_parse_mark_rejects_mismatched_stem_and_bar_fill(tmp_path):
    script = importlib.import_module("scripts.generate_favicons")
    bad_svg = tmp_path / "bad.svg"
    bad_svg.write_text(
        _VALID_SVG.replace(
            '<rect x="32" y="18" width="42" height="13" fill="#14171A"/>',
            '<rect x="32" y="18" width="42" height="13" fill="#000000"/>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stem and top bar are expected to share one fill color"):
        script._parse_mark(bad_svg)


def test_generate_favicons(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.generate_favicons")

    monkeypatch.setattr(script, "ICONS_DIR", tmp_path)
    monkeypatch.setattr(script, "LOGO_SVG", script.ROOT / "assets" / "icons" / "logo.svg")

    script.main()

    expected = {
        "favicon-16x16.png": (16, 16),
        "favicon-32x32.png": (32, 32),
        "favicon.ico": (32, 32),
        "apple-touch-icon.png": (180, 180),
        "icon-192.png": (192, 192),
        "icon-512.png": (512, 512),
    }

    for filename in expected:
        output = tmp_path / filename
        assert output.is_file(), f"Expected {filename} to be generated"
        assert output.stat().st_size > 0, f"{filename} is empty"

        with Image.open(output) as image:
            assert image.width > 0
            assert image.height > 0

        with Image.open(output) as image:
            assert image.size == expected[filename]


def test_render_card_writes_a_1200x630_png_for_both_themes(tmp_path, monkeypatch):
    # The fast-suite check (#933): one card through the same render path as main(), into a
    # temp dir; the full set of cards is rendered by the slow test below.
    script = importlib.import_module("scripts.generate_og_images")
    monkeypatch.setitem(script.THEMES["dark"], "out_dir", tmp_path / "og")
    monkeypatch.setitem(script.THEMES["light"], "out_dir", tmp_path / "og-light")
    for colors in script.THEMES.values():
        colors["out_dir"].mkdir(parents=True, exist_ok=True)

    script.render_card_both_themes("home", kicker="Kicker", title="A title that is long enough to wrap "
                                   "onto a second line of the card", subtitle="A subtitle.")

    for directory in (tmp_path / "og", tmp_path / "og-light"):
        output = directory / "home.png"
        assert output.is_file() and output.stat().st_size > 0
        with Image.open(output) as image:
            assert image.size == (1200, 630)


@pytest.mark.slow   # renders every explainer card in both themes (about 6 s); CI runs it (#933)
def test_generate_og_images(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.generate_og_images")

    # main() computes each out_dir's relative_to(script.ROOT) for its summary print,
    # so the scratch dir has to live under script.ROOT - it can't just be tmp_path.
    # mkdtemp (not tmp_path.name, which repeats across separate pytest invocations)
    # creates it atomically with a unique name, so concurrent runs can't collide.
    test_root = Path(tempfile.mkdtemp(dir=script.ROOT, prefix=".tmp-test-"))

    try:
        monkeypatch.setitem(script.THEMES["dark"], "out_dir", test_root / "og")
        monkeypatch.setitem(script.THEMES["light"], "out_dir", test_root / "og-light")

        script.main()

        expected = ["home.png", "profiler.png"]

        for entry in json.loads(
            script.DATA_JSON.read_text(encoding="utf-8")
        ):
            expected.append(f"{entry['slug']}.png")

        for directory in (test_root / "og", test_root / "og-light"):
            for filename in expected:
                output = directory / filename
                assert output.is_file(), f"Expected {output} to be generated"
                assert output.stat().st_size > 0, f"{output} is empty"

                with Image.open(output) as image:
                    assert image.size == (1200, 630)
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def _scratch_themes(script, monkeypatch, tmp_path):
    monkeypatch.setitem(script.THEMES["dark"], "out_dir", tmp_path / "og")
    monkeypatch.setitem(script.THEMES["light"], "out_dir", tmp_path / "og-light")
    for colors in script.THEMES.values():
        colors["out_dir"].mkdir(parents=True, exist_ok=True)


def test_an_existing_card_is_skipped_unless_forced(tmp_path, monkeypatch):
    # #1144: PNG bytes differ per Pillow/zlib build, so only a missing PNG is rendered by default.
    script = importlib.import_module("scripts.generate_og_images")
    _scratch_themes(script, monkeypatch, tmp_path)
    kwargs = dict(kicker="Kicker", title="A title", subtitle="A subtitle.")

    assert script.render_card_both_themes("one", force=False, **kwargs) == 2
    dark, light = tmp_path / "og" / "one.png", tmp_path / "og-light" / "one.png"
    dark.write_bytes(b"sentinel")                      # not a valid PNG: a re-render would replace it
    light.unlink()

    assert script.render_card_both_themes("one", force=False, **kwargs) == 1
    assert dark.read_bytes() == b"sentinel" and light.is_file()
    assert script.render_card_both_themes("one", force=True, **kwargs) == 2
    assert dark.read_bytes() != b"sentinel"


def test_main_leaves_existing_cards_alone_and_force_rewrites_them(tmp_path, monkeypatch, capsys):
    script = importlib.import_module("scripts.generate_og_images")
    test_root = Path(tempfile.mkdtemp(dir=script.ROOT, prefix=".tmp-test-"))
    try:
        monkeypatch.setitem(script.THEMES["dark"], "out_dir", test_root / "og")
        monkeypatch.setitem(script.THEMES["light"], "out_dir", test_root / "og-light")
        monkeypatch.setattr(script, "DATA_JSON", test_root / "explainers.json")
        script.DATA_JSON.write_text(json.dumps([{"slug": "first", "title": "First", "summary": "One."}]), encoding="utf-8")

        script.main()
        assert "Wrote 8 of 8" in capsys.readouterr().out
        sentinel = test_root / "og" / "first.png"
        sentinel.write_bytes(b"sentinel")

        script.DATA_JSON.write_text(json.dumps([{"slug": "first", "title": "First", "summary": "One."},
                                                {"slug": "second", "title": "Second", "summary": "Two."}]), encoding="utf-8")
        script.main()
        assert "Wrote 2 of 10" in capsys.readouterr().out
        assert sentinel.read_bytes() == b"sentinel"
        assert (test_root / "og" / "second.png").is_file() and (test_root / "og-light" / "second.png").is_file()

        script.main(force=True)
        assert "Wrote 10 of 10" in capsys.readouterr().out
        assert sentinel.read_bytes() != b"sentinel"
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
