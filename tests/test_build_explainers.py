import importlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_explainers_ui_js_stays_in_render_parity_with_build_explainers():
    # assets/explainers-ui.js's renderMarkdown/parseTable is a client-side
    # port of scripts/build_explainers.py's render_markdown/split_row, but
    # nothing exercises it on a real explainer page, so it has silently
    # drifted before (#552, #553). Source-level guard (same idea as
    # test_js_parity's DOM-renderer checks) that the two fixes are present in
    # the JS too.
    js = (REPO_ROOT / "assets" / "explainers-ui.js").read_text(encoding="utf-8")

    # #553: +1 heading offset, capped at h6
    assert "Math.min(headingMatch[1].length + 1, 6)" in js
    # #552: split table rows on unescaped "|" and unescape "\|" -> "|". The split is a scan, not a
    # lookbehind regex (#956); tests/test_js_no_lookbehind.py checks it against Python's re.split.
    assert "const splitUnescaped" in js and "text[i] === '|'" in js
    assert r"replace(/\\\|/g, '|')" in js


def test_build_package_mirror_copies_data_and_markdown(tmp_path, monkeypatch):
    # faircode/_explainers/ ships the package's own copy of explainers/*.md +
    # explainers-data.json (issue #388) - a real `pip install faircode[mcp]`
    # never has the repo-root explainers//assets/ directories on disk, so
    # mcp_server.py's list_explainers/get_explainer need this mirror instead.
    script = importlib.import_module("scripts.build_explainers")

    explainers_dir = tmp_path / "explainers"
    explainers_dir.mkdir()
    (explainers_dir / "sample-topic.md").write_text("# Sample Topic\n", encoding="utf-8")
    data_json = tmp_path / "assets" / "explainers-data.json"
    data_json.parent.mkdir()
    entries = [{"slug": "sample-topic", "title": "Sample Topic"}]
    data_json.write_text(json.dumps(entries), encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_explainers"

    monkeypatch.setattr(script, "EXPLAINERS_DIR", explainers_dir)
    monkeypatch.setattr(script, "DATA_JSON", data_json)
    monkeypatch.setattr(script, "PACKAGE_MIRROR_DIR", mirror_dir)

    script.build_package_mirror(entries)

    assert (mirror_dir / "sample-topic.md").read_text(encoding="utf-8") == "# Sample Topic\n"
    assert json.loads((mirror_dir / "data.json").read_text(encoding="utf-8")) == entries


def test_build_package_mirror_removes_a_stale_slug(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.build_explainers")

    explainers_dir = tmp_path / "explainers"
    explainers_dir.mkdir()
    (explainers_dir / "still-here.md").write_text("# Still Here\n", encoding="utf-8")
    data_json = tmp_path / "assets" / "explainers-data.json"
    data_json.parent.mkdir()
    data_json.write_text("[]", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_explainers"
    mirror_dir.mkdir(parents=True)
    (mirror_dir / "removed-topic.md").write_text("# Gone\n", encoding="utf-8")

    monkeypatch.setattr(script, "EXPLAINERS_DIR", explainers_dir)
    monkeypatch.setattr(script, "DATA_JSON", data_json)
    monkeypatch.setattr(script, "PACKAGE_MIRROR_DIR", mirror_dir)

    script.build_package_mirror([{"slug": "still-here", "title": "Still Here"}])

    assert not (mirror_dir / "removed-topic.md").exists()
    assert (mirror_dir / "still-here.md").is_file()


def test_parse_table_accepts_two_dash_separator_row():
    # explainers/false-positives-vs-false-negatives.md's real separator row
    # is |--|--|--| (2 dashes/cell) - valid GFM, but this parser used to
    # require 3+ dashes and would silently fall through to garbled
    # plain-text lines instead of a real <table> (#324).
    script = importlib.import_module("scripts.build_explainers")
    lines = [
        "| | False Positive | False Negative |",
        "|--|--|--|",
        "| What happens | flags risk | says low risk |",
    ]

    result = script.parse_table(lines, 0)

    assert result is not None
    headers, body_rows, next_index = result
    assert headers == ["", "False Positive", "False Negative"]
    assert body_rows == [["What happens", "flags risk", "says low risk"]]


def test_parse_table_still_accepts_three_dash_separator_row():
    script = importlib.import_module("scripts.build_explainers")
    lines = [
        "| A | B |",
        "|---|---|",
        "| 1 | 2 |",
    ]

    result = script.parse_table(lines, 0)

    assert result is not None
    headers, _body_rows, _next_index = result
    assert headers == ["A", "B"]


def test_parse_table_honors_escaped_pipe_inside_a_cell():
    # A literal pipe in a cell must be written "\|" (GFM) and must not start a
    # new column; the "\" is stripped in the rendered cell. reject-inference.md
    # and base-rate-fallacy.md both hit this (#552).
    script = importlib.import_module("scripts.build_explainers")
    lines = [
        "| Method | Formula |",
        "|---|---|",
        r"| IPW | w(X) = P(S = 1 \| X) |",
    ]

    headers, body_rows, _ = script.parse_table(lines, 0)

    assert headers == ["Method", "Formula"]
    assert body_rows == [["IPW", "w(X) = P(S = 1 | X)"]]


def test_render_markdown_offsets_heading_levels_by_one():
    # The explainer page's hero already renders a real <h1>, so the markdown
    # body's headings are shifted down one level (h1 -> h2, capped at h6).
    # assets/explainers-ui.js's renderMarkdown must match this (#553).
    script = importlib.import_module("scripts.build_explainers")

    html = script.render_markdown("# Top\n\n## Sub\n\n###### Deep\n", set())

    assert '<h2 id="top">Top</h2>' in html
    assert '<h3 id="sub">Sub</h3>' in html
    assert '<h6 id="deep">Deep</h6>' in html   # h6 + 1 stays h6, not h7
