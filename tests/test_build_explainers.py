import importlib
import json
import re
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


def test_code_blocks_are_focusable_in_both_renderers():
    script = importlib.import_module("scripts.build_explainers")
    html = script.render_markdown("```python\nprint('ok')\n```\n", set())

    assert '<pre tabindex="0" aria-label="Scrollable code block"><code class="language-python">' in html
    js = (REPO_ROOT / "assets" / "explainers-ui.js").read_text(encoding="utf-8")
    assert '<pre tabindex="0" aria-label="Scrollable code block"><code${codeLang ?' in js


def test_every_generated_code_block_is_keyboard_focusable():
    code_blocks = [
        tag
        for page in (REPO_ROOT / "explainers").glob("*.html")
        for tag in re.findall(r"<pre\b[^>]*>", page.read_text(encoding="utf-8"))
    ]

    assert code_blocks
    assert all('tabindex="0"' in tag and 'aria-label="Scrollable code block"' in tag for tag in code_blocks)


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


def test_render_markdown_gives_every_table_header_a_column_scope():
    # #1172: screen readers tie a cell to its column header only reliably
    # when the <th> has scope="col". assets/explainers-ui.js renders the same.
    script = importlib.import_module("scripts.build_explainers")

    html = script.render_markdown("| Group | Rate |\n| --- | --- |\n| A | 0.5 |\n", set())

    assert '<thead><tr><th scope="col">Group</th><th scope="col">Rate</th></tr></thead>' in html
    assert "<th>" not in html
    js = (REPO_ROOT / "assets" / "explainers-ui.js").read_text(encoding="utf-8")
    assert '`<th scope="col">${inlineMarkdown(cell)}</th>`' in js


def test_render_markdown_offsets_heading_levels_by_one():
    # The explainer page's hero already renders a real <h1>, so the markdown
    # body's headings are shifted down one level (h1 -> h2, capped at h6).
    # assets/explainers-ui.js's renderMarkdown must match this (#553).
    script = importlib.import_module("scripts.build_explainers")

    html = script.render_markdown("# Top\n\n## Sub\n\n###### Deep\n", set())

    assert '<h2 id="top">Top</h2>' in html
    assert '<h3 id="sub">Sub</h3>' in html
    assert '<h6 id="deep">Deep</h6>' in html   # h6 + 1 stays h6, not h7


def test_render_markdown_flushes_lists_quotes_rules_and_code():
    script = importlib.import_module("scripts.build_explainers")
    source = (
        "Paragraph before a list.\n\n"
        "- first item\n- second item\n\n"
        "1. numbered item\n\n"
        "> quoted line\n\n"
        "---\n\n"
        "```python\nprint('ok')\n```\n"
    )

    html = script.render_markdown(source, set())

    assert "<p>Paragraph before a list.</p>" in html
    assert "<ul><li>first item</li><li>second item</li></ul>" in html
    assert "<ol><li>numbered item</li></ol>" in html
    assert "<blockquote><p>quoted line</p></blockquote>" in html
    assert "<hr>" in html
    assert '<pre tabindex="0" aria-label="Scrollable code block"><code class="language-python">print(&#x27;ok&#x27;)</code></pre>' in html


def test_resolve_link_target_handles_unknown_markdown_and_project_paths():
    script = importlib.import_module("scripts.build_explainers")

    assert script.resolve_link_target("unknown-topic.md?view=full#intro", {"known-topic"}) == "unknown-topic.html?view=full#intro"
    assert script.resolve_link_target("../Benefits%20Denial/", set()) == "../index.html#project-benefits"
    assert script.resolve_link_target("../notebooks/example.ipynb", set()) == "../notebooks/example.ipynb"


def test_faq_skips_nonparagraph_content_and_truncates_long_why_answer():
    script = importlib.import_module("scripts.build_explainers")
    why = "First sentence. " + ("additional context " * 40)
    markdown = (
        "## Definition\n\n| Term | Meaning |\n| --- | --- |\n\n"
        "A definition paragraph.\n\n"
        "## Why It Matters\n\n- a list item\n\n"
        f"{why}\n\n## Example\n\nLater text.\n"
    )

    pairs = script._faq(_entry("sample", "Sample Topic"), markdown)

    assert pairs[0] == ("What is Sample Topic?", "A definition paragraph.")
    assert pairs[1][0] == "Why does this matter for fairness?"
    assert len(pairs[1][1]) <= 501
    assert pairs[1][1].startswith("First sentence.")


def test_seo_title_uses_the_short_title_and_rejects_a_long_one():
    # #1105: a <title> over 65 characters is cut in search results.
    script = importlib.import_module("scripts.build_explainers")
    assert script.seo_title({"slug": "x", "title": "Short Title"}) == "Short Title"
    long_entry = {"slug": "x", "title": "A" * 60, "seo_title": "Shorter Title"}
    assert script.seo_title(long_entry) == "Shorter Title"
    import pytest
    with pytest.raises(ValueError, match="seo_title"):
        script.seo_title({"slug": "x", "title": "A" * 60})


def test_every_explainer_title_fits_the_limit():
    script = importlib.import_module("scripts.build_explainers")
    entries = json.loads((REPO_ROOT / "assets" / "explainers-data.json").read_text(encoding="utf-8"))
    for entry in entries:
        assert len(script.seo_title(entry) + script.TITLE_SUFFIX) <= script.TITLE_MAX, entry["slug"]


def test_meta_description_prefers_the_data_file_and_cuts_at_a_sentence():
    # #1104: search results cut a meta description at about 160 characters.
    import pytest
    script = importlib.import_module("scripts.build_explainers")
    short = "A short, clear summary of exactly one sentence that is comfortably inside the allowed range."
    assert script.meta_description({"slug": "x", "summary": short}) == short
    assert script.meta_description({"slug": "x", "summary": "S" * 300, "description": short}) == short
    long = "First sentence that carries the point of the page in a handful of words. " + "Second sentence " * 20
    cut = script.meta_description({"slug": "x", "summary": long})
    assert cut == "First sentence that carries the point of the page in a handful of words."
    one_long_sentence = "word " * 80
    assert script.meta_description({"slug": "x", "summary": one_long_sentence}).endswith("...")
    with pytest.raises(ValueError, match="description"):
        script.meta_description({"slug": "x", "summary": "Too short."})


def test_every_explainer_description_is_between_70_and_160_characters():
    script = importlib.import_module("scripts.build_explainers")
    entries = json.loads((REPO_ROOT / "assets" / "explainers-data.json").read_text(encoding="utf-8"))
    for entry in entries:
        assert script.DESCRIPTION_MIN <= len(script.meta_description(entry)) <= script.DESCRIPTION_MAX, entry["slug"]


# ── The whole build, on a two-explainer fixture (#1110) ──────────────────────────────────────

import pytest  # noqa: E402

SAMPLE_MD = """## The One-Sentence Definition

**Sample topic** is a short, plain definition used to test the generator.

## Why It Matters

It matters because the page needs a body. See [Other Topic](other-topic.md) and [`Benefits Denial/`](../Benefits%20Denial/).

```python
print("hello")
```

| a | b |
|---|---|
| 1 | 2 |

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
"""


def _entry(slug, title, summary=None, **extra):
    return {"slug": slug, "title": title, "subtitle": f"{title} subtitle",
            "summary": summary or f"A clear one sentence summary of {title} that is long enough to be a good meta description.",
            "tags": ["metrics"], **extra}


@pytest.fixture
def site(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.build_explainers")
    entries = [_entry("sample-topic", "Sample Topic"), _entry("other-topic", "Other Topic")]
    (tmp_path / "explainers").mkdir()
    (tmp_path / "assets" / "og").mkdir(parents=True)
    (tmp_path / "assets" / "og-light").mkdir()
    (tmp_path / "assets" / "explainers-data.json").write_text(json.dumps(entries), encoding="utf-8")
    for e in entries:
        (tmp_path / "explainers" / f"{e['slug']}.md").write_text(SAMPLE_MD, encoding="utf-8")
        for folder in ("og", "og-light"):
            (tmp_path / "assets" / folder / f"{e['slug']}.png").write_bytes(b"\x89PNG fixture")
    for name, value in {
        "ROOT": tmp_path, "EXPLAINERS_DIR": tmp_path / "explainers",
        "DATA_JSON": tmp_path / "assets" / "explainers-data.json", "DATA_JS": tmp_path / "assets" / "explainers-data.js",
        "SITEMAP": tmp_path / "sitemap.xml", "LLMS_FULL": tmp_path / "llms-full.txt",
        "OG_DIR": tmp_path / "assets" / "og", "OG_LIGHT_DIR": tmp_path / "assets" / "og-light",
        "PACKAGE_MIRROR_DIR": tmp_path / "faircode" / "_explainers", "CODE_DIR": tmp_path / "explainers" / "code",
    }.items():
        monkeypatch.setattr(script, name, value)
    monkeypatch.setattr(script, "_git_lastmod", lambda relpath: "2026-01-02")
    monkeypatch.setattr(script, "_git_first_commit", lambda relpath: "2026-01-01")
    return script, tmp_path, entries


def test_the_build_writes_every_output_for_every_entry(site, capsys):
    script, root, _ = site
    script.main()
    assert "Generated 2 explainer pages" in capsys.readouterr().out
    for slug in ("sample-topic", "other-topic"):
        assert (root / "explainers" / f"{slug}.html").is_file()
        assert (root / "faircode" / "_explainers" / f"{slug}.md").is_file()
    assert (root / "faircode" / "_explainers" / "data.json").is_file()
    assert (root / "assets" / "explainers-data.js").read_text(encoding="utf-8").startswith("window.FAIR_CODE_EXPLAINERS = [")


def test_a_page_has_the_head_tags_search_and_social_need(site):
    script, root, _ = site
    script.main()
    html = (root / "explainers" / "sample-topic.html").read_text(encoding="utf-8")
    canonical = "https://www.thefaircode.xyz/explainers/sample-topic.html"
    assert "<title>Sample Topic · Fair Code</title>" in html
    assert f'<link rel="canonical" href="{canonical}">' in html
    assert f'<meta property="og:url" content="{canonical}">' in html
    assert 'og:image" content="https://www.thefaircode.xyz/assets/og/sample-topic.png"' in html
    assert "assets/og-light/sample-topic.png" in html
    assert len(re.findall(r"<h1[ >]", html)) == 1
    description = re.search(r'<meta name="description" content="(.*?)"', html).group(1)
    assert 'content="%s"' % description in html.split('og:description')[1][:300]
    assert 70 <= len(description) <= 160


def test_a_page_has_parseable_json_ld_with_the_expected_types(site):
    script, root, _ = site
    script.main()
    html = (root / "explainers" / "sample-topic.html").read_text(encoding="utf-8")
    blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
    assert blocks
    graph = json.loads(blocks[0])["@graph"]
    assert {node["@type"] for node in graph} >= {"DefinedTerm", "BreadcrumbList"}
    term = next(node for node in graph if node["@type"] == "DefinedTerm")
    assert term["name"] == "Sample Topic"


def test_the_markdown_becomes_html_and_internal_links_resolve(site):
    script, root, _ = site
    script.main()
    html = (root / "explainers" / "sample-topic.html").read_text(encoding="utf-8")
    assert '<a href="other-topic.html">Other Topic</a>' in html
    assert "<table" in html and "<pre" in html


def test_every_page_starts_with_a_skip_link_to_the_content(site):
    # #1170 (WCAG 2.4.1): the first focusable element skips the top bar and
    # lands on the hero, which has the matching id and can take focus.
    script, root, entries = site
    script.main()
    for e in entries:
        html = (root / "explainers" / f"{e['slug']}.html").read_text(encoding="utf-8")
        body = html.split('<body class="explainer-page">', 1)[1]
        first_focusable = re.search(r"<(a|button)\b[^>]*>", body).group(0)
        assert first_focusable == '<a class="skip-link" href="#explainer-start">'
        assert '<section class="explainer-hero" id="explainer-start" tabindex="-1">' in body
    css = (REPO_ROOT / "assets" / "explainers.css").read_text(encoding="utf-8")
    assert ".skip-link:focus" in css


def test_the_sitemap_lists_the_site_pages_and_each_explainer_with_lastmod(site):
    script, root, _ = site
    script.main()
    xml = (root / "sitemap.xml").read_text(encoding="utf-8")
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    urls = re.findall(r"<loc>(.*?)</loc>", xml)
    assert urls[:3] == ["https://www.thefaircode.xyz/", "https://www.thefaircode.xyz/profiler.html",
                        "https://www.thefaircode.xyz/benchmark.html"]
    assert "https://www.thefaircode.xyz/explainers/sample-topic.html" in urls
    assert xml.count("<lastmod>2026-01-02</lastmod>") == len(urls)


def test_llms_full_has_a_section_per_entry(site):
    script, root, _ = site
    script.main()
    text = (root / "llms-full.txt").read_text(encoding="utf-8")
    assert text.startswith("# Fair Code - Full Text for LLMs")
    for title in ("Sample Topic", "Other Topic"):
        assert f"# {title}\nURL: https://www.thefaircode.xyz/explainers/" in text
    assert text.count("## The One-Sentence Definition") == 2


def test_the_build_refuses_a_missing_markdown_file(site):
    script, root, _ = site
    (root / "explainers" / "other-topic.md").unlink()
    with pytest.raises(SystemExit, match="Missing markdown file.*other-topic"):
        script.main()


def test_the_build_refuses_a_missing_or_empty_og_image(site):
    script, root, _ = site
    (root / "assets" / "og" / "sample-topic.png").write_bytes(b"")
    with pytest.raises(SystemExit, match="Open Graph.*sample-topic"):
        script.main()
    (root / "assets" / "og" / "sample-topic.png").write_bytes(b"x")
    (root / "assets" / "og-light" / "other-topic.png").unlink()
    with pytest.raises(SystemExit, match="Open Graph.*other-topic"):
        script.main()


def test_the_build_rejects_a_data_entry_without_a_slug(site):
    script, root, entries = site
    del entries[0]["slug"]
    (root / "assets" / "explainers-data.json").write_text(json.dumps(entries), encoding="utf-8")

    with pytest.raises(KeyError, match="slug"):
        script.main()


def test_the_mirror_drops_a_stale_explainer(site):
    script, root, _ = site
    stale = root / "faircode" / "_explainers" / "removed-topic.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    script.main()
    assert not stale.exists()


def test_a_page_title_or_description_out_of_range_stops_the_build(site):
    script, root, entries = site
    entries[0]["title"] = "T" * 70
    (root / "assets" / "explainers-data.json").write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="page title"):
        script.main()
    entries[0]["title"] = "Sample Topic"
    entries[0]["summary"] = "Too short."
    (root / "assets" / "explainers-data.json").write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="meta description"):
        script.main()


# ── The runnable code of each explainer, as a file (#1125) ───────────────────────────────────

CODE_MD = """## Detection Code

Run from the repo root (`PYTHONPATH=. python3 explainers/code/sample-topic.py`; that file is generated from the block below).

```python
# runnable
# slow
import math

print(math.sqrt(16))
```

```python
print("a second block that is not the one named above")
```
"""


def test_extract_code_takes_the_block_after_the_named_path_without_the_markers():
    script = importlib.import_module("scripts.build_explainers")
    code = script.extract_code("sample-topic", CODE_MD)
    assert code.startswith("# Generated from explainers/sample-topic.md by scripts/build_explainers.py")
    assert "import math" in code and "print(math.sqrt(16))" in code
    assert "# runnable" not in code and "# slow" not in code and "second block" not in code
    assert code.endswith("\n") and not code.endswith("\n\n")
    compile(code, "sample-topic.py", "exec")


def test_extract_code_is_none_without_a_pointer_and_an_error_without_a_block():
    script = importlib.import_module("scripts.build_explainers")
    assert script.extract_code("sample-topic", "## Detection Code\n\n```python\nprint(1)\n```\n") is None
    with pytest.raises(ValueError, match="no python block"):
        script.extract_code("sample-topic", "Run explainers/code/sample-topic.py now.\n")


def test_the_build_writes_code_files_and_removes_stale_ones(site):
    script, root, _ = site
    (root / "explainers" / "sample-topic.md").write_text(CODE_MD, encoding="utf-8")
    stale = root / "explainers" / "code" / "removed-topic.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("print('old')\n", encoding="utf-8")
    script.main()
    written = root / "explainers" / "code" / "sample-topic.py"
    assert written.is_file() and "import math" in written.read_text(encoding="utf-8")
    assert not (root / "explainers" / "code" / "other-topic.py").exists()
    assert not stale.exists()


def test_every_code_file_in_the_repo_is_current_and_compiles():
    script = importlib.import_module("scripts.build_explainers")
    pointed = 0
    for md in sorted((REPO_ROOT / "explainers").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        expected = script.extract_code(md.stem, text)
        target = REPO_ROOT / "explainers" / "code" / f"{md.stem}.py"
        if expected is None:
            assert not target.exists(), f"{target.name} has no explainer pointing at it"
            continue
        pointed += 1
        assert target.read_text(encoding="utf-8") == expected, f"{target.name} is stale: run make build-explainers"
        compile(expected, target.name, "exec")
    assert pointed >= 40
