"""scripts/check_broken_links.py catches broken in-page anchors and relative links (#1109)."""

import importlib

import pytest

cbl = importlib.import_module("scripts.check_broken_links")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A temporary repo: write(rel_path, text) creates a file that git 'tracks'."""
    tracked = set()

    def write(rel, text=""):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        tracked.add(rel)
        return path

    monkeypatch.setattr(cbl, "ROOT", tmp_path)
    monkeypatch.setattr(cbl, "_tracked_paths", lambda: set(tracked))
    monkeypatch.setattr(
        cbl, "_tracked_markdown_files",
        lambda: [tmp_path / rel for rel in sorted(tracked) if rel.endswith(".md")],
    )
    write.root = tmp_path
    return write


def _run(capsys):
    code = cbl.main()
    return code, capsys.readouterr().out


def test_clean_repo_passes_and_reports_the_counts(repo, capsys):
    repo("README.md", "# Title\n\nSee [guide](docs/guide.md) and [top](#title).\n")
    repo("docs/guide.md", "# Guide\n")
    code, out = _run(capsys)
    assert code == 0
    assert out.startswith("OK: 2 internal links checked across 2 markdown files")


def test_a_relative_link_to_a_missing_file_fails_with_file_line_and_target(repo, capsys):
    repo("README.md", "# Title\n\nSee [guide](docs/missing.md).\n")
    code, out = _run(capsys)
    assert code == 1
    assert "Broken internal links found:" in out
    assert "README.md:3: [docs/missing.md]" in out
    assert "does not exist" in out


def test_an_in_page_anchor_must_match_a_heading(repo, capsys):
    repo("README.md", "# Title\n\n[ok](#title) and [bad](#nope)\n")
    code, out = _run(capsys)
    assert code == 1
    assert "[#nope] - no matching heading on this page" in out
    assert "[#title]" not in out


def test_an_anchor_in_another_file_must_match_a_heading_there(repo, capsys):
    repo("README.md", "[ok](docs/guide.md#setup) [bad](docs/guide.md#teardown)\n")
    repo("docs/guide.md", "# Guide\n\n## Setup\n")
    code, out = _run(capsys)
    assert code == 1
    assert "no heading 'teardown' in docs/guide.md" in out
    assert "#setup" not in out


def test_a_link_to_a_directory_is_fine(repo, capsys):
    repo("README.md", "[folder](docs/)\n")
    repo("docs/guide.md", "# Guide\n")
    code, _ = _run(capsys)
    assert code == 0


def test_external_links_are_ignored(repo, capsys):
    repo("README.md", "[a](https://example.invalid/x) [b](mailto:a@b.c) [c](ftp://h/f)\n")
    code, _ = _run(capsys)
    assert code == 0


def test_images_are_not_checked_as_links(repo, capsys):
    repo("README.md", "![logo](assets/missing.png)\n")
    code, _ = _run(capsys)
    assert code == 0


def test_links_inside_fenced_code_and_inline_code_are_ignored(repo, capsys):
    repo("README.md", "# T\n\n```md\n[bad](nope.md) [bad2](#nowhere)\n```\n\nUse `[x](gone.md)` here.\n")
    code, _ = _run(capsys)
    assert code == 0


def test_headings_inside_fences_do_not_count_as_anchors(repo, capsys):
    repo("README.md", "# T\n\n```md\n# Fake\n```\n\n[x](#fake)\n")
    code, out = _run(capsys)
    assert code == 1
    assert "[#fake]" in out


def test_a_link_title_is_stripped_from_the_target(repo, capsys):
    repo("README.md", '[guide](docs/guide.md "The guide")\n')
    repo("docs/guide.md", "# Guide\n")
    code, _ = _run(capsys)
    assert code == 0


def test_percent_encoded_paths_are_decoded(repo, capsys):
    repo("README.md", "[a](docs/my%20file.md)\n")
    repo("docs/my file.md", "# X\n")
    code, _ = _run(capsys)
    assert code == 0


def test_blob_permalink_style_paths_are_skipped(repo, capsys):
    repo(".github/TEMPLATE.md", "[x](../blob/main/README.md)\n")
    code, _ = _run(capsys)
    assert code == 0


def test_a_file_on_disk_that_git_does_not_track_is_a_broken_link(repo, capsys):
    repo("README.md", "[x](docs/guide.md)\n")
    (repo.root / "docs").mkdir()
    (repo.root / "docs" / "guide.md").write_text("# G\n", encoding="utf-8")   # not tracked
    code, out = _run(capsys)
    assert code == 1
    assert "docs/guide.md" in out


def test_a_link_that_escapes_the_repo_falls_back_to_the_filesystem(repo, capsys):
    repo("README.md", "[x](../outside.md)\n")
    code, out = _run(capsys)
    assert code == 1
    assert "../outside.md" in out


def test_every_bad_link_is_reported_not_just_the_first(repo, capsys):
    repo("README.md", "[a](one.md)\n[b](two.md)\n")
    code, out = _run(capsys)
    assert code == 1
    assert "README.md:1: [one.md]" in out and "README.md:2: [two.md]" in out


@pytest.mark.parametrize("heading,slug", [
    ("Plain Title", "plain-title"),
    ("COMPAS - Criminal Justice Bias", "compas--criminal-justice-bias"),
    ("Auto-Detection", "auto-detection"),
    ("What is `K`?", "what-is-k"),
    ("**Bold** and [link](x.md)", "bold-and-link"),
])
def test_slugify_matches_github_style_anchors(heading, slug):
    assert cbl._slugify(heading) == slug


def test_duplicate_headings_get_numbered_anchors(repo, capsys):
    repo("README.md", "# Notes\n\n# Notes\n\n[second](#notes-1)\n")
    code, _ = _run(capsys)
    assert code == 0


def test_generated_explainer_mirror_is_excluded_by_default():
    assert cbl.ALLOW_PREFIXES == ("faircode/_explainers/",)


def test_the_real_repo_has_no_broken_links(capsys):
    assert cbl.main() == 0
    capsys.readouterr()
