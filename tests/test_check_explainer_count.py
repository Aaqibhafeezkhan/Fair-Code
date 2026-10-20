import importlib
from pathlib import Path

check = importlib.import_module("scripts.check_explainer_count")


def _site(root: Path, slugs, readme_slugs, llms_slugs) -> Path:
    (root / "explainers").mkdir()
    for slug in slugs:
        (root / "explainers" / f"{slug}.md").write_text("# x\n", encoding="utf-8")
    (root / "README.md").write_text(
        "".join(f"- [{s}](explainers/{s}.md)\n" for s in readme_slugs), encoding="utf-8"
    )
    (root / "llms.txt").write_text(
        "".join(f"- [{s}](https://www.thefaircode.xyz/explainers/{s}.html)\n" for s in llms_slugs),
        encoding="utf-8",
    )
    return root


def test_every_explainer_listed_passes(tmp_path):
    root = _site(tmp_path, ["a", "b"], ["a", "b"], ["b", "a"])
    assert check.missing_listings(root) == []


def test_unlisted_explainer_is_named_per_file(tmp_path):
    root = _site(tmp_path, ["a", "b", "c"], ["a"], ["a", "b"])
    assert check.missing_listings(root) == [("README.md", ["b", "c"]), ("llms.txt", ["c"])]


def test_readme_accepts_html_links(tmp_path):
    root = _site(tmp_path, ["a"], [], ["a"])
    (root / "README.md").write_text("[a](explainers/a.html)\n", encoding="utf-8")
    assert check.missing_listings(root) == []


def test_repository_lists_every_explainer():
    assert check.missing_listings() == []


def test_main_fails_and_names_the_missing_slug(tmp_path, monkeypatch, capsys):
    root = _site(tmp_path, ["a", "new-one"], ["a"], ["a", "new-one"])
    monkeypatch.setattr(check, "ROOT", root)
    monkeypatch.setattr(check, "EXPLAINERS_DIR", root / "explainers")
    monkeypatch.setattr(check, "CHECKS", [])
    assert check.main() == 1
    assert "README.md: does not list new-one" in capsys.readouterr().err
