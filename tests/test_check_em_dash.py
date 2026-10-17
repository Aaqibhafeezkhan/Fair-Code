"""The em-dash check scans every tracked prose-like file type, not only Markdown and code (#1020)."""

import importlib

check_em_dash = importlib.import_module("scripts.check_em_dash")


def test_scan_covers_notebooks_text_toml_json_and_cff():
    for suffix in (".ipynb", ".txt", ".toml", ".cff", ".json", ".md", ".py"):
        assert suffix in check_em_dash.SCAN_EXT, suffix


def test_tracked_notebooks_and_requirements_are_scanned_and_clean():
    scanned = set(check_em_dash._tracked_files())
    assert "requirements.txt" in scanned
    assert any(path.endswith(".ipynb") for path in scanned)
    assert check_em_dash.main() == 0
