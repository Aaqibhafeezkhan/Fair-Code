"""The em-dash check scans every tracked prose-like file type, not only Markdown and code (#1020)."""

import importlib
import subprocess

import pytest

check_em_dash = importlib.import_module("scripts.check_em_dash")


def test_scan_covers_notebooks_text_toml_json_and_cff():
    for suffix in (".ipynb", ".txt", ".toml", ".cff", ".json", ".md", ".py"):
        assert suffix in check_em_dash.SCAN_EXT, suffix


def test_tracked_notebooks_and_requirements_are_scanned_and_clean():
    scanned = set(check_em_dash._tracked_files())
    assert "requirements.txt" in scanned
    assert any(path.endswith(".ipynb") for path in scanned)
    assert check_em_dash.main() == 0


@pytest.fixture
def temp_git_repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_tracked_em_dash_reports_path_line_and_ignores_untracked_file(
    temp_git_repo, capsys
):
    tracked = temp_git_repo / "notes.md"
    tracked.write_text("clean line\nfound an \u2014 em dash\n", encoding="utf-8")
    (temp_git_repo / "untracked.md").write_text(
        "untracked \u2014 text\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "notes.md"], cwd=temp_git_repo, check=True)

    assert check_em_dash.main() == 1

    output = capsys.readouterr().err
    assert "notes.md:2:10: found an \u2014 em dash" in output
    assert "1 em dash(es) across 1 file(s)." in output
    assert "untracked.md" not in output


def test_allowlisted_tracked_path_passes(temp_git_repo, capsys):
    (temp_git_repo / "CONTRIBUTING.md").write_text(
        "rule example: \u2014\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "CONTRIBUTING.md"], cwd=temp_git_repo, check=True)

    assert list(check_em_dash._tracked_files()) == []
    assert check_em_dash.main() == 0
    assert (
        capsys.readouterr().out == "OK: no em dashes in tracked source/prose files.\n"
    )


def test_tracked_binary_file_is_skipped(temp_git_repo, capsys):
    (temp_git_repo / "binary.md").write_bytes(
        b"binary with em dash: \xe2\x80\x94 and invalid byte: \xff\n"
    )
    subprocess.run(["git", "add", "binary.md"], cwd=temp_git_repo, check=True)

    assert list(check_em_dash._tracked_files()) == ["binary.md"]
    assert check_em_dash.main() == 0
    assert (
        capsys.readouterr().out == "OK: no em dashes in tracked source/prose files.\n"
    )
