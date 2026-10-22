"""Tests for scripts/freeze_paper_results.py: mirror_for_mcp() and mirror_current_for_mcp()
(never touch the directory they read from) and, in the second half of the file, the whole
freeze() flow against a temporary tree with git stubbed (#1111).
"""

import importlib
import shutil
import sys
import types

import pytest


def test_mirror_for_mcp_copies_the_three_frozen_csvs(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.freeze_paper_results")

    frozen_dir = tmp_path / "paper" / "results-frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "results_fairness.csv").write_text("audit,metric\ncompas,dp\n", encoding="utf-8")
    (frozen_dir / "results_performance.csv").write_text("audit,metric\ncompas,acc\n", encoding="utf-8")
    (frozen_dir / "summary.csv").write_text("audit\ncompas\n", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_results_frozen"

    monkeypatch.setattr(script, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(script, "MCP_MIRROR_DIR", mirror_dir)

    script.mirror_for_mcp()

    for name in ("results_fairness.csv", "results_performance.csv", "summary.csv"):
        assert (mirror_dir / name).read_text(encoding="utf-8") == (frozen_dir / name).read_text(encoding="utf-8")


def test_mirror_for_mcp_never_writes_to_frozen_dir(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.freeze_paper_results")

    frozen_dir = tmp_path / "paper" / "results-frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "results_fairness.csv").write_text("audit\ncompas\n", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_results_frozen"

    monkeypatch.setattr(script, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(script, "MCP_MIRROR_DIR", mirror_dir)

    before = {p.name: p.read_text(encoding="utf-8") for p in frozen_dir.iterdir()}
    script.mirror_for_mcp()
    after = {p.name: p.read_text(encoding="utf-8") for p in frozen_dir.iterdir()}

    assert before == after


def test_mirror_current_for_mcp_copies_the_live_results(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.freeze_paper_results")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for name in ("results_fairness.csv", "results_performance.csv", "summary.csv"):
        (results_dir / name).write_text(f"audit\n{name}\n", encoding="utf-8")
    mirror_dir = tmp_path / "faircode" / "_results_current"

    monkeypatch.setattr(script, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(script, "MCP_CURRENT_MIRROR_DIR", mirror_dir)

    before = {p.name: p.read_text(encoding="utf-8") for p in results_dir.iterdir()}
    script.mirror_current_for_mcp()

    for name, text in before.items():
        assert (mirror_dir / name).read_text(encoding="utf-8") == text
    assert {p.name: p.read_text(encoding="utf-8") for p in results_dir.iterdir()} == before


def test_package_mirrors_match_their_sources():
    script = importlib.import_module("scripts.freeze_paper_results")
    for name in script.MCP_MIRROR_FILES:
        assert (script.MCP_CURRENT_MIRROR_DIR / name).read_bytes() == (script.RESULTS_DIR / name).read_bytes()
        assert (script.MCP_MIRROR_DIR / name).read_bytes() == (script.FROZEN_DIR / name).read_bytes()


# ---- freeze() itself (#1111) ----------------------------------------------------------------
# Everything runs inside tmp_path: the module's directory constants are monkeypatched, git is
# stubbed through _run, and the manifest discovery is replaced, so nothing outside tmp_path is read
# or written.


@pytest.fixture
def frozen_env(tmp_path, monkeypatch):
    script = importlib.import_module("scripts.freeze_paper_results")
    root = tmp_path / "repo"
    results = root / "results"
    (results / "figures").mkdir(parents=True)
    (results / "results_fairness.csv").write_text("audit,metric\ncompas,dp\n", encoding="utf-8")
    (results / "results_performance.csv").write_text("audit,metric\ncompas,acc\n", encoding="utf-8")
    (results / "summary.csv").write_text("audit\ncompas\n", encoding="utf-8")
    (results / "figures" / "a.png").write_bytes(b"png-bytes")
    lock = root / "requirements-lock.txt"
    lock.write_text("pandas==2.0.0\n", encoding="utf-8")

    monkeypatch.setattr(script, "REPO_ROOT", root)
    monkeypatch.setattr(script, "RESULTS_DIR", results)
    monkeypatch.setattr(script, "FROZEN_DIR", root / "paper" / "results-frozen")
    monkeypatch.setattr(script, "LOCKFILE", lock)
    monkeypatch.setattr(script, "MCP_MIRROR_DIR", root / "faircode" / "_results_frozen")
    monkeypatch.setattr(script, "MCP_CURRENT_MIRROR_DIR", root / "faircode" / "_results_current")
    monkeypatch.setattr(script, "_discover_manifests", lambda: ["COMPAS/audit.yaml", "Tenant Screening/audit.yaml"])
    monkeypatch.setattr(script, "_package_version", lambda name: f"{name}-9.9")

    commands = []
    git = {"rev-parse HEAD": "abc123", "rev-parse --abbrev-ref HEAD": "main", "status --porcelain": ""}

    def fake_run(cmd):
        commands.append(cmd)
        return git.get(" ".join(cmd[1:]), "")

    monkeypatch.setattr(script, "_run", fake_run)
    return types.SimpleNamespace(script=script, root=root, tmp=tmp_path, git=git, commands=commands)


def _manifest(root):
    return (root / "paper" / "results-frozen" / "MANIFEST.md").read_text(encoding="utf-8")


def test_freeze_copies_results_figures_and_lockfile(frozen_env):
    e = frozen_env
    frozen = e.script.freeze()
    assert frozen == e.root / "paper" / "results-frozen"
    for name in ("results_fairness.csv", "results_performance.csv", "summary.csv"):
        assert (frozen / name).read_text(encoding="utf-8") == (e.root / "results" / name).read_text(encoding="utf-8")
    assert (frozen / "figures" / "a.png").read_bytes() == b"png-bytes"
    assert (frozen / "requirements-lock.txt").read_text(encoding="utf-8") == "pandas==2.0.0\n"


def test_freeze_manifest_records_commit_branch_versions_and_audits(frozen_env):
    e = frozen_env
    e.script.freeze()
    text = _manifest(e.root)
    assert "- **Git commit:** `abc123`\n" in text
    assert "WORKING TREE HAD UNCOMMITTED CHANGES" not in text
    assert "- **Git branch:** `main`" in text
    for line in ("- **scikit-learn:** sklearn-9.9", "- **fairlearn:** fairlearn-9.9",
                 "- **pandas:** pandas-9.9", "- **numpy:** numpy-9.9", "- **Python:** "):
        assert line in text
    assert "## Audits included (2 domains)" in text
    assert "- `COMPAS/audit.yaml`" in text and "- `Tenant Screening/audit.yaml`" in text
    assert "git checkout abc123" in text
    assert "## Tag" not in text


def test_freeze_flags_an_uncommitted_working_tree(frozen_env, capsys):
    e = frozen_env
    e.git["status --porcelain"] = " M README.md"
    e.script.freeze()
    assert "(WORKING TREE HAD UNCOMMITTED CHANGES AT FREEZE TIME)" in _manifest(e.root)
    assert "[DIRTY]" in capsys.readouterr().out


def test_freeze_without_git_output_says_unknown(frozen_env):
    e = frozen_env
    e.git["rev-parse HEAD"] = ""
    e.git["rev-parse --abbrev-ref HEAD"] = ""
    e.script.freeze()
    text = _manifest(e.root)
    assert "`unknown (not a git checkout?)`" in text
    assert "- **Git branch:** `unknown`" in text


def test_freeze_with_tag_prints_but_never_runs_a_git_tag_command(frozen_env, capsys):
    e = frozen_env
    e.script.freeze(tag="v9-paper")
    text = _manifest(e.root)
    assert "## Tag" in text and "git tag -a v9-paper" in text and "git push origin v9-paper" in text
    out = capsys.readouterr().out
    assert 'next step: git tag -a v9-paper -m "Results frozen for paper citation"' in out
    # only read-only git commands ran: rev-parse twice and status
    assert [c[1] for c in e.commands] == ["rev-parse", "rev-parse", "status"]


def test_freeze_without_tag_suggests_the_tag_flag(frozen_env, capsys):
    frozen_env.script.freeze()
    assert "pass --tag vX.Y-paper" in capsys.readouterr().out


def test_freeze_replaces_an_existing_snapshot_cleanly(frozen_env):
    e = frozen_env
    frozen = e.root / "paper" / "results-frozen"
    (frozen / "figures").mkdir(parents=True)
    (frozen / "stale.csv").write_text("old", encoding="utf-8")
    (frozen / "figures" / "old.png").write_bytes(b"old")
    e.script.freeze()
    assert not (frozen / "stale.csv").exists()
    assert not (frozen / "figures" / "old.png").exists()
    assert (frozen / "figures" / "a.png").exists()


def test_freeze_refreshes_both_mcp_mirrors(frozen_env):
    e = frozen_env
    e.script.freeze()
    for mirror in ("_results_frozen", "_results_current"):
        for name in ("results_fairness.csv", "results_performance.csv", "summary.csv"):
            assert (e.root / "faircode" / mirror / name).read_text(encoding="utf-8") == (
                e.root / "results" / name).read_text(encoding="utf-8")


def test_freeze_skips_missing_optional_inputs(frozen_env):
    e = frozen_env
    (e.root / "results" / "summary.csv").unlink()
    (e.root / "results" / "figures" / "a.png").unlink()
    (e.root / "results" / "figures").rmdir()
    (e.root / "requirements-lock.txt").unlink()
    frozen = e.script.freeze()
    assert not (frozen / "summary.csv").exists()
    assert not (frozen / "figures").exists()
    assert not (frozen / "requirements-lock.txt").exists()
    assert (frozen / "results_fairness.csv").exists() and (frozen / "MANIFEST.md").exists()


def test_freeze_fails_cleanly_when_results_are_missing(frozen_env, capsys):
    e = frozen_env
    shutil.rmtree(e.root / "results")
    with pytest.raises(SystemExit) as exc:
        e.script.freeze()
    assert exc.value.code == 2
    assert "does not exist - run `faircode benchmark` first" in capsys.readouterr().err
    assert not (e.root / "paper").exists()


def test_freeze_writes_nothing_outside_the_intended_directories(frozen_env):
    e = frozen_env
    before = set(e.tmp.rglob("*"))
    e.script.freeze()
    new = set(e.tmp.rglob("*")) - before
    assert new, "freeze() should have written something"
    allowed = (e.root / "paper", e.root / "faircode")
    assert all(any(p == a or a in p.parents for a in allowed) for p in new)


def test_main_passes_the_tag_option_through(frozen_env, monkeypatch):
    e = frozen_env
    monkeypatch.setattr(sys, "argv", ["freeze_paper_results.py", "--tag", "v3-paper"])
    e.script.main()
    assert "git tag -a v3-paper" in _manifest(e.root)


def test_package_version_reports_installed_missing_and_unversioned_modules():
    script = importlib.import_module("scripts.freeze_paper_results")
    assert script._package_version("definitely_not_a_real_module_xyz") == "not installed"
    assert script._package_version("keyword") == "unknown"          # importable, has no __version__
    assert script._package_version("json") == __import__("json").__version__


def test_run_returns_stripped_stdout():
    script = importlib.import_module("scripts.freeze_paper_results")
    assert script._run([sys.executable, "-c", "print('  hi  ')"]) == "hi"


def test_discover_manifests_lists_the_real_audits_sorted():
    pytest.importorskip("yaml", reason="manifest discovery needs the optional benchmark extra")
    script = importlib.import_module("scripts.freeze_paper_results")
    found = script._discover_manifests()
    assert found == sorted(found)
    assert any(path.replace("\\", "/") == "COMPAS/audit.yaml" for path in found)
