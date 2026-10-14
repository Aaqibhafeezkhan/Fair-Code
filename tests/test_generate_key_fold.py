"""scripts/generate_key_fold.py rebuilds the JS KEY_FOLD table (#939)."""

import importlib
import re
import shutil
import subprocess

import pytest

HAS_NODE = shutil.which("node") is not None
needs_node = pytest.mark.skipif(not HAS_NODE, reason="node is needed to compare with toLowerCase()")


@pytest.fixture
def script(tmp_path, monkeypatch):
    mod = importlib.import_module("scripts.generate_key_fold")
    copy = tmp_path / "profiler-engine.js"
    shutil.copy(mod.ENGINE, copy)
    monkeypatch.setattr(mod, "ENGINE", copy)
    return mod


@needs_node
def test_generated_block_carries_the_known_folds_and_the_unicode_versions(script):
    block = script.build_block()
    assert block.startswith(script.BEGIN) and block.endswith(script.END)
    assert "'\\u00df':'\\u0073\\u0073'" in block          # sharp s -> ss
    assert "'\\ufb01':'\\u0066\\u0069'" in block          # fi ligature
    assert "(Unicode" in block and "node v" in block
    assert block.isascii()


@needs_node
def test_check_agrees_after_a_rewrite_reports_what_it_skipped_and_detects_a_stale_table(script, capsys, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["generate_key_fold.py"])
    assert script.main() == 0                                   # rewrite in place
    monkeypatch.setattr(sys, "argv", ["generate_key_fold.py", "--check"])
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "agrees with Python on every code point both runtimes assign" in out
    assert "table built with Python" in out and "(Unicode" in out
    # the skip is reported, not hidden (#946)
    import re
    counts = re.search(r"compared ([\d,]+) code points; not compared: ([\d,]+) assigned only in node, ([\d,]+) only in Python", out)
    assert counts and int(counts.group(1).replace(",", "")) > 100_000

    text = script.ENGINE.read_text(encoding="utf-8")
    script.ENGINE.write_text(text.replace("'\\u00df':'\\u0073\\u0073',", ""), encoding="utf-8")
    assert script.main() == 1                                   # a missing entry is a real disagreement
    stale = capsys.readouterr().out
    assert "STALE" in stale and "U+00DF" in stale and "make key-fold" in stale


@needs_node
def test_check_notes_when_the_runtimes_know_a_newer_unicode_than_the_table(script, capsys, monkeypatch):
    import sys
    text = script.ENGINE.read_text(encoding="utf-8")
    old = re.sub(r"Built with: Python \S+ \(Unicode \S+\) and node \S+ \(Unicode \S+\)",
                 "Built with: Python 3.8.0 (Unicode 12.1.0) and node v14.0.0 (Unicode 13.0)", text)
    script.ENGINE.write_text(old, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate_key_fold.py", "--check"])
    script.main()
    assert "knows a newer Unicode than the table was built with" in capsys.readouterr().out


def test_check_skips_when_node_is_missing(script, capsys, monkeypatch):
    import sys
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(sys, "argv", ["generate_key_fold.py", "--check"])
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "node is needed for this check (install Node 18+); skipping" in out


def test_check_fails_when_node_is_missing_and_require_is_passed(script, capsys, monkeypatch):
    import sys
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(sys, "argv", ["generate_key_fold.py", "--check", "--require"])
    assert script.main() == 2
    out = capsys.readouterr().out
    assert "node is needed for this check (install Node 18+); skipping" in out


def test_rewrite_fails_when_node_is_missing(script, monkeypatch):
    import sys
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(sys, "argv", ["generate_key_fold.py"])
    with pytest.raises((FileNotFoundError, subprocess.CalledProcessError)):
        script.main()


def test_the_repo_copy_still_has_its_markers():
    mod = importlib.import_module("scripts.generate_key_fold")
    text = mod.ENGINE.read_text(encoding="utf-8")
    assert mod.BEGIN in text and mod.END in text and "var KEY_FOLD = {" in text

