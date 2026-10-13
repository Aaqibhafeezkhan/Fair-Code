"""scripts/generate_key_fold.py rebuilds the JS KEY_FOLD table (#939)."""

import importlib
import shutil
import subprocess

import pytest

if shutil.which("node") is None:  # pragma: no cover - CI has node
    pytest.skip("node is needed to compare with toLowerCase()", allow_module_level=True)


@pytest.fixture
def script(tmp_path, monkeypatch):
    mod = importlib.import_module("scripts.generate_key_fold")
    copy = tmp_path / "profiler-engine.js"
    shutil.copy(mod.ENGINE, copy)
    monkeypatch.setattr(mod, "ENGINE", copy)
    return mod


def test_generated_block_carries_the_known_folds_and_the_unicode_versions(script):
    block = script.build_block()
    assert block.startswith(script.BEGIN) and block.endswith(script.END)
    assert "'\\u00df':'\\u0073\\u0073'" in block          # sharp s -> ss
    assert "'\\ufb01':'\\u0066\\u0069'" in block          # fi ligature
    assert "(Unicode" in block and "node v" in block
    assert block.isascii()


def test_rewrite_then_check_is_a_fixed_point_and_the_check_detects_drift(script):
    assert script.main.__module__ == script.__name__
    import sys
    old = sys.argv
    try:
        sys.argv = ["generate_key_fold.py"]
        assert script.main() == 0                          # rewrite in place
        sys.argv = ["generate_key_fold.py", "--check"]
        assert script.main() == 0                          # immediately up to date
        text = script.ENGINE.read_text(encoding="utf-8")
        script.ENGINE.write_text(text.replace("'\\u00df':'\\u0073\\u0073',", ""), encoding="utf-8")
        assert script.main() == 1                          # a missing entry is reported
    finally:
        sys.argv = old


def test_the_repo_copy_still_has_its_markers():
    mod = importlib.import_module("scripts.generate_key_fold")
    text = mod.ENGINE.read_text(encoding="utf-8")
    assert mod.BEGIN in text and mod.END in text and "var KEY_FOLD = {" in text
