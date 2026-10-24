import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPLAINERS_DIR = REPO_ROOT / "explainers"
PYTHON_BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
RUNNABLE_MARKER = "# runnable"
SLOW_MARKER = "# slow"


def runnable_blocks():
    for path in sorted(EXPLAINERS_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for block_number, match in enumerate(PYTHON_BLOCK.finditer(content), start=1):
            lines = match.group(1).splitlines()
            first_content = next((line.strip() for line in lines if line.strip()), "")
            if first_content != RUNNABLE_MARKER:
                continue
            slow = SLOW_MARKER in [line.strip() for line in lines[:3]]
            code = "\n".join(
                line for line in lines if line.strip() not in {RUNNABLE_MARKER, SLOW_MARKER}
            )
            marks = [pytest.mark.slow] if slow else []
            yield pytest.param(
                path,
                block_number,
                code,
                slow,
                id=f"{path.name}-python-block-{block_number}",
                marks=marks,
            )


@pytest.mark.parametrize("path,block_number,code,slow", list(runnable_blocks()))
def test_runnable_explainer_python_blocks(path, block_number, code, slow):
    # A block that loads an audit needs the benchmark extra; the profiler CI job installs only the core package.
    if "faircode.benchmark" in code:
        for module in ("yaml", "sklearn"):
            pytest.importorskip(module)
    timeout = 120 if slow else 30
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{path.relative_to(REPO_ROOT)} Python block {block_number} timed out "
            f"after {timeout}s.\nstdout: {exc.stdout or ''}\nstderr: {exc.stderr or ''}"
        )

    assert result.returncode == 0, (
        f"{path.relative_to(REPO_ROOT)} Python block {block_number} exited "
        f"with status {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_at_least_one_explainer_python_block_is_marked_runnable():
    assert list(runnable_blocks()), (
        "No runnable explainer Python blocks found. Add '# runnable' as the first "
        "nonblank line inside a Python fence in explainers/*.md."
    )
