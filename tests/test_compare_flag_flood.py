"""compare() must not turn a high-cardinality column into tens of thousands of drift flags (#1052)."""

import numpy as np
import pandas as pd

from faircode import compare, profile
from faircode.profiler import FLAG_GROUP_CAP


def _zip_frame(n, labels, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["m", "f"], n),
        "zip": rng.integers(10000, 10000 + labels, n).astype(str),
    })


def test_compare_summarises_appeared_and_disappeared_groups():
    a = profile(_zip_frame(6000, 5000, 6))
    b = profile(_zip_frame(6000, 5000, 7))
    cmp = compare(a, b)
    zip_flags = [f for f in cmp["flags"] if f.startswith("zip:")]
    assert len(cmp["flags"]) < 30
    assert any(f.endswith("groups appeared") for f in zip_flags)
    assert any(f.endswith("groups disappeared") for f in zip_flags)
    assert cmp["drift_detected"]
    zip_dim = next(d for d in cmp["dimensions"] if d["name"] == "zip")
    assert sum(g["status"] == "appeared" for g in zip_dim["groups"]) > FLAG_GROUP_CAP


def test_few_appeared_groups_keep_one_flag_each():
    a = pd.DataFrame({"race": ["x"] * 80 + ["y"] * 80})
    b = pd.DataFrame({"race": ["x"] * 80 + ["z"] * 80})
    flags = compare(profile(a), profile(b))["flags"]
    assert any("'z' appeared" in f for f in flags)
    assert any("'y' disappeared" in f for f in flags)


def test_js_engine_summarises_compare_flags_like_python(tmp_path):
    import json
    import subprocess

    from js_helpers import REPO_ROOT
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _zip_frame(3000, 1500, 8).to_csv(a, index=False)
    _zip_frame(3000, 1500, 9).to_csv(b, index=False)
    py = compare(profile(pd.read_csv(a)), profile(pd.read_csv(b)))
    out = subprocess.run(["node", "scripts/engine-js.js", "compare", str(a), str(b)], capture_output=True,
                         text=True, encoding="utf-8", cwd=REPO_ROOT)
    assert out.returncode == 0, out.stderr[-300:]
    js = json.loads(out.stdout)
    assert [f for f in js["flags"] if f.startswith("zip:")] == [f for f in py["flags"] if f.startswith("zip:")]
