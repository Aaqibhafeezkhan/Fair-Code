"""A high-cardinality column must not flood the profile flag list (#1043)."""

import json
import subprocess

import numpy as np
import pandas as pd

from faircode import compare, profile
from faircode.report import MAX_DISPLAYED_FLAGS, compare_to_html, compare_to_terminal, to_html, to_terminal
from js_helpers import REPO_ROOT


def _zip_frame(n, labels, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex": rng.choice(["m", "f"], n),
        "zip": rng.integers(10000, 10000 + labels, n).astype(str),
    })


def test_profile_summarises_per_group_flags_for_a_high_cardinality_column():
    result = profile(_zip_frame(6000, 3000, 1))
    assert len(result["flags"]) < 30
    assert any(f.startswith("zip: ") and "groups have only" in f for f in result["flags"])
    # The detail is not lost: every group is still in the result.
    zip_dim = next(d for d in result["dimensions"] if d["name"] == "zip")
    assert zip_dim["n_groups"] > 1000


def test_few_groups_keep_one_flag_each():
    df = pd.DataFrame({"sex": ["m"] * 300 + ["f"] * 4 + ["x"] * 3})
    flags = profile(df)["flags"]
    assert any("'f' has only 4 rows" in f for f in flags)
    assert any("'x' has only 3 rows" in f for f in flags)


def test_terminal_and_html_reports_cap_the_flag_list():
    flags = [f"flag {i}" for i in range(MAX_DISPLAYED_FLAGS + 25)]
    result = dict(profile(_zip_frame(500, 20, 2)), flags=flags)
    assert to_terminal(result).count("  ! flag") == MAX_DISPLAYED_FLAGS
    assert "and 25 more flags" in to_terminal(result)
    assert to_html(result).count("<li>flag") == MAX_DISPLAYED_FLAGS
    assert "and 25 more flags" in to_html(result)
    cmp = dict(compare(profile(_zip_frame(500, 20, 3)), profile(_zip_frame(500, 20, 4))), flags=flags)
    assert compare_to_terminal(cmp).count("  ! flag") == MAX_DISPLAYED_FLAGS
    assert compare_to_html(cmp).count("<li>flag") == MAX_DISPLAYED_FLAGS


def test_js_engine_summarises_like_python(tmp_path):
    csv = tmp_path / "z.csv"
    _zip_frame(3000, 1500, 8).to_csv(csv, index=False)
    py = profile(pd.read_csv(csv))
    out = subprocess.run(["node", "scripts/engine-js.js", "profile", str(csv)], capture_output=True,
                         text=True, encoding="utf-8", check=True, cwd=REPO_ROOT)
    js = json.loads(out.stdout)
    assert js["flags"] == py["flags"]
