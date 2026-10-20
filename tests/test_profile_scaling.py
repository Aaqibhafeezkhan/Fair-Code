"""A high-cardinality geography column must not make profile() quadratic (#1042)."""

import time

import numpy as np
import pandas as pd

from faircode import profile


def test_profile_with_many_distinct_labels_is_not_quadratic():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "sex": rng.choice(["m", "f"], 60000),
        "zip": rng.integers(10000, 60000, 60000).astype(str),
    })
    start = time.time()
    result = profile(df)
    assert time.time() - start < 5
    assert next(d for d in result["dimensions"] if d["name"] == "zip")["n_groups"] > 20000
