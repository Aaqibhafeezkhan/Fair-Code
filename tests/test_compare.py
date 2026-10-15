"""Tests for the dataset-comparison (representation drift) module.

Run from the repo root:  pytest tests/ -q
"""

import pandas as pd

from faircode import compare, profile
from faircode.compare import _drift_level, _name_tokens, _psi_term


# ── Helpers ──────────────────────────────────────────────────────────────────
def _find(dims, name):
    return next(d for d in dims if d["name"] == name)


def _group(groups, label):
    return next(g for g in groups if g["label"] == label)


# ── PSI internals ─────────────────────────────────────────────────────────────
def test_psi_term_zero_when_shares_equal():
    assert _psi_term(0.5, 0.5) == 0.0


def test_psi_term_positive_when_shares_differ():
    # PSI contributions are always non-negative regardless of shift direction.
    assert _psi_term(0.6, 0.8) > 0
    assert _psi_term(0.8, 0.6) > 0


def test_drift_level_thresholds():
    assert _drift_level(0.05) == "none"
    assert _drift_level(0.10) == "moderate"
    assert _drift_level(0.24) == "moderate"
    assert _drift_level(0.25) == "significant"


# ── No drift: identical datasets ──────────────────────────────────────────────
def test_identical_datasets_show_no_drift():
    df = pd.DataFrame({"sex": ["M", "F"] * 100})
    cmp = compare(profile(df), profile(df))
    assert cmp["score_delta"] == 0
    dim = _find(cmp["dimensions"], "sex")
    assert dim["psi"] == 0.0
    assert dim["tvd"] == 0.0
    assert dim["drift_level"] == "none"
    assert all(g["status"] == "unchanged" and g["share_delta"] == 0.0
               for g in dim["groups"])
    assert cmp["flags"] == []


# ── Significant drift: distribution collapses onto one group ──────────────────
def test_large_shift_is_significant_and_flagged():
    a = pd.DataFrame({"race": ["White"] * 50 + ["Black"] * 50})   # 50/50
    b = pd.DataFrame({"race": ["White"] * 90 + ["Black"] * 10})   # 90/10
    cmp = compare(profile(a), profile(b), name_a="a.csv", name_b="b.csv")
    dim = _find(cmp["dimensions"], "race")
    assert dim["drift_level"] == "significant"
    assert dim["psi"] >= 0.25
    white = _group(dim["groups"], "White")
    assert white["share_delta"] > 0
    assert any("significant representation drift" in f for f in cmp["flags"])


# ── Appeared / disappeared groups ─────────────────────────────────────────────
def test_appeared_and_disappeared_groups():
    a = pd.DataFrame({"region": ["North"] * 60 + ["South"] * 40})
    b = pd.DataFrame({"region": ["North"] * 60 + ["West"] * 40})
    cmp = compare(profile(a), profile(b))
    dim = _find(cmp["dimensions"], "region")
    south = _group(dim["groups"], "South")
    west = _group(dim["groups"], "West")
    assert south["status"] == "disappeared"
    assert west["status"] == "appeared"
    assert any("'South' disappeared" in f for f in cmp["flags"])
    assert any("'West' appeared" in f for f in cmp["flags"])


# ── Added / removed dimensions ────────────────────────────────────────────────
def test_added_and_removed_dimensions():
    a = pd.DataFrame({"sex": ["M", "F"] * 50, "race": ["White", "Black"] * 50})
    b = pd.DataFrame({"sex": ["M", "F"] * 50, "region": ["North", "South"] * 50})
    cmp = compare(profile(a), profile(b), name_a="a.csv", name_b="b.csv")
    assert cmp["removed_dimensions"] == ["race"]
    assert cmp["added_dimensions"] == ["region"]
    # sex is shared, so it still gets compared
    assert _find(cmp["dimensions"], "sex")["drift_level"] == "none"
    assert any("only in b.csv" in f for f in cmp["flags"])
    assert any("only in a.csv" in f for f in cmp["flags"])


# ── Groups ordered by descending absolute shift ───────────────────────────────
def test_groups_ordered_by_shift_magnitude():
    a = pd.DataFrame({"race": ["White"] * 50 + ["Black"] * 30 + ["Asian"] * 20})
    b = pd.DataFrame({"race": ["White"] * 55 + ["Black"] * 10 + ["Asian"] * 35})
    dim = _find(compare(profile(a), profile(b))["dimensions"], "race")
    deltas = [abs(g["share_delta"]) for g in dim["groups"]]
    assert deltas == sorted(deltas, reverse=True)


# ── Age-banding mismatch (#318) ─────────────────────────────────────────────
def test_age_banding_mismatch_skips_drift_instead_of_reporting_100_percent():
    # A: raw birthdates (never banded, per _looks_like_dates()). B: numeric
    # ages (banded into "18-30"/"30-45"). `kind` is "age" on both sides -
    # set from the column name, not the values - so it alone can't catch
    # this; before the fix, every label showed as appeared/disappeared and
    # PSI came out far past the "significant" threshold for no real reason.
    df_a = pd.DataFrame({"DOB": ["15/05/1980"] * 50 + ["20/06/1985"] * 50})
    df_b = pd.DataFrame({"DOB": [18] * 50 + [35] * 50})
    cmp = compare(profile(df_a), profile(df_b))
    dim = _find(cmp["dimensions"], "DOB")

    assert dim["kind_mismatch"] is True
    assert dim["kind_a"] == dim["kind_b"] == "age"
    assert dim["psi"] == 0.0
    assert dim["drift_level"] == "none"
    assert dim["groups"] == []
    assert any("banded" in f and "DOB" in f for f in cmp["flags"])


def test_matching_age_banding_still_compares_normally():
    # Same-shape ages on both sides (both banded) must not be treated as a
    # mismatch - only an asymmetry between the two sides should skip it.
    df_a = pd.DataFrame({"age": [20] * 50 + [40] * 50})
    df_b = pd.DataFrame({"age": [20] * 30 + [40] * 70})
    cmp = compare(profile(df_a), profile(df_b))
    dim = _find(cmp["dimensions"], "age")

    assert dim["kind_mismatch"] is False
    assert dim["groups"]


# ── Score-drop flag ───────────────────────────────────────────────────────────
def test_overall_score_drop_is_flagged():
    a = pd.DataFrame({"sex": ["M", "F"] * 100})            # balanced -> high score
    b = pd.DataFrame({"sex": ["M"] * 190 + ["F"] * 10})    # skewed   -> low score
    cmp = compare(profile(a), profile(b))
    assert cmp["score_delta"] < 0
    assert any("overall representation score dropped" in f for f in cmp["flags"])


def test_unmeasured_profile_has_no_score_delta():
    measured = profile(pd.DataFrame({"sex": ["M", "F"] * 20}))
    unmeasured = profile(pd.DataFrame({"id": range(40)}))

    cmp = compare(measured, unmeasured)

    assert cmp["score_delta"] is None
    assert cmp["b"]["dimensions_detected"] is False
    assert not any("overall representation score dropped" in f for f in cmp["flags"])


# --- #868 / #866 ---------------------------------------------------------------------------

def _cmp(a, b, **kw):
    from faircode.compare import compare
    from faircode.profiler import profile
    return compare(profile(a), profile(b), "a.csv", "b.csv", **kw)


def test_compare_flags_implausible_ages_that_differ_between_datasets():
    import pandas as pd
    a = pd.DataFrame({"sex": ["M", "F"] * 4, "age": [25, 30, 41, 55, 62, 19, 33, 70]})
    b = pd.DataFrame({"sex": ["M", "F"] * 4, "age": [25, 30, 41, 200, 62, 19, 150, 70]})
    result = _cmp(a, b)
    age = next(d for d in result["dimensions"] if d["name"] == "age")
    assert (age["implausible_a"], age["implausible_b"]) == (0, 2)
    assert any("implausible age values differ (0 in a.csv, 2 in b.csv)" in f for f in result["flags"])
    # equal counts, or none at all, add neither the keys nor a flag
    same = _cmp(b, b)
    assert not any("implausible" in f for f in same["flags"])
    clean = next(d for d in _cmp(a, a)["dimensions"] if d["name"] == "age")
    assert "implausible_a" not in clean


def test_compare_notes_a_side_where_nothing_was_recognised_by_name():
    import pandas as pd
    plain = pd.DataFrame({"colA": ["x", "y"] * 4, "colB": ["p", "q"] * 4})
    named = pd.DataFrame({"sex": ["M", "F"] * 4, "colB": ["p", "q"] * 4})
    flags = _cmp(plain, named)["flags"]
    assert any(f.startswith("a.csv: no column name was recognised") and "--map COL=KIND" in f
               for f in flags)
    assert not any(f.startswith("b.csv: no column name") for f in flags)


def test_compare_suggests_a_renamed_dimension_without_pairing_it():
    import pandas as pd
    a = pd.DataFrame({"sex": ["M", "F"] * 4, "race": ["White", "Black", "Asian", "White"] * 2})
    b = pd.DataFrame({"sex": ["M", "F"] * 4, "ethnicity": ["White", "Black", "Asian", "White"] * 2})
    result = _cmp(a, b)
    assert result["possible_renames"] == [{"a": "race", "b": "ethnicity", "overlap": 1.0, "reason": "labels"}]
    assert any("'race' (a.csv) and 'ethnicity' (b.csv) look like the same dimension" in f
               for f in result["flags"])
    assert [d["name"] for d in result["dimensions"]] == ["sex"]   # never auto-paired
    assert result["drift_detected"] is True


def test_compare_does_not_suggest_unrelated_or_different_kind_dimensions():
    import pandas as pd
    a = pd.DataFrame({"sex": ["M", "F"] * 4, "race": ["White", "Black"] * 4})
    b = pd.DataFrame({"sex": ["M", "F"] * 4, "region": ["North", "South"] * 4})
    assert "possible_renames" not in _cmp(a, b)
    c = pd.DataFrame({"sex": ["M", "F"] * 4, "colX": ["White", "Black"] * 4})   # categorical, not race
    assert "possible_renames" not in _cmp(a, c)


def test_name_tokens_keep_unicode_words_whole():
    # #917
    assert _name_tokens("género") == {"genero"}
    assert _name_tokens("genero") == {"genero"}
    assert _name_tokens("Geschlecht_código") == {"geschlecht", "codigo"}
    assert _name_tokens("性别") == {"性别"}
    assert _name_tokens("性别代码") == {"性别代码"}
    assert _name_tokens("SexGender") == {"sex", "gender"}


def test_rename_compares_a_renamed_dimension_as_one_without_touching_the_tables():
    # #913
    import pytest
    a = profile(pd.DataFrame({"race": ["x", "y", "x", "y"] * 20}))
    b = profile(pd.DataFrame({"ethnicity": ["x", "y", "x", "x"] * 20}))
    plain = compare(a, b)
    assert plain["possible_renames"][0]["b"] == "ethnicity" and plain["added_dimensions"] == ["ethnicity"]
    joined = compare(a, b, renames={"race": "ethnicity"})
    assert [d["name"] for d in joined["dimensions"]] == ["race"]
    assert joined["added_dimensions"] == [] == joined["removed_dimensions"]
    assert "possible_renames" not in joined
    assert joined["renamed_dimensions"] == [{"a": "race", "b": "ethnicity"}]
    assert any("compared as 'race'" in f for f in joined["flags"])
    assert [d["name"] for d in b["dimensions"]] == ["ethnicity"]     # input result untouched
    for bad, msg in (({"nope": "ethnicity"}, "first dataset"), ({"race": "nope"}, "second dataset")):
        with pytest.raises(ValueError, match=msg):
            compare(a, b, renames=bad)
    c = profile(pd.DataFrame({"race": ["x", "y"] * 40, "ethnicity": ["x", "y"] * 40}))
    with pytest.raises(ValueError, match="already has a dimension named 'race'"):
        compare(a, c, renames={"race": "ethnicity"})


def test_rename_rejects_one_b_dimension_used_for_two_a_dimensions():
    # #968: two A names pointing at the same B dimension (and a B name repeated) are refused
    # with the exact message; a dict can't hold one A name twice, so this is the only
    # "same target" shape left.
    import pytest
    a = profile(pd.DataFrame({"race": ["x", "y"] * 40, "sex": ["m", "f"] * 40}))
    b = profile(pd.DataFrame({"ethnicity": ["x", "y"] * 40, "gender": ["m", "f"] * 40}))
    with pytest.raises(ValueError, match=r"^--rename: 'ethnicity' is renamed more than once$"):
        compare(a, b, renames={"race": "ethnicity", "sex": "ethnicity"})
    assert compare(a, b, renames={"race": "ethnicity", "sex": "gender"})["renamed_dimensions"] == [
        {"a": "race", "b": "ethnicity"}, {"a": "sex", "b": "gender"}]


def test_an_age_dimension_with_no_groups_is_not_an_age_banding_mismatch():
    # #976: an all-missing age column has no group labels on its side; the guard returns False
    # instead of comparing an empty list's all() with the other side's.
    from faircode.compare import _age_banding_mismatch
    empty = profile(pd.DataFrame({"age": [None] * 20}), {"age": "age"})["dimensions"][0]
    banded = profile(pd.DataFrame({"age": [20, 40, 60, 80] * 5}), {"age": "age"})["dimensions"][0]
    assert empty["groups"] == [] and banded["groups"]
    assert _age_banding_mismatch(empty, banded) is False and _age_banding_mismatch(banded, empty) is False
    assert compare(profile(pd.DataFrame({"age": [None] * 20}), {"age": "age"}),
                   profile(pd.DataFrame({"age": [20, 40, 60, 80] * 5}), {"age": "age"}))["dimensions"][0]["kind_mismatch"] is False
