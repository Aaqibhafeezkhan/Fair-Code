"""Tests for the opt-in chi-squared proxy hints (needs the scipy extra).

Run from the repo root:  pytest tests/ -q
"""

from pathlib import Path

import pandas as pd
import pytest

from faircode import profile
from faircode.proxy import proxy_hints

pytest.importorskip("scipy", reason="proxy hints need the optional scipy extra")


def test_benefits_denial_native_proxy_summary_is_not_stale():
    # Regression test for #733: the top-of-file proxy summary must keep
    # Native applicants separate from Black applicants, with Native's 20.3%
    # rate preserved instead of the stale ~15% lumping that appears elsewhere.
    source = (Path(__file__).resolve().parents[1] / "Benefits Denial" / "unfair.py").read_text(encoding="utf-8")
    assert "Native applicants are at\n#                   20.3% (~77% of White's rate)." in source
    assert "Native applicants are at\n#                   ~15.5%" not in source


def test_perfect_proxy_is_flagged():
    # occupation is a perfect function of sex -> maximal association.
    df = pd.DataFrame({
        "sex": ["male", "female"] * 100,
        "occupation": ["engineer", "nurse"] * 100,
    })
    dims = profile(df)["dimensions"]
    hints = proxy_hints(df, dims)
    pair = next(h for h in hints
                if {h["a"], h["b"]} == {"sex", "occupation"})
    assert pair["p_value"] < 0.05
    assert pair["cramers_v"] > 0.9   # near-perfect association (Yates-corrected)


def test_held_out_column_catches_proxy_for_a_dropped_attribute():
    # "we dropped race so it's fine": race isn't in the profiled dataframe at
    # all, so it never becomes a dimension - proxy_hints() would otherwise
    # have no way to flag zip_code as a proxy for it (#328).
    race = (["A"] * 100 + ["B"] * 100)
    zip_code = (["111"] * 100 + ["222"] * 100)  # perfectly aligned with race
    df = pd.DataFrame({"zip_code": zip_code, "sex": ["M", "F"] * 100})

    held_out = {"race": pd.Series(race, index=df.index)}
    hints = proxy_hints(df, profile(df)["dimensions"], held_out=held_out)

    pair = next(h for h in hints if {h["a"], h["b"]} == {"zip_code", "race"})
    assert pair["p_value"] < 0.05
    assert pair["cramers_v"] > 0.9


def test_held_out_column_not_flagged_against_an_independent_column():
    # sex cycles every 2 rows, race every 3 - deliberately different periods
    # so the two are independent (verified: no hint), unlike the perfectly-
    # aligned zip_code/race case above.
    df = pd.DataFrame({"sex": ["male", "female"] * 150})
    held_out = {"race": pd.Series((["A", "B", "C"] * 100), index=df.index)}

    hints = proxy_hints(df, profile(df)["dimensions"], held_out=held_out)

    assert not any("race" in (h["a"], h["b"]) for h in hints)


def test_independent_columns_not_flagged():
    # Deterministic independence: sex alternates every row, grp every 3 rows,
    # so the two are (near) independent and should not be flagged.
    df = pd.DataFrame({
        "sex": ["male", "female"] * 150,
        "grp": ["x", "y", "z"] * 100,
    })
    hints = proxy_hints(df, profile(df)["dimensions"])
    assert not any({h["a"], h["b"]} == {"sex", "grp"} for h in hints)


def test_adjust_p_values_bonferroni_and_holm_known_values():
    """#806: hand-computed. Holm on [0.01, 0.04, 0.03], m=3: sorted -> 0.03,
    0.06, 0.04; the running max makes it 0.03, 0.06, 0.06 (input order below)."""
    from faircode.proxy import adjust_p_values

    assert adjust_p_values([0.01, 0.04, 0.03], "bonferroni") == pytest.approx([0.03, 0.12, 0.09])
    assert adjust_p_values([0.01, 0.04, 0.03], "holm") == pytest.approx([0.03, 0.06, 0.06])
    assert adjust_p_values([0.6, 0.9], "bonferroni") == [1.0, 1.0]
    with pytest.raises(ValueError):
        adjust_p_values([0.1], "fdr")


@pytest.mark.parametrize(
    "p_values,method,expected",
    [
        # Empty set of tested pairs (m=0)
        ([], "bonferroni", []),
        ([], "holm", []),
        # Single tested pair (m=1: adjusted equals raw)
        ([0.042], "bonferroni", [0.042]),
        ([0.042], "holm", [0.042]),
        ([0.8], "bonferroni", [0.8]),
        ([0.8], "holm", [0.8]),
        # Ties in Holm: identical p-values break ties stably and monotonicity holds
        ([0.02, 0.02, 0.04], "holm", [0.06, 0.06, 0.06]),
        ([0.05, 0.01, 0.01, 0.03], "holm", [0.06, 0.04, 0.04, 0.06]),
        ([0.03, 0.03], "bonferroni", [0.06, 0.06]),
        ([0.03, 0.03], "holm", [0.06, 0.06]),
        ([0.1, 0.1, 0.1], "holm", [0.3, 0.3, 0.3]),
    ],
)
def test_adjust_p_values_ties_and_edge_cases(p_values, method, expected):
    """#820: covers ties in Holm correction, single tested pair (m=1), and empty input (m=0)."""
    from faircode.proxy import adjust_p_values

    assert adjust_p_values(p_values, method) == pytest.approx(expected)


def test_proxy_hints_zero_testable_pairs_constant_columns():
    """#820: when all columns are constant, contingency tables are < 2x2 and tested pairs is empty."""
    pytest.importorskip("scipy")
    from faircode.proxy import proxy_hints

    df = pd.DataFrame({"col_a": ["const_a"] * 20, "col_b": ["const_b"] * 20, "col_c": ["const_c"] * 20})
    dims = [{"name": c, "kind": "categorical"} for c in df.columns]
    for correction in (None, "bonferroni", "holm"):
        hints = proxy_hints(df, dims, alpha=0.05, correction=correction)
        assert hints == []


def test_proxy_hints_single_testable_pair_adjusted_equals_raw():
    """#820: with exactly one testable pair (m=1), adjusted p-value equals raw p-value."""
    pytest.importorskip("scipy")
    from faircode.proxy import proxy_hints

    df = pd.DataFrame({"col_a": ["m", "f"] * 30, "col_b": ["a", "b"] * 30})
    dims = [{"name": c, "kind": "categorical"} for c in df.columns]
    plain = proxy_hints(df, dims, alpha=1.0)
    assert len(plain) == 1
    raw_p = plain[0]["p_value"]
    for correction in ("bonferroni", "holm"):
        corrected = proxy_hints(df, dims, alpha=1.0, correction=correction)
        assert len(corrected) == 1
        assert corrected[0]["p_adjusted"] == pytest.approx(raw_p)



def _three_dim_frame():
    n = 120
    sex = ["m", "f"] * (n // 2)
    race = ["a" if (i % 2 == 0) == (i % 10 != 0) else "b" for i in range(n)]
    age = [55 if (sex[i] == "m" and i % 3 == 0) or i % 11 == 0 else 25 for i in range(n)]
    return pd.DataFrame({"sex": sex, "race": race, "age": age})


def test_proxy_hints_correction_adds_p_adjusted_and_is_stricter():
    pytest.importorskip("scipy")
    from faircode.detect import detect_columns
    from faircode.proxy import proxy_hints

    df = _three_dim_frame()
    dims = [{"name": d["name"], "kind": d["kind"]} for d in detect_columns(df)]
    plain = proxy_hints(df, dims, alpha=1.0)
    corrected = proxy_hints(df, dims, alpha=1.0, correction="bonferroni")
    assert len(plain) == 3 and "p_adjusted" not in plain[0]
    for h in corrected:
        assert h["p_adjusted"] == pytest.approx(min(1.0, h["p_value"] * 3))
    strict = proxy_hints(df, dims, alpha=0.05, correction="holm")
    assert len(strict) <= len(proxy_hints(df, dims, alpha=0.05))
    with pytest.raises(ValueError, match="correction"):
        proxy_hints(df, dims, correction="nope")


def test_proxy_hints_flag_small_expected_cells():
    """#810: a sparse table is marked low_expected; a well-populated one is not."""
    sparse = pd.DataFrame({
        "sex": ["M", "F"] * 12,
        "race": ["a", "b", "c", "d", "e", "f"] * 4,
    })
    dense = pd.DataFrame({"sex": ["M", "F"] * 40, "race": ["a", "b"] * 40})
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    (small,) = proxy_hints(sparse, dims, alpha=1.0)
    (big,) = proxy_hints(dense, dims, alpha=1.0)
    assert small["low_expected"] is True and small["low_expected_share"] > 0.2
    assert big["low_expected"] is False and big["low_expected_share"] == 0.0


def test_terminal_report_marks_small_cell_hints():
    from faircode.report import to_terminal
    result = profile(pd.DataFrame({"sex": ["M", "F"] * 12, "race": ["a", "b", "c", "d", "e", "f"] * 4}))
    result["proxy_hints"] = [{"a": "sex", "b": "race", "p_value": 0.01, "cramers_v": 0.4,
                              "chi2": 5.0, "low_expected_share": 0.5, "low_expected": True}]
    assert "small cells" in to_terminal(result)


def test_proxy_hints_record_how_many_pairs_were_tested():
    """#821: n_tests is the family size m; constant-column pairs do not count."""
    df = pd.DataFrame({
        "sex": ["M", "F"] * 20,
        "race": ["a", "b"] * 20,
        "region": ["x", "y", "z", "w"] * 10,
        "country": ["US"] * 40,  # constant - every pair with it is skipped
    })
    dims = [{"name": n, "kind": k} for n, k in
            (("sex", "sex"), ("race", "race"), ("region", "geography"), ("country", "geography"))]
    hints = proxy_hints(df, dims, alpha=1.0, correction="bonferroni")
    assert hints and all(h["n_tests"] == 3 for h in hints)  # sex-race, sex-region, race-region
    for h in hints:
        assert h["p_adjusted"] == pytest.approx(min(1.0, h["p_value"] * h["n_tests"]))
    assert all(h["n_tests"] == 3 for h in proxy_hints(df, dims, alpha=1.0))


def test_terminal_report_shows_family_size_next_to_adjusted_p():
    from faircode.report import to_terminal
    result = profile(pd.DataFrame({"sex": ["M", "F"] * 12, "race": ["a", "b"] * 12}))
    result["proxy_hints"] = [{"a": "sex", "b": "race", "p_value": 0.01, "cramers_v": 0.4,
                              "chi2": 5.0, "p_adjusted": 0.03, "n_tests": 3}]
    assert "adj p=0.03 (m=3 pairs)" in to_terminal(result)


# --- #822: join key for held-out files ---------------------------------------

def _joined_frames():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "zip": ["111", "111", "222", "222"]})
    held = pd.DataFrame({"id": [4, 2, 3, 1, 9], "race": ["B", "A", "B", "A", "Z"]})
    return df, held


def test_held_out_key_matches_rows_by_key_not_position():
    from faircode.proxy import parse_held_out_specs
    df, held = _joined_frames()
    out = parse_held_out_specs(["h.csv=race:id"], df, lambda _p: held)
    assert list(out["race"]) == ["A", "A", "B", "B"]  # re-sorted file, extra key 9 ignored
    assert list(out["race"].index) == list(df.index)


def test_held_out_without_key_still_requires_equal_row_counts():
    from faircode.proxy import parse_held_out_specs
    df, held = _joined_frames()
    with pytest.raises(ValueError, match="rows must align 1:1"):
        parse_held_out_specs(["h.csv=race"], df, lambda _p: held)


def test_held_out_encoding_suffix_overrides_the_shared_reader_encoding():
    from faircode.proxy import parse_held_out_specs, split_held_out_encoding

    df = pd.DataFrame({"id": [1, 2]})
    held = pd.DataFrame({"race": ["Café", "Thé"]})
    seen = []

    def read(_path, encoding="latin-1"):
        seen.append(encoding)
        return held

    assert split_held_out_encoding("h.csv=race@utf-8") == ("h.csv=race", "utf-8")
    out = parse_held_out_specs(["h.csv=race@utf-8"], df, read)
    assert list(out["race"]) == ["Café", "Thé"]
    assert seen == ["utf-8"]


def test_held_out_encoding_composes_with_key_normalize_marker():
    from faircode.proxy import parse_held_out_specs

    df = pd.DataFrame({"id": ["A1", "B2"]})
    held = pd.DataFrame({"id": [" a1 ", "b2"], "race": ["x", "y"]})
    out = parse_held_out_specs(
        ["h.csv=race:id~@utf-8"],
        df,
        lambda _path, encoding=None: held,
    )
    assert list(out["race"]) == ["x", "y"]


def test_held_out_encoding_rejects_unknown_codec():
    from faircode.proxy import split_held_out_encoding

    with pytest.raises(ValueError, match="unknown held-out encoding"):
        split_held_out_encoding("h.csv=race@definitely-not-a-codec")


@pytest.mark.parametrize("held_override, df_override, message", [
    ({"id": [1, 1, 2, 3, 4], "race": list("ABABA")}, None, "duplicate"),
    ({"id": [1, 2], "race": list("AB")}, None, "no row for 2 key"),
    ({"id": [1, None, 3, 4], "race": list("ABAB")}, None, "empty values"),
    (None, {"id": [1, 2, 2, 4], "zip": list("aabb")}, "duplicate"),
])
def test_held_out_key_rejects_bad_keys(held_override, df_override, message):
    from faircode.proxy import parse_held_out_specs
    df, held = _joined_frames()
    if held_override:
        held = pd.DataFrame(held_override)
    if df_override:
        df = pd.DataFrame(df_override)
    with pytest.raises(ValueError, match=message):
        parse_held_out_specs(["h.csv=race:id"], df, lambda _p: held)


def test_held_out_key_must_exist_in_both_files_and_differ_from_the_column():
    from faircode.proxy import parse_held_out_specs
    df, held = _joined_frames()
    # #869: a suffix that is not a profiled column falls back to the whole
    # string as the column name, so race:nope looks for column "race:nope".
    with pytest.raises(ValueError, match="column 'race:nope' not found"):
        parse_held_out_specs(["h.csv=race:nope"], df, lambda _p: held)
    with pytest.raises(ValueError, match="not found in h.csv"):
        parse_held_out_specs(["h.csv=race:zip"], df, lambda _p: held)
    with pytest.raises(ValueError, match="invalid"):
        parse_held_out_specs(["h.csv=race:"], df, lambda _p: held)


def test_held_out_key_feeds_proxy_hints(tmp_path):
    from faircode.cli import main
    import json as _json
    df = pd.DataFrame({"id": range(200), "zip_code": ["111"] * 100 + ["222"] * 100})
    held = pd.DataFrame({"id": range(200), "race": ["A"] * 100 + ["B"] * 100}).sample(
        frac=1, random_state=3)  # shuffled: position alignment would be wrong
    df.to_csv(tmp_path / "d.csv", index=False)
    held.to_csv(tmp_path / "h.csv", index=False)
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["profile", str(tmp_path / "d.csv"), "--proxy-hints", "--json",
                     "--proxy-hints-with", f"{tmp_path / 'h.csv'}=race:id"])
    assert code == 0
    result = _json.loads(buf.getvalue())
    assert any({h["a"], h["b"]} == {"zip_code", "race"} for h in result["proxy_hints"])
    entry = result["provenance"]["proxy_hints_with"][0]
    assert entry["column"] == "race" and entry["key"] == "id"


# --- #869: held-out column names that contain a colon --------------------------

def test_held_out_column_with_colon_falls_back_when_suffix_is_not_a_key():
    """PATH=a:b keeps column a:b when b is not a profiled column (#869)."""
    from faircode.proxy import parse_held_out_specs, split_held_out_spec
    df = pd.DataFrame({"zip": ["1", "1", "2", "2"]})
    held = pd.DataFrame({"a:b": ["A", "A", "B", "B"], "x": [1, 2, 3, 4]})
    assert split_held_out_spec("h.csv=a:b", df.columns) == ("h.csv", "a:b", None)
    out = parse_held_out_specs(["h.csv=a:b"], df, lambda _p: held)
    assert list(out["a:b"]) == ["A", "A", "B", "B"]


def test_held_out_column_with_colon_escape_always_keeps_literal_colon():
    """PATH=a\\:b is column a:b even when b is a profiled column (#869)."""
    from faircode.proxy import parse_held_out_specs, split_held_out_spec
    df = pd.DataFrame({"b": [1, 2, 3, 4], "zip": ["1", "1", "2", "2"]})
    held = pd.DataFrame({"a:b": ["A", "A", "B", "B"]})
    assert split_held_out_spec(r"h.csv=a\:b", df.columns) == ("h.csv", "a:b", None)
    out = parse_held_out_specs([r"h.csv=a\:b"], df, lambda _p: held)
    assert list(out["a:b"]) == ["A", "A", "B", "B"]


def test_held_out_escaped_colon_column_still_accepts_a_join_key():
    """PATH=a\\:b:id joins on id with held-out column a:b (#869)."""
    from faircode.proxy import parse_held_out_specs
    df = pd.DataFrame({"id": [1, 2, 3, 4], "zip": ["1", "1", "2", "2"]})
    held = pd.DataFrame({"id": [4, 2, 3, 1], "a:b": ["B", "A", "B", "A"]})
    out = parse_held_out_specs([r"h.csv=a\:b:id"], df, lambda _p: held)
    assert list(out["a:b"]) == ["A", "A", "B", "B"]


def test_held_out_colon_column_cli_repro(tmp_path):
    """Issue #869 repro: profile with PATH=a:b no longer misreads column a."""
    from faircode.cli import main
    import io
    from contextlib import redirect_stdout, redirect_stderr
    (tmp_path / "d.csv").write_text("zip\n1\n1\n2\n2\n", encoding="utf-8")
    (tmp_path / "h.csv").write_text("a:b,x\nA,1\nA,2\nB,3\nB,4\n", encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(["profile", str(tmp_path / "d.csv"), "--proxy-hints",
                     "--proxy-hints-with", f"{tmp_path / 'h.csv'}=a:b", "--json"])
    assert code == 0, err.getvalue()
    assert "column 'a' not found" not in err.getvalue()
    import json as _json
    result = _json.loads(out.getvalue())
    assert result["provenance"]["proxy_hints_with"][0]["column"] == "a:b"
    assert "key" not in result["provenance"]["proxy_hints_with"][0]


# --- #859: composite keys and key normalisation ---------------------------------------

def _composite_frames():
    df = pd.DataFrame({"id": ["007", "007", "B2"], "visit": ["v1", "v2", "v1"], "zip": list("aab")})
    held = pd.DataFrame({"id": [" 7", "7", "b2"], "visit": ["V2", "V1", "v1"], "race": list("xyz")})
    return df, held


def test_composite_key_joins_on_every_part():
    from faircode.proxy import parse_held_out_specs
    df = pd.DataFrame({"id": [1, 1, 2], "visit": ["a", "b", "a"], "zip": list("xyz")})
    held = pd.DataFrame({"id": [2, 1, 1], "visit": ["a", "b", "a"], "race": ["Z", "Y", "X"]})
    out = parse_held_out_specs(["h.csv=race:id+visit"], df, lambda _p: held)
    assert list(out["race"]) == ["X", "Y", "Z"]


def test_keys_are_exact_text_unless_normalised():
    from faircode.proxy import parse_held_out_specs
    df, held = _composite_frames()
    with pytest.raises(ValueError, match="no row for"):
        parse_held_out_specs(["h.csv=race:id+visit"], df, lambda _p: held)
    out = parse_held_out_specs(["h.csv=race:id+visit"], df, lambda _p: held, normalize_keys=True)
    assert list(out["race"]) == ["y", "x", "z"]


def test_normalisation_can_expose_duplicates_and_bad_composites_are_rejected():
    from faircode.proxy import normalize_key_text, parse_held_out_specs
    df = pd.DataFrame({"id": ["a", "A"], "zip": ["x", "y"]})
    held = pd.DataFrame({"id": ["a", "A"], "race": ["p", "q"]})
    assert normalize_key_text("  0042 ") == "42" and normalize_key_text("000") == "0"
    with pytest.raises(ValueError, match="duplicate"):
        parse_held_out_specs(["h.csv=race:id"], df, lambda _p: held, normalize_keys=True)
    # an unrecognised key suffix is read as part of the column name (#869), so it fails there
    with pytest.raises(ValueError, match="not found in h.csv"):
        parse_held_out_specs(["h.csv=race:id+nope"], df, lambda _p: held)


def test_cli_key_normalize_flag_and_provenance(tmp_path, capsys):
    import json as _json
    from faircode.cli import main
    pd.DataFrame({"id": [" A1", "B2"] * 100, "zip_code": ["1"] * 100 + ["2"] * 100}).drop_duplicates().to_csv(
        tmp_path / "d.csv", index=False)
    df = pd.DataFrame({"id": [f"A{i}" for i in range(200)], "zip_code": ["1"] * 100 + ["2"] * 100})
    df.to_csv(tmp_path / "d.csv", index=False)
    held = pd.DataFrame({"id": [f" a{i} " for i in range(200)], "race": ["A"] * 100 + ["B"] * 100})
    held.to_csv(tmp_path / "h.csv", index=False)
    argv = ["profile", str(tmp_path / "d.csv"), "--proxy-hints", "--json",
            "--proxy-hints-with", f"{tmp_path / 'h.csv'}=race:id"]
    with pytest.raises(SystemExit):
        main(argv)
    capsys.readouterr()
    assert main(argv + ["--proxy-key-normalize"]) == 0
    result = _json.loads(capsys.readouterr().out)
    assert any({h["a"], h["b"]} == {"zip_code", "race"} for h in result["proxy_hints"])
    entry = result["provenance"]["proxy_hints_with"][0]
    assert entry["key"] == "id" and entry["key_normalize"] is True
    assert main(["profile", str(tmp_path / "d.csv"), "--proxy-key-normalize"]) == 2


# --- #861: exact-test fallback ------------------------------------------------------------

def test_exact_replaces_p_for_small_2x2_tables_with_fishers_test():
    from scipy.stats import fisher_exact
    df = pd.DataFrame({"sex": ["M"] * 5 + ["F"] * 9,
                       "race": ["x"] * 4 + ["y"] * 1 + ["x"] * 1 + ["y"] * 8})
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    (plain,) = proxy_hints(df, dims, alpha=1.0)
    (exact,) = proxy_hints(df, dims, alpha=1.0, exact=True)
    assert plain["low_expected"] and "p_method" not in plain
    assert exact["p_method"] == "fisher" and exact["p_chi2"] == pytest.approx(plain["p_value"])
    assert exact["p_value"] == pytest.approx(fisher_exact([[4, 1], [1, 8]])[1])


def test_exact_uses_a_seeded_permutation_for_larger_tables_and_leaves_dense_ones_alone():
    from faircode.proxy import PERMUTATIONS, permutation_p_value
    sparse = pd.DataFrame({"sex": ["M", "F"] * 12, "race": list("abcdef") * 4})
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    (h,) = proxy_hints(sparse, dims, alpha=1.0, exact=True)
    assert h["p_method"] == "permutation" and 1 / (PERMUTATIONS + 1) <= h["p_value"] <= 1
    assert proxy_hints(sparse, dims, alpha=1.0, exact=True)[0]["p_value"] == h["p_value"]  # deterministic
    assert permutation_p_value([0, 1] * 12, [0, 1, 2, 3, 4, 5] * 4, 2, 6) == h["p_value"]

    dense = pd.DataFrame({"sex": ["M", "F"] * 60, "race": ["a", "b"] * 60})
    (d,) = proxy_hints(dense, dims, alpha=1.0, exact=True)
    assert d["p_method"] == "chi2" and "p_chi2" not in d


def test_exact_flag_notes_and_csv_column(tmp_path, capsys):
    import csv as _csv
    import io as _io
    from faircode.report import _write_proxy_rows, _hint_notes
    hint = {"a": "sex", "b": "race", "p_value": 0.01, "cramers_v": 0.4, "p_method": "fisher",
            "low_expected": True, "n_tests": 1}
    assert "fisher exact p" in _hint_notes(hint)
    buf = _io.StringIO()
    _write_proxy_rows(_csv.writer(buf), [hint])
    assert "p_method" in buf.getvalue().splitlines()[0]
    from faircode.cli import main
    path = tmp_path / "d.csv"
    path.write_text("sex,race\n" + "\n".join(f"{'M' if i < 5 else 'F'},{'x' if i in (0, 1, 2, 3, 5) else 'y'}"
                                              for i in range(14)) + "\n")
    assert main(["profile", str(path), "--proxy-exact"]) == 2
    assert "--proxy-exact needs --proxy-hints" in capsys.readouterr().err
    assert main(["profile", str(path), "--proxy-hints", "--proxy-alpha", "1", "--proxy-exact", "--json",
                 "--no-provenance"]) == 0
    import json as _json
    assert _json.loads(capsys.readouterr().out)["proxy_hints"][0]["p_method"] == "fisher"


def test_normalize_key_text_composes_unicode_and_folds_case():
    # #884: a combining accent equals the precomposed letter, and ß equals ss.
    from faircode.proxy import normalize_key_text as n
    assert n("é") == n("é")
    assert n("Straße") == n("STRASSE")
    assert n("  0042 ") == "42"


def test_exact_marks_small_cell_tables_over_the_row_cap_as_chi2_capped():
    # #885: sparse and too big for the permutation fallback, so the chi-squared p stays
    # but the hint says why. The cap is lowered via max_rows so the table stays small (#906).
    from faircode.report import _hint_notes
    n = 600
    df = pd.DataFrame({"sex": [f"s{i // 2}" for i in range(n)], "race": [f"r{i // 3}" for i in range(n)]})
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    (h,) = proxy_hints(df, dims, alpha=1.0, exact=True, max_rows=n - 1)
    assert h["low_expected"] and h["p_method"] == "chi2-capped" and "p_chi2" not in h
    assert "too large for the exact fallback" in _hint_notes(h)
    (ok,) = proxy_hints(df, dims, alpha=1.0, exact=True, max_rows=n)
    assert ok["p_method"] == "permutation"


def test_default_permutation_row_cap_is_twenty_thousand():
    from faircode.proxy import PERMUTATION_MAX_ROWS
    assert PERMUTATION_MAX_ROWS == 20000   # #909: re-measured after the #886 speed-up


_MASK32 = 0xFFFFFFFF


def scalar_mulberry32(seed):
    """The plain-Python reference mulberry32 (ported verbatim to JS): the oracle that the
    vectorised `_mulberry32_block` used in production is checked against (#927)."""
    state = seed & _MASK32

    def next_float():
        nonlocal state
        state = (state + 0x6D2B79F5) & _MASK32
        t = ((state ^ (state >> 15)) * (1 | state)) & _MASK32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & _MASK32)) ^ t) & _MASK32
        return ((t ^ (t >> 14)) & _MASK32) / 4294967296

    return next_float


def test_permutation_p_value_matches_the_scalar_mulberry_reference():
    # #886: the vectorised PRNG block and sparse statistic must equal a plain-loop reference.
    import random
    from faircode.proxy import PERMUTATION_SEED, _mulberry32_block, permutation_p_value
    rand = scalar_mulberry32(PERMUTATION_SEED)
    assert list(_mulberry32_block(PERMUTATION_SEED, 0, 40)) == [rand() for _ in range(40)]
    rng = random.Random(3)
    rows = [rng.randrange(5) for _ in range(120)]
    cols = [rng.randrange(30) for _ in range(120)]
    n_r, n_c = 5, 30
    rt = [rows.count(r) for r in range(n_r)]
    ct = [cols.count(c) for c in range(n_c)]

    def stat(cs):
        counts = [0] * (n_r * n_c)
        for r, c in zip(rows, cs):
            counts[r * n_c + c] += 1
        return sum((counts[r * n_c + c] - rt[r] * ct[c] / 120) ** 2 / (rt[r] * ct[c] / 120)
                   for r in range(n_r) for c in range(n_c))

    rand = scalar_mulberry32(PERMUTATION_SEED)
    shuffled, hits, observed = list(cols), 0, stat(cols)
    for _ in range(100):
        for i in range(119, 0, -1):
            j = int(rand() * (i + 1))
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        hits += stat(shuffled) >= observed - 1e-9
    assert permutation_p_value(rows, cols, n_r, n_c, permutations=100) == (1 + hits) / 101


def test_per_spec_tilde_marker_normalises_only_that_specs_key(tmp_path):
    # #892: PATH=COLUMN:KEY~ normalises one spec; others stay exact.
    from faircode.proxy import parse_held_out_specs, strip_key_normalize_marker
    from faircode.provenance import held_out_entries
    df = pd.DataFrame({"id": ["A1", "B2", "C3"], "x": [1, 2, 3]})
    messy = pd.DataFrame({"id": [" a1", "b2", "C3 "], "race": ["u", "v", "w"]})
    clean = pd.DataFrame({"id": ["A1", "B2", "C3"], "age": [5, 6, 7]})
    tables = {"messy.csv": messy, "clean.csv": clean}
    read = lambda path: tables[path]  # noqa: E731

    held = parse_held_out_specs(["messy.csv=race:id~", "clean.csv=age:id"], df, read)
    assert list(held["race"]) == ["u", "v", "w"] and list(held["age"]) == [5, 6, 7]
    with pytest.raises(ValueError, match="no row for"):
        parse_held_out_specs(["messy.csv=race:id"], df, read)
    with pytest.raises(ValueError, match="no row for"):
        parse_held_out_specs(["messy.csv=race:id~"], df.assign(id=["A1", "B2", "Z9"]), read)

    assert strip_key_normalize_marker("m.csv=race:id~", ["id"]) == ("m.csv=race:id", True)
    assert strip_key_normalize_marker("m.csv=race~", ["id"]) == ("m.csv=race~", False)   # no key: not a marker
    entries = held_out_entries(["messy.csv=race:id~", "clean.csv=age:id"], ["id", "x"])
    assert [e.get("key_normalize") for e in entries] == [True, None]
    assert [e["key"] for e in entries] == ["id", "id"]


def test_per_spec_bang_marker_keeps_one_key_exact_under_the_global_flag():
    # #907: with normalisation on globally, PATH=COLUMN:KEY! stays exact.
    from faircode.proxy import parse_held_out_specs, resolve_key_marker
    from faircode.provenance import held_out_entries
    df = pd.DataFrame({"id": ["A1", "B2"], "x": [1, 2]})
    messy = pd.DataFrame({"id": ["a1", "b2"], "race": ["u", "v"]})
    read = lambda path: messy  # noqa: E731
    assert list(parse_held_out_specs(["m.csv=race:id"], df, read, normalize_keys=True)["race"]) == ["u", "v"]
    with pytest.raises(ValueError, match="no row for"):
        parse_held_out_specs(["m.csv=race:id!"], df, read, normalize_keys=True)
    assert resolve_key_marker("m.csv=race:id!", ["id"], True) == ("m.csv=race:id", False)
    assert resolve_key_marker("m.csv=race:id", ["id"], True) == ("m.csv=race:id", True)
    assert resolve_key_marker("m.csv=race!", ["id"], True) == ("m.csv=race!", True)   # no key: not a marker
    entries = held_out_entries(["a.csv=race:id!", "b.csv=age:id"], ["id"], key_normalize=True)
    assert [e.get("key_normalize") for e in entries] == [None, True]


def test_exact_permutations_and_row_cap_are_parameters():
    # #906: a different shuffle count changes the resolution; a lowered cap labels the pair.
    from faircode.proxy import PERMUTATIONS
    sparse = pd.DataFrame({"sex": ["M", "F"] * 12, "race": list("abcdef") * 4})
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    (default,) = proxy_hints(sparse, dims, alpha=1.0, exact=True)
    (fewer,) = proxy_hints(sparse, dims, alpha=1.0, exact=True, permutations=99)
    (more,) = proxy_hints(sparse, dims, alpha=1.0, exact=True, permutations=3000)
    assert default["p_value"] * (PERMUTATIONS + 1) == pytest.approx(round(default["p_value"] * (PERMUTATIONS + 1)))
    assert fewer["p_value"] * 100 == pytest.approx(round(fewer["p_value"] * 100))
    assert more["p_value"] * 3001 == pytest.approx(round(more["p_value"] * 3001))
    (capped,) = proxy_hints(sparse, dims, alpha=1.0, exact=True, max_rows=10)
    assert capped["p_method"] == "chi2-capped" and "p_chi2" not in capped
    for bad in ({"permutations": 0}, {"max_rows": -1}, {"permutations": 2.5}, {"max_rows": True}):
        with pytest.raises(ValueError, match="positive integer"):
            proxy_hints(sparse, dims, alpha=1.0, exact=True, **bad)


def test_near_unique_held_out_columns_are_skipped_with_a_note():
    # #908: an ID-like column (more distinct values than the cap, nearly one per row) is not
    # tested; a column with few values or a mid-cardinality one in a big table is kept.
    from faircode.proxy import near_unique_note, parse_held_out_specs
    n = 100
    df = pd.DataFrame({"sex": ["m", "f"] * (n // 2)})
    held = pd.DataFrame({"ident": [f"id{i}" for i in range(n)], "race": ["a", "b", "c", "d"] * (n // 4),
                         "age": [20 + i % 30 for i in range(n)]})
    read = lambda path: held  # noqa: E731
    notes = []
    out = parse_held_out_specs(["h.csv=ident", "h.csv=race"], df, read, notes=notes, max_categorical_card=20)
    assert list(out) == ["race"] and notes == [near_unique_note("ident", n, 20)]
    assert "identifier" in notes[0]
    notes = []
    assert list(parse_held_out_specs(["h.csv=age"], df, read, notes=notes, max_categorical_card=20)) == ["age"] and not notes
    assert list(parse_held_out_specs(["h.csv=ident"], df, read, notes=notes, max_categorical_card=200)) == ["ident"]
    assert list(parse_held_out_specs(["h.csv=ident"], df, read)) == ["ident"]   # no notes list: not screened


def test_a_small_numeric_age_is_kept_but_a_text_id_is_skipped():
    # #937: the near-unique screen only applies to non-numeric text, so a raw age on a
    # small table is tested; an ID-like string column is still skipped.
    from faircode.proxy import near_unique_note, parse_held_out_specs
    df = pd.DataFrame({"sex": ["m", "f"] * 20})
    held = pd.DataFrame({"age": [20 + i % 30 + (i // 30) for i in range(40)],
                         "ident": [f"id{i}" for i in range(40)]})
    read = lambda path: held  # noqa: E731
    notes = []
    out = parse_held_out_specs(["h.csv=age", "h.csv=ident"], df, read, notes=notes, max_categorical_card=20)
    assert list(out) == ["age"] and notes == [near_unique_note("ident", 40, 20)]


def test_permutations_have_an_upper_bound():
    # #926: 10**12 shuffles would never finish; the limit is documented and enforced.
    from faircode.proxy import MAX_PERMUTATIONS
    sparse = pd.DataFrame({"sex": ["M", "F"] * 12, "race": list("abcdef") * 4})
    dims = [{"name": "sex", "kind": "sex"}, {"name": "race", "kind": "race"}]
    with pytest.raises(ValueError, match="at most 1,000,000"):
        proxy_hints(sparse, dims, alpha=1.0, exact=True, permutations=MAX_PERMUTATIONS + 1)
    with pytest.raises(ValueError, match="at most"):
        proxy_hints(sparse, dims, alpha=1.0, exact=True, permutations=10**12)
    assert MAX_PERMUTATIONS == 1_000_000


def test_split_held_out_encoding_edge_cases_and_missing_key_errors():
    # #961: the empty suffix, an unknown codec, an @ in the path, and the two join-key raises.
    from faircode.proxy import parse_held_out_specs, split_held_out_encoding
    assert split_held_out_encoding("h.csv=race") == ("h.csv=race", None)
    assert split_held_out_encoding("h.csv=race@latin-1") == ("h.csv=race", "latin-1")
    assert split_held_out_encoding("me@home/h.csv=race") == ("me@home/h.csv=race", None)   # @ before = is the path
    assert split_held_out_encoding("me@home/h.csv=race:id~@utf-16") == ("me@home/h.csv=race:id~", "utf-16")
    assert split_held_out_encoding("noequals@x") == ("noequals@x", None)
    with pytest.raises(ValueError, match="expected @ENCODING"):
        split_held_out_encoding("h.csv=race@")
    with pytest.raises(ValueError, match="unknown held-out encoding 'nope'"):
        split_held_out_encoding("h.csv=race@nope")
    df = pd.DataFrame({"id": [1, 2], "x": [1, 2]})
    held = pd.DataFrame({"id": [1, 2], "race": ["a", "b"]})
    with pytest.raises(ValueError, match="column 'race:nokey' not found in h.csv"):   # unknown key: the whole RHS is the column (#869)
        parse_held_out_specs(["h.csv=race:nokey"], df, lambda p: held)
    with pytest.raises(ValueError, match="join key 'x' not found in h.csv"):
        parse_held_out_specs(["h.csv=race:x"], df, lambda p: held)


def test_held_out_path_may_contain_an_escaped_equals_sign(tmp_path):
    # #958: `\=` is a literal equals sign in the PATH part; without it the error says what to do.
    from faircode.loaders import read_table
    from faircode.provenance import held_out_entries
    from faircode.proxy import parse_held_out_specs, partition_held_out_path, split_held_out_encoding, split_held_out_spec
    held = tmp_path / "data=v2.csv"
    held.write_text("id,race\n" + "\n".join(f"{i},{'ab'[i % 2]}" for i in range(6)) + "\n")
    df = pd.DataFrame({"id": range(6), "x": range(6)})
    spec = f"{tmp_path}/data\\=v2.csv=race:id"
    assert partition_held_out_path(spec) == (f"{tmp_path}/data=v2.csv", "=", "race:id")
    assert partition_held_out_path("plain.csv=race") == ("plain.csv", "=", "race")
    assert partition_held_out_path("noequals") == ("noequals", "", "")
    assert partition_held_out_path("a\\=b") == ("a=b", "", "")
    assert split_held_out_spec(spec, ["id", "x"]) == (f"{tmp_path}/data=v2.csv", "race", "id")
    assert list(parse_held_out_specs([spec], df, read_table)["race"]) == list("ababab")
    # composes with the key marker and the encoding suffix
    full = spec + "~@utf-8"
    assert split_held_out_encoding(full)[1] == "utf-8"
    assert list(parse_held_out_specs([full], df, read_table)["race"]) == list("ababab")
    entry = held_out_entries([full], ["id", "x"])[0]
    assert entry["path"] == f"{tmp_path}/data=v2.csv" and entry["key_normalize"] is True and entry["encoding"] == "utf-8"
    with pytest.raises(ValueError, match=r"does not exist but .*data=v2.csv' does - write the = inside the file name as \\="):
        parse_held_out_specs([f"{tmp_path}/data=v2.csv=race:id"], df, read_table)


@pytest.mark.parametrize("digits", ["١٢٣", "１２３", "१२३"])
def test_non_ascii_digits_are_not_numbers_for_the_identifier_screen(digits):
    # #967: Arabic-Indic, full-width and Devanagari digits are values, not numbers (the web
    # engine's \d is 0-9 only), so the near-unique screen treats them like any other text.
    from faircode.proxy import _all_numeric
    assert not _all_numeric(pd.Series([digits, digits[::-1]]))
    assert _all_numeric(pd.Series(["123", "-4.5", ".5", "1e3"]))


def test_a_column_named_twice_is_rejected_whether_the_first_was_kept_or_skipped():
    # #980: the duplicate check used to look only at kept columns, so a repeated near-unique
    # column slipped past it and was silently skipped twice.
    from faircode.proxy import parse_held_out_specs
    n = 100
    df = pd.DataFrame({"sex": ["m", "f"] * (n // 2)})
    held = pd.DataFrame({"ident": [f"id{i}" for i in range(n)], "race": ["a", "b"] * (n // 2)})
    read = lambda path: held  # noqa: E731
    for column in ("race", "ident"):
        with pytest.raises(ValueError, match=f"column '{column}' was already supplied by an earlier"):
            parse_held_out_specs([f"h.csv={column}", f"h.csv={column}"], df, read, notes=[], max_categorical_card=20)
