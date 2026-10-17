import csv
import io

import pandas as pd
import pytest

# Adjust the import path to match your package structure (e.g., faircode.report)
from faircode.compare import compare
from faircode.profiler import profile
from faircode.report import (
    compare_to_csv, compare_to_html, compare_to_terminal, to_csv, to_html, to_terminal,
)


@pytest.fixture
def mock_profile_result():
    return {
        "n_rows": 1250,
        "n_cols": 8,
        "overall_score": 88,
        "grade": "B+",
        "dimensions": [
            {
                "name": "Gender",
                "kind": "Demographic",
                "dimension_score": 85,
                "under_represented": ["Non-binary"],
                "n_groups": 2,
                "imbalance_ratio": 1.6,
                "missing_pct": 0.02,
                "skewness": 0.3,
                "groups": [
                    {
                        "label": "Female",
                        "share": 0.45,
                        "count": 562,
                        "ci_low": 0.42,
                        "ci_high": 0.48,
                    }
                ],
            }
        ],
        "flags": ["Gender under-representation detected"],
    }


@pytest.fixture
def mock_compare_result():
    return {
        "score_delta": -5,
        "a": {"name": "Dataset A", "overall_score": 90, "n_rows": 1000, "grade": "A"},
        "b": {"name": "Dataset B", "overall_score": 85, "n_rows": 1200, "grade": "B"},
        "added_dimensions": ["NewDim"],
        "removed_dimensions": ["OldDim"],
        "flags": ["High drift in Gender"],
        "dimensions": [
            {
                "name": "Gender",
                "kind": "Demographic",
                "kind_mismatch": False,
                "drift_level": "significant",
                "psi": 0.125,
                "tvd": 0.082,
                "dimension_score_a": 92,
                "dimension_score_b": 85,
                "dimension_score_delta": -7,
                "groups": [
                    {
                        "label": "Female",
                        "status": "shifted",
                        "share_a": 0.50,
                        "share_b": 0.40,
                        "share_delta": -0.10,
                    }
                ],
            }
        ],
    }


def test_to_html_smoke(mock_profile_result):
    """Smoke test for profile report rendering."""
    html_out = to_html(mock_profile_result)

    assert html_out.startswith("<!DOCTYPE html>")
    assert "</html>" in html_out
    assert "Dataset Representation Profile" in html_out


def test_to_html_renders_key_figures(mock_profile_result):
    """Verify raw calculated metrics appear in the rendered HTML."""
    html_out = to_html(mock_profile_result)

    assert "1,250" in html_out
    assert "88/100" in html_out
    assert "B+" in html_out
    assert "Gender" in html_out
    assert "imbalance 1.6x" in html_out
    assert "missing 2.0%" in html_out
    assert "skew +0.30" in html_out


def test_to_html_reports_groups_omitted_by_display_cap(mock_profile_result):
    dimension = mock_profile_result["dimensions"][0]
    dimension["groups"] = [
        {
            "label": f"Group {index}",
            "share": 1 / 15,
            "count": 10,
            "ci_low": None,
            "ci_high": None,
        }
        for index in range(15)
    ]

    html_out = to_html(mock_profile_result)

    assert "Group 11" in html_out
    assert "Group 12" not in html_out
    assert '<div class="dim-more">… and 3 more groups</div>' in html_out


def test_to_html_renders_proxy_hints(mock_profile_result):
    mock_profile_result["proxy_hints"] = [
        {"a": "sex", "b": "region", "p_value": 1.64e-22, "cramers_v": 0.98}
    ]

    html_out = to_html(mock_profile_result)

    assert "Proxy Hints" in html_out
    assert "sex" in html_out and "region" in html_out
    assert "0.98" in html_out


def test_to_html_omits_proxy_hints_section_when_absent(mock_profile_result):
    html_out = to_html(mock_profile_result)

    assert "Proxy Hints" not in html_out


def test_to_html_renders_reference_baseline_section(mock_profile_result):
    mock_profile_result["dimensions"][0]["reference"] = {
        "deviation": 0.2,
        "groups": [
            {"label": "Female", "expected": 0.5, "actual": 0.45, "delta": -0.05},
        ],
    }

    html_out = to_html(mock_profile_result)

    assert "Reference" in html_out
    assert "deviation 20.0%" in html_out
    assert "50.0%" in html_out and "45.0%" in html_out
    assert "-5.0 pp" in html_out


def test_to_html_reports_reference_groups_omitted_by_display_cap(mock_profile_result):
    """Mirrors test_to_html_reports_groups_omitted_by_display_cap for the
    reference-baseline sub-table, which was silently truncated with no
    dim-more notice."""
    mock_profile_result["dimensions"][0]["reference"] = {
        "deviation": 0.1,
        "groups": [
            {"label": f"Group {i}", "expected": 1 / 15, "actual": 1 / 15, "delta": 0.0}
            for i in range(15)
        ],
    }

    html_out = to_html(mock_profile_result)

    assert "Group 11" in html_out
    assert "Group 12" not in html_out
    assert '<div class="dim-more">… and 3 more groups</div>' in html_out


def test_to_html_omits_reference_section_when_absent(mock_profile_result):
    html_out = to_html(mock_profile_result)

    assert '<div class="reference">' not in html_out


def test_compare_to_html_smoke(mock_compare_result):
    """Smoke test for comparison drift report rendering."""
    html_out = compare_to_html(mock_compare_result)

    assert html_out.startswith("<!DOCTYPE html>")
    assert "</html>" in html_out
    assert "Representation Drift" in html_out


def test_compare_to_html_renders_key_figures(mock_compare_result):
    """Verify key drift metrics appear in the comparison report."""
    html_out = compare_to_html(mock_compare_result)

    assert "PSI" in html_out
    assert "0.125" in html_out
    assert "0.082" in html_out
    assert "significant" in html_out
    assert "Dataset A" in html_out
    assert "Dataset B" in html_out


def test_compare_to_html_styles_all_three_drift_levels(mock_compare_result):
    # Only .drift-badge.significant was styled; "moderate" and "none" fell
    # back to the flat grey base style, indistinguishable from each other
    # in the exported/CLI --html report (closes #322).
    html_out = compare_to_html(mock_compare_result)

    assert ".drift-badge.none" in html_out
    assert ".drift-badge.moderate" in html_out
    assert ".drift-badge.significant" in html_out


def test_compare_to_html_renders_added_removed_dimensions(mock_compare_result):
    """Mirrors test_compare_to_terminal_renders_added_removed_dimensions for
    the HTML branch, which had no equivalent coverage (#302)."""
    html_out = compare_to_html(mock_compare_result)

    assert "Only in B" in html_out and "NewDim" in html_out
    assert "Only in A" in html_out and "OldDim" in html_out


def test_compare_to_html_renders_appeared_disappeared_groups(mock_compare_result):
    """Mirrors compare_to_terminal's appeared/disappeared coverage for the
    HTML branch's class="gone"/"new" row styling and status tag (#302)."""
    mock_compare_result["dimensions"][0]["groups"] = [
        {"label": "Female", "status": "shifted", "share_a": 0.50, "share_b": 0.40, "share_delta": -0.10},
        {"label": "Non-binary", "status": "appeared", "share_a": 0.0, "share_b": 0.05, "share_delta": 0.05},
        {"label": "Prefer not to say", "status": "disappeared", "share_a": 0.03, "share_b": 0.0, "share_delta": -0.03},
    ]

    html_out = compare_to_html(mock_compare_result)

    assert 'class="drift-row new"' in html_out
    assert 'class="drift-row gone"' in html_out
    assert '<span class="tag">appeared</span>' in html_out
    assert '<span class="tag">disappeared</span>' in html_out
    assert "Non-binary" in html_out
    assert "Prefer not to say" in html_out


def test_to_terminal_smoke(mock_profile_result):
    """Smoke test for the default (no --json/--html) profile report."""
    out = to_terminal(mock_profile_result)

    assert out.startswith("=" * 20)
    assert "FAIR CODE - DATASET REPRESENTATION PROFILE" in out


def test_to_terminal_renders_key_figures(mock_profile_result):
    out = to_terminal(mock_profile_result)

    assert "1,250" in out
    assert "88/100" in out
    assert "B+" in out
    assert "Gender" in out
    assert "Female" in out
    assert "<- under-represented" not in out  # Female isn't in under_represented


def test_to_terminal_marks_under_represented_group(mock_profile_result):
    mock_profile_result["dimensions"][0]["under_represented"] = ["Female"]

    out = to_terminal(mock_profile_result)

    assert "Female" in out
    assert "<- under-represented" in out


def test_to_terminal_widens_for_a_long_group_label_and_keeps_alignment(mock_profile_result):
    label = "Not Hispanic or Latino"  # 22 chars: the column widens instead of cutting it (#1021)
    mock_profile_result["dimensions"][0]["groups"][0]["label"] = label

    out = to_terminal(mock_profile_result)

    rows = [line for line in out.splitlines() if "(n=" in line][:2]
    assert label in rows[0] and " 45.0%" in rows[0]
    assert len({line.index("%") for line in rows}) == 1      # every row's percentage lands in the same column


def test_to_terminal_marks_a_cut_reference_group_label(mock_profile_result):
    ref_label = "Some Long Reference Group Name That Is Way Past The Cap"  # over the 28-char maximum
    mock_profile_result["dimensions"][0]["reference"] = {
        "deviation": 0.05,
        "groups": [
            {"label": ref_label, "expected": 0.5, "actual": 0.45, "delta": -0.05},
        ],
    }

    out = to_terminal(mock_profile_result)

    assert ref_label[:27] + "…" in out
    assert ref_label not in out


def test_to_terminal_reports_reference_groups_omitted_by_display_cap(mock_profile_result):
    """SPEC section 7: every report surface that truncates a group list must
    state how many groups were omitted. The reference sub-table used to be
    silently capped at DISPLAY_GROUPS with no notice."""
    mock_profile_result["dimensions"][0]["reference"] = {
        "deviation": 0.1,
        "groups": [
            {"label": f"Group {i}", "expected": 1 / 15, "actual": 1 / 15, "delta": 0.0}
            for i in range(15)
        ],
    }

    out = to_terminal(mock_profile_result)

    assert "Group 11" in out
    assert "Group 12" not in out
    assert "… and 3 more groups" in out


def test_to_terminal_renders_flags_section(mock_profile_result):
    out = to_terminal(mock_profile_result)

    assert "FLAGS" in out
    assert "Gender under-representation detected" in out


def test_to_terminal_renders_proxy_hints(mock_profile_result):
    mock_profile_result["proxy_hints"] = [
        {"a": "sex", "b": "region", "p_value": 1.64e-22, "cramers_v": 0.98}
    ]

    out = to_terminal(mock_profile_result)

    assert "PROXY HINTS" in out
    assert "sex" in out and "region" in out


def test_to_terminal_no_dimensions_detected():
    result = {
        "n_rows": 10, "n_cols": 2, "overall_score": None, "grade": None,
        "dimensions_detected": False, "note": "No demographic columns detected.",
        "dimensions": [], "flags": [],
    }

    out = to_terminal(result)
    html_out = to_html(result)

    assert "No demographic columns detected." in out
    assert "Grade F" not in out
    assert "0/100" not in out
    assert "Not measured" in html_out
    assert "None/100" not in html_out


def test_compare_to_terminal_smoke(mock_compare_result):
    """Smoke test for the default (no --json/--html) compare report."""
    out = compare_to_terminal(mock_compare_result)

    assert out.startswith("=" * 20)
    assert "FAIR CODE - REPRESENTATION DRIFT" in out


def test_compare_to_terminal_renders_key_figures(mock_compare_result):
    out = compare_to_terminal(mock_compare_result)

    assert "Dataset A" in out
    assert "Dataset B" in out
    assert "PSI 0.125" in out
    assert "significant drift" in out
    assert "TVD 0.082" in out


def test_compare_to_terminal_renders_added_removed_dimensions(mock_compare_result):
    out = compare_to_terminal(mock_compare_result)

    assert "Only in B: NewDim" in out
    assert "Only in A: OldDim" in out


def test_compare_to_terminal_renders_drift_flags(mock_compare_result):
    out = compare_to_terminal(mock_compare_result)

    assert "DRIFT FLAGS" in out
    assert "High drift in Gender" in out


def test_compare_to_terminal_no_shared_dimensions():
    out = compare_to_terminal({
        "score_delta": 0,
        "a": {"name": "A", "overall_score": 100, "n_rows": 10, "grade": "A"},
        "b": {"name": "B", "overall_score": 100, "n_rows": 10, "grade": "A"},
        "added_dimensions": [], "removed_dimensions": [], "flags": [], "dimensions": [],
    })

    assert "No shared demographic dimensions to compare." in out


def test_compare_reports_unmeasured_score_without_formatting_none():
    result = {
        "score_delta": None,
        "a": {"name": "A", "overall_score": 100, "n_rows": 10, "grade": "A"},
        "b": {"name": "B", "overall_score": None, "n_rows": 10, "grade": None},
        "added_dimensions": [], "removed_dimensions": [], "flags": [], "dimensions": [],
    }

    terminal_out = compare_to_terminal(result)
    html_out = compare_to_html(result)

    assert "score not measured" in terminal_out
    assert "Overall score change: not available" in terminal_out
    assert "score change not available" in html_out
    assert "None" not in html_out


def test_compare_does_not_render_negative_zero_share_delta():
    """A tiny negative share_delta that rounds to 0.0 pp must not print as
    '-0.0 pp' in either the terminal or the HTML compare report."""
    result = {
        "score_delta": 0,
        "a": {"name": "A", "overall_score": 100, "n_rows": 10000, "grade": "A"},
        "b": {"name": "B", "overall_score": 100, "n_rows": 10000, "grade": "A"},
        "added_dimensions": [], "removed_dimensions": [], "flags": [],
        "dimensions": [
            {
                "name": "Gender", "kind": "Demographic", "kind_mismatch": False,
                "drift_level": "none",
                "psi": 0.0, "tvd": 0.0,
                "dimension_score_a": 100, "dimension_score_b": 100,
                "dimension_score_delta": 0,
                "groups": [
                    {"label": "Female", "status": "shifted",
                     "share_a": 0.5, "share_b": 0.4999, "share_delta": -0.0001},
                    {"label": "Male", "status": "shifted",
                     "share_a": 0.5, "share_b": 0.5001, "share_delta": 0.0001},
                ],
            }
        ],
    }

    assert "-0.0 pp" not in compare_to_terminal(result)
    assert "-0.0 pp" not in compare_to_html(result)


def test_compare_renderers_special_case_kind_mismatch():
    """A kind_mismatch dimension (#519's own compare() logic zeroes psi/tvd
    and empties groups since the comparison genuinely can't be measured, not
    because there's no drift) must render as "comparison skipped" in both
    the terminal and HTML compare reports, not as a fabricated "none drift"
    result sitting right next to the flag that says it was skipped (#619)."""
    result = {
        "score_delta": -1,
        "a": {"name": "A", "overall_score": 100, "n_rows": 40, "grade": "A"},
        "b": {"name": "B", "overall_score": 99, "n_rows": 50, "grade": "A"},
        "added_dimensions": [], "removed_dimensions": [],
        "flags": ["DOB: age values are banded in one dataset but left raw in the other - drift comparison skipped"],
        "dimensions": [
            {
                "name": "DOB", "kind": "age", "kind_a": "age", "kind_b": "age",
                "kind_mismatch": True,
                "dimension_score_a": 100, "dimension_score_b": 99,
                "dimension_score_delta": -1,
                "psi": 0.0, "tvd": 0.0, "drift_level": "none", "groups": [],
                "missing_pct_a": 0.0, "missing_pct_b": 0.0, "missing_pct_delta": 0.0,
            }
        ],
    }

    term = compare_to_terminal(result)
    html = compare_to_html(result)

    assert "comparison skipped" in term
    assert "none drift" not in term
    assert "PSI 0.000" not in term

    assert "comparison skipped" in html
    assert "none drift" not in html
    assert "PSI 0.000" not in html


def test_to_terminal_does_not_render_negative_zero_skew():
    """A tiny negative skewness that rounds to 0.00 must not print as
    '-0.00' in the terminal profile report."""
    result = {
        "n_rows": 60, "n_cols": 1, "overall_score": 88, "grade": "B+",
        "dimensions": [
            {
                "name": "Age", "kind": "Numeric", "dimension_score": 85,
                "under_represented": [], "n_groups": 1,
                "imbalance_ratio": None, "missing_pct": 0.0,
                "skewness": -0.0039,
                "groups": [{"label": "all", "share": 1.0, "count": 60}],
            }
        ],
        "flags": [],
    }

    assert "-0.00" not in to_terminal(result)
    assert "skew +0.00" in to_terminal(result)


def test_to_html_does_not_render_negative_zero_skew():
    """A tiny negative skewness that rounds to 0.00 must not print as
    '-0.00' in the HTML profile report."""
    result = {
        "n_rows": 60, "n_cols": 1, "overall_score": 88, "grade": "B+",
        "dimensions": [
            {
                "name": "Age", "kind": "Numeric", "dimension_score": 85,
                "under_represented": [], "n_groups": 1,
                "imbalance_ratio": None, "missing_pct": 0.0,
                "skewness": -0.0039,
                "groups": [{"label": "all", "share": 1.0, "count": 60}],
            }
        ],
        "flags": [],
    }

    assert "-0.00" not in to_html(result)
    assert "skew +0.00" in to_html(result)


def test_to_terminal_does_not_render_negative_zero_reference_delta():
    """A tiny negative reference delta that rounds to 0.0 pp must not print
    as '-0.0 pp' in the terminal profile report."""
    result = {
        "n_rows": 1000, "n_cols": 2, "overall_score": 88, "grade": "B+",
        "dimensions": [
            {
                "name": "Gender", "kind": "Demographic", "dimension_score": 85,
                "under_represented": [], "n_groups": 2,
                "imbalance_ratio": 1.6, "missing_pct": 0.0,
                "skewness": None,
                "groups": [{"label": "Female", "share": 0.5, "count": 500}],
                "reference": {
                    "deviation": 0.0,
                    "groups": [
                        {"label": "Female", "expected": 0.5, "actual": 0.4997, "delta": -0.0003},
                    ],
                },
            }
        ],
        "flags": [],
    }

    assert "-0.0 pp" not in to_terminal(result)
    assert "+0.0 pp" in to_terminal(result)


def test_to_html_does_not_render_negative_zero_reference_delta():
    """A tiny negative reference delta that rounds to 0.0 pp must not print
    as '-0.0 pp' in the HTML profile report."""
    result = {
        "n_rows": 1000, "n_cols": 2, "overall_score": 88, "grade": "B+",
        "dimensions": [
            {
                "name": "Gender", "kind": "Demographic", "dimension_score": 85,
                "under_represented": [], "n_groups": 2,
                "imbalance_ratio": 1.6, "missing_pct": 0.0,
                "skewness": None,
                "groups": [{"label": "Female", "share": 0.5, "count": 500}],
                "reference": {
                    "deviation": 0.0,
                    "groups": [
                        {"label": "Female", "expected": 0.5, "actual": 0.4997, "delta": -0.0003},
                    ],
                },
            }
        ],
        "flags": [],
    }

    assert "-0.0 pp" not in to_html(result)
    assert "+0.0 pp" in to_html(result)


def test_to_csv_has_one_row_per_group_across_all_dimensions():
    df = pd.DataFrame({
        "sex": ["M"] * 70 + ["F"] * 30,
        "race": ["White"] * 60 + ["Black"] * 25 + ["Asian"] * 15,
    })
    result = profile(df)

    rows = list(csv.reader(io.StringIO(to_csv(result))))
    header = rows[0]
    assert header == ["dimension", "kind", "label", "count", "share", "ci_low",
                       "ci_high", "under_represented", "small_group"]

    group_rows = rows[1:1 + sum(d["n_groups"] for d in result["dimensions"])]
    assert len(group_rows) == 5  # 2 sex groups + 3 race groups
    assert {r[0] for r in group_rows} == {"sex", "race"}

    # a blank line, then the flag section header, then one row per flag
    blank_index = 1 + len(group_rows)
    assert rows[blank_index] == []
    assert rows[blank_index + 1] == ["flag"]
    assert len(rows) == blank_index + 2 + len(result["flags"])


def test_to_csv_flags_under_represented_groups():
    df = pd.DataFrame({"sex": ["M"] * 195 + ["F"] * 5})
    result = profile(df)

    rows = list(csv.DictReader(io.StringIO(to_csv(result))))
    female = next(r for r in rows if r.get("label") == "F")
    assert female["under_represented"] == "True"
    male = next(r for r in rows if r.get("label") == "M")
    assert male["under_represented"] == "False"


def test_compare_to_csv_has_group_rows_a_summary_section_and_flags():
    df_a = pd.DataFrame({"sex": ["M"] * 70 + ["F"] * 30})
    df_b = pd.DataFrame({"sex": ["M"] * 50 + ["F"] * 50})
    cmp = compare(profile(df_a), profile(df_b), name_a="A", name_b="B")

    text = compare_to_csv(cmp)
    blocks = text.split("\r\n\r\n")
    assert len(blocks) == 3  # group rows, dimension summary, flags

    group_rows = list(csv.DictReader(io.StringIO(blocks[0])))
    assert {r["label"] for r in group_rows} == {"M", "F"}
    assert group_rows[0]["dimension"] == "sex"

    summary_rows = list(csv.DictReader(io.StringIO(blocks[1])))
    assert summary_rows[0]["dimension"] == "sex"
    assert summary_rows[0]["drift_level"] in ("none", "minor", "moderate", "significant")

    flag_rows = list(csv.DictReader(io.StringIO(blocks[2])))
    assert [r["flag"] for r in flag_rows] == cmp["flags"]


def test_to_csv_includes_proxy_hints_section_only_when_present():
    base = {"dimensions": [], "flags": []}
    assert "proxy_hint_a" not in to_csv(base)
    hints = [{"a": "sex", "b": "race", "p_value": 0.001, "cramers_v": 0.42, "chi2": 9.0}]
    rows = list(csv.reader(io.StringIO(to_csv({**base, "proxy_hints": hints}))))
    i = rows.index(["proxy_hint_a", "proxy_hint_b", "p_value", "cramers_v", "n_tests", "low_expected"])
    assert rows[i + 1] == ["sex", "race", "0.001", "0.42", "", "False"]


def test_compare_to_csv_includes_labelled_proxy_hints_per_dataset():
    base = {"dimensions": [], "flags": []}
    hint = {"a": "sex", "b": "race", "p_value": 0.01, "cramers_v": 0.3, "chi2": 5.0}
    text = compare_to_csv({**base, "proxy_hints_a": [hint], "proxy_hints_b": [hint]})
    rows = list(csv.reader(io.StringIO(text)))
    assert ["A", "sex", "race", "0.01", "0.3", "", "False"] in rows
    assert ["B", "sex", "race", "0.01", "0.3", "", "False"] in rows
    assert "dataset" in text


def test_to_csv_writes_a_reference_section_when_a_baseline_was_used():
    result = {"flags": [], "dimensions": [{
        "name": "sex", "kind": "sex", "under_represented": [],
        "groups": [{"label": "f", "count": 1, "share": 0.5, "ci_low": 0.1, "ci_high": 0.9,
                    "small_group": False}],
        "reference": {"deviation": 0.1, "groups": [
            {"label": "f", "expected": 0.5, "actual": 0.4, "delta": -0.1}]},
    }]}
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    i = rows.index(["dimension", "reference_label", "expected", "actual", "delta",
                    "reference_deviation"])
    assert rows[i + 1] == ["sex", "f", "0.5", "0.4", "-0.1", "0.1"]


def test_csv_exports_defuse_spreadsheet_formula_injection():
    """#791: untrusted labels that start like a formula get a ' prefix; real
    negative numbers are left numeric."""
    result = {"flags": ["=cmd"], "dimensions": [{
        "name": "+x", "kind": "categorical", "under_represented": [],
        "groups": [{"label": '=HYPERLINK("http://x")', "count": 1, "share": -0.5,
                    "ci_low": None, "ci_high": None, "small_group": False}],
    }]}
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    assert rows[1][0] == "'+x" and rows[1][2] == "'=HYPERLINK(\"http://x\")"
    assert rows[1][4] == "-0.5"
    assert ["'=cmd"] in rows


def test_terminal_reports_escape_control_characters_from_dataset_values():
    """#845: an escape sequence in a group label or column name must not reach the terminal."""
    df = pd.DataFrame({"sex": ["\x1b[31mRED", "F", "M", "F"], "age\x1b]0;pwned\x07": [20, 30, 40, 50]})
    result = profile(df)
    out = to_terminal(result)
    assert "\x1b" not in out and "\x07" not in out
    assert "\\x1b[31mRED" in out
    # JSON-style data stays lossless
    assert any(g["label"] == "\x1b[31mRED" for d in result["dimensions"] for g in d["groups"])


def test_compare_terminal_escapes_control_characters():
    a = profile(pd.DataFrame({"sex": ["\x1b[2JM", "F", "M", "F"]}))
    b = profile(pd.DataFrame({"sex": ["\x1b[2JM", "F", "F", "F"]}))
    out = compare_to_terminal(compare(a, b, name_a="a\x1b[1m.csv", name_b="b.csv"))
    assert "\x1b" not in out
    assert "\\x1b[1m" in out


def test_compare_csv_lists_dimensions_present_in_only_one_dataset():
    """#842: removed/added dimensions get dimension,present_in rows, not just a flag."""
    base = {"dimensions": [], "flags": [], "removed_dimensions": ["race"],
            "added_dimensions": ["region", "age"]}
    rows = list(csv.reader(io.StringIO(compare_to_csv(base))))
    i = rows.index(["dimension", "present_in"])
    assert rows[i + 1:i + 4] == [["race", "a_only"], ["region", "b_only"], ["age", "b_only"]]
    plain = compare_to_csv({"dimensions": [], "flags": [], "removed_dimensions": [],
                            "added_dimensions": []})
    assert "present_in" not in plain


def _compare_with_hints():
    import pandas as pd
    from faircode import compare, profile
    a = profile(pd.DataFrame({"sex": ["m", "f"] * 30}))
    cmp = compare(a, a, "a.csv", "b.csv")
    hint = {"a": "sex", "b": "<occupation>", "p_value": 0.00123, "cramers_v": 0.42, "chi2": 10.0,
            "low_expected_share": 0.5, "low_expected": True, "p_method": "permutation", "n_tests": 1}
    capped = dict(hint, a="age", b="job & title", p_method="chi2-capped")
    cmp["proxy_hints_a"] = [hint, capped]
    cmp["proxy_hints_b"] = [dict(hint, b="region", low_expected=False, p_method="chi2")]
    return cmp


def test_compare_terminal_report_renders_proxy_hint_sections_for_both_sides():
    # #960
    from faircode.report import compare_to_terminal
    text = compare_to_terminal(_compare_with_hints())
    assert "PROXY HINTS - A" in text and "PROXY HINTS - B" in text
    assert "sex ↔ <occupation>" in text and "χ² p=0.00123" in text and "Cramér's V=0.42" in text
    assert "small cells (permutation exact p)" in text
    assert "too large for the exact fallback" in text            # chi2-capped note (#885)
    assert text.index("PROXY HINTS - A") < text.index("PROXY HINTS - B")
    plain = _compare_with_hints()
    del plain["proxy_hints_a"], plain["proxy_hints_b"]
    assert "PROXY HINTS" not in compare_to_terminal(plain)


def test_compare_html_report_renders_and_escapes_proxy_hint_sections():
    # #960: column names are user data, so the HTML must escape them.
    from faircode.report import compare_to_html
    html = compare_to_html(_compare_with_hints())
    assert "Proxy Hints - A" in html and "Proxy Hints - B" in html
    assert "sex ↔ &lt;occupation&gt;" in html and "job &amp; title" in html
    assert "<occupation>" not in html and "small cells (permutation exact p)" in html
    assert "too large for the exact fallback" in html


def test_terminal_and_html_reports_say_imbalance_inf_for_an_empty_subgroup():
    # #975
    from js_helpers import result_with_an_empty_subgroup
    result = result_with_an_empty_subgroup()
    assert "(imbalance inf (empty subgroup))" in to_terminal(result)
    assert "imbalance inf (empty subgroup)" in to_html(result)


def _fifteen_group_profile():
    df = pd.DataFrame({"tier": [f"grp{i % 15:02d}" for i in range(150)]})
    return profile(df, {"tier": "categorical"}, {"max_categorical_card": 30})


def test_terminal_reports_cap_a_long_dimension_with_an_and_n_more_groups_line():
    # #976: DISPLAY_GROUPS (12) rows are shown, the other three are counted in a closing line.
    result = _fifteen_group_profile()
    assert len(result["dimensions"][0]["groups"]) == 15
    text = to_terminal(result)
    assert text.count("  … and 3 more groups") == 1
    assert not any(line.startswith("  grp14") for line in text.splitlines())   # (it still appears in the flags)
    drift = compare_to_terminal(compare(result, result, "a.csv", "b.csv"))
    assert drift.count("  … and 3 more groups") == 1
    assert not any(line.startswith("  grp14") for line in drift.splitlines())


def test_compare_terminal_names_the_two_detected_kinds_for_a_kind_mismatch():
    # #976: sex on one side and race on the other (what --map does) skips the drift comparison
    # and says why.
    side_a = profile(pd.DataFrame({"col": ["m", "f"] * 40}), {"col": "sex"})
    side_b = profile(pd.DataFrame({"col": ["x", "y"] * 40}), {"col": "race"})
    text = compare_to_terminal(compare(side_a, side_b, "a.csv", "b.csv"))
    assert "  detected as sex vs race on the two datasets" in text


def test_compare_html_caps_a_long_dimension_and_survives_all_zero_shares():
    # #1001: the HTML drift card has its own "more groups" line and its own bar-scaling guard.
    result = _fifteen_group_profile()
    cmp_result = compare(result, result, "a.csv", "b.csv")
    html_out = compare_to_html(cmp_result)
    assert html_out.count('<div class="dim-more">… and 3 more groups</div>') == 1
    assert "grp11" in html_out and "grp12" not in html_out
    for group in cmp_result["dimensions"][0]["groups"]:
        group.update(share_a=0.0, share_b=0.0, share_delta=0.0)
    flat = compare_to_html(cmp_result)           # max_share would be 0: must not divide by zero
    assert "nan" not in flat.lower() and "inf" not in flat.lower().replace("information", "")


def test_terminal_reports_keep_long_group_labels_apart():
    # #1021: two census-style labels that share their first 18 characters must not print alike.
    long_a, long_b = "Hispanic or Latino (any race)", "Hispanic or Latino (white)"
    df = pd.DataFrame({"race": [long_a, long_b] * 20, "sex": ["m", "f", "f"] * 13 + ["m"]})
    result = profile(df)
    text = to_terminal(result)
    assert long_a in text and long_b in text            # both fit in the widened column
    very_long = "x" * 40
    clipped = to_terminal(profile(pd.DataFrame({"race": [very_long, very_long + "y"] * 20})))
    rows = [line for line in clipped.splitlines() if "(n=" in line]
    assert rows and all("x" * 29 + "…" in line and very_long not in line for line in rows)   # cut with a marker
    ref = profile(df, opts={"reference": {"race": {long_a: 0.5, long_b: 0.5}}})
    assert "Hispanic or Latino (any race)" in to_terminal(ref).split("reference (deviation")[1]
    drift = compare_to_terminal(compare(result, result, "a.csv", "b.csv"))
    assert long_a in drift and long_b in drift


def test_a_requested_proxy_check_with_no_hint_says_so():
    # #1022: an empty list is "tested, none found" (or "no pair possible"); an absent key prints nothing.
    two = profile(pd.DataFrame({"sex": ["m", "f"] * 20, "race": ["a", "a", "b", "b"] * 10}))
    one = profile(pd.DataFrame({"sex": ["m", "f"] * 20}))
    assert "PROXY HINTS" not in to_terminal(two) and "Proxy Hints" not in to_html(two)
    for result, text in ((dict(two, proxy_hints=[]), "none found: no pair of dimensions is significantly associated"),
                         (dict(one, proxy_hints=[]), "none tested: a pair needs at least two dimensions (found 1)")):
        assert text in to_terminal(result) and text in to_html(result)
    cmp_result = dict(compare(two, two, "a.csv", "b.csv"), proxy_hints_a=[], proxy_hints_b=[])
    assert compare_to_terminal(cmp_result).count("none found") == 2
    assert compare_to_html(cmp_result).count("none found") == 2
