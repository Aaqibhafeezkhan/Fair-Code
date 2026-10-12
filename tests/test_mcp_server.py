"""Tests for the MCP server's tool-logic functions (needs the mcp extra).

These exercise the plain `_*_impl` functions directly rather than going
through a live stdio session - the module docstring explains why (keeps the
`mcp` SDK's own API out of the unit of testing). server-registration/schema
smoke coverage is one small test at the bottom.

Run from the repo root:  pytest tests/ -q
"""

import importlib.util

import pytest

pytest.importorskip("mcp", reason="MCP tools need the optional mcp extra")

from faircode.mcp_server import (  # noqa: E402
    _compare_datasets_impl,
    RESULTS_FROZEN_FILES,
    _get_benchmark_results_impl,
    _get_explainer_impl,
    _list_explainers_impl,
    _profile_dataset_impl,
    _proxy_hints_impl,
    build_server,
)

requires_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="optional 'proxy' extra not installed",
)

requires_openpyxl = pytest.mark.skipif(
    importlib.util.find_spec("openpyxl") is None,
    reason="optional 'excel' extra not installed",
)


@requires_scipy
def test_proxy_hints_held_out_file_can_override_the_dataset_encoding(tmp_path):
    dataset = tmp_path / "dataset-latin1.csv"
    held = tmp_path / "held-utf8.csv"
    dataset.write_bytes(("sex\n" + "M\nF\n" * 20).encode("latin-1"))
    held.write_bytes(("ráce\n" + "A\nB\n" * 20).encode("utf-8"))

    result = _proxy_hints_impl(
        str(dataset),
        held_out_with=[f"{held}=ráce@utf-8"],
        encoding="latin-1",
    )

    entry = result["provenance"]["proxy_hints_with"][0]
    assert entry["column"] == "ráce"
    assert entry["encoding"] == "utf-8"


def _write_multi_sheet_xlsx(path, first_col, first_rows):
    import openpyxl

    wb = openpyxl.Workbook()
    first = wb.active
    first.title = "Data"
    first.append([first_col])
    for row in first_rows:
        first.append([row])
    wb.create_sheet("Notes")
    wb.save(path)


def test_profile_dataset_matches_the_shape_profile_returns(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\n" + "M\n" * 8 + "F\n" * 2, encoding="utf-8")

    result = _profile_dataset_impl(str(path))

    assert set(result) == {"n_rows", "n_cols", "overall_score", "grade",
                          "dimensions_detected", "note", "dimensions",
                          "intersections", "flags", "provenance"}
    assert result["n_rows"] == 10


def test_profile_dataset_exposes_unmeasured_state(tmp_path):
    path = tmp_path / "identifiers.csv"
    path.write_text("id\n" + "\n".join(str(i) for i in range(40)), encoding="utf-8")

    result = _profile_dataset_impl(str(path), include_provenance=False)

    assert result["overall_score"] is None
    assert result["grade"] is None
    assert result["dimensions_detected"] is False
    assert result["note"] == "No demographic columns detected."


def test_profile_dataset_provenance_default_on_and_matches_the_file_hash(tmp_path):
    import hashlib

    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path))

    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["provenance"]["dataset_hash"] == expected
    assert result["provenance"]["engine"] == "python"


def test_profile_dataset_include_provenance_false_omits_the_block(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path), include_provenance=False)

    assert "provenance" not in result


def test_profile_dataset_unknown_file_raises_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        _profile_dataset_impl(str(tmp_path / "does-not-exist.csv"))


def test_profile_dataset_rejects_stdin_shorthand():
    # "-" reads real stdin at the CLI, but MCP runs over a stdio transport
    # where stdin IS the JSON-RPC channel - must be rejected, not attempted.
    with pytest.raises(ValueError, match="stdin input"):
        _profile_dataset_impl("-")


def test_compare_datasets_rejects_stdin_shorthand(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stdin input"):
        _compare_datasets_impl(str(path), "-")


def test_proxy_hints_rejects_stdin_shorthand():
    with pytest.raises(ValueError, match="stdin input"):
        _proxy_hints_impl("-")


def test_profile_dataset_unknown_override_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overrides column"):
        _profile_dataset_impl(str(path), overrides={"not_a_real_column": "sex"})


@requires_openpyxl
def test_profile_dataset_reports_ignored_sheets(tmp_path):
    path = tmp_path / "multi_sheet.xlsx"
    _write_multi_sheet_xlsx(path, "sex", ["M"] * 5 + ["F"] * 5)

    result = _profile_dataset_impl(str(path))

    assert result["sheet_note"] == "read sheet 'Data' - 1 other sheet(s) ignored"


def test_profile_dataset_single_sheet_xlsx_has_no_sheet_note(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path))

    assert "sheet_note" not in result


@requires_openpyxl
def test_profile_dataset_xlsx_encoding_adds_notes(tmp_path):
    """#870: MCP returns a notes entry when encoding is set for .xlsx."""
    path = tmp_path / "data.xlsx"
    _write_multi_sheet_xlsx(path, "sex", ["M", "F"])

    result = _profile_dataset_impl(str(path), encoding="latin-1")

    assert result["notes"] == ["encoding has no effect on .xlsx/.parquet files"]


def test_profile_dataset_csv_encoding_has_no_notes(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path), encoding="latin-1")

    assert "notes" not in result


def test_profile_dataset_invalid_override_kind_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid kind"):
        _profile_dataset_impl(str(path), overrides={"sex": "not_a_real_kind"})


def test_profile_dataset_cross_with_one_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex,race\nM,W\nF,B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cross expects exactly two"):
        _profile_dataset_impl(str(path), cross=["sex"])


def test_profile_dataset_cross_unknown_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex,race\n" + "M,W\nF,B\n" * 25, encoding="utf-8")

    with pytest.raises(ValueError, match="raace"):
        _profile_dataset_impl(str(path), cross=["sex", "raace"])


def test_profile_dataset_cross_same_column_twice_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex,race\n" + "M,W\nF,B\n" * 25, encoding="utf-8")

    with pytest.raises(ValueError, match="cross needs two different columns, got 'sex' twice"):
        _profile_dataset_impl(str(path), cross=["sex", "sex"])


def test_profile_dataset_reference_path_adds_reference_hash_to_provenance(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\n" + "M\n" * 5 + "F\n" * 5, encoding="utf-8")
    ref = tmp_path / "ref.csv"
    ref.write_text("column,group,share\nsex,M,0.5\nsex,F,0.5\n", encoding="utf-8")

    result = _profile_dataset_impl(str(path), reference_path=str(ref))

    assert "reference_hash" in result["provenance"]
    assert result["dimensions"][0]["reference"] is not None


def test_compare_datasets_matches_the_shape_compare_returns(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\n" + "M\n" * 8 + "F\n" * 2, encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\n" + "M\n" * 5 + "F\n" * 5, encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert set(result) >= {"a", "b", "score_delta", "dimensions",
                           "added_dimensions", "removed_dimensions", "flags"}
    assert "provenance" in result


def test_compare_datasets_provenance_has_both_dataset_hashes(tmp_path):
    import hashlib

    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\nM\nF\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\nM\nM\n", encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert result["provenance"]["dataset_hash_a"] == "sha256:" + hashlib.sha256(path_a.read_bytes()).hexdigest()
    assert result["provenance"]["dataset_hash_b"] == "sha256:" + hashlib.sha256(path_b.read_bytes()).hexdigest()


def test_compare_datasets_unknown_override_column_raises(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\nM\nF\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\nM\nM\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overrides column"):
        _compare_datasets_impl(str(path_a), str(path_b), overrides={"nope": "sex"})


def test_profile_and_compare_thread_max_categorical_card_and_max_dimension_groups(tmp_path):
    # Same cases as the profiler's own tunable tests: 30 distinct codes sit
    # outside the default generic-categorical window, 60 "ethnicity" groups
    # sit past the default identifier/date-like cutoff (#759).
    card = tmp_path / "card.csv"
    card.write_text("occupation_code\n" + "".join(f"occ_{i % 30}\n" for i in range(300)),
                    encoding="utf-8")
    groups = tmp_path / "groups.csv"
    groups.write_text("ethnicity\n" + "".join(f"group_{i % 60}\n" for i in range(600)),
                      encoding="utf-8")

    assert _profile_dataset_impl(str(card))["dimensions"] == []
    result = _profile_dataset_impl(str(card), max_categorical_card=30)
    assert [d["name"] for d in result["dimensions"]] == ["occupation_code"]
    assert result["provenance"]["params"]["max_categorical_card"] == 30

    assert _profile_dataset_impl(str(groups))["dimensions"] == []
    result = _profile_dataset_impl(str(groups), max_dimension_groups=60)
    assert [d["name"] for d in result["dimensions"]] == ["ethnicity"]

    assert _compare_datasets_impl(str(card), str(card))["dimensions"] == []
    result = _compare_datasets_impl(str(card), str(card), max_categorical_card=30)
    assert [d["name"] for d in result["dimensions"]] == ["occupation_code"]
    result = _compare_datasets_impl(str(groups), str(groups), max_dimension_groups=60)
    assert [d["name"] for d in result["dimensions"]] == ["ethnicity"]


@requires_openpyxl
def test_compare_datasets_reports_ignored_sheets_for_both_files(tmp_path):
    path_a = tmp_path / "a.xlsx"
    path_b = tmp_path / "b.xlsx"
    _write_multi_sheet_xlsx(path_a, "sex", ["M"] * 5 + ["F"] * 5)
    _write_multi_sheet_xlsx(path_b, "sex", ["M"] * 5 + ["F"] * 5)

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert result["sheet_note_a"] == "read sheet 'Data' - 1 other sheet(s) ignored"
    assert result["sheet_note_b"] == "read sheet 'Data' - 1 other sheet(s) ignored"


def test_compare_datasets_proxy_hints_defaults_off(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("sex\nM\nF\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("sex\nM\nM\n", encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b))

    assert "proxy_hints_a" not in result
    assert "proxy_hints_b" not in result


@requires_scipy
def test_compare_datasets_proxy_hints_attaches_hints_for_both_datasets(tmp_path):
    # occupation is a perfect function of sex in both files -> maximal association.
    rows = ["sex,occupation"] + [
        f"{'male' if i % 2 == 0 else 'female'},{'engineer' if i % 2 == 0 else 'nurse'}"
        for i in range(100)
    ]
    path_a = tmp_path / "a.csv"
    path_a.write_text("\n".join(rows) + "\n", encoding="utf-8")
    path_b = tmp_path / "b.csv"
    path_b.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = _compare_datasets_impl(str(path_a), str(path_b), proxy_hints=True)

    for key in ("proxy_hints_a", "proxy_hints_b"):
        pair = next(h for h in result[key] if {h["a"], h["b"]} == {"sex", "occupation"})
        assert pair["p_value"] < 0.05


@requires_scipy
def test_proxy_hints_returns_a_dict_with_a_hints_key(tmp_path):
    # occupation is a perfect function of sex -> maximal association.
    rows = ["sex,occupation"] + [
        f"{'male' if i % 2 == 0 else 'female'},{'engineer' if i % 2 == 0 else 'nurse'}"
        for i in range(100)
    ]
    path = tmp_path / "a.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = _proxy_hints_impl(str(path))

    assert set(result) == {"hints", "provenance"}
    pair = next(h for h in result["hints"] if {h["a"], h["b"]} == {"sex", "occupation"})
    assert pair["p_value"] < 0.05


def test_proxy_hints_with_no_significant_pairs_returns_an_empty_list_not_an_error(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    result = _proxy_hints_impl(str(path))

    assert result["hints"] == []
    assert "provenance" in result


def test_proxy_hints_runtime_error_propagates_with_a_clean_message(tmp_path, monkeypatch):
    import faircode.mcp_server as mcp_server

    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    def raise_runtime(_df, _dimensions, **_kwargs):
        raise RuntimeError("proxy hints need scipy (install with: pip install faircode[proxy])")

    monkeypatch.setattr(mcp_server, "compute_proxy_hints", raise_runtime)

    with pytest.raises(RuntimeError, match="proxy hints need scipy"):
        _proxy_hints_impl(str(path))


@requires_openpyxl
def test_proxy_hints_reports_ignored_sheets_for_main_path_and_held_out(tmp_path):
    path = tmp_path / "a.xlsx"
    held_path = tmp_path / "held.xlsx"
    _write_multi_sheet_xlsx(path, "sex", ["M"] * 5 + ["F"] * 5)
    _write_multi_sheet_xlsx(held_path, "race", ["A"] * 5 + ["B"] * 5)

    result = _proxy_hints_impl(str(path), held_out_with=[f"{held_path}=race"])

    assert result["sheet_notes"] == [
        "read sheet 'Data' - 1 other sheet(s) ignored",
        "read sheet 'Data' - 1 other sheet(s) ignored",
    ]


@requires_scipy
def test_proxy_hints_held_out_with_flags_a_dropped_column(tmp_path):
    # zip_code is perfectly aligned with race, which has been dropped from
    # the profiled dataset - only visible via held_out_with.
    path = tmp_path / "dropped.csv"
    held_path = tmp_path / "full.csv"
    zip_code = (["111"] * 100 + ["222"] * 100)
    race = (["A"] * 100 + ["B"] * 100)
    path.write_text("zip_code\n" + "\n".join(zip_code), encoding="utf-8")
    held_path.write_text("zip_code,race\n" +
                         "\n".join(f"{z},{r}" for z, r in zip(zip_code, race)),
                         encoding="utf-8")

    result = _proxy_hints_impl(str(path), held_out_with=[f"{held_path}=race"])

    pair = next(h for h in result["hints"] if {h["a"], h["b"]} == {"zip_code", "race"})
    assert pair["p_value"] < 0.05


def test_proxy_hints_held_out_with_malformed_spec_raises_value_error(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("sex\nM\nF\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid held_out_with 'noequalssign'"):
        _proxy_hints_impl(str(path), held_out_with=["noequalssign"])


def test_proxy_hints_held_out_with_column_collision_raises_value_error(tmp_path):
    path = tmp_path / "a.csv"
    held_path = tmp_path / "b.csv"
    path.write_text("sex,race\nM,A\nF,B\n", encoding="utf-8")
    held_path.write_text("race\nX\nY\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column 'race' already exists"):
        _proxy_hints_impl(str(path), held_out_with=[f"{held_path}=race"])


def test_list_explainers_returns_metadata_for_every_published_explainer():
    result = _list_explainers_impl()

    assert result["explainers"]  # the real, current published set - not empty
    slugs = {e["slug"] for e in result["explainers"]}
    assert "proxy-variables" in slugs
    entry = next(e for e in result["explainers"] if e["slug"] == "proxy-variables")
    assert set(entry) == {"slug", "title", "subtitle", "summary", "tags"}
    assert entry["title"] == "Proxy Variables"


def test_list_explainers_filters_by_tag():
    result = _list_explainers_impl(tag="detection")

    assert result["explainers"]
    assert all("detection" in e["tags"] for e in result["explainers"])


def test_list_explainers_unknown_tag_raises():
    with pytest.raises(ValueError, match="no explainer has tag"):
        _list_explainers_impl(tag="not-a-real-tag")


def test_get_explainer_returns_content_and_metadata():
    result = _get_explainer_impl("proxy-variables")

    assert result["slug"] == "proxy-variables"
    assert result["title"] == "Proxy Variables"
    assert "# Explainer: What is a Proxy Variable?" in result["content"]


def test_get_explainer_unknown_slug_raises():
    with pytest.raises(FileNotFoundError, match="no explainer found for slug"):
        _get_explainer_impl("not-a-real-explainer")


@pytest.mark.parametrize("slug", [
    "../README",
    "../../README",
    "../CLAUDE",
    "explainers/../../README",
    "sub/dir",
    "UPPERCASE",
    "has space",
    "trailing.md.md",
])
def test_get_explainer_rejects_path_traversal_and_malformed_slugs(slug):
    # slug is caller-controlled text used to build a filesystem path - a
    # traversal-shaped value used to escape explainers/ and read an
    # arbitrary file elsewhere in the repo (issue #387).
    with pytest.raises(FileNotFoundError, match="no explainer found for slug"):
        _get_explainer_impl(slug)


def test_get_benchmark_results_filters_to_matching_rows():
    # A real, verifiable anchor: COMPAS logistic_regression baseline's race
    # equal_opportunity_diff is cited as 0.926 in explainers/equal-opportunity.md.
    result = _get_benchmark_results_impl(
        audit="compas", model="logistic_regression", strategy="baseline",
        protected_attribute="race", metric="equal_opportunity_diff")

    assert result["total_matches"] == 1
    assert result["truncated"] is False
    row = result["results"][0]
    assert round(row["value"], 3) == 0.926
    assert row["significant"] is True


def test_get_benchmark_results_rejects_filter_on_column_absent_from_kind():
    # results_performance.csv has no protected_attribute column. Silently
    # dropping the filter handed the caller the whole unfiltered table back
    # with no signal their filter was ignored (#512); it now raises, like
    # profile_dataset/compare_datasets do for an unknown overrides column.
    with pytest.raises(ValueError, match="protected_attribute.*performance"):
        _get_benchmark_results_impl(
            kind="performance", audit="compas", model="logistic_regression",
            protected_attribute="race")

    # The same filter against the fairness kind, which does have the column,
    # still works.
    ok = _get_benchmark_results_impl(
        kind="fairness", audit="compas", model="logistic_regression",
        protected_attribute="race")
    assert ok["total_matches"] > 0


def test_get_benchmark_results_nan_cells_become_none():
    # results_performance.csv's auc rows have no ci_low/ci_high.
    result = _get_benchmark_results_impl(
        kind="performance", audit="compas", model="logistic_regression",
        strategy="baseline", metric="auc")

    assert result["total_matches"] == 1
    assert result["results"][0]["ci_low"] is None
    assert result["results"][0]["ci_high"] is None


def test_get_benchmark_results_rounds_float_precision():
    # Raw CSV value is -0.06555690919543475 (17 significant digits) - the row
    # cap alone still leaves a large payload once every float carries that
    # many digits it doesn't need for a fairness-metric lookup (#395).
    result = _get_benchmark_results_impl(
        audit="ai_fair_recruitment", model="logistic_regression",
        strategy="baseline", protected_attribute="gender",
        metric="demographic_parity_diff")

    row = result["results"][0]
    assert row["value"] == -0.065557
    for key in ("value", "ci_low", "ci_high", "p_value"):
        decimals = str(float(row[key])).split(".")[-1]
        assert len(decimals) <= 6


def test_get_benchmark_results_no_match_returns_empty_not_an_error():
    result = _get_benchmark_results_impl(audit="not_a_real_audit")

    assert result == {"results": [], "total_matches": 0, "truncated": False}


def test_get_benchmark_results_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        _get_benchmark_results_impl(kind="not_a_real_kind")


def test_get_benchmark_results_missing_mirror_names_path_and_safe_recovery(
        tmp_path, monkeypatch):
    missing = tmp_path / "faircode" / "_results_frozen" / "results_fairness.csv"
    monkeypatch.setitem(RESULTS_FROZEN_FILES, "fairness", missing)

    with pytest.raises(FileNotFoundError) as exc_info:
        _get_benchmark_results_impl()

    message = str(exc_info.value)
    assert str(missing) in message
    assert "scripts.freeze_paper_results.mirror_for_mcp()" in message
    assert "may not have been frozen" not in message


@pytest.mark.parametrize("bad_value", [{"x": 1}, ["a", "b"], (1, 2)])
def test_get_benchmark_results_non_scalar_filter_raises_clean_error(bad_value):
    # A dict raised an uncaught NotImplementedError and a list raised a raw
    # pandas ValueError with internal shape-tuple wording - neither is the
    # clean, caught error every other validation failure in this file gives
    # (issue #390).
    with pytest.raises(ValueError, match="audit must be a plain string"):
        _get_benchmark_results_impl(audit=bad_value)


def test_build_server_registers_all_phase_one_and_phase_two_tools():
    import asyncio

    server = build_server()
    tools = asyncio.run(server.list_tools())

    assert {t.name for t in tools} == {
        "profile_dataset", "compare_datasets", "proxy_hints",
        "list_explainers", "get_explainer", "get_benchmark_results",
    }


def test_tool_errors_surface_the_anticipated_message_not_a_generic_one():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool("profile_dataset", {"path": "definitely-missing.csv"})

    with pytest.raises(ToolError, match="file not found: definitely-missing.csv"):
        asyncio.run(call())


def test_compare_datasets_tool_error_surfaces_via_call_tool():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool(
            "compare_datasets", {"path_a": "definitely-missing.csv", "path_b": "also-missing.csv"})

    with pytest.raises(ToolError, match="file not found: definitely-missing.csv"):
        asyncio.run(call())


def test_proxy_hints_tool_error_surfaces_via_call_tool():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool("proxy_hints", {"path": "definitely-missing.csv"})

    with pytest.raises(ToolError, match="file not found: definitely-missing.csv"):
        asyncio.run(call())


def test_get_explainer_tool_error_surfaces_via_call_tool():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool("get_explainer", {"slug": "not-a-real-explainer"})

    with pytest.raises(ToolError, match="no explainer found for slug"):
        asyncio.run(call())


def test_get_benchmark_results_tool_error_surfaces_via_call_tool():
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    server = build_server()

    async def call():
        return await server.call_tool("get_benchmark_results", {"kind": "not_a_real_kind"})

    with pytest.raises(ToolError, match="kind must be one of"):
        asyncio.run(call())


@requires_scipy
def test_compare_datasets_held_out_with_a_flags_a_dropped_column_on_that_side(tmp_path):
    """#782: compare_datasets mirrors the CLI's --proxy-hints-with-a/-b."""
    zip_code = ["111"] * 100 + ["222"] * 100
    race = ["A"] * 100 + ["B"] * 100
    a = tmp_path / "a.csv"
    a.write_text("zip_code\n" + "\n".join(zip_code), encoding="utf-8")
    full = tmp_path / "full.csv"
    full.write_text("zip_code,race\n" + "\n".join(f"{z},{r}" for z, r in zip(zip_code, race)),
                    encoding="utf-8")
    b = tmp_path / "b.csv"
    b.write_text("zip_code\n" + "\n".join(zip_code), encoding="utf-8")

    result = _compare_datasets_impl(str(a), str(b), proxy_hints=True,
                                    held_out_with_a=[f"{full}=race"])

    assert any({h["a"], h["b"]} == {"zip_code", "race"} for h in result["proxy_hints_a"])
    assert result["proxy_hints_b"] == []


def test_compare_datasets_held_out_needs_proxy_hints(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("sex\nM\nF\n", encoding="utf-8")
    with pytest.raises(ValueError, match="need proxy_hints"):
        _compare_datasets_impl(str(a), str(a), held_out_with_a=[f"{a}=sex"])


@requires_scipy
def test_proxy_hints_alpha_is_tunable_and_validated(tmp_path):
    """#786: a stricter alpha drops a borderline pair; invalid alpha is rejected."""
    rows = ["sex,occupation"] + [
        f"{'male' if i % 2 == 0 else 'female'},{'engineer' if i % 2 == 0 else 'nurse'}"
        for i in range(100)
    ]
    p = tmp_path / "d.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert _proxy_hints_impl(str(p))["hints"]
    assert _proxy_hints_impl(str(p), alpha=1e-300)["hints"] == []
    with pytest.raises(ValueError, match="alpha"):
        _proxy_hints_impl(str(p), alpha=0)


@requires_scipy
def test_proxy_hints_correction_param_adds_p_adjusted_and_rejects_unknown(tmp_path):
    """#806: MCP mirrors --proxy-correction."""
    rows = ["sex,occupation"] + [
        f"{'male' if i % 2 == 0 else 'female'},{'engineer' if i % 2 == 0 else 'nurse'}"
        for i in range(100)
    ]
    p = tmp_path / "d.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert "p_adjusted" in _proxy_hints_impl(str(p), correction="holm")["hints"][0]
    assert "p_adjusted" not in _proxy_hints_impl(str(p))["hints"][0]
    with pytest.raises(ValueError, match="correction"):
        _proxy_hints_impl(str(p), correction="fdr")


def test_as_format_csv_matches_the_cli_writers_and_rejects_unknown(tmp_path):
    """#807: MCP's csv format is byte-identical to faircode's to_csv/compare_to_csv."""
    from faircode.mcp_server import _as_format
    from faircode.report import compare_to_csv, to_csv

    p = tmp_path / "a.csv"
    p.write_text("sex\nM\nF\nM\nF\n", encoding="utf-8")
    prof = _profile_dataset_impl(str(p))
    out = _as_format(prof, "csv", to_csv)
    assert out == {"csv": to_csv(prof, provenance=prof["provenance"])}
    assert out["csv"].startswith("dimension,kind,label,count,share,")
    assert "provenance_key" in out["csv"]
    plain = _as_format(_profile_dataset_impl(str(p), include_provenance=False), "csv", to_csv)
    assert "provenance_key" not in plain["csv"]
    cmp = _compare_datasets_impl(str(p), str(p))
    assert _as_format(cmp, "csv", compare_to_csv)["csv"].startswith("dimension,kind_a,kind_b,")
    assert _as_format(prof, "json", to_csv) is prof
    with pytest.raises(ValueError, match="format must be"):
        _as_format(prof, "xml", to_csv)


def test_proxy_hints_provenance_default_on_and_includes_params_and_held_out(tmp_path):
    """#860: proxy_hints attaches provenance by default with dataset_hash, params, and held_out."""
    import hashlib

    p = tmp_path / "data.csv"
    p.write_text("sex,age\nM,25\nF,30\nM,45\nF,50\n", encoding="utf-8")
    held = tmp_path / "held.csv"
    held.write_text("race\nA\nB\nA\nB\n", encoding="utf-8")

    result = _proxy_hints_impl(str(p), held_out_with=[f"{held}=race"], alpha=0.01, correction="holm", overrides={"age": "age"})
    assert "provenance" in result
    prov = result["provenance"]
    assert prov["dataset_hash"] == "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    assert prov["engine"] == "python"
    assert prov["params"]["alpha"] == 0.01
    assert prov["params"]["correction"] == "holm"
    assert prov["overrides"] == {"age": "age"}
    assert len(prov["proxy_hints_with"]) == 1
    assert prov["proxy_hints_with"][0]["column"] == "race"
    assert prov["proxy_hints_with"][0]["sha256"] == "sha256:" + hashlib.sha256(held.read_bytes()).hexdigest()


def test_proxy_hints_include_provenance_false_omits_the_block(tmp_path):
    """#860: include_provenance=False omits provenance from proxy_hints."""
    p = tmp_path / "data.csv"
    p.write_text("sex,age\nM,25\nF,30\n", encoding="utf-8")

    result = _proxy_hints_impl(str(p), include_provenance=False)
    assert "provenance" not in result



@requires_scipy
def test_proxy_hints_provenance_records_held_out_encoding(tmp_path):
    # #891
    path = tmp_path / "a.csv"
    path.write_text("sex,occupation\n" + "\n".join(
        f"{'m' if i % 2 else 'f'},{'x' if i % 2 else 'y'}" for i in range(40)) + "\n", encoding="utf-8")
    held = tmp_path / "h.csv"
    held.write_text("race\n" + "\n".join("ab"[i % 2] for i in range(40)) + "\n", encoding="utf-8")
    result = _proxy_hints_impl(str(path), held_out_with=[f"{held}=race"], encoding="latin-1")
    assert result["provenance"]["proxy_hints_with"][0]["encoding"] == "latin-1"
    plain = _proxy_hints_impl(str(path), held_out_with=[f"{held}=race"])
    assert "encoding" not in plain["provenance"]["proxy_hints_with"][0]


@requires_scipy
def test_proxy_hints_exact_tuning_parameters_reach_hints_and_provenance(tmp_path):
    # #906
    path = tmp_path / "s.csv"
    path.write_text("sex,race\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{'abcdef'[i % 6]}" for i in range(24)) + "\n", encoding="utf-8")
    result = _proxy_hints_impl(str(path), alpha=1.0, exact=True, exact_permutations=99, exact_max_rows=10)
    assert result["hints"][0]["p_method"] == "chi2-capped"
    params = result["provenance"]["params"]
    assert (params["exact_permutations"], params["exact_max_rows"]) == (99, 10)
    with pytest.raises(ValueError, match="positive integer"):
        _proxy_hints_impl(str(path), alpha=1.0, exact=True, exact_permutations=0)


@requires_scipy
def test_proxy_hints_skips_near_unique_held_out_columns_with_a_note(tmp_path):
    # #908
    path = tmp_path / "a.csv"
    path.write_text("sex,occupation\n" + "\n".join(
        f"{'m' if i % 2 else 'f'},{'x' if i % 2 else 'y'}" for i in range(60)) + "\n", encoding="utf-8")
    held = tmp_path / "h.csv"
    held.write_text("ident\n" + "\n".join(f"id{i}" for i in range(60)) + "\n", encoding="utf-8")
    result = _proxy_hints_impl(str(path), alpha=1.0, held_out_with=[f"{held}=ident"])
    assert any("held-out column 'ident'" in n and "identifier" in n for n in result["notes"])
    assert all("ident" not in (h["a"], h["b"]) for h in result["hints"])
    wide = _proxy_hints_impl(str(path), alpha=1.0, held_out_with=[f"{held}=ident"], max_categorical_card=100)
    assert "notes" not in wide
