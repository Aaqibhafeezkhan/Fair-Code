"""Tests for audit.yaml loading/validation (faircode.manifest).

Run from the repo root:  pytest tests/ -q
"""

from pathlib import Path

import pandas as pd
import pytest

yaml = pytest.importorskip("yaml", reason="manifest loading needs the optional pyyaml extra")

from faircode.manifest import (
    ProtectedAttribute,
    RowFilter,
    TargetSpec,
    discover_manifests,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_manifest(tmp_path, data):
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.dump(data))
    return path


def _minimal_manifest_dict(**overrides):
    data = {
        "name": "toy",
        "dataset": {"path": "toy.csv"},
        "target": {"column": "label", "method": "binary"},
        "protected_attributes": [
            {"name": "group", "type": "categorical", "column": "group",
             "disadvantaged_values": ["a"], "advantaged_values": ["b"]},
        ],
        "core_features": ["x1", "x2"],
    }
    data.update(overrides)
    return data


# ── Loading a well-formed manifest ───────────────────────────────────────────
def test_load_minimal_manifest(tmp_path):
    path = _write_manifest(tmp_path, _minimal_manifest_dict())
    manifest = load_manifest(path)
    assert manifest.name == "toy"
    assert manifest.title == "toy"   # defaults to name when not given
    assert manifest.dataset_path == tmp_path / "toy.csv"
    assert manifest.core_features == ["x1", "x2"]
    assert manifest.proxy_features == []   # optional, defaults to empty
    assert manifest.random_state == 42     # documented default
    assert manifest.test_size == 0.2
    assert len(manifest.protected_attributes) == 1


def test_manifest_defaults_are_overridable(tmp_path):
    path = _write_manifest(tmp_path, _minimal_manifest_dict(
        title="Toy Audit", random_state=7, test_size=0.3, proxy_features=["p1"]))
    manifest = load_manifest(path)
    assert manifest.title == "Toy Audit"
    assert manifest.random_state == 7
    assert manifest.test_size == 0.3
    assert manifest.proxy_features == ["p1"]


# ── Failing loudly on bad input ──────────────────────────────────────────────
def test_malformed_yaml_syntax_raises(tmp_path):
    path = tmp_path / "audit.yaml"
    path.write_text("name: toy\n  bad_indent: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_manifest(path)


def test_missing_required_target_raises(tmp_path):
    data = _minimal_manifest_dict()
    del data["target"]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(KeyError):
        load_manifest(path)


def test_missing_required_name_raises(tmp_path):
    data = _minimal_manifest_dict()
    del data["name"]
    path = _write_manifest(tmp_path, data)
    with pytest.raises(KeyError):
        load_manifest(path)


def test_empty_protected_attributes_raises(tmp_path):
    path = _write_manifest(tmp_path, _minimal_manifest_dict(protected_attributes=[]))
    with pytest.raises(ValueError, match="protected_attributes"):
        load_manifest(path)


def test_unknown_target_method_fails_loudly():
    spec = TargetSpec(column="x", method="not_a_real_method")
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="unknown target method"):
        spec.compute(df)


def test_target_spec_equals_needs_a_value_field():
    with pytest.raises(ValueError, match="'equals' needs a 'value' field"):
        TargetSpec(column="flag", method="equals")


def test_target_spec_isin_needs_a_values_field():
    with pytest.raises(ValueError, match="'isin' needs a 'values' field"):
        TargetSpec(column="flag", method="isin")


def test_unknown_protected_attribute_type_fails_loudly():
    pa = ProtectedAttribute(name="g", type="not_a_real_type", column="g")
    df = pd.DataFrame({"g": [1, 2, 3]})
    with pytest.raises(ValueError, match="unknown protected attribute type"):
        pa.disadvantaged_mask(df)


def test_categorical_protected_attribute_needs_a_values_list():
    pa = ProtectedAttribute(name="g", type="categorical", column="g")
    df = pd.DataFrame({"g": ["a", "b"]})
    with pytest.raises(ValueError, match="need disadvantaged_values or advantaged_values"):
        pa.disadvantaged_mask(df)


def test_numeric_threshold_protected_attribute_needs_a_threshold():
    with pytest.raises(ValueError, match="needs a 'threshold' field"):
        ProtectedAttribute(name="age", type="numeric_threshold", column="age")


def test_age_interval_threshold_protected_attribute_needs_a_threshold():
    with pytest.raises(ValueError, match="needs a 'threshold' field"):
        ProtectedAttribute(name="age", type="age_interval_threshold", column="age")


def test_numeric_threshold_rejects_a_typo_d_disadvantaged_value():
    with pytest.raises(ValueError, match="disadvantaged must be 'below' or 'above'"):
        ProtectedAttribute(name="age", type="numeric_threshold", column="age",
                           threshold=30, disadvantaged="Below")


def test_age_interval_threshold_rejects_a_typo_d_disadvantaged_value():
    with pytest.raises(ValueError, match="disadvantaged must be 'below' or 'above'"):
        ProtectedAttribute(name="age", type="age_interval_threshold", column="age",
                           threshold=70, disadvantaged="ABOVE")


# ── TargetSpec / RowFilter / ProtectedAttribute behaviour ───────────────────
def test_target_spec_methods():
    df = pd.DataFrame({"income": [10, 60, 30, 90], "flag": ["yes", "no", "yes", "no"]})
    assert TargetSpec("income", "above_median").compute(df)[0].tolist() == [0, 1, 0, 1]
    assert TargetSpec("flag", "equals", value="yes").compute(df)[0].tolist() == [1, 0, 1, 0]
    assert TargetSpec("flag", "isin", values=["yes"]).compute(df)[0].tolist() == [1, 0, 1, 0]


def test_target_spec_methods_all_report_known_true_with_no_missing_labels():
    df = pd.DataFrame({"income": [10, 60, 30, 90], "flag": ["yes", "no", "yes", "no"]})
    assert TargetSpec("income", "above_median").compute(df)[1].tolist() == [True] * 4
    assert TargetSpec("flag", "equals", value="yes").compute(df)[1].tolist() == [True] * 4
    assert TargetSpec("flag", "isin", values=["yes"]).compute(df)[1].tolist() == [True] * 4


@pytest.mark.parametrize("method,kwargs", [
    ("binary", {}),
    ("equals", {"value": 1}),
    ("isin", {"values": [1]}),
    ("above_median", {}),
])
def test_target_spec_excludes_missing_labels_via_known_mask(method, kwargs):
    # Regression test for #488: a NaN row used to silently become a plain
    # negative-class (0) row for equals/isin/above_median (NaN == value,
    # NaN in [...], and NaN > median are all False in pandas), and raised
    # IntCastingNaNError for binary - inconsistent, and neither excludes
    # the row the way ProtectedAttribute.disadvantaged_mask() already does
    # for an unknown protected-attribute value.
    df = pd.DataFrame({"outcome": [1, 0, 1, float("nan"), 0]})
    y, known = TargetSpec("outcome", method, **kwargs).compute(df)
    assert known.tolist() == [True, True, True, False, True]
    # The excluded row's y value is a don't-care placeholder - what matters
    # is that known_mask flags it, not what y happens to be there.
    assert len(y) == len(df)


def test_row_filter_isin_and_notna():
    df = pd.DataFrame({"race": ["A", "B", "C", None]})
    kept = RowFilter(column="race", isin=["A", "B"]).apply(df)
    assert kept["race"].tolist() == ["A", "B"]
    kept = RowFilter(column="race", notna=True).apply(df)
    assert len(kept) == 3


def test_protected_attribute_categorical_complement():
    # Only advantaged_values given -> disadvantaged is "everything else".
    pa = ProtectedAttribute(name="g", type="categorical", column="g", advantaged_values=["white"])
    df = pd.DataFrame({"g": ["white", "black", "asian"]})
    disadv, known = pa.disadvantaged_mask(df)
    assert disadv.tolist() == [False, True, True]
    assert known.all()


@pytest.mark.parametrize("kwargs", [
    {"advantaged_values": ["white"]},
    {"disadvantaged_values": ["black"]},
])
def test_categorical_single_list_excludes_nan_via_known_mask(kwargs):
    # A NaN can't be classified when only one list is given, so known_mask
    # must be False for it - previously the single-list branches hardcoded
    # known=True and silently routed NaN to whichever side isin() landed it
    # on (#547).
    pa = ProtectedAttribute(name="g", type="categorical", column="g", **kwargs)
    df = pd.DataFrame({"g": ["white", "black", None]})

    disadv, known = pa.disadvantaged_mask(df)

    assert known.tolist() == [True, True, False]
    assert not bool(disadv.iloc[2])   # the NaN row is not counted as disadvantaged


def test_row_filter_needs_at_least_one_operator():
    # MANIFEST_SPEC.md requires exactly one operator; a bare column silently
    # kept every row instead of raising (#549).
    with pytest.raises(ValueError, match="row filter needs one of"):
        RowFilter(column="race")

    # notna=True counts as an operator; so does any of the others.
    assert RowFilter(column="race", notna=True).column == "race"
    assert RowFilter(column="race", equals="X").column == "race"


def test_protected_attribute_numeric_threshold():
    pa = ProtectedAttribute(name="age", type="numeric_threshold", column="age",
                            threshold=30, disadvantaged="below")
    df = pd.DataFrame({"age": [20, 30, 40, None]})
    disadv, known = pa.disadvantaged_mask(df)
    assert disadv.tolist() == [True, False, False, False]
    assert known.tolist() == [True, True, True, False]


def test_protected_attribute_age_interval_threshold():
    pa = ProtectedAttribute(name="age", type="age_interval_threshold", column="age",
                            threshold=70, disadvantaged="above")
    df = pd.DataFrame({"age": ["[60-70)", "[70-80)", "[80-90)"]})
    disadv, known = pa.disadvantaged_mask(df)
    assert disadv.tolist() == [False, True, True]


# ── Every shipped audit.yaml loads and discovers correctly ──────────────────
def test_discover_manifests_finds_all_seven_shipped_audits():
    manifests = discover_manifests(REPO_ROOT)
    names = {load_manifest(p).name for p in manifests}
    assert names == {
        "compas", "ai_fair_recruitment", "german_credit_lending", "insurance_denial",
        "benefits_denial", "healthcare_readmission", "tenant_screening",
    }


@pytest.mark.parametrize("path", sorted(discover_manifests(REPO_ROOT)), ids=lambda p: p.parent.name)
def test_every_shipped_manifest_loads_without_error(path):
    manifest = load_manifest(path)
    assert manifest.dataset_path.exists(), f"{manifest.name}: dataset file missing at {manifest.dataset_path}"
    assert manifest.core_features
    assert manifest.protected_attributes
    assert manifest.random_state == 42   # the pinned reproducibility convention (see MANIFEST_SPEC.md)

    df = pd.read_csv(manifest.dataset_path)

    assert manifest.target.column in df.columns

    for attr in manifest.protected_attributes:
        assert attr.column in df.columns

    for row_filter in manifest.row_filters:
        assert row_filter.column in df.columns, (
            f"{manifest.name}: row_filters references column "
            f"'{row_filter.column}', which isn't in {manifest.dataset_path.name}"
        )


_FILTER_FRAME = pd.DataFrame({"race": ["A", "B", "C", None, "A"], "n": [1, 2, 3, 4, 5]})


@pytest.mark.parametrize("operator, value, expected_index", [
    ("isin", ["A", "B"], [0, 1, 4]),
    ("not_isin", ["A", "B"], [2, 3]),          # a NaN is not in the list, so it is kept
    ("equals", "A", [0, 4]),
    ("not_equals", "A", [1, 2, 3]),            # NaN != "A", so the missing row is kept
])
def test_row_filter_operators_select_the_expected_rows(operator, value, expected_index):
    # #977: one test per operator, with a NaN in the column.
    kept = RowFilter(column="race", **{operator: value}).apply(_FILTER_FRAME)
    assert kept.index.tolist() == expected_index


@pytest.mark.parametrize("operator, value", [
    ("isin", ["A"]), ("not_isin", ["A"]), ("equals", "A"), ("not_equals", "A"), ("notna", True),
])
def test_row_filter_on_a_missing_column_raises_keyerror(operator, value):
    # #977: a typo'd column must not silently keep (or drop) every row.
    with pytest.raises(KeyError, match="nope"):
        RowFilter(column="nope", **{operator: value}).apply(_FILTER_FRAME)


def test_a_row_filter_with_two_operators_is_rejected():
    # #999: MANIFEST_SPEC says exactly one operator per filter; two filters AND together instead.
    with pytest.raises(ValueError, match="exactly one operator, got not_isin and equals"):
        RowFilter(column="race", not_isin=["C"], equals="A")
    with pytest.raises(ValueError, match="got isin and notna"):
        RowFilter(column="race", isin=["A"], notna=True)
    first, second = RowFilter(column="race", not_isin=["C"]), RowFilter(column="race", equals="A")
    assert second.apply(first.apply(_FILTER_FRAME)).index.tolist() == [0, 4]
    assert RowFilter(column="race", notna=False, equals="A").column == "race"   # notna: false is not an operator


def test_threshold_attribute_needs_a_valid_direction_and_a_threshold():
    # #977: validation errors listed in MANIFEST_SPEC.
    with pytest.raises(ValueError, match="needs a 'threshold' field"):
        ProtectedAttribute(name="age", type="numeric_threshold", column="age")
    with pytest.raises(ValueError, match="disadvantaged must be 'below' or 'above'"):
        ProtectedAttribute(name="age", type="numeric_threshold", column="age", threshold=40, disadvantaged="sideways")


def test_manifest_without_protected_attributes_is_rejected(tmp_path):
    path = _write_manifest(tmp_path, _minimal_manifest_dict(protected_attributes=[]))
    with pytest.raises(ValueError, match="protected_attributes must have at least one entry"):
        load_manifest(path)


def test_a_value_in_both_categorical_lists_is_rejected():
    with pytest.raises(ValueError, match=r"value\(s\) \['x'\] appear in both"):
        ProtectedAttribute(name="g", type="categorical", column="g",
                           disadvantaged_values=["x", "y"], advantaged_values=["x", "z"])


def test_categorical_attribute_with_both_lists_leaves_other_values_unknown():
    pa = ProtectedAttribute(name="g", type="categorical", column="g",
                            disadvantaged_values=["a"], advantaged_values=["b"])
    disadv, known = pa.disadvantaged_mask(pd.DataFrame({"g": ["a", "b", "c", None]}))
    assert known.tolist() == [True, True, False, False]
    assert disadv.tolist() == [True, False, False, False]


def test_manifest_audit_dir_is_the_manifest_folder(tmp_path):
    (tmp_path / "toy.csv").write_text("label,group,x1,x2\n1,a,1,2\n")
    manifest = load_manifest(_write_manifest(tmp_path, _minimal_manifest_dict()))
    assert manifest.audit_dir == tmp_path
