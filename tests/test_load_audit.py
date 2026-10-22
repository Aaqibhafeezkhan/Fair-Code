"""load_audit() prepares an audit exactly like the benchmark harness (#1078)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="the benchmark harness needs the optional benchmark extra")
pytest.importorskip("fairlearn", reason="the benchmark harness needs the optional benchmark extra")
pytest.importorskip("yaml", reason="the benchmark harness needs the optional benchmark extra")

from faircode import benchmark
from faircode.benchmark import AuditData, load_audit, run_audit
from faircode.manifest import load_manifest
from faircode.strategies import STRATEGIES, strategy_features

AUDIT = Path(__file__).resolve().parent.parent / "German Credit Lending"


def test_load_audit_accepts_a_folder_or_a_manifest_path():
    by_folder = load_audit(AUDIT)
    by_file = load_audit(AUDIT / "audit.yaml")
    assert isinstance(by_folder, AuditData)
    assert by_folder.manifest.name == by_file.manifest.name == "german_credit_lending"
    assert np.array_equal(by_folder.idx_test, by_file.idx_test)


def test_load_audit_missing_folder_lists_available_audits(tmp_path):
    audit = tmp_path / "German Credit Lending"
    audit.mkdir()
    (audit / "audit.yaml").touch()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_audit(tmp_path / "Nope")

    message = str(exc_info.value)
    assert str(tmp_path / "Nope" / "audit.yaml") in message
    assert "audit folder or a manifest path" in message
    assert "German Credit Lending" in message


def test_load_audit_folder_without_manifest_names_attempted_file(tmp_path):
    audit = tmp_path / "Incomplete Audit"
    audit.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_audit(audit)

    message = str(exc_info.value)
    assert str(audit / "audit.yaml") in message
    assert "audit folder or a manifest path" in message


def test_load_audit_close_misspelling_suggests_audit_folder(tmp_path):
    audit = tmp_path / "Benefits Denial"
    audit.mkdir()
    (audit / "audit.yaml").touch()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_audit(tmp_path / "Benefits Denia")

    message = str(exc_info.value)
    assert str(tmp_path / "Benefits Denia" / "audit.yaml") in message
    assert "Benefits Denial" in message
    assert "Did you mean 'Benefits Denial'?" in message


def test_split_features_and_masks_equal_what_run_audit_uses(monkeypatch):
    seen = {}

    def spy(strategy, model_name, manifest, X_all, y, masks, idx_train, idx_test):
        seen.setdefault("calls", []).append((strategy, model_name))
        seen["X_all"], seen["y"], seen["masks"] = X_all, y, masks
        seen["idx_train"], seen["idx_test"] = idx_train, idx_test
        n = len(idx_test)
        return y.iloc[idx_test].to_numpy(), np.zeros(n, dtype=int), np.full(n, 0.5)

    monkeypatch.setattr(benchmark, "_run_strategy", spy)
    run_audit(load_manifest(AUDIT / "audit.yaml"), n_resamples=20, n_permutations=20)
    data = load_audit(AUDIT)
    assert len(seen["calls"]) == len(STRATEGIES) * 3
    assert np.array_equal(data.idx_train, seen["idx_train"]) and np.array_equal(data.idx_test, seen["idx_test"])
    assert data.X_all.equals(seen["X_all"]) and data.y.equals(seen["y"])
    assert data.masks.keys() == seen["masks"].keys()
    assert all(data.masks[k].equals(seen["masks"][k]) for k in data.masks)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_x_holds_the_strategys_feature_columns(strategy):
    data = load_audit(AUDIT, strategy=strategy)
    m = data.manifest
    expected = strategy_features(strategy, m.core_features, m.proxy_features, [pa.column for pa in m.protected_attributes])
    assert list(data.X.columns) == expected


def test_default_strategy_is_the_core_feature_set():
    data = load_audit(AUDIT)
    assert list(data.X.columns) == list(dict.fromkeys(data.manifest.core_features))
    assert data.random_state == data.manifest.random_state


def test_random_state_override_changes_the_split():
    assert not np.array_equal(load_audit(AUDIT).idx_test, load_audit(AUDIT, random_state=7).idx_test)
