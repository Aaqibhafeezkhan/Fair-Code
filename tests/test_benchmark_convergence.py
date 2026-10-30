"""No benchmark fit of the logistic regression may stop at max_iter (#1205).

#1152 found by hand that the benchmark's logistic regression warned on two audits and no test noticed. This fits
the model the way `run_audit` does (the same `_run_strategy`, so the reweighted fits inside in-processing and the
calibration fits in post-processing are included) for every audit and strategy with `ConvergenceWarning` turned
into an error. The combinations that warn today are listed in KNOWN_NOT_CONVERGING and expected to fail
(strict), so fixing one makes the test say so and the entry has to be removed; a new warning anywhere else fails.
"""
import warnings
from pathlib import Path

import pytest

pytest.importorskip("yaml", reason="faircode.benchmark needs the optional benchmark extra")
pytest.importorskip("sklearn", reason="faircode.benchmark needs the optional benchmark extra")
pytest.importorskip("fairlearn", reason="faircode.benchmark needs the optional benchmark extra")

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from faircode.benchmark import _run_strategy, load_audit  # noqa: E402
from faircode.strategies import STRATEGIES  # noqa: E402

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parent.parent
AUDITS = ["COMPAS", "AI Fair Recruitment", "German Credit Lending", "Insurance Denial", "Benefits Denial",
          "Healthcare Readmission", "Tenant Screening"]
ALL = set(STRATEGIES)
# Audit -> strategies whose logistic regression fit stops at max_iter on the committed code (#1152, closed as not
# planned for now; Benefits Denial's two unscaled feature sets were found while writing this test).
KNOWN_NOT_CONVERGING = {
    "German Credit Lending": ALL,
    "Healthcare Readmission": ALL,
    "Benefits Denial": {"baseline", "unawareness"},
}
# Healthcare Readmission's in_processing fit warns too but takes about a minute and a half, so it is not run.
SKIPPED_FOR_RUNTIME = {("Healthcare Readmission", "in_processing")}


def _params():
    for audit in AUDITS:
        for strategy in STRATEGIES:
            marks = []
            if (audit, strategy) in SKIPPED_FOR_RUNTIME:
                marks.append(pytest.mark.skip(reason="known not to converge (#1152) and about 100 s"))
            elif strategy in KNOWN_NOT_CONVERGING.get(audit, ()):
                marks.append(pytest.mark.xfail(strict=True, raises=ConvergenceWarning, reason="#1152"))
            yield pytest.param(audit, strategy, marks=marks, id=f"{audit}-{strategy}")


@pytest.mark.parametrize("audit,strategy", list(_params()))
def test_logistic_regression_converges(audit, strategy):
    data = load_audit(ROOT / audit)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        _run_strategy(strategy, "logistic_regression", data.manifest, data.X_all, data.y, data.masks,
                      data.idx_train, data.idx_test)


def test_known_exceptions_name_real_audits_and_strategies():
    assert set(KNOWN_NOT_CONVERGING) <= set(AUDITS)
    assert all(strategies <= ALL for strategies in KNOWN_NOT_CONVERGING.values())
