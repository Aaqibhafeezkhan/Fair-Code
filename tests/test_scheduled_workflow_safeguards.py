from pathlib import Path

import pytest
import yaml


WORKFLOWS_DIR = Path(__file__).parents[1] / ".github/workflows"
EXPECTED_SCHEDULED_PERMISSIONS = {
    "codeowners-access.yml": {"contents": "read"},
    "codeql.yml": {
        "actions": "read",
        "contents": "read",
        "security-events": "write",
    },
    "external-links.yml": {"contents": "read", "issues": "write"},
    "traction-live.yml": {"contents": "read", "issues": "write"},
}


def _scheduled_workflows():
    workflows = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        if "schedule" in workflow.get("on", {}):
            workflows.append((path, workflow))
    return workflows


SCHEDULED_WORKFLOWS = _scheduled_workflows()


def test_scheduled_workflows_have_explicit_minimal_permissions():
    actual = {path.name for path, _ in SCHEDULED_WORKFLOWS}
    assert actual == set(EXPECTED_SCHEDULED_PERMISSIONS)

    for path, workflow in SCHEDULED_WORKFLOWS:
        permissions = workflow.get("permissions")
        assert permissions == EXPECTED_SCHEDULED_PERMISSIONS[path.name]
        assert permissions["contents"] == "read"


@pytest.mark.parametrize(
    "path,workflow",
    SCHEDULED_WORKFLOWS,
    ids=[path.name for path, _ in SCHEDULED_WORKFLOWS],
)
def test_scheduled_workflows_have_concurrency_controls(path, workflow):
    concurrency = workflow.get("concurrency")
    assert isinstance(concurrency, dict), path.name
    assert concurrency.get("group", "").strip(), path.name
    assert "cancel-in-progress" in concurrency, path.name


@pytest.mark.parametrize(
    "path,workflow",
    SCHEDULED_WORKFLOWS,
    ids=[path.name for path, _ in SCHEDULED_WORKFLOWS],
)
def test_scheduled_workflow_jobs_have_timeouts(path, workflow):
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, path.name

    for job_name, job in jobs.items():
        timeout = job.get("timeout-minutes")
        assert timeout is not None, f"{path.name}:{job_name} has no timeout"
        assert 1 <= int(timeout) <= 360, f"{path.name}:{job_name} has invalid timeout"
