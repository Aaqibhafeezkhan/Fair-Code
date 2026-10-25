from pathlib import Path

import yaml


def test_benchmark_diff_separates_smoke_and_full_rerun_concurrency_groups():
    workflow_path = Path(__file__).parents[1] / ".github/workflows/benchmark-diff.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    concurrency = workflow["concurrency"]
    group = concurrency["group"]

    assert concurrency["cancel-in-progress"] is True
    assert group == (
        "${{ github.workflow }}-${{ github.event.pull_request.number || inputs.pr }}-"
        "${{ (github.event_name == 'workflow_dispatch' || "
        "(github.event_name == 'pull_request' && github.event.action == 'labeled' && "
        "github.event.label.name == 'rerun-benchmark')) && 'full' || "
        "(github.event_name == 'pull_request' && github.event.action != 'labeled') && 'smoke' || 'noop' }}"
    )
