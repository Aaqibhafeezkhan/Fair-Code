from pathlib import Path
import re


def test_benchmark_diff_separates_smoke_and_full_rerun_concurrency_groups():
    workflow_path = Path(__file__).parents[1] / ".github/workflows/benchmark-diff.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^concurrency:\n(?P<block>(?:^[ \t]+.*(?:\n|$)|^[ \t]*\n)*)",
        workflow,
    )
    assert match is not None
    concurrency = match.group("block")

    group_match = re.search(r"(?m)^  group: (.+)$", concurrency)
    assert group_match is not None

    assert group_match.group(1) == (
        "${{ github.workflow }}-${{ github.event.pull_request.number || inputs.pr }}-"
        "${{ (github.event_name == 'workflow_dispatch' || "
        "(github.event_name == 'pull_request' && github.event.action == 'labeled' && "
        "github.event.label.name == 'rerun-benchmark')) && 'full' || "
        "(github.event_name == 'pull_request' && github.event.action != 'labeled') && 'smoke' || 'noop' }}"
    )
    assert re.search(r"(?m)^  cancel-in-progress: true\s*$", concurrency)
