import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
WORKFLOW_DOC = WORKFLOW_DIR / "README.md"


def test_workflow_reference_documents_every_workflow_file():
    rows = [line for line in WORKFLOW_DOC.read_text().splitlines() if line.startswith("|")]
    assert rows[0] == "| Workflow | Trigger | Merge gate | Script or action |"
    documented = set(re.findall(r"\]\(([^)]+\.ya?ml)\)", "\n".join(rows)))
    actual = {
        path.name
        for path in WORKFLOW_DIR.iterdir()
        if path.suffix in {".yml", ".yaml"} and path != WORKFLOW_DOC
    }

    assert documented == actual
