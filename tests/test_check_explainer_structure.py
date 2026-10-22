"""scripts/check_explainer_structure.py enforces the explainer structure from CONTRIBUTING.md (#1118)."""

import importlib

import pytest

check = importlib.import_module("scripts.check_explainer_structure")

GOOD = """# Explainer: Thing

## The One-Sentence Definition

A thing.

## Why It Matters

Because.

## Concrete Example: COMPAS - Audit 01

An example.

## Detection Code

```python
print(1)
```

## Limitations

Some.

## Related Concepts

- other

## Further Reading

- [One](https://example.com/1)
- [Two](https://example.com/2)

---

*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
"""


def _replace(text, old, new=""):
    assert old in text
    return text.replace(old, new, 1)


def test_a_complete_explainer_has_no_problems():
    assert check.problems_in(GOOD) == set()


@pytest.mark.parametrize("old,new", [
    ("## Why It Matters", "## Why This Matters"),
    ("## Limitations", "## Limitations and Trade-offs"),
    ("## Limitations", "## Limitations / Trade-offs"),
    ("## Detection Code", "## Detection/Implementation Code"),
    ("## Detection Code", "## How to Detect It in Python"),
    ("## Concrete Example: COMPAS - Audit 01", "## Real-World Proof: COMPAS Analysis"),
    ("## Related Concepts", "## Related Projects in This Repo"),
    ("## The One-Sentence Definition", "## The Formal Definition"),
])
def test_accepted_heading_variants_pass(old, new):
    assert check.problems_in(_replace(GOOD, old, new)) == set()


@pytest.mark.parametrize("heading,name", [
    ("## The One-Sentence Definition", "definition"),
    ("## Why It Matters", "why it matters"),
    ("## Concrete Example: COMPAS - Audit 01", "concrete example"),
    ("## Detection Code", "detection code"),
    ("## Limitations", "limitations"),
    ("## Related Concepts", "related concepts"),
    ("## Further Reading", "further reading"),
])
def test_a_missing_section_is_named(heading, name):
    problems = check.problems_in(_replace(GOOD, heading, "## Something Else"))
    assert name in problems


def test_sections_out_of_order_are_reported():
    text = _replace(GOOD, "## Limitations\n\nSome.\n\n", "")
    text = _replace(text, "## Why It Matters", "## Limitations\n\nSome.\n\n## Why It Matters")
    assert "order" in check.problems_in(text)


def test_a_missing_python_block_is_reported():
    assert check.problems_in(_replace(GOOD, "```python", "```text")) == {"python block"}


def test_headings_inside_code_fences_do_not_count():
    # the "Limitations" below is a comment-style heading inside a fence, not a section
    text = _replace(GOOD, "## Limitations\n\nSome.\n\n", "```text\n## Limitations\n```\n\n")
    assert "limitations" in check.problems_in(text)


def test_too_few_further_reading_entries_are_reported():
    text = _replace(GOOD, "- [Two](https://example.com/2)\n")
    assert check.problems_in(text) == {"further reading entries"}


def test_a_missing_footer_is_reported():
    text = _replace(GOOD, "*Part of [The Fair Code Project]", "*Made by someone else")
    assert check.problems_in(text) == {"footer"}


def test_check_reports_file_and_problem_and_applies_exceptions(tmp_path):
    (tmp_path / "ok.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "bad.md").write_text(_replace(GOOD, "## Limitations", "## Nope"), encoding="utf-8")
    messages = check.check(tmp_path, exceptions={})
    assert messages == ["bad.md: missing or wrong: limitations"]
    assert check.check(tmp_path, exceptions={"bad": {"limitations"}}) == []


def test_an_exception_that_is_no_longer_needed_fails_so_the_list_only_shrinks(tmp_path):
    (tmp_path / "fixed.md").write_text(GOOD, encoding="utf-8")
    messages = check.check(tmp_path, exceptions={"fixed": {"limitations"}})
    assert len(messages) == 1
    assert "fixed.md: no longer needs the exception for limitations" in messages[0]


def test_an_exception_for_a_page_that_does_not_exist_fails(tmp_path):
    (tmp_path / "ok.md").write_text(GOOD, encoding="utf-8")
    messages = check.check(tmp_path, exceptions={"ghost": {"limitations"}})
    assert messages == ["KNOWN_EXCEPTIONS names 'ghost', but explainers/ghost.md does not exist"]


def test_main_exits_1_and_prints_to_stderr_on_a_problem(tmp_path, monkeypatch, capsys):
    (tmp_path / "bad.md").write_text(_replace(GOOD, "## Limitations", "## Nope"), encoding="utf-8")
    monkeypatch.setattr(check, "EXPLAINERS_DIR", tmp_path)
    monkeypatch.setattr(check, "KNOWN_EXCEPTIONS", {})
    assert check.main() == 1
    err = capsys.readouterr().err
    assert "bad.md: missing or wrong: limitations" in err


def test_main_exits_0_when_everything_passes(tmp_path, monkeypatch, capsys):
    (tmp_path / "ok.md").write_text(GOOD, encoding="utf-8")
    monkeypatch.setattr(check, "EXPLAINERS_DIR", tmp_path)
    monkeypatch.setattr(check, "KNOWN_EXCEPTIONS", {})
    assert check.main() == 0
    assert capsys.readouterr().out.startswith("OK: 1 explainers")


def test_the_real_explainers_pass():
    assert check.check() == []
