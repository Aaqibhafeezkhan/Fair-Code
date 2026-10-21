"""scripts/check_traction.py keeps the traction figures in step across README, ROADMAP and METRICS (#1097)."""
import importlib
from pathlib import Path

check = importlib.import_module("scripts.check_traction")

README = """| Metric | Count |
|--------|------:|
| GitHub Stars | 51 |
| External Contributors | 53 |
| Forks | 63 |
| Watching | 8 |
| Combined Social Reach (Instagram + LinkedIn) | 30K+ |
| Countries Reached (Website Visitors) | 20 |
| Code Audits Published | 7 |

<details>
<summary><strong>Show 2 completed items</strong></summary>

- [x] One
- [x] Two
- [ ] Not done

</details>
"""
ROADMAP = """| Stars | Contributors | Forks | Watching | Social Reach | Countries | Audits | Explainers | CI |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 51 | 54 | 63 | 8 | 30K+ | 20 | 7 | 82 | ok |
"""
METRICS = """![Stars](https://img.shields.io/badge/Stars-51-brightgreen?style=flat-square)
![Contributors](https://img.shields.io/badge/Contributors-54-blue?style=flat-square)
![Forks](https://img.shields.io/badge/Forks-63-orange?style=flat-square)
![Watching](https://img.shields.io/badge/Watching-8-yellow?style=flat-square)
![Countries](https://img.shields.io/badge/Countries-20-informational?style=flat-square)

| Week | Stars | Forks | Watching | Contributors | Social Views | Countries | Issues Closed | Code Audits |
|------|------:|------:|---------:|-------------:|-------------:|----------:|--------------:|------------:|
| 2026-W42 | 50 | 58 | 8 | 53 | 30K+ total | 20 | 123 | 7 total |
| 2026-W43 | 51 | 63 | 8 | 54 | 30K+ total | 20 | 28 | 7 total |
"""


def _site(root: Path, readme=README, roadmap=ROADMAP, metrics=METRICS) -> Path:
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "ROADMAP.md").write_text(roadmap, encoding="utf-8")
    (root / "METRICS.md").write_text(metrics, encoding="utf-8")
    return root


def test_matching_figures_pass(tmp_path):
    assert check.check(_site(tmp_path)) == []


def test_a_changed_traction_figure_names_every_place_that_disagrees(tmp_path):
    problems = check.check(_site(tmp_path, roadmap=ROADMAP.replace("| 63 |", "| 58 |")))
    assert len(problems) == 1
    assert problems[0].startswith("forks disagree")
    assert "ROADMAP.md" in problems[0] and "says 58" in problems[0]
    assert "README.md (Forks) says 63" in problems[0] and "METRICS.md (badge Forks) says 63" in problems[0]


def test_only_the_newest_weekly_row_is_compared(tmp_path):
    # W42 says 50 stars; that is history, not a disagreement.
    assert check.check(_site(tmp_path)) == []
    stale = METRICS.replace("| 2026-W43 | 51 |", "| 2026-W43 | 52 |")
    assert any(p.startswith("stars disagree") and "W43" in p for p in check.check(_site(tmp_path, metrics=stale)))


def test_readme_external_contributors_is_expected_one_below_the_total(tmp_path):
    assert check.check(_site(tmp_path)) == []
    problems = check.check(_site(tmp_path, readme=README.replace("| 53 |", "| 54 |")))
    assert any(p.startswith("contributors disagree") for p in problems)


def test_social_reach_text_is_compared_too(tmp_path):
    problems = check.check(_site(tmp_path, readme=README.replace("30K+", "27K+")))
    assert any(p.startswith("social disagree") for p in problems)


def test_completed_item_count_must_match_the_checked_lines(tmp_path):
    assert check.completed_items(README) == (2, 2)
    problems = check.check(_site(tmp_path, readme=README.replace("Show 2", "Show 3")))
    assert problems == ["README.md: 'Show 3 completed items' but the section has 2 '- [x]' lines"]


def test_missing_rows_are_reported_not_ignored(tmp_path):
    problems = check.check(_site(tmp_path, roadmap="no table here\n"))
    assert "ROADMAP.md: no 'Current traction' table found" in problems


def test_the_real_repo_agrees():
    assert check.check() == []
