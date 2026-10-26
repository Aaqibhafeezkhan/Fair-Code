from pathlib import Path
import re


def test_requirements_lock_versions_satisfy_pyproject_minimums():
    root = Path(__file__).resolve().parents[1]

    pyproject = root / "pyproject.toml"
    requirements_lock = root / "requirements-lock.txt"

    assert pyproject.is_file(), f"Could not find {pyproject}"
    assert requirements_lock.is_file(), f"Could not find {requirements_lock}"

    pyproject_text = pyproject.read_text(encoding="utf-8")
    lock_text = requirements_lock.read_text(encoding="utf-8")

    pyproject_versions = {
        normalize_dependency(name): minimum
        for name, minimum in re.findall(
            r'([A-Za-z0-9_.-]+)\s*>=\s*([0-9]+(?:\.[0-9]+)*)',
            pyproject_text,
        )
    }

    locked_versions = {
        normalize_dependency(name): version
        for name, version in re.findall(
            r'^([A-Za-z0-9_.-]+)==([0-9]+(?:\.[0-9]+)*)',
            lock_text,
            re.MULTILINE,
        )
    }

    for dependency, minimum in pyproject_versions.items():
        locked = locked_versions.get(dependency)

        if locked is None:
            continue

        assert version_tuple(locked) >= version_tuple(minimum), (
            f"{dependency} is locked to {locked}, "
            f"but pyproject.toml requires >= {minimum}"
        )


def test_every_project_dependency_is_locked_or_documented_as_excluded():
    root = Path(__file__).resolve().parents[1]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock_text = (root / "requirements-lock.txt").read_text(encoding="utf-8")

    core_section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[project\.optional-dependencies\])", pyproject_text)
    optional_section = re.search(r"(?ms)^\[project\.optional-dependencies\]\s*(.*?)(?=^\[)", pyproject_text)
    assert core_section is not None and optional_section is not None
    dependency_arrays = re.findall(
        r"(?ms)^\s*dependencies\s*=\s*\[(.*?)^\s*\]", core_section.group(1)
    )
    dependency_arrays.extend(
        re.findall(r"(?ms)^\s*[A-Za-z0-9_-]+\s*=\s*\[(.*?)\]", optional_section.group(1))
    )
    declared = set()
    for array in dependency_arrays:
        for requirement in re.findall(r'[\"\']([^\"\']+)[\"\']', array):
            match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
            assert match is not None, f"could not parse dependency requirement: {requirement}"
            declared.add(normalize_dependency(match.group(1)))
    locked = {
        normalize_dependency(name)
        for name in re.findall(r"^([A-Za-z0-9_.-]+)==", lock_text, re.MULTILINE)
    }
    excluded = {
        normalize_dependency(name)
        for name in re.findall(r"^#   ([A-Za-z0-9_.-]+) - .+", lock_text, re.MULTILINE)
    }

    assert declared - locked == excluded
    assert excluded, "document intentional exclusions in requirements-lock.txt"


def version_tuple(version):
    return tuple(int(part) for part in version.split("."))


def normalize_dependency(name):
    return name.lower().replace("-", "_").replace(".", "_")
