"""Python/JS parity of the age handling: implausible, sentinel, birth-year and scientific values. (split out of test_js_parity_profile.py, #985)."""

import json
import subprocess

import pandas as pd

from faircode import compare, profile


def test_python_js_profiler_parity_detects_dates_appended_after_numeric_ages(tmp_path):
    """A merged-in birthdate block cannot become a fabricated elderly group."""
    csv = tmp_path / "mixed-age-and-birthdate.csv"
    ages = [str(age) for age in range(20, 80)]
    dates = ["1985-03-21", "1990-07-14", "2001-11-02"] * 20
    csv.write_text("age\n" + "\n".join(ages + dates) + "\n", encoding="utf-8")

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    assert all(dim["name"] != "age" for dim in python_result["dimensions"])


def test_python_js_profiler_parity_handles_scientific_and_nonfinite_ages(tmp_path):
    """Scientific notation follows pandas' numeric inference; inf/nan stay missing."""
    csv = tmp_path / "scientific-and-nonfinite-ages.csv"
    csv.write_text(
        "age,sex\n"
        "1e2,F\n"
        "25,M\n"
        "40,F\n"
        "60,M\n"
        "inf,F\n"
        "nan,M\n",
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    labels = {group["label"] for group in age["groups"]}
    assert "75+" in labels
    assert "inf" not in labels
    assert "nan" not in labels
    assert age["missing_pct"] == 0.3333


def test_python_js_profiler_handles_overflowing_scientific_age_as_missing(tmp_path):
    """A value such as 1e400 parses to Infinity in JS and must not become an age group."""
    csv = tmp_path / "overflow-age.csv"
    csv.write_text(
        "age\n"
        "25\n"
        "40\n"
        "60\n"
        "1e400\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)
    age = next(d for d in javascript_result["dimensions"] if d["name"] == "age")
    labels = {group["label"] for group in age["groups"]}

    assert "inf" not in labels
    assert "1e400" not in labels
    assert age["missing_pct"] == 0.25


def test_python_js_profiler_parity_rejects_negative_age_sentinels(tmp_path):
    """Signed sentinel ages stay missing in both profiler engines."""
    csv = tmp_path / "negative-age-sentinels.csv"
    csv.write_text(
        "age,sex\n25,F\n30,M\nage -5,F\n-1,M\n45,F\n",
        encoding="utf-8",
    )

    python_result = profile(pd.read_csv(csv))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    assert "75+" not in {group["label"] for group in age["groups"]}
    assert age["missing_pct"] == 0.4


def test_python_js_parity_for_implausible_ages(tmp_path):
    """#840: ages above max_age are flagged and left out of the bands identically in
    both engines, including in the intersection and with a custom --max-age."""
    path = tmp_path / "ages.csv"
    path.write_text("sex,age\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[25, 33, 47, 62, 80, 150, 200, 1985][i % 8]}" for i in range(64)) + "\n")
    for opts in ({}, {"max_age": 70}):
        opts_file = tmp_path / "opts.json"
        opts_file.write_text(json.dumps({"overrides": {}, "opts": opts}))
        python_result = dict(profile(pd.read_csv(path), None, opts))
        completed = subprocess.run(
            ["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
            capture_output=True, text=True, encoding="utf-8", check=True)
        javascript_result = json.loads(completed.stdout)
        py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
        assert javascript_result == python_result
        age = next(d for d in python_result["dimensions"] if d["name"] == "age")
        assert age["implausible_values"] == (24 if not opts else 32)
        implausible = [f for f in py_flags if "implausible age" in f]
        assert implausible and implausible == [f for f in js_flags if "implausible age" in f]


def test_python_js_parity_for_negative_sentinel_ages(tmp_path):
    """#863: negative sentinel ages are flagged identically in both engines."""
    path = tmp_path / "neg_ages.csv"
    path.write_text("sex,age\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[25, 33, -1, 62, -9, 150, 40, 200][i % 8]}" for i in range(64)) + "\n")
    opts_file = tmp_path / "opts.json"
    opts_file.write_text(json.dumps({"overrides": {}, "opts": {}}))
    python_result = dict(profile(pd.read_csv(path), None, {}))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    javascript_result = json.loads(completed.stdout)
    py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
    assert javascript_result == python_result
    age = next(d for d in python_result["dimensions"] if d["name"] == "age")
    assert age["implausible_values"] == 32
    assert age["has_negative_ages"] is True
    sentinel_flags = [f for f in py_flags if "sentinel/implausible age" in f]
    assert sentinel_flags and sentinel_flags == [f for f in js_flags if "sentinel/implausible age" in f]


def test_python_js_parity_for_birth_year_conversion(tmp_path):
    """#862: age_reference_year turns birth years into ages identically in both engines,
    including the intersection, and a future year stays implausible."""
    path = tmp_path / "yob.csv"
    path.write_text("sex,yob\n" + "\n".join(
        f"{'M' if i % 2 else 'F'},{[1985, 1990, 1972, 2001, 1950, 2010, 40, 2090][i % 8]}" for i in range(64)) + "\n")
    opts_file = tmp_path / "opts.json"
    opts_file.write_text(json.dumps({"overrides": {}, "opts": {"age_reference_year": 2026}}))
    python_result = dict(profile(pd.read_csv(path), None, {"age_reference_year": 2026}))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(path), str(opts_file)],
        capture_output=True, text=True, encoding="utf-8", check=True)
    javascript_result = json.loads(completed.stdout)
    py_flags, js_flags = python_result.pop("flags"), javascript_result.pop("flags")
    assert javascript_result == python_result
    yob = next(d for d in python_result["dimensions"] if d["name"] == "yob")
    assert yob["implausible_values"] == 8 and yob["n_groups"] >= 4  # only 2090 is implausible
    assert [f for f in py_flags if "implausible" in f] == [f for f in js_flags if "implausible" in f]


def test_python_js_intersection_parity_keeps_non_numeric_age_sentinels(tmp_path):
    """labelize() gives a non-numeric age sentinel its own crosstab label on
    both engines, instead of mapping it to null and dropping the row (#524)."""
    csv = tmp_path / "age_sentinels.csv"
    ages = ["25", "30", "45", "unknown", "unknown", "unknown",
            "prefer not to say", "22", "33", "41"] * 3
    sexes = ["M", "F"] * 15
    csv.write_text(
        "age,sex\n" + "\n".join(a + "," + s for a, s in zip(ages, sexes)) + "\n",
        encoding="utf-8",
    )

    opts = {"cross": ["age", "sex"]}
    python_result = profile(pd.read_csv(csv, dtype={"age": str}), opts=opts)

    opts_path = tmp_path / "opts.json"
    opts_path.write_text(json.dumps({"opts": opts}), encoding="utf-8")
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv), str(opts_path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)
    assert javascript_result == python_result

    a_labels = {c["a"] for c in python_result["intersections"][0]["cells"]}
    assert "prefer not to say" in a_labels


def test_python_js_compare_parity_age_banding_mismatch(tmp_path):
    """The age-banding-mismatch guard (#318) agrees between engines too -
    `kind` alone can't detect it (it's set from the column name and is
    identical on both sides), so this exercises isAgeBandLabel()'s port of
    _is_age_band_label() directly, not just the already-covered common path.
    """
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    path_a.write_text("DOB\n" + "\n".join(["15/05/1980"] * 50 + ["20/06/1985"] * 50))
    path_b.write_text("DOB\n" + "\n".join(["18"] * 50 + ["35"] * 50))

    python_result = compare(
        profile(pd.read_csv(path_a)), profile(pd.read_csv(path_b)), "a.csv", "b.csv"
    )

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "compare", str(path_a), str(path_b)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result
    dim = python_result["dimensions"][0]
    assert dim["kind_mismatch"] is True
    assert dim["kind_a"] == dim["kind_b"] == "age"
