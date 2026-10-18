"""Python/JS parity of the CSV, JSON and XLSX parsers and their edge cases. (split out of test_js_parity_profile.py, #985)."""

import json
import re
import subprocess

import pandas as pd
import pytest

from faircode import profile
from faircode.loaders import read_table
from js_helpers import FIXTURES, REPO_ROOT, _extract, requires_openpyxl


def test_sheetjs_cdn_url_matches():
    engine = (REPO_ROOT / "assets" / "profiler-engine.js").read_text(encoding="utf-8")
    cli = (REPO_ROOT / "scripts" / "engine-js.js").read_text(encoding="utf-8")

    engine_url = _extract(r'script\.src\s*=\s*"([^"]+)"', engine)
    cli_url = _extract(r'XLSX_CDN_URL\s*=\s*"([^"]+)"', cli)

    assert engine_url == cli_url


def test_python_js_profiler_parity_preserves_mid_field_quotes(tmp_path):
    """Quotes after field content are literal in pandas and the browser parser."""
    csv = tmp_path / "space_before_quote.csv"
    csv.write_text(
        'sex, race, age\n'
        'Male, "White", 25\n'
        'Female, "Black", 30\n'
        'Male, "White", 45\n'
        'Female, "Asian", 22\n',
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


def test_python_js_profiler_parity_sniffs_quoted_newlines(tmp_path):
    """Embedded newlines do not split logical rows during delimiter sniffing."""
    dataset = tmp_path / "sniff_quoted_newline.dat"
    dataset.write_text(
        'sex;race;age;notes\n'
        'M;White;25;"single line"\n'
        'F;Black;30;"multi\nline note"\n'
        'M;White;45;"ok"\n'
        'F;Asian;22;"fine"\n'
        'M;White;50;"good"\n',
        encoding="utf-8",
    )

    python_result = profile(read_table(str(dataset)))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(dataset)],
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


def test_python_js_na_token_parity_on_literal_na_and_none(tmp_path):
    """NA_TOKENS / isMissing() must match pandas' default STR_NA_VALUES
    exactly and case-sensitively: literal "None" is missing, bare lowercase
    "na" is a real category. The JS engine used to have both backwards and
    lower-cased the cell before comparing (#491)."""
    csv = tmp_path / "na_test.csv"
    csv.write_text(
        "status,x\n"
        "active,1\ninactive,2\nna,3\nna,4\nNone,5\nNone,6\n"
        "active,7\ninactive,8\nactive,9\ninactive,10\n",
        encoding="utf-8",
    )

    python_result = profile(read_table(str(csv)))
    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(csv)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)
    assert javascript_result == python_result

    status = next(d for d in python_result["dimensions"] if d["name"] == "status")
    labels = {g["label"] for g in status["groups"]}
    assert "na" in labels          # bare lowercase "na" is NOT a pandas NA token
    assert "None" not in labels    # "None" IS a pandas NA token
    assert status["missing_pct"] == 0.2


def test_python_js_json_parity_inconsistent_keys():
    """Records-orient JSON where later records add columns the first one
    doesn't have (#144). The JS parseJSON() used to derive columns from only
    the first record, silently dropping any column that first appeared later
    - pandas' read_json unions keys across every record instead."""

    json_path = FIXTURES / "inconsistent_keys.json"

    python_result = profile(pd.read_json(json_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-json", str(json_path)],
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


def test_python_js_json_parity_columns_orientation():
    """Columns-orient JSON ({"col": {"0": v, ...}}, pandas' read_json default
    for a plain object) - #155 documented and tested this for the CLI, but
    the JS engine's parseJSON() only handled records/split and threw on it.
    Now handled the same way as the records branch (union of index keys)."""

    json_path = FIXTURES / "columns_orient.json"

    python_result = profile(pd.read_json(json_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-json", str(json_path)],
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


@requires_openpyxl
def test_python_js_xlsx_parity():
    """.xlsx support (#158) - the JS engine's parseXLSX() (via SheetJS,
    fetched from the same pinned CDN profiler.html loads) should agree with
    pandas.read_excel() on the same workbook. Skips if the CDN is
    unreachable rather than failing the suite - see scripts/engine-js.js.
    """
    xlsx_path = FIXTURES / "adult_sample.xlsx"

    python_result = profile(pd.read_excel(xlsx_path))

    completed = subprocess.run(
        ["node", "scripts/engine-js.js", "profile-xlsx", str(xlsx_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode == 3:
        pytest.skip("SheetJS CDN unreachable: " + completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr

    javascript_result = json.loads(completed.stdout)

    python_result = dict(python_result)
    javascript_result = dict(javascript_result)
    python_result.pop("flags", None)
    javascript_result.pop("flags", None)

    assert javascript_result == python_result


def test_python_js_sample_dataset_is_byte_identical():
    """faircode profile --sample (faircode/sample_data.py) and the web
    profiler's "Try it with a sample dataset" button
    (assets/profiler-ui.js's buildSampleCSV()) must produce the exact same
    CSV text, so a first-time user sees an identical demo either way (#741)."""
    from faircode.sample_data import build_sample_csv

    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")
    match = re.search(r"function buildSampleCSV\(\) \{[\s\S]*?\n  \}", src)
    assert match, "could not find buildSampleCSV() in profiler-ui.js"

    script = match.group(0) + ";process.stdout.write(buildSampleCSV());"
    completed = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, encoding="utf-8", check=True,
    )
    assert completed.stdout == build_sample_csv()


@pytest.mark.parametrize("name,text", [
    ("duplicate_headers", "sex,sex,race\nM,F,A\nM,F,B\nM,F,A\nF,M,B\n"),
    ("duplicate_headers_colliding_suffix", "a,a,a,a.1\n1,2,3,4\n"),
    ("blank_header_cell", "sex,,race\nM,a,x\nF,b,y\nM,a,x\nF,b,y\n"),
    ("all_blank_header_row", ",,\nM,a,x\nF,b,y\nM,a,x\nF,b,y\n"),
    ("blank_header_next_to_unnamed_name", "Unnamed: 1,,race\nM,a,x\nF,b,y\nM,a,x\nF,b,y\n"),
    ("quoted_empty_single_column_row", 'region\r\n""\r\nN\r\nS\r\n'),
    ("quoted_empty_last_line", 'region\r\nN\r\n""'),
    ("non_ascii_digit_ages", "age\n٣٠\n25\n40\n٣٠\n"),
])
def test_python_js_parity_on_parser_edge_cases(tmp_path, name, text):
    """#834 (pandas-style header de-duplication), #835 (a quoted-empty row is a
    missing value, not a blank line), #836 (only ASCII digits are ages) and #1046
    (a blank header cell is "Unnamed: N", not an empty name):
    the CSV parsers and the profile of the result agree on each."""
    path = tmp_path / f"{name}.csv"
    path.write_text(text, encoding="utf-8", newline="")
    df = pd.read_csv(path)
    script = (
        "require(process.argv[1]);var fs=require('fs');"
        "var t=globalThis.FairCodeProfiler.parseCSV(fs.readFileSync(process.argv[2],'utf-8'));"
        "process.stdout.write(JSON.stringify({columns:t.columns,n:t.rows.length}));"
    )
    parsed = json.loads(subprocess.run(
        ["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert parsed["columns"] == list(df.columns)
    assert parsed["n"] == len(df)

    js = json.loads(subprocess.run(
        ["node", "scripts/engine-js.js", "profile", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    py = dict(profile(df))
    js.pop("flags", None)
    py.pop("flags", None)
    assert js == py


def test_python_js_sniff_delimiter_agree_on_the_edge_cases():
    """#964: both engines pick the same delimiter for every awkward sample."""
    from js_helpers import SNIFF_CASES
    from faircode.loaders import _sniff_delimiter
    script = (
        "require(process.argv[1]);var E=globalThis.FairCodeProfiler;"
        "process.stdout.write(JSON.stringify(JSON.parse(process.argv[2]).map(function(t){return E.sniffDelimiter(t);})));"
    )
    texts = [text for text, _ in SNIFF_CASES]
    out = json.loads(subprocess.run(["node", "-e", script, str(REPO_ROOT / "assets" / "profiler-engine.js"), json.dumps(texts)],
                                    capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out == [_sniff_delimiter(t) for t in texts] == [expected for _, expected in SNIFF_CASES]


def test_python_js_skip_array_valued_columns_like_unhashable_cells(tmp_path):
    """#973: a column of arrays next to a normal one is not a dimension in either engine."""
    path = tmp_path / "nested.json"
    path.write_text(json.dumps([{"sex": "m" if i % 2 else "f", "tags": [i % 2], "note": f"n{i % 3}"}
                                for i in range(20)]), encoding="utf-8")
    from faircode.loaders_extra import read_table as read_any_table
    py = profile(read_any_table(str(path)), None, {})
    js = json.loads(subprocess.run(["node", "scripts/engine-js.js", "profile-json", str(path)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert [d["name"] for d in py["dimensions"]] == ["sex", "note"]
    assert js == json.loads(json.dumps(py))


def test_python_js_flatten_nested_json_objects_into_dotted_columns(tmp_path):
    """#992: {"loc": {"state": "TX"}} is the column loc.state in both engines; arrays stay cells."""
    from faircode.loaders_extra import read_table as read_any_table
    path = tmp_path / "nested.json"
    path.write_text(json.dumps([
        {"sex": "m" if i % 2 else "f", "loc": {"state": "TX" if i % 3 else "CA", "geo": {"zone": i % 2}}, "tags": [i % 2]}
        for i in range(30)]), encoding="utf-8")
    py = profile(read_any_table(str(path)), None, {})
    js = json.loads(subprocess.run(["node", "scripts/engine-js.js", "profile-json", str(path)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert [d["name"] for d in py["dimensions"]] == ["sex", "loc.state", "loc.geo.zone"]
    assert js == json.loads(json.dumps(py))


def test_python_js_count_one_group_for_a_number_and_its_string_form(tmp_path):
    """#993: 1 and "1" in one JSON column are one group labelled 1 in both engines."""
    from faircode.loaders_extra import read_table as read_any_table
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps([{"race": v} for v in [1, "1", "a", "b"] * 10]), encoding="utf-8")
    py = profile(read_any_table(str(path)), None, {})
    js = json.loads(subprocess.run(["node", "scripts/engine-js.js", "profile-json", str(path)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert [(g["label"], g["count"]) for g in py["dimensions"][0]["groups"]] == [("1", 20), ("a", 10), ("b", 10)]
    assert js == json.loads(json.dumps(py))


def test_python_js_reject_an_empty_file_with_the_same_message(tmp_path):
    """#1010: zero bytes (or only whitespace) is an error in both engines; a header row alone is not."""
    from faircode.loaders_extra import EMPTY_DATASET_MESSAGE, read_table as read_any_table
    for text in ("", "\n\n", "﻿"):
        path = tmp_path / "empty.csv"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="dataset is empty"):
            read_any_table(str(path))
        done = subprocess.run(["node", "scripts/engine-js.js", "profile", str(path)],
                              capture_output=True, text=True, encoding="utf-8")
        assert done.returncode != 0 and EMPTY_DATASET_MESSAGE in done.stderr, repr(text)
    header = tmp_path / "header.csv"
    header.write_text("sex,race\n", encoding="utf-8")
    assert subprocess.run(["node", "scripts/engine-js.js", "profile", str(header)],
                          capture_output=True, text=True, encoding="utf-8").returncode == 0


def _label_parity(tmp_path, name, text, command="profile"):
    """Profile one file in both engines and return (python result, js result)."""
    from faircode.loaders_extra import read_table as read_any_table
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    py = json.loads(json.dumps(profile(read_any_table(str(path)), None, {})))
    js = json.loads(subprocess.run(["node", "scripts/engine-js.js", command, str(path)],
                                   capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert js == py
    return py


def _labels(result, index=0):
    return [(g["label"], g["count"]) for g in result["dimensions"][index]["groups"]]


def test_csv_numeric_code_columns_are_labelled_like_pandas(tmp_path):
    """#1011: an integer column with a blank is float64 in pandas but its labels stay 1 and 2; leading
    zeros, exponents and spaced numbers follow the number, lowercase booleans become True/False."""
    sexes = "".join(f"{'m' if i % 2 else 'f'}\n" for i in range(30)).split()
    def csv(column):
        return "sex,race\n" + "".join(f"{s},{column[i % len(column)]}\n" for i, s in enumerate(sexes))
    assert _labels(_label_parity(tmp_path, "a.csv", csv(["1", "2", ""])), 1) == [("1", 10), ("2", 10)]
    assert _labels(_label_parity(tmp_path, "b.csv", csv(["007", "7.0", "8"])), 1) == [("7", 20), ("8", 10)]
    assert _labels(_label_parity(tmp_path, "c.csv", csv(["1.50", "1.5", "2e3"])), 1) == [("1.5", 20), ("2000", 10)]
    assert _labels(_label_parity(tmp_path, "d.csv", csv(["true", "False", "TRUE"])), 1) == [("True", 20), ("False", 10)]
    assert _labels(_label_parity(tmp_path, "e.csv", csv(["1", "2", "x"])), 1) == [("1", 10), ("2", 10), ("x", 10)]


def test_json_booleans_floats_and_exponents_are_labelled_alike(tmp_path):
    """#1012, #1013, #1015: JSON true is True, 1.0 is 1 and 1e-7 is written 1e-07 in both engines."""
    def records(values):
        return json.dumps([{"race": values[i % len(values)]} for i in range(30)])
    assert _labels(_label_parity(tmp_path, "b.json", records([True, False, True]), "profile-json")) == [("True", 20), ("False", 10)]
    assert _labels(_label_parity(tmp_path, "f.json", records([1.0, 2.0, 1]), "profile-json")) == [("1", 20), ("2", 10)]
    exp = _labels(_label_parity(tmp_path, "e2.json", records([1e-7, 1.5e-7, 0.00001]), "profile-json"))
    assert sorted(label for label, _ in exp) == ["1.5e-07", "1e-05", "1e-07"]


def test_json_integers_above_2_53_keep_every_digit(tmp_path):
    """#1014: 9007199254740993 and 9007199254740992 are two groups in the web engine as in Python."""
    text = "[" + ",".join('{"race": %s}' % v for v in ["9007199254740993", "9007199254740992", "-9007199254740993"] * 10) + "]"
    assert _labels(_label_parity(tmp_path, "big.json", text, "profile-json")) == [
        ("-9007199254740993", 10), ("9007199254740992", 10), ("9007199254740993", 10)]


def test_ties_are_ordered_by_code_point_in_both_engines(tmp_path):
    """#1016: a label outside the BMP sorts after full-width letters, as in Python, including in the
    intersection cells."""
    text = json.dumps([{"race": v, "sex": "m" if i % 2 else "f"} for i, v in enumerate(["\U0001F600", "ｚ", "a"] * 10)])
    result = _label_parity(tmp_path, "astral.json", text, "profile-json")
    assert [label for label, _ in _labels(result)] == ["a", "ｚ", "😀"]


def test_python_js_reject_a_nul_byte_with_the_same_stem(tmp_path):
    """#1025: a NUL inside a cell is an error in both engines, not 'b' in one and 'b\\x00c' in the other;
    UTF-16 read with the right encoding (full of zero bytes) still loads."""
    from faircode.loaders_extra import read_table as read_any_table
    path = tmp_path / "nul.csv"
    path.write_bytes(b"sex,race\nm,a\nf,b\x00c\nm,a\n")
    with pytest.raises(ValueError, match="contains NUL bytes"):
        read_any_table(str(path))
    done = subprocess.run(["node", "scripts/engine-js.js", "profile", str(path)],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode != 0 and "contains NUL bytes" in done.stderr
    utf16 = tmp_path / "u16.csv"
    utf16.write_bytes("sex,race\nm,a\nf,b\n".encode("utf-16"))
    assert list(read_any_table(str(utf16)).columns) == ["sex", "race"]
    assert list(read_any_table(str(utf16), encoding="utf-16").columns) == ["sex", "race"]


@pytest.mark.parametrize("name, text, message", [
    ("ragged.csv", "sex,race\nm,a\nf,b,extra\nm,a\nf,b\n", "line 3 has 3 fields but the header has 2"),
    ("unclosed.csv", 'sex,race\nm,a\nf,"b\nm,a\n', "a quoted field starting at row 2 is never closed"),
])
def test_python_js_refuse_malformed_rows_with_the_same_message(tmp_path, name, text, message):
    """#1023, #1024: a row longer than the header and a quote that never closes are errors in both
    engines, in the same words (not pandas' 'C error' text, and not silently trimmed or swallowed)."""
    from faircode.loaders_extra import read_table as read_any_table
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_any_table(str(path))
    done = subprocess.run(["node", "scripts/engine-js.js", "profile", str(path)],
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode != 0 and message in done.stderr


def test_a_short_row_is_still_padded_with_missing_values(tmp_path):
    """The other direction is not an error: pandas and the JS parser both fill the gap with a blank."""
    result = _label_parity(tmp_path, "short.csv", "sex,race\nm,a\nf\nm,a\nf,b\n")
    assert result["dimensions"][1]["missing_pct"] == 0.25


def test_python_js_accept_nan_and_infinity_literals_in_json(tmp_path):
    """#1026: NaN is a missing value and Infinity is the text inf in both engines."""
    records = ",".join('{"sex":"%s","race":%s}' % ("m" if i % 2 else "f", ["NaN", "Infinity", '"a"', "-Infinity"][i % 4])
                       for i in range(40))
    result = _label_parity(tmp_path, "nan.json", "[" + records + "]", "profile-json")
    assert _labels(result, 1) == [("-inf", 10), ("a", 10), ("inf", 10)]
    assert result["dimensions"][1]["missing_pct"] == 0.25


def test_python_js_read_json_lines_files(tmp_path):
    """#1027: .jsonl and .ndjson are one record per line in both engines, with nested objects flattened;
    a bad line is named, and a one-record file is not mistaken for a plain JSON object."""
    from faircode.loaders_extra import read_table as read_any_table
    lines = "\n".join(json.dumps({"sex": "m" if i % 2 else "f", "loc": {"state": "TX" if i % 3 else "CA"}}) for i in range(30))
    for name in ("rows.jsonl", "rows.ndjson"):
        result = _label_parity(tmp_path, name, lines + "\n\n", "profile-json")
        assert [d["name"] for d in result["dimensions"]] == ["sex", "loc.state"] and result["n_rows"] == 30
    one = _label_parity(tmp_path, "one.jsonl", '{"sex": "m"}\n', "profile-json")
    assert one["n_rows"] == 1
    for text, message in (('{"sex": "m"}\n{oops}\n', "line 2 is not valid JSON"), ('{"sex": "m"}\n[1, 2]\n', "line 2 is not a JSON object")):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_any_table(str(bad))
        done = subprocess.run(["node", "scripts/engine-js.js", "profile-json", str(bad)],
                              capture_output=True, text=True, encoding="utf-8")
        assert done.returncode != 0 and message in done.stderr
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset is empty"):
        read_any_table(str(empty))
