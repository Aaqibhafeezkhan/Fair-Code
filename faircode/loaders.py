"""Read a tabular dataset file into a DataFrame, regardless of format.

Supports .csv, .tsv, and .xlsx (the last requires the optional `openpyxl`
extra: `pip install faircode[excel]`). Files with an unrecognized or missing
extension fall back to sniffing the delimiter from their content, so a
tab-separated export saved with a `.csv` extension still reads correctly.
A few well-known formats that are not supported (.xls, .ods and so on) are
rejected by name instead, because reading them as text only produces an
encoding error (#1018).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Formats people commonly try that are binary and not supported. Reading one as text fails with a
# "not valid utf-8" error that points at --encoding, which cannot help, so name the format instead (#1018).
UNSUPPORTED_SUFFIXES = (".xls", ".ods", ".feather", ".pkl", ".zip", ".7z")

SNIFF_DELIMITERS = ",\t;|"
SNIFF_SAMPLE_BYTES = 8192

# Byte-order marks, longest first (the UTF-32 LE mark begins with the UTF-16 LE one).
_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


def sniff_bom_encoding(head: bytes) -> str | None:
    """The encoding a leading BOM announces, or None when there is no BOM (#843).

    The "utf-16"/"utf-32" codecs consume the BOM and pick the byte order from
    it; "utf-8-sig" drops the UTF-8 one.
    """
    for bom, name in _BOMS:
        if head.startswith(bom):
            return name
    return None


def resolve_encoding(path: str, encoding: str | None = None) -> str:
    """The encoding to read `path` with: the caller's explicit choice, else
    whatever its BOM announces, else UTF-8."""
    if encoding:
        return encoding
    try:
        with open(path, "rb") as fh:
            return sniff_bom_encoding(fh.read(4)) or "utf-8"
    except OSError:
        return "utf-8"


def read_table(path: str, encoding: str | None = None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".xlsx":
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError(
                "reading .xlsx files requires the 'openpyxl' package "
                "(install with: pip install faircode[excel])"
            ) from exc

    if suffix in (".tsv", ".csv"):
        _reject_nul_bytes(path, encoding, NUL_MESSAGE)
        return _read_delimited(path, default="\t" if suffix == ".tsv" else ",", encoding=encoding)

    # A missing file keeps its "file not found" error: only an existing file is reported as the wrong format.
    if suffix in UNSUPPORTED_SUFFIXES and Path(path).exists():
        raise ValueError(f"unsupported format {suffix}: save it as .xlsx or .csv")

    # An unknown suffix is sniffed as text, but a file that is not text at all must not "profile" (#1019).
    _reject_nul_bytes(path, encoding, "{path} looks like a binary file, not a table (it contains NUL bytes)")
    return _read_delimited(path, default=",", encoding=encoding)


NUL_MESSAGE = "{path} contains NUL bytes: is it UTF-16 without a byte-order mark? try --encoding utf-16"


def _reject_nul_bytes(path: str, encoding: str | None, message: str) -> None:
    """Raise ValueError(message.format(path=path)) if the decoded text of `path` holds a NUL character.

    Checked on decoded text, so a UTF-16 file read with the right encoding (whose bytes are full of
    zeros) passes. pandas' C parser would cut a field at a NUL without a word (#1025)."""
    encoding = resolve_encoding(path, encoding)
    with open(path, "r", encoding=encoding, errors="replace", newline="") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                return
            if "\x00" in chunk:
                raise ValueError(message.format(path=path))


def _read_delimited(path: str, *, default: str, encoding: str | None = None) -> pd.DataFrame:
    """Read delimited text, using the extension's convention only as fallback.

    `encoding` overrides the default (a BOM-announced encoding, else UTF-8).
    """
    encoding = resolve_encoding(path, encoding)
    with open(path, "r", encoding=encoding, errors="replace", newline="") as fh:
        sample = fh.read(SNIFF_SAMPLE_BYTES)
    try:
        return pd.read_csv(path, sep=_sniff_delimiter(sample, default=default), encoding=encoding)
    except pd.errors.ParserError as exc:
        raise ValueError(_plain_parser_message(str(exc))) from exc


def _plain_parser_message(text: str) -> str:
    """pandas' tokenizer errors in the words the web engine uses (#1023, #1024)."""
    ragged = re.search(r"Expected (\d+) fields? in line (\d+), saw (\d+)", text)
    if ragged:
        expected, line, saw = ragged.groups()
        return f"line {line} has {saw} fields but the header has {expected}"
    unclosed = re.search(r"EOF inside string starting at row (\d+)", text)
    if unclosed:
        return f"a quoted field starting at row {unclosed.group(1)} is never closed"
    return text


def _logical_row_delimiter_counts(text: str, delimiter: str, max_rows: int = 5) -> list[int]:
    """Per-row delimiter counts for the first `max_rows` quote-aware logical rows.

    Mirrors assets/profiler-engine.js's logicalRowDelimiterCounts() exactly, so
    both engines sniff the same file the same way. csv.Sniffer() used to do this
    job but scans the *whole* sample for consistency: one messy row anywhere in
    an 8KB sample (e.g. a free-text field with a stray, unquoted delimiter) broke
    it and silently fell back to the extension-implied default, while the JS
    engine - checking only the first 5 logical rows - still sniffed correctly.
    See issue #729.
    """
    counts: list[int] = []
    count = 0
    in_quotes = False
    at_field_start = True
    has_content = False
    i = 0
    n = len(text)
    while i < n and len(counts) < max_rows:
        c = text[i]
        if in_quotes:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    i += 1
                else:
                    in_quotes = False
        elif c == '"' and at_field_start:
            in_quotes = True
            has_content = True
        elif c == delimiter:
            count += 1
            at_field_start = True
            has_content = True
        elif c in ("\n", "\r"):
            if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 1
            if has_content:
                counts.append(count)
            count = 0
            at_field_start = True
            has_content = False
        else:
            at_field_start = False
            has_content = True
        i += 1
    if len(counts) < max_rows and has_content and not in_quotes:
        counts.append(count)
    return counts


def _sniff_delimiter(sample: str, default: str = ",") -> str:
    sample = sample[:SNIFF_SAMPLE_BYTES]
    if not sample:
        return default
    best, best_count = default, -1
    for d in SNIFF_DELIMITERS:
        counts = _logical_row_delimiter_counts(sample, d)
        if not counts:
            continue
        first = counts[0]
        if first <= 0:
            continue
        if all(c == first for c in counts) and first > best_count:
            best_count = first
            best = d
    return best
