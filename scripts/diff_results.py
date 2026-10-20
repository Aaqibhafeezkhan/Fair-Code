#!/usr/bin/env python3
"""Diff the live results/ against the paper/results-frozen/ reference snapshot.

`results/` moves every time the harness is rerun; `paper/results-frozen/` is the earlier snapshot
kept as a point of comparison. This joins the two result tables on the columns that identify a row
and prints the per-row change, so nobody has to diff two CSVs by hand:

    python3 scripts/diff_results.py                       # fairness + performance, moved rows only
    python3 scripts/diff_results.py --threshold 0.02      # flag rows that moved by more than 0.02
    python3 scripts/diff_results.py --kind fairness --all # every row, not just the ones that moved
    python3 scripts/diff_results.py --json                # machine-readable
    python3 scripts/diff_results.py --fail-on-change      # exit 1 if any row is flagged (for CI)
    python3 scripts/diff_results.py --audit COMPAS        # only these audits (repeatable)
    python3 scripts/diff_results.py --summary             # one line per audit and kind

Join keys: fairness = audit, strategy, model, protected_attribute, metric; performance = audit,
strategy, model, metric. A row present on one side only is reported as `only_current` or
`only_reference` (a new audit, strategy or metric, or one that was dropped) and always counts as
flagged. A row on both sides is `moved` when |current - reference| > --threshold (default 0.01),
else `unchanged`. A value that is empty on one side only (e.g. an AUC row's missing interval)
counts as moved. Reads only; it never writes to either directory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = REPO_ROOT / "results"
REFERENCE_DIR = REPO_ROOT / "paper" / "results-frozen"
DEFAULT_THRESHOLD = 0.01

KINDS = {
    "fairness": ("results_fairness.csv", ["audit", "strategy", "model", "protected_attribute", "metric"]),
    "performance": ("results_performance.csv", ["audit", "strategy", "model", "metric"]),
}


def diff_table(current: pd.DataFrame, reference: pd.DataFrame, keys: list[str],
               threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """One dict per row on either side: the key columns plus status, reference, current, delta."""
    def index(df):
        out = {}
        for _, row in df.iterrows():
            out[tuple(str(row[k]) for k in keys)] = row["value"]
        return out

    cur, ref = index(current), index(reference)
    rows = []
    for key in sorted(set(cur) | set(ref)):
        entry = dict(zip(keys, key))
        c, r = cur.get(key), ref.get(key)
        entry["reference"] = None if r is None or pd.isna(r) else float(r)
        entry["current"] = None if c is None or pd.isna(c) else float(c)
        if key not in ref:
            entry["status"], entry["delta"] = "only_current", None
        elif key not in cur:
            entry["status"], entry["delta"] = "only_reference", None
        elif entry["current"] is None and entry["reference"] is None:
            entry["status"], entry["delta"] = "unchanged", 0.0
        elif entry["current"] is None or entry["reference"] is None:
            entry["status"], entry["delta"] = "moved", None
        else:
            delta = entry["current"] - entry["reference"]
            entry["delta"] = delta
            entry["status"] = "moved" if abs(delta) > threshold and not math.isclose(delta, threshold) else "unchanged"
        rows.append(entry)
    return rows


def diff_dirs(current_dir: Path, reference_dir: Path, kinds=("fairness", "performance"),
              threshold: float = DEFAULT_THRESHOLD, audits=None) -> dict:
    """With `audits`, only rows whose audit is one of those names are joined."""
    out = {}
    for kind in kinds:
        filename, keys = KINDS[kind]
        cur_path, ref_path = Path(current_dir) / filename, Path(reference_dir) / filename
        for p in (cur_path, ref_path):
            if not p.is_file():
                raise FileNotFoundError(f"{p} not found")
        current, reference = pd.read_csv(cur_path), pd.read_csv(ref_path)
        if audits:
            wanted = set(audits)
            current = current[current["audit"].astype(str).isin(wanted)]
            reference = reference[reference["audit"].astype(str).isin(wanted)]
        out[kind] = diff_table(current, reference, keys, threshold)
    return out


def known_audits(current_dir: Path, reference_dir: Path, kinds=("fairness", "performance")) -> set[str]:
    """Every audit name on either side, for checking --audit."""
    names = set()
    for kind in kinds:
        filename = KINDS[kind][0]
        for directory in (current_dir, reference_dir):
            path = Path(directory) / filename
            if path.is_file():
                names.update(pd.read_csv(path)["audit"].astype(str))
    return names


def summarize(diff: dict) -> dict:
    """Per kind, one dict per audit: rows compared, rows flagged, largest absolute delta (None if no numeric delta)."""
    out = {}
    for kind, rows in diff.items():
        per_audit = {}
        for r in rows:
            s = per_audit.setdefault(r["audit"], {"audit": r["audit"], "compared": 0, "flagged": 0,
                                                  "max_abs_delta": None})
            s["compared"] += 1
            s["flagged"] += _flagged(r)
            if r["delta"] is not None and (s["max_abs_delta"] is None or abs(r["delta"]) > s["max_abs_delta"]):
                s["max_abs_delta"] = abs(r["delta"])
        out[kind] = [per_audit[a] for a in sorted(per_audit)]
    return out


def render_summary(summary: dict) -> str:
    lines = []
    for kind, audits in summary.items():
        lines.append(f"{kind}:")
        for s in audits:
            largest = "n/a" if s["max_abs_delta"] is None else f"{s['max_abs_delta']:.4f}"
            lines.append(f"  {s['audit']}: {s['compared']} compared, {s['flagged']} flagged, largest |delta| {largest}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _flagged(row: dict) -> bool:
    return row["status"] != "unchanged"


def render(diff: dict, show_all: bool = False, threshold: float = DEFAULT_THRESHOLD) -> str:
    lines = []
    for kind, rows in diff.items():
        keys = KINDS[kind][1]
        flagged = [r for r in rows if _flagged(r)]
        lines.append(f"{kind}: {len(rows)} rows compared, {len(flagged)} flagged "
                     f"(moved by more than {threshold:g}, or present on one side only)")
        for r in rows if show_all else flagged:
            label = " / ".join(r[k] for k in keys)
            if r["status"] == "only_current":
                detail = f"only in results/ (current {r['current']:.4f})" if r["current"] is not None else "only in results/"
            elif r["status"] == "only_reference":
                detail = (f"only in paper/results-frozen/ (reference {r['reference']:.4f})"
                          if r["reference"] is not None else "only in paper/results-frozen/")
            else:
                ref = "-" if r["reference"] is None else f"{r['reference']:.4f}"
                cur = "-" if r["current"] is None else f"{r['current']:.4f}"
                delta = "n/a" if r["delta"] is None else f"{r['delta']:+.4f}"
                detail = f"{ref} -> {cur} ({delta})"
            mark = "!" if _flagged(r) else " "
            lines.append(f"  {mark} {label}: {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--current", type=Path, default=CURRENT_DIR, help="directory holding the live results (default: results/)")
    p.add_argument("--reference", type=Path, default=REFERENCE_DIR,
                   help="directory holding the reference snapshot (default: paper/results-frozen/)")
    p.add_argument("--kind", choices=["fairness", "performance", "both"], default="both")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"flag a row whose value moved by more than this (default {DEFAULT_THRESHOLD})")
    p.add_argument("--all", action="store_true", help="list unchanged rows too")
    p.add_argument("--json", action="store_true", help="print the diff as JSON")
    p.add_argument("--fail-on-change", action="store_true", help="exit 1 if any row is flagged")
    p.add_argument("--audit", action="append", metavar="NAME",
                   help="only compare this audit (repeatable; default: every audit)")
    p.add_argument("--summary", action="store_true",
                   help="print one line per audit and kind instead of every row")
    args = p.parse_args(argv)
    if args.threshold < 0:
        p.error("--threshold must be zero or positive")
    kinds = ("fairness", "performance") if args.kind == "both" else (args.kind,)
    try:
        if args.audit:
            unknown = sorted(set(args.audit) - known_audits(args.current, args.reference, kinds))
            if unknown:
                print(f"error: unknown audit(s): {', '.join(unknown)}", file=sys.stderr)
                return 2
        diff = diff_dirs(args.current, args.reference, kinds, args.threshold, args.audit)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.summary:
        summary = summarize(diff)
        print(json.dumps(summary, indent=2) if args.json else render_summary(summary), end="\n" if args.json else "")
    elif args.json:
        print(json.dumps(diff, indent=2))
    else:
        print(render(diff, args.all, args.threshold), end="")
    if args.fail_on_change and any(_flagged(r) for rows in diff.values() for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
