#!/usr/bin/env python3
"""Reports dead external links in the markdown docs and explainers (#1127).

scripts/check_broken_links.py checks internal links on every run. External links are different: a publisher
or a news site moves a page, and nobody notices until a reader does. They are also slow and flaky to check
(many sites answer a script with 403), so this is a report, never a gate: the weekly workflow runs it and
keeps one issue up to date, and nothing in `make lint` or a pull request depends on it.

    python3 scripts/check_external_links.py                  # explainers/*.md and the top-level docs
    python3 scripts/check_external_links.py --limit 20       # only the first 20 links
    python3 scripts/check_external_links.py --json           # machine-readable, for the workflow
    python3 scripts/check_external_links.py --fail-on-broken # exit 1 when something is broken (off by default)

Each URL is requested with a browser-like user agent and a timeout (HEAD first, then GET when the server does
not allow HEAD). 2xx and 3xx (after redirects) are fine. 404, 410 and a name that does not resolve are
broken. Everything else (403, 429, 5xx, timeouts, refused connections) is unknown: it is listed but never
counted as broken, because it usually means the site blocks scripts or is down for a moment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "Mozilla/5.0 (compatible; FairCodeLinkCheck/1.0; +https://github.com/yakew7/Fair-Code)"
LINK = re.compile(r"\]\((https?://[^)\s]+)\)")
BARE = re.compile(r"<(https?://[^>\s]+)>")
SKIP_HOSTS = {"localhost", "127.0.0.1", "example.com", "www.example.com"}
BROKEN_STATUSES = {404, 410}


def _ssl_context():
    """The system trust store, or certifi's bundle when the Python build has none (a python.org macOS install
    ships without root certificates until `Install Certificates` is run), so a missing bundle is not mistaken
    for a dead link."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def default_paths():
    return sorted((ROOT / "explainers").glob("*.md")) + sorted(ROOT.glob("*.md"))


def collect_urls(paths, skip_hosts=SKIP_HOSTS) -> dict[str, list[str]]:
    """{url: [files that cite it]} for every external markdown link in `paths`."""
    found: dict[str, list[str]] = {}
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for url in LINK.findall(text) + BARE.findall(text):
            url = url.rstrip(".,;")
            host = re.match(r"https?://([^/:]+)", url)
            if host and host.group(1).lower() in skip_hosts:
                continue
            files = found.setdefault(url, [])
            name = str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() and ROOT in Path(path).parents else str(path)
            if name not in files:
                files.append(name)
    return found


def check_url(url: str, timeout: float = 15.0) -> tuple[str, str]:
    """(status class, detail) with the class one of 'ok', 'broken' or 'unknown'."""
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                return "ok", str(response.status)
        except urllib.error.HTTPError as error:
            if error.code in BROKEN_STATUSES:
                return "broken", str(error.code)
            if error.code in (405, 501) and method == "HEAD":
                continue                      # the server does not allow HEAD: try GET
            return "unknown", str(error.code)
        except urllib.error.URLError as error:
            if isinstance(error.reason, socket.gaierror):
                return "broken", "name does not resolve"
            return "unknown", str(error.reason)
        except (TimeoutError, OSError) as error:
            return "unknown", type(error).__name__
    return "unknown", "no response"


def run(urls: dict[str, list[str]], limit: int | None = None, workers: int = 8, timeout: float = 15.0) -> dict:
    items = sorted(urls)[:limit] if limit else sorted(urls)
    results = {"ok": [], "broken": [], "unknown": []}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for url, (kind, detail) in zip(items, pool.map(lambda u: check_url(u, timeout), items)):
            results[kind].append({"url": url, "detail": detail, "files": urls[url]})
    return {"checked": len(items), **{kind: results[kind] for kind in results}}


def render(report: dict) -> str:
    lines = [f"Checked {report['checked']} external links: {len(report['ok'])} ok, "
             f"{len(report['broken'])} broken, {len(report['unknown'])} unknown."]
    for title, key in (("Broken (404, 410 or a name that does not resolve)", "broken"),
                       ("Unknown (blocked, rate limited or unreachable; not counted as broken)", "unknown")):
        if report[key]:
            lines += ["", title + ":"]
            lines += [f"  {item['detail']:>22}  {item['url']}  ({', '.join(item['files'][:3])})" for item in report[key]]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="markdown files to check (default: explainers/*.md and the top-level docs)")
    parser.add_argument("--limit", type=int, help="check at most this many links")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--check-local", action="store_true", help="also check localhost and example.com links (for tests)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--fail-on-broken", action="store_true", help="exit 1 when a link is broken (off by default)")
    args = parser.parse_args(argv)
    report = run(collect_urls(args.paths or default_paths(), () if args.check_local else SKIP_HOSTS), args.limit, timeout=args.timeout)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if args.fail_on_broken and report["broken"] else 0


if __name__ == "__main__":
    sys.exit(main())
