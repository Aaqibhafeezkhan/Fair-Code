"""scripts/check_external_links.py classifies links and never fails the repository's checks (#1127)."""

import http.server
import importlib
import io
import json
import threading
import urllib.error

import pytest

script = importlib.import_module("scripts.check_external_links")


class _Handler(http.server.BaseHTTPRequestHandler):
    requests = []

    def _respond(self):
        user_agent = self.headers.get("User-Agent", "")
        self.requests.append((self.path, self.command, user_agent))
        if self.path == "/always404":
            code = 404
        elif self.path == "/kaggle":
            code = 200 if "Chrome/" in user_agent and "FairCodeLinkCheck/1.0" in user_agent else 404
        else:
            code = {"/ok": 200, "/gone": 404, "/removed": 410, "/blocked": 403, "/slow-down": 429, "/down": 503,
                    "/nohead": 200}.get(self.path, 404)
        if self.path == "/nohead" and self.command == "HEAD":
            code = 405
        redirects = {"/redirect": (302, "/ok"), "/temporary": (307, "/ok"), "/moved": (301, "/ok"),
                     "/moved-permanently": (308, "/ok"), "/moved-twice": (301, "/redirect"),
                     "/moved-to-gone": (301, "/gone")}
        if self.path in redirects:
            status, location = redirects[self.path]
            self.send_response(status)
            self.send_header("Location", location)
            self.end_headers()
            return
        self.send_response(code)
        self.end_headers()

    do_GET = do_HEAD = _respond

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    thread.join()


@pytest.mark.parametrize("path,kind", [("/ok", "ok"), ("/redirect", "ok"), ("/temporary", "ok"), ("/nohead", "ok"),
                                       ("/moved", "moved"), ("/moved-permanently", "moved"), ("/gone", "broken"),
                                       ("/removed", "broken"), ("/blocked", "unknown"), ("/slow-down", "unknown"),
                                       ("/down", "unknown")])
def test_each_status_class(server, path, kind):
    assert script.check_url(server + path, timeout=5)[0] == kind


def test_404_retries_once_with_project_browser_user_agent(server):
    _Handler.requests.clear()

    assert script.check_url(server + "/kaggle", timeout=5) == ("ok", "200")

    requests = [request for request in _Handler.requests if request[0] == "/kaggle"]
    assert [(request[1], request[2]) for request in requests] == [
        ("HEAD", script.USER_AGENT),
        ("GET", script.BROWSER_USER_AGENT),
    ]
    assert len(requests) == 2


def test_404_stays_broken_after_the_single_retry(server):
    _Handler.requests.clear()

    assert script.check_url(server + "/always404", timeout=5) == ("broken", "404")
    assert len([request for request in _Handler.requests if request[0] == "/always404"]) == 2


@pytest.mark.parametrize(
    "resolver_body,expected",
    [
        (b'{"responseCode": 1}', ("ok", "DOI exists")),
        (b'{"responseCode": 100}', ("broken", "DOI not found")),
        (b'{"responseCode": 200}', ("broken", "DOI has no resolution values")),
        (b'{"responseCode": 2}', ("unknown", "DOI resolver response 2")),
    ],
)
def test_403_doi_page_uses_handle_resolver(monkeypatch, resolver_body, expected):
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return resolver_body

    def fake_urlopen(request, **_kwargs):
        calls.append(request.full_url)
        if request.full_url.startswith("https://doi.org/api/handles/"):
            return Response()
        raise urllib.error.HTTPError(request.full_url, 403, "blocked", {}, io.BytesIO())

    monkeypatch.setattr(script.urllib.request, "urlopen", fake_urlopen)

    assert script.check_url("https://publisher.example/doi/10.1234/example", timeout=1) == expected
    assert calls == [
        "https://publisher.example/doi/10.1234/example",
        "https://doi.org/api/handles/10.1234/example",
    ]


def test_missing_doi_is_broken_when_resolver_returns_404(monkeypatch):
    def fake_urlopen(request, **_kwargs):
        if request.full_url.startswith("https://doi.org/api/handles/"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, io.BytesIO())
        raise urllib.error.HTTPError(request.full_url, 403, "blocked", {}, io.BytesIO())

    monkeypatch.setattr(script.urllib.request, "urlopen", fake_urlopen)

    assert script.check_url("https://publisher.example/doi/10.1234/missing", timeout=1) == (
        "broken", "DOI not found"
    )


def test_a_name_that_does_not_resolve_is_broken_and_a_refused_connection_is_unknown():
    assert script.check_url("http://this-host-does-not-exist.invalid/page", timeout=5)[0] == "broken"
    assert script.check_url("http://127.0.0.1:9/", timeout=5)[0] == "unknown"


def test_collect_urls_finds_markdown_and_bare_links_and_skips_local_hosts(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("See [a](https://example.org/a), <https://example.org/b> and [local](http://localhost:8000/x).\n"
                   "Again [a](https://example.org/a). A [relative](other.md) link is ignored.\n", encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text("[a](https://example.org/a)\n", encoding="utf-8")
    found = script.collect_urls([doc, other])
    assert set(found) == {"https://example.org/a", "https://example.org/b"}
    assert len(found["https://example.org/a"]) == 2


def test_main_reports_without_failing_unless_asked(server, tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text(f"[ok]({server}/ok) [gone]({server}/gone) [blocked]({server}/blocked)\n", encoding="utf-8")
    assert script.main([str(doc), "--timeout", "5", "--check-local"]) == 0
    out = capsys.readouterr().out
    assert "1 ok, 1 broken, 1 unknown" in out and f"{server}/gone" in out
    assert script.main([str(doc), "--timeout", "5", "--fail-on-broken", "--check-local"]) == 1
    capsys.readouterr()
    assert script.main([str(doc), "--timeout", "5", "--json", "--limit", "2", "--check-local"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checked"] == 2 and set(report) == {"checked", "ok", "moved", "broken", "unknown", "ignored"}


def test_a_permanent_redirect_reports_the_new_address(server):
    assert script.check_url(server + "/moved", timeout=5) == ("moved", server + "/ok")
    assert script.check_url(server + "/moved-permanently", timeout=5) == ("moved", server + "/ok")
    # The first hop decides: a 301 to a page that then redirects again still ends on its final address.
    assert script.check_url(server + "/moved-twice", timeout=5) == ("moved", server + "/ok")
    # A temporary redirect keeps the old address, which is still the one to cite.
    assert script.check_url(server + "/redirect", timeout=5) == ("ok", "200")


def test_a_permanent_redirect_to_a_dead_page_is_broken_not_moved(server):
    assert script.check_url(server + "/moved-to-gone", timeout=5) == ("broken", "404")


def test_main_lists_moved_links_with_their_new_address_and_still_exits_0(server, tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text(f"[ok]({server}/ok) [moved]({server}/moved)\n", encoding="utf-8")
    assert script.main([str(doc), "--timeout", "5", "--check-local", "--fail-on-broken"]) == 0
    out = capsys.readouterr().out
    assert "1 ok, 0 broken, 0 unknown. 1 moved." in out
    assert f"{server}/moved  ->  {server}/ok  ({doc})" in out
    assert script.main([str(doc), "--timeout", "5", "--check-local", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["moved"] == [{"url": f"{server}/moved", "detail": f"{server}/ok", "files": [str(doc)]}]


def test_ignore_list_skips_a_url_and_reports_it(server, tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text(f"[ok]({server}/ok) [gone]({server}/gone)\n", encoding="utf-8")
    ignore = tmp_path / "ignore.txt"
    ignore.write_text(f"# a comment\n\n{server}/gone needs a login\n", encoding="utf-8")
    assert script.main([str(doc), "--timeout", "5", "--check-local", "--ignore-file", str(ignore)]) == 0
    out = capsys.readouterr().out
    assert "0 broken" in out and "1 ignored" in out and "(needs a login)" in out
    assert script.main([str(doc), "--timeout", "5", "--check-local", "--json", "--ignore-file", str(ignore)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checked"] == 1 and [item["url"] for item in report["ignored"]] == [f"{server}/gone"]
    assert report["ignored"][0]["files"] == [str(doc)]


def test_ignore_list_requires_a_reason_and_tolerates_a_missing_file(tmp_path):
    bad = tmp_path / "ignore.txt"
    bad.write_text("https://example.org/a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ignore.txt:1"):
        script.load_ignore_list(bad)
    assert script.load_ignore_list(tmp_path / "missing.txt") == {}


def test_the_committed_ignore_list_parses_and_names_the_two_known_links():
    ignored = script.load_ignore_list()
    assert "https://github.com/TanishGoyal-Dev" in ignored
    assert "https://github.com/yakew7/Fair-Code/security/code-scanning" in ignored
