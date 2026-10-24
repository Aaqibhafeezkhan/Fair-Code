"""scripts/check_external_links.py classifies links and never fails the repository's checks (#1127)."""

import http.server
import importlib
import json
import threading

import pytest

script = importlib.import_module("scripts.check_external_links")


class _Handler(http.server.BaseHTTPRequestHandler):
    def _respond(self):
        code = {"/ok": 200, "/gone": 404, "/removed": 410, "/blocked": 403, "/slow-down": 429, "/down": 503,
                "/nohead": 200}.get(self.path, 404)
        if self.path == "/nohead" and self.command == "HEAD":
            code = 405
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
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


@pytest.mark.parametrize("path,kind", [("/ok", "ok"), ("/redirect", "ok"), ("/nohead", "ok"), ("/gone", "broken"),
                                       ("/removed", "broken"), ("/blocked", "unknown"), ("/slow-down", "unknown"),
                                       ("/down", "unknown")])
def test_each_status_class(server, path, kind):
    assert script.check_url(server + path, timeout=5)[0] == kind


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
    assert report["checked"] == 2 and set(report) == {"checked", "ok", "broken", "unknown"}
