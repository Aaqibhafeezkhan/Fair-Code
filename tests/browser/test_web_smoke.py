"""Browser smoke test for the static pages (#988).

Every other web test runs the engine under node with a hand-written DOM stub. This one opens the
real pages in Chromium and fails on any console error or page error, so a syntax error only a
browser reports, a wrong element id between HTML and script, a `hidden` rule overridden by CSS or a
script load-order mistake is caught.

Needs the optional Playwright package and a browser, so it is skipped when they are missing:

    make setup-browser   # pip install -e ".[browser]" and python -m playwright install chromium
    make test-browser    # python -m pytest tests/browser -q

CI runs it in its own, non-required workflow (.github/workflows/browser-smoke.yml).
"""

import functools
import http.server
import threading
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed: run `make setup-browser`")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
pytestmark = pytest.mark.browser


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site():
    handler = functools.partial(_QuietHandler, directory=str(REPO_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch()
        except Exception as exc:   # noqa: BLE001 - no browser binary installed
            pytest.skip(f"Chromium is not available (run `make setup-browser`): {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, site):
    context = browser.new_context(accept_downloads=True)
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=site)
    # Analytics scripts (Vercel's /_vercel/* exist only on the host; Umami is third-party) are not
    # under test and may be unreachable: answer them with an empty script.
    context.route("**/*", lambda route: route.continue_()
                  if route.request.url.startswith(site) and "/_vercel/" not in route.request.url
                  else route.fulfill(status=200, content_type="application/javascript", body=""))
    page = context.new_page()
    page.problems = []
    page.on("console", lambda msg: page.problems.append(f"console.{msg.type}: {msg.text}")
            if msg.type in ("error", "warning") else None)
    page.on("pageerror", lambda err: page.problems.append(f"pageerror: {err}"))
    page.on("response", lambda response: page.problems.append(f"HTTP {response.status}: {response.url}")
            if response.status >= 400 and response.url.startswith(site) else None)
    yield page
    context.close()


def _clean(page):
    assert page.problems == []


def test_profile_demo_shows_results_and_the_copy_buttons_work(page, site):
    page.goto(f"{site}/profiler.html?demo")
    page.wait_for_selector("#results:not([hidden])")
    assert page.locator("#dimensions .dim-card, #dimensions > *").count() > 0
    page.click("#copyCliBtn")
    assert "faircode profile" in page.evaluate("navigator.clipboard.readText()")
    page.click("#copyJsonBtn")
    assert '"dimensions"' in page.evaluate("navigator.clipboard.readText()")
    _clean(page)


def test_profile_demo_proxy_hint_button_reveals_the_hints(page, site):
    page.goto(f"{site}/profiler.html?demo")
    page.wait_for_selector("#results:not([hidden])")
    page.click("#proxyHintsBtn")
    page.wait_for_function("document.getElementById('proxyHintsResults') && document.getElementById('proxyHintsResults').children.length > 0")
    _clean(page)


def test_compare_demo_shows_the_drift_results(page, site):
    page.goto(f"{site}/profiler.html?demo=compare")
    page.wait_for_selector("#compareResults:not([hidden])")
    assert page.locator("#compareResults").inner_text().strip() != ""
    page.click("#compareCopyCliBtn")
    assert "faircode compare" in page.evaluate("navigator.clipboard.readText()")
    page.click("#compareProxyHintsBtn")
    page.wait_for_function("document.getElementById('compareProxyHintsResults') && document.getElementById('compareProxyHintsResults').children.length > 0")
    _clean(page)


def test_benchmark_dashboard_renders_without_errors(page, site):
    page.goto(f"{site}/benchmark.html")
    page.wait_for_load_state("networkidle")
    assert page.locator("body").inner_text().strip() != ""
    _clean(page)


def test_home_page_loads_without_errors(page, site):
    page.goto(f"{site}/index.html")
    page.wait_for_load_state("networkidle")
    _clean(page)
