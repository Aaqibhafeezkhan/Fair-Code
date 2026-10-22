"""Browser smoke test for the static pages (#988).

Every other web test runs the engine under node with a hand-written DOM stub. This one opens the
real pages in Chromium and fails on any console error or page error, so a syntax error only a
browser reports, a wrong element id between HTML and script, a `hidden` rule overridden by CSS or a
script load-order mistake is caught.

Needs the optional Playwright package and a browser, so it is skipped when they are missing:

    make setup-browser   # pip install -e ".[browser]" and python -m playwright install chromium
    make test-browser    # python -m pytest tests/browser -q

CI does not run it (the browser-smoke workflow was removed); run `make test-browser` before changing the web pages.
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


# ── Interactions (#1000): the controls real people use first ─────────────────────────────────

_CSV = "sex,race,age\n" + "".join(
    f"{'male' if i % 2 else 'female'},{['White', 'Black', 'Asian'][i % 3]},{20 + i % 40}\n" for i in range(90))


def _upload(page, tmp_path, text=_CSV, name="people.csv", selector="#fileInput"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    page.set_input_files(selector, str(path))


def _open_details(page):
    page.evaluate("document.querySelectorAll('details').forEach(function (d) { d.open = true; })")


def test_uploading_a_file_shows_results_and_a_threshold_edit_reprofiles(page, site, tmp_path):
    page.goto(f"{site}/profiler.html")
    _upload(page, tmp_path)
    page.wait_for_selector("#results:not([hidden])")
    _open_details(page)
    page.fill("#minShareInput", "0.4")
    page.wait_for_function("document.getElementById('results').innerText.includes('under-represented')")
    page.fill("#minShareInput", "1.5")          # rejected by the engine: the panel must come back
    page.wait_for_selector("#results:not([hidden])")
    _clean(page)


def test_a_bad_file_shows_an_error_not_a_console_error(page, site, tmp_path):
    page.goto(f"{site}/profiler.html")
    _upload(page, tmp_path, text="", name="empty.csv")
    page.wait_for_selector("#error:not([hidden])")
    assert "empty" in page.inner_text("#error")
    _clean(page)


def test_keywords_are_remembered_and_the_other_view_picks_them_up(page, site, tmp_path):
    page.goto(f"{site}/profiler.html?demo")
    page.wait_for_selector("#results:not([hidden])")
    _open_details(page)
    page.fill("#keywordsInput", '{"sex": ["gndr"]}')
    page.wait_for_function("localStorage.getItem('faircode.keywords.v1') !== null")
    page.wait_for_function("document.getElementById('compareKeywordsInput').value.includes('gndr')")
    page.fill("#keywordsInput", '{"colour": ["x"]}')
    page.wait_for_function("!document.getElementById('keywordsError').hidden")
    assert "unknown key" in page.inner_text("#keywordsError")
    page.reload()
    page.wait_for_selector("#results", state="attached")
    _clean(page)


def test_a_held_out_row_feeds_the_proxy_check(page, site, tmp_path):
    page.goto(f"{site}/profiler.html")
    _upload(page, tmp_path, text="sex,age\n" + "".join(f"{'male' if i % 2 else 'female'},{20 + i % 40}\n" for i in range(90)))
    page.wait_for_selector("#results:not([hidden])")
    _open_details(page)
    held = tmp_path / "held.csv"
    held.write_text("race\n" + "".join(f"{['White', 'Black', 'Asian'][i % 3]}\n" for i in range(90)), encoding="utf-8")
    page.set_input_files("#heldOutRows input[type=file]", str(held))
    page.fill("#heldOutRows input[type=text]", "race")
    page.click("#proxyHintsBtn")
    page.wait_for_function("document.getElementById('proxyHintsResults').children.length > 0")
    _clean(page)


def test_the_exact_inputs_follow_the_checkbox_and_report_a_bad_value_inline(page, site):
    page.goto(f"{site}/profiler.html?demo")
    page.wait_for_selector("#results:not([hidden])")
    _open_details(page)
    assert page.is_disabled("#proxyExactPermutationsInput")
    page.check("#proxyExactInput")
    assert page.is_enabled("#proxyExactPermutationsInput")
    page.fill("#proxyExactPermutationsInput", "0")
    page.click("#proxyHintsBtn")
    page.wait_for_selector("#proxyExactError:not([hidden])")
    assert "Shuffles" in page.inner_text("#proxyExactError")
    _clean(page)


# ── High-cardinality columns keep the flag lists capped in the real DOM (#1080) ──────────────

def _zip_csv(seed, n=20000, labels=18000):
    import random
    rng = random.Random(seed)
    return "sex,zip\n" + "".join(f"{rng.choice(['m', 'f'])},{10000 + rng.randrange(labels)}\n" for _ in range(n))


def test_a_high_cardinality_column_keeps_the_profile_flag_list_capped(page, site, tmp_path):
    page.goto(f"{site}/profiler.html")
    _upload(page, tmp_path, text=_zip_csv(1), name="zips.csv")
    page.wait_for_selector("#results:not([hidden])", timeout=120000)
    items = page.locator("#flagsList li").count()
    assert 1 <= items <= 51, items        # 50 flags plus the "and N more" line; it used to be 71,633
    _clean(page)


def test_a_high_cardinality_column_keeps_the_compare_flag_list_capped(page, site, tmp_path):
    page.goto(f"{site}/profiler.html")
    (tmp_path / "a.csv").write_text(_zip_csv(2), encoding="utf-8")
    (tmp_path / "b.csv").write_text(_zip_csv(3), encoding="utf-8")
    page.set_input_files("#fileA", str(tmp_path / "a.csv"))
    page.set_input_files("#fileB", str(tmp_path / "b.csv"))
    page.wait_for_selector("#compareResults:not([hidden])", timeout=120000)
    items = page.locator("#compareResults .flags-block li").count()
    assert 1 <= items <= 51, items        # it used to list one line per appeared or disappeared zip code
    _clean(page)
