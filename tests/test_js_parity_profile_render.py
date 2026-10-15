"""Source-level checks of the dashboard card renderers. (split out of test_js_parity_profile.py, #985)."""

from js_helpers import REPO_ROOT


def test_compare_card_renderers_special_case_kind_mismatch():
    """driftCard() and buildCompareHtmlReport()'s per-dimension section must
    both read cd.kind_mismatch, so a skipped comparison isn't drawn as a
    "none drift" badge next to a real score change (#519). Source-level check
    (mirrors test_sheetjs_cdn_url_matches) - these renderers are DOM-coupled
    and have no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")

    drift_card = src[src.index("function driftCard("):]
    drift_card = drift_card[: drift_card.index("\n  }\n")]
    assert "kind_mismatch" in drift_card
    assert "comparison skipped" in drift_card

    report = src[src.index("function buildCompareHtmlReport("):]
    assert "if (cd.kind_mismatch)" in report
    # the skipped badge is styled in both the live css and the report's own <style>
    assert ".drift-badge.skipped" in src
    assert ".drift-badge.skipped" in (REPO_ROOT / "assets" / "profiler.css").read_text(encoding="utf-8")


def test_dim_card_renders_an_expand_control_past_display_groups():
    """profiler-ui.js's dimCard() must render extra groups past DISPLAY_GROUPS
    up front (hidden) with a toggle button, rather than the old static "...and
    N more groups" text with no way to actually see them (#740). Source-level
    check (mirrors test_threshold_input_recovers_panel_after_invalid_value) -
    this is DOM-coupled and has no unit harness."""
    src = (REPO_ROOT / "assets" / "profiler-ui.js").read_text(encoding="utf-8")

    marker = "  function dimCard(d) {"
    fn = src[src.index(marker):]
    fn = fn[: fn.index("\n  function renderIntersections")]

    assert "dim-extra-groups" in fn
    assert 'aria-expanded="false"' in fn
    assert "dim-more-btn" in fn
    # the click handler must flip the hidden attribute and the aria state together
    assert "extra.hidden = expanded" in fn
    assert "btn.setAttribute('aria-expanded'" in fn
    # the old dead-end text is gone
    assert "… and " not in fn


def test_drift_card_renders_an_expand_control_past_display_groups():
    """assets/profiler-compare.js's driftCard() gets the same #740 treatment,
    with a resultsEl-level delegated click handler (driftCard rebuilds via an
    innerHTML string, not a DOM node dimCard() can attach a listener to
    directly)."""
    src = (REPO_ROOT / "assets" / "profiler-compare.js").read_text(encoding="utf-8")

    card_fn_marker = "  function driftCard(cd) {"
    card_fn = src[src.index(card_fn_marker):]
    card_fn = card_fn[: card_fn.index("\n\n  // ── Report export")]
    assert "dim-extra-groups" in card_fn
    assert "dim-more-btn" in card_fn
    assert "… and " not in card_fn

    delegated_marker = "resultsEl.addEventListener('click'"
    assert delegated_marker in src
    handler = src[src.index(delegated_marker):]
    handler = handler[: handler.index("\n  });")]
    assert "closest" in handler
    assert "extra.hidden = expanded" in handler
