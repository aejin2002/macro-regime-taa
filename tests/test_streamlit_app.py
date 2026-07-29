"""Smoke tests for the Macro Regime TAA Streamlit dashboard
(app/streamlit_app.py), using Streamlit's own AppTest harness.

Integration smoke tests, not unit tests of strategy logic -- they assert
the app starts, all five tabs render without raising, key widgets work,
Project 60/40 never appears anywhere user-facing, and there is no
hardcoded 2026-06-30 date literal anywhere in the app source.

Assumes `data/processed/production_v13_daily.parquet` (and the sibling
v1_2/benchmark parquet artifacts) already exist -- i.e. `python -m
macro_regime.cli update-all` has been run at least once. Skips rather
than triggering a live fetch if they're missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
APP_SOURCE = Path(APP_PATH).read_text()
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from ui import data_loader  # noqa: E402


class _MockedLivePipelineFailure(RuntimeError):
    pass


def _always_fail_live_pipeline(*, refresh_cache=True):
    raise _MockedLivePipelineFailure("mocked: live pipeline disabled by default in this test file")


@pytest.fixture(autouse=True)
def _isolate_live_data_state(monkeypatch):
    """Every test in this file gets a clean slate for the live-data
    caches, AND the live pipeline is forced to fail immediately by
    default (network-free) -- so every existing test in this file
    exercises the exact same release-fallback path the app has always
    used, matching every pre-existing assertion. Tests that specifically
    exercise live/session-fallback mode override
    `data_loader.run_live_production_pipeline` themselves, AFTER this
    fixture runs, via their own `monkeypatch.setattr`."""
    data_loader._run_live_pipeline_cached.clear()
    data_loader._live_result_holder.clear()
    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _always_fail_live_pipeline)
    yield
    data_loader._run_live_pipeline_cached.clear()
    data_loader._live_result_holder.clear()

REQUIRED_ARTIFACTS = [
    PROCESSED_DIR / "production_v13_daily.parquet",
    PROCESSED_DIR / "v1_2_regression_daily.parquet",
    PROCESSED_DIR / "benchmarks_daily.parquet",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED_ARTIFACTS),
    reason="canonical v1.3 parquet artifacts not present -- run "
    "`python -m macro_regime.cli update-all` first "
    "(this test never triggers a live fetch itself).",
)

TAB_LABELS = ["Overview", "Markets", "Performance", "Signals", "Methodology"]
OVERVIEW, MARKETS, PERFORMANCE, SIGNALS, METHODOLOGY = range(5)


def _run() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def _headers(tab) -> list[str]:
    return [m.value for m in tab.markdown if m.value and m.value.startswith("####")]


def _plotly_specs(tab) -> list[dict]:
    return [json.loads(c.proto.spec) for c in tab.get("plotly_chart")]


# =============================================================================
# Basic structure
# =============================================================================


def test_app_renders_without_exception():
    at = _run()
    assert not at.exception, f"App raised: {[str(e) for e in at.exception]}"


def test_five_tabs_present_and_no_more():
    at = _run()
    assert [tab.label for tab in at.tabs] == TAB_LABELS


def test_no_hardcoded_2026_06_30_in_source():
    assert "2026-06-30" not in APP_SOURCE


def test_project_6040_never_rendered_in_any_tab():
    """Project 60/40 may be mentioned in a code comment explaining why
    it's excluded (that's fine), but must never appear in anything
    actually rendered to the user -- markdown text, checkbox/radio
    labels, dataframe content, or captions, on any tab."""
    at = _run()
    for tab in at.tabs:
        rendered_text = " ".join(m.value for m in tab.markdown if m.value)
        rendered_text += " ".join(c.value for c in tab.caption if c.value)
        assert "roject 60/40" not in rendered_text, f"leaked in tab '{tab.label}' markdown/caption"
        widget_labels = (
            [w.label for w in tab.checkbox] + [w.label for w in tab.radio] + [w.label for w in tab.selectbox]
        )
        assert not any("roject 60/40" in (lbl or "") for lbl in widget_labels), (
            f"leaked in tab '{tab.label}' widget label"
        )


def test_v1_3_visible_by_default_on_overview():
    at = _run()
    overview_text = " ".join(m.value for m in at.tabs[OVERVIEW].markdown if m.value)
    assert "Macro Regime TAA v1.3" in overview_text


def test_page_config_favicon_and_title():
    assert 'page_title="Macro Regime TAA"' in APP_SOURCE
    assert 'page_icon="🩵"' in APP_SOURCE


def test_slow_fast_tagline_removed():
    assert "Macro는 느리게" not in APP_SOURCE
    assert "Crisis는 빠르게" not in APP_SOURCE


def test_one_line_description_present():
    at = _run()
    overview_text = " ".join(m.value for m in at.tabs[OVERVIEW].markdown if m.value)
    description = (
        "성장과 물가 환경에 따라 기본 자산배분을 전환하고, "
        "금리 위험과 금융시장 위기를 별도 레이어로 관리합니다."
    )
    assert description in overview_text


def test_partial_month_disclosure_renders_safely():
    """Whether or not the latest data happens to fall in a partial month,
    rendering that section must never raise."""
    at = _run()
    assert not at.exception


# =============================================================================
# Overview layout -- Current Positioning content folded in
# =============================================================================


def test_overview_renders_all_former_current_positioning_content():
    at = _run()
    headers = _headers(at.tabs[OVERVIEW])
    for expected in (
        "Current Risk State",
        "Last Rebalance / Next Expected Macro Update",
        "Current Partial-Month Performance",
        "Growth Signal Evidence",
        "Inflation Signal Evidence",
        "Current Allocation",
        "Latest Contribution by Asset",
    ):
        assert any(expected in h for h in headers), f"missing '{expected}' on Overview"


def test_overview_section_order_matches_spec():
    at = _run()
    headers = _headers(at.tabs[OVERVIEW])

    def idx(name: str) -> int:
        matches = [i for i, h in enumerate(headers) if name in h]
        assert matches, f"'{name}' not found in Overview headers: {headers}"
        return matches[0]

    risk = idx("Current Risk State")
    rebalance = idx("Last Rebalance / Next Expected Macro Update")
    partial = idx("Current Partial-Month Performance")
    growth = idx("Growth Signal Evidence")
    inflation = idx("Inflation Signal Evidence")
    allocation = idx("Current Allocation")
    contrib = idx("Latest Contribution by Asset")

    assert risk < rebalance < partial < growth < inflation < allocation < contrib


def test_overview_comparison_graph_removed():
    at = _run()
    headers = _headers(at.tabs[OVERVIEW])
    assert not any("Comparison" in h for h in headers)


def test_overview_no_leftover_current_positioning_tab():
    at = _run()
    assert [tab.label for tab in at.tabs] == TAB_LABELS
    assert "Current Positioning" not in [tab.label for tab in at.tabs]


# =============================================================================
# Reflation regime styling
# =============================================================================


def test_regime_tone_mapping_reflation_distinct_from_crisis_and_others():
    from ui import components
    from ui.theme import TONE_BG, TONE_BORDER, TONE_TEXT

    assert components.regime_tone_for("REFLATION") == "reflation"
    assert components.regime_tone_for("GOLDILOCKS") == "good"
    assert components.regime_tone_for("STAGFLATION") == "risk"
    assert components.regime_tone_for("CONTRACTION") == "neutral"
    assert components.regime_tone_for("UNKNOWN") == "neutral"
    assert components.regime_tone_for(None) == "neutral"

    # Reflation must never collide visually with the risk/crisis tone.
    assert TONE_TEXT["reflation"] != TONE_TEXT["risk"]
    assert TONE_BG["reflation"] != TONE_BG["risk"]
    assert TONE_BORDER["reflation"] != TONE_BORDER["risk"]
    assert TONE_TEXT["reflation"] == "#9A5B00"
    assert TONE_BG["reflation"] == "#FFF3D6"
    assert TONE_BORDER["reflation"] == "#E6B85C"


def test_overview_macro_regime_card_uses_regime_tone():
    """The live artifact's current regime (whatever it is) must render
    with the color the REGIME_TONE mapping assigns it -- not a hardcoded
    tone -- verified by finding the Macro Regime card's HTML and checking
    it carries that tone's colors."""
    from ui import components
    from ui.theme import TONE_BG

    at = _run()
    v13 = pd.read_parquet(PROCESSED_DIR / "production_v13_daily.parquet").set_index("date").sort_index()
    current_regime = str(v13.iloc[-1]["macro_regime"])
    expected_tone = components.regime_tone_for(current_regime)
    expected_bg = TONE_BG[expected_tone]

    overview_html = " ".join(
        m.value for m in at.tabs[OVERVIEW].markdown if m.value and "sac-card" in m.value
    )
    assert expected_bg in overview_html


# =============================================================================
# Next Expected Macro Update must not copy Last Rebalance
# =============================================================================


def test_next_expected_macro_update_not_equal_last_rebalance():
    at = _run()
    overview = at.tabs[OVERVIEW]
    metrics = {m.label: m.value for m in overview.metric}
    assert "Last rebalance event" in metrics
    assert "Next expected macro update" in metrics
    last_rebalance = metrics["Last rebalance event"]
    next_update = metrics["Next expected macro update"]
    assert last_rebalance != "n/a"
    assert next_update != "n/a"
    assert next_update != last_rebalance


def test_next_expected_macro_update_is_after_current_market_date():
    at = _run()
    overview = at.tabs[OVERVIEW]
    metrics = {m.label: m.value for m in overview.metric}
    next_update = pd.Timestamp(metrics["Next expected macro update"])
    v13 = pd.read_parquet(PROCESSED_DIR / "production_v13_daily.parquet")
    current_market_date = pd.to_datetime(v13["date"]).max()
    assert next_update > current_market_date


# =============================================================================
# Growth / Inflation Evidence (compact, Overview) -- artifact-only, no CSV
# =============================================================================


def test_evidence_render_path_never_reads_fred_wide_csv_anywhere():
    assert "fred_wide" not in APP_SOURCE


def test_overview_evidence_shows_effective_and_observed_with_dates():
    at = _run()
    body = " ".join(m.value for m in at.tabs[OVERVIEW].markdown if m.value)
    caption_body = " ".join(c.value for c in at.tabs[OVERVIEW].caption if c.value)
    combined = body + " " + caption_body
    assert "Effective:" in combined
    assert "Latest observed:" in combined
    assert "Effective since" in combined
    assert "Observed as of" in combined
    assert "Expected effective" in combined
    assert "month lag" in combined


def test_overview_evidence_no_csv_missing_warning():
    at = _run()
    warning_text = " ".join(w.value for w in at.tabs[OVERVIEW].warning if w.value)
    assert "fred_wide" not in warning_text
    assert "not found" not in warning_text


# =============================================================================
# Markets -- AGG / MALOX / VIX
# =============================================================================


def test_benchmarks_artifact_has_agg_and_malox_columns():
    df = pd.read_parquet(PROCESSED_DIR / "benchmarks_daily.parquet")
    assert {"date", "benchmark_id", "daily_return", "nav", "available"}.issubset(df.columns)
    ids = set(df["benchmark_id"].unique())
    assert "agg" in ids
    assert "malox" in ids


def test_markets_selector_lists_agg_malox_vix():
    at = _run()
    markets = at.tabs[MARKETS]
    series_selector = [m for m in markets.multiselect if m.label == "Series"]
    assert len(series_selector) == 1
    options = series_selector[0].options
    assert any("AGG" in o for o in options)
    assert any("MALOX" in o for o in options)
    assert any("VIX" in o for o in options)


def test_markets_single_selection_renders_one_new_chart():
    """Selecting exactly one benchmark series renders exactly one figure
    for it (in addition to the unrelated, pre-existing asset grid and
    Normalized Performance charts, which this task does not touch)."""
    at = _run()
    markets = at.tabs[MARKETS]
    series_selector = [m for m in markets.multiselect if m.label == "Series"][0]
    series_selector.set_value(["AGG — US Aggregate Bond ETF"]).run()
    markets2 = at.tabs[MARKETS]
    specs = _plotly_specs(markets2)
    titles = [s.get("layout", {}).get("title", {}).get("text", "") for s in specs]
    agg_titles = [t for t in titles if t.startswith("AGG")]
    malox_titles = [t for t in titles if t.startswith("MALOX")]
    assert len(agg_titles) == 1
    assert len(malox_titles) == 0


def test_markets_three_selection_renders_independent_single_series_charts():
    at = _run()
    markets = at.tabs[MARKETS]
    series_selector = [m for m in markets.multiselect if m.label == "Series"][0]
    series_selector.set_value(
        ["AGG — US Aggregate Bond ETF", "MALOX — Allocation Fund NAV", "VIX — CBOE Volatility Index"]
    ).run()
    markets2 = at.tabs[MARKETS]
    specs = _plotly_specs(markets2)
    by_title = {
        s.get("layout", {}).get("title", {}).get("text", ""): s
        for s in specs
        if s.get("layout", {}).get("title")
    }
    agg_spec = next(s for t, s in by_title.items() if t.startswith("AGG"))
    malox_spec = next(s for t, s in by_title.items() if t.startswith("MALOX"))
    assert len(agg_spec["data"]) == 1
    assert len(malox_spec["data"]) == 1
    # No chart mixes AGG and MALOX (or any two market series) into one figure.
    assert not any(t.startswith("AGG") and t.count("MALOX") for t in by_title)


def test_markets_vix_selection_shows_no_fake_chart_and_no_crash():
    at = _run()
    markets = at.tabs[MARKETS]
    series_selector = [m for m in markets.multiselect if m.label == "Series"][0]
    series_selector.set_value(["VIX — CBOE Volatility Index"]).run()
    markets2 = at.tabs[MARKETS]
    assert not at.exception
    caption_text = " ".join(c.value for c in markets2.caption if c.value)
    assert "unavailable" in caption_text.lower()
    specs = _plotly_specs(markets2)
    titles = [s.get("layout", {}).get("title", {}).get("text", "") for s in specs]
    assert not any(t.startswith("VIX") for t in titles)


def test_markets_range_selector_still_present_and_works():
    at = _run()
    markets = at.tabs[MARKETS]
    range_radio = [r for r in markets.radio if r.label == "Range"]
    assert len(range_radio) == 1
    assert set(range_radio[0].options) == {"1M", "3M", "6M", "1Y", "2Y", "ALL"}
    range_radio[0].set_value("ALL").run()
    assert not at.exception


# =============================================================================
# Performance / Signals / Methodology (index shifted, otherwise unchanged)
# =============================================================================


def test_malox_optional_and_unchecked_by_default_on_performance():
    at = _run()
    perf = at.tabs[PERFORMANCE]
    malox_checkbox = [cb for cb in perf.checkbox if cb.label == "MALOX"]
    assert len(malox_checkbox) == 1
    assert malox_checkbox[0].value is False


def test_signals_tab_recent_signal_table_and_filters():
    at = _run()
    signals = at.tabs[SIGNALS]
    headers = _headers(signals)
    assert any("Recent Signal Table" in h for h in headers)
    show_all = [cb for cb in signals.checkbox if "Show all daily states" in cb.label]
    assert len(show_all) == 1
    show_all[0].check().run()
    assert not at.exception


def test_signals_tab_still_shows_full_evidence_unchanged():
    """Signals tab's full-detail Evidence (with the artifact-based data
    flow fixed previously) must keep working exactly as before -- this
    task only adds a compact copy on Overview, it does not touch Signals."""
    at = _run()
    signals = at.tabs[SIGNALS]
    body = " ".join(m.value for m in signals.markdown if m.value)
    assert "Effective signal" in body
    assert "Observed signal" in body
    assert "Why Growth is" in body
    assert "Why Inflation is" in body
    warning_text = " ".join(w.value for w in signals.warning if w.value)
    assert "fred_wide" not in warning_text
    assert "not found" not in warning_text


def test_methodology_tab_has_version_history_and_malox_treatment():
    at = _run()
    methodology = at.tabs[METHODOLOGY]
    headers = _headers(methodology)
    assert any("Version History" in h for h in headers)
    assert any("MALOX Data Treatment" in h for h in headers)
    assert any("Data Freshness Rules" in h for h in headers)


# =============================================================================
# Performance regression -- unchanged figures
# =============================================================================


def test_performance_figures_unchanged():
    at = _run()
    overview = at.tabs[OVERVIEW]
    metrics = {m.label: m.value for m in overview.metric}
    assert metrics["CAGR"] == "11.12%"
    assert metrics["Sharpe"] == "1.14"
    assert metrics["Daily MDD"] == "-16.64%"
    assert metrics["Final Wealth"] == "6.14x"


# =============================================================================
# Live production data: mode, caching, manual refresh, fallback chain
#
# Every test below explicitly overrides `data_loader.run_live_production_pipeline`
# (which the autouse fixture above already points at a network-free stub by
# default) -- none of this ever touches FRED or Yahoo Finance.
# =============================================================================


def _mock_live_result(fetched_at=None):
    v13 = pd.read_parquet(PROCESSED_DIR / "production_v13_daily.parquet")
    bench = pd.read_parquet(PROCESSED_DIR / "benchmarks_daily.parquet")
    latest = pd.Timestamp(v13["date"].max())
    # VIX index intentionally ends at the SAME latest date as v1_3_df so
    # the default mock never trips the "stale" heuristic in ordinary
    # live-mode tests -- tests that specifically want a stale live
    # result build their own dict instead of using this helper.
    vix = pd.Series(
        [15.0, 16.5, 14.2], index=pd.date_range(end=latest, periods=3), name="^VIX"
    )
    return {
        "v1_3_df": v13,
        "bench_df": bench,
        "vix_series": vix,
        "vix_source": "yahoo_^VIX",
        "fetched_at": fetched_at or pd.Timestamp("2026-07-29 03:00:00"),
        "pipeline_run_id": "test-run-id",
        "source_mode": "live_pipeline",
    }


def _mock_live_success(*, refresh_cache=True, calls=None):
    if calls is not None:
        calls.append(1)
    return _mock_live_result()


def test_live_mode_shows_live_badge_and_last_refresh_time(monkeypatch):
    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _mock_live_success)
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "🟢 Live" in sidebar_text
    assert "2026-07-29 03:00:00" in sidebar_text
    assert not at.exception


def test_release_fallback_never_shows_live_badge():
    """Default fixture state (live pipeline always fails, no prior
    success) -- must show the fallback badge, never claim to be live."""
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "🔵 Release fallback" in sidebar_text
    assert "🟢 Live" not in sidebar_text


def test_repeated_run_within_ttl_does_not_recall_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(
        data_loader, "run_live_production_pipeline", lambda **kw: _mock_live_success(calls=calls, **kw)
    )
    _run()
    _run()
    assert len(calls) == 1, "second AppTest run must reuse the cached live result, not re-fetch"


def test_manual_refresh_button_forces_recall(monkeypatch):
    calls = []
    monkeypatch.setattr(
        data_loader, "run_live_production_pipeline", lambda **kw: _mock_live_success(calls=calls, **kw)
    )
    at = _run()
    assert len(calls) == 1
    at.sidebar.button[0].click().run()
    assert len(calls) == 2, "clicking Refresh latest data must bypass the TTL cache and re-fetch"
    assert not at.exception


def test_live_failure_with_prior_success_falls_back_to_session_cache(monkeypatch):
    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _mock_live_success)
    _run()  # populates _live_result_holder with a successful live result

    def _fail(**kw):
        raise RuntimeError("mock: live refresh failed on retry")

    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _fail)
    data_loader._run_live_pipeline_cached.clear()  # simulate TTL expiry forcing a fresh attempt
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "🟡 Cached live (stale)" in sidebar_text
    assert not at.exception


def test_live_failure_with_no_prior_success_falls_back_to_release():
    """Default fixture state IS this scenario -- named explicitly here
    per the spec's required fallback-chain test."""
    at = _run()
    assert not at.exception
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "🔵 Release fallback" in sidebar_text
    # And the app still renders real numbers from the static artifact --
    # fallback is never blank/broken.
    overview = at.tabs[OVERVIEW]
    metrics = {m.label: m.value for m in overview.metric}
    assert metrics["CAGR"] == "11.12%"


def test_vix_renders_real_chart_in_live_mode(monkeypatch):
    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _mock_live_success)
    at = _run()
    markets = at.tabs[MARKETS]
    series_selector = [m for m in markets.multiselect if m.label == "Series"][0]
    series_selector.set_value(["VIX — CBOE Volatility Index"]).run()
    markets2 = at.tabs[MARKETS]
    specs = _plotly_specs(markets2)
    titles = [s.get("layout", {}).get("title", {}).get("text", "") for s in specs]
    assert any(t.startswith("VIX") for t in titles)
    caption_text = " ".join(c.value for c in markets2.caption if c.value)
    assert "unavailable" not in caption_text.lower()


def test_vix_still_shows_unavailable_in_release_fallback_mode():
    """Default fixture state (fallback mode) -- VIX must still show the
    honest 'unavailable' message, never fabricated data."""
    at = _run()
    markets = at.tabs[MARKETS]
    series_selector = [m for m in markets.multiselect if m.label == "Series"][0]
    series_selector.set_value(["VIX — CBOE Volatility Index"]).run()
    markets2 = at.tabs[MARKETS]
    caption_text = " ".join(c.value for c in markets2.caption if c.value)
    assert "unavailable" in caption_text.lower()


def test_release_fallback_error_never_shown_as_stack_trace(monkeypatch):
    """A live failure's error text is compact (an exception message),
    never a raw Python traceback dumped into the general UI."""
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "Traceback (most recent call last)" not in sidebar_text
    for tab in at.tabs:
        rendered = " ".join(m.value for m in tab.markdown if m.value)
        assert "Traceback (most recent call last)" not in rendered


# =============================================================================
# Freshness contract: a past artifact must never be shown as if it were
# the current live result.
# =============================================================================


def test_release_fallback_shows_exact_recommended_snapshot_message():
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    sidebar_captions = " ".join(c.value for c in at.sidebar.caption if c.value)
    combined = sidebar_text + " " + sidebar_captions
    assert "Live production data is unavailable." in combined
    assert "Showing the last validated release snapshot as of" in combined
    # The artifact's own as-of date, not "today" -- confirm SOME real
    # ISO date follows the phrase.
    v13 = pd.read_parquet(PROCESSED_DIR / "production_v13_daily.parquet")
    artifact_date = pd.to_datetime(v13["date"]).max().date().isoformat()
    assert artifact_date in combined


def test_release_fallback_warning_also_shown_prominently_on_overview():
    """Not just the sidebar (easy to miss/collapse) -- Overview itself
    must carry the warning directly above Current Risk State."""
    at = _run()
    overview = at.tabs[OVERVIEW]
    warning_text = " ".join(w.value for w in overview.warning if w.value)
    assert "Live production data is unavailable" in warning_text


def test_release_fallback_never_shows_a_live_refresh_timestamp():
    """The artifact's download/resolution time must never be presented
    as a live-refresh timestamp -- fallback mode must show "None this
    session", never a fabricated or borrowed timestamp."""
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "Last successful live refresh" in sidebar_text
    assert "None this session" in sidebar_text


def test_live_mode_never_shows_release_fallback_message(monkeypatch):
    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _mock_live_success)
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    sidebar_captions = " ".join(c.value for c in at.sidebar.caption if c.value)
    assert "Live production data is unavailable" not in (sidebar_text + sidebar_captions)
    assert not at.tabs[OVERVIEW].warning  # no fallback banner in live mode


def test_session_cached_live_result_carries_explicit_provenance_metadata(monkeypatch):
    """A "Cached live" result is only ever trusted because it explicitly
    carries `source_mode == "live_pipeline"` -- not merely because a
    DataFrame happens to be present in the session holder."""
    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _mock_live_success)
    _run()
    holder = data_loader._live_result_holder()
    assert holder["last_good"]["source_mode"] == "live_pipeline"
    assert holder["last_good"]["pipeline_run_id"]
    assert holder["last_good"]["fetched_at"] is not None


def test_session_value_without_provenance_metadata_is_not_trusted_as_live(monkeypatch):
    """Simulates a corrupted/legacy holder entry (no `source_mode`) --
    must NOT be surfaced as "Cached live"; must fall through to Release
    fallback instead, since a bare DataFrame is not proof of a live run."""

    def _fail(**kw):
        raise RuntimeError("mock: live refresh fails")

    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _fail)
    holder = data_loader._live_result_holder()
    v13 = pd.read_parquet(PROCESSED_DIR / "production_v13_daily.parquet")
    holder["last_good"] = {"v1_3_df": v13, "bench_df": None, "vix_series": None, "fetched_at": None}
    # deliberately no "source_mode" / "pipeline_run_id" key

    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "🔵 Release fallback" in sidebar_text
    assert "🟡 Cached live" not in sidebar_text


def test_stale_market_date_flagged_even_in_live_mode(monkeypatch):
    """A "live" result whose underlying strategy market date is far
    behind today (simulated) is flagged stale rather than shown as an
    unqualified green "Live" -- staleness is about the DATA's own date,
    not just whether the fetch attempt itself succeeded."""

    def _stale_live(**kw):
        result = _mock_live_result()
        v13 = result["v1_3_df"].copy()
        v13["date"] = pd.to_datetime(v13["date"]) - pd.Timedelta(days=30)
        result["v1_3_df"] = v13
        return result

    monkeypatch.setattr(data_loader, "run_live_production_pipeline", _stale_live)
    at = _run()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown if m.value)
    assert "🟢 Live (data delayed)" in sidebar_text
