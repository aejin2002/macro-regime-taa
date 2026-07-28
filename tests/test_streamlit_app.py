"""Smoke tests for the v1.3 Streamlit dashboard (app/streamlit_app.py),
using Streamlit's own AppTest harness.

Integration smoke tests, not unit tests of strategy logic -- they assert
the app starts, all six tabs render without raising, key widgets work,
Project 60/40 never appears anywhere user-facing, and there is no
hardcoded 2026-06-30 date literal anywhere in the app source.

Assumes `data/processed/production_v13_daily.parquet` (and the sibling
v1_2/benchmark parquet artifacts) already exist -- i.e. `python -m
macro_regime.cli update-all` has been run at least once. Skips rather
than triggering a live fetch if they're missing.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
APP_SOURCE = Path(APP_PATH).read_text()
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

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

TAB_LABELS = ["Overview", "Current Positioning", "Markets", "Performance", "Signals", "Methodology"]


def _run() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def test_app_renders_without_exception():
    at = _run()
    assert not at.exception, f"App raised: {[str(e) for e in at.exception]}"


def test_all_six_tabs_present_and_no_more():
    at = _run()
    assert [tab.label for tab in at.tabs] == TAB_LABELS


def test_no_hardcoded_2026_06_30_in_source():
    assert "2026-06-30" not in APP_SOURCE


def test_project_6040_never_rendered_in_any_tab():
    """Project 60/40 may be mentioned in a code comment explaining why
    it's excluded (that's fine), but must never appear in anything
    actually rendered to the user -- markdown text, checkbox/radio
    labels, dataframe content, or captions, on any of the six tabs."""
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
    overview_text = " ".join(m.value for m in at.tabs[0].markdown if m.value)
    assert "Macro Regime TAA v1.3" in overview_text


def test_us_6040_default_checked_and_disabled_on_overview():
    at = _run()
    overview = at.tabs[0]
    us_checkbox = [cb for cb in overview.checkbox if cb.label == "US 60/40"]
    assert len(us_checkbox) == 1
    assert us_checkbox[0].value is True
    assert us_checkbox[0].disabled is True


def test_malox_optional_and_unchecked_by_default_on_performance():
    at = _run()
    perf = at.tabs[3]
    malox_checkbox = [cb for cb in perf.checkbox if cb.label == "MALOX"]
    assert len(malox_checkbox) == 1
    assert malox_checkbox[0].value is False


def test_toggling_malox_on_overview_does_not_raise():
    at = _run()
    overview = at.tabs[0]
    malox_checkboxes = [cb for cb in overview.checkbox if cb.label == "MALOX"]
    if not malox_checkboxes:
        pytest.skip("MALOX unavailable in this data snapshot")
    malox_checkboxes[0].check().run()
    assert not at.exception


def test_markets_tab_range_selector_works():
    at = _run()
    markets = at.tabs[2]
    range_radio = [r for r in markets.radio if r.label == "Range"]
    assert len(range_radio) == 1
    range_radio[0].set_value("1Y").run()
    assert not at.exception


def test_signals_tab_recent_signal_table_and_filters():
    at = _run()
    signals = at.tabs[4]
    headers = [m.value for m in signals.markdown if m.value and m.value.startswith("####")]
    assert any("Recent Signal Table" in h for h in headers)
    show_all = [cb for cb in signals.checkbox if "Show all daily states" in cb.label]
    assert len(show_all) == 1
    show_all[0].check().run()
    assert not at.exception


def test_methodology_tab_has_version_history_and_malox_treatment():
    at = _run()
    methodology = at.tabs[5]
    headers = [m.value for m in methodology.markdown if m.value and m.value.startswith("####")]
    assert any("Version History" in h for h in headers)
    assert any("MALOX Data Treatment" in h for h in headers)
    assert any("Data Freshness Rules" in h for h in headers)


def test_partial_month_disclosure_renders_safely():
    """Whether or not the latest data happens to fall in a partial month,
    rendering that section must never raise."""
    at = _run()
    assert not at.exception


def test_current_positioning_tab_renders_risk_state():
    at = _run()
    positioning = at.tabs[1]
    headers = [m.value for m in positioning.markdown if m.value and m.value.startswith("####")]
    assert not any("Target vs Drifted Allocation" in h for h in headers)
    assert any("Current Risk State" in h for h in headers)


def test_signals_tab_evidence_render_path_never_reads_fred_wide_csv():
    """`app/streamlit_app.py` must not read data/processed/fred_wide.csv to
    render Growth/Inflation Signal Evidence -- that file is gitignored and
    never present in a deployment checkout. Source-level guard: no
    fred_wide.csv path is constructed between the Growth Signal Evidence
    header and the Inflation Signal Evidence block."""
    start = APP_SOURCE.index('st.markdown("#### Growth Signal Evidence")')
    end = APP_SOURCE.index('st.markdown("#### BEI Duration Gate")')
    evidence_source = APP_SOURCE[start:end]
    assert "fred_wide" not in evidence_source
    assert "read_csv" not in evidence_source


def test_signals_tab_shows_effective_and_observed_growth_signal():
    at = _run()
    signals = at.tabs[4]
    body = " ".join(m.value for m in signals.markdown if m.value)
    assert "Effective signal" in body
    assert "Observed signal" in body
    assert "Why Growth is" in body


def test_signals_tab_shows_effective_and_observed_inflation_signal():
    at = _run()
    signals = at.tabs[4]
    body = " ".join(m.value for m in signals.markdown if m.value)
    assert "Why Inflation is" in body
    assert body.count("Effective signal") == 2
    assert body.count("Observed signal") == 2


def test_signals_tab_evidence_shows_dates():
    import pandas as pd

    at = _run()
    signals = at.tabs[4]
    body = " ".join(m.value for m in signals.markdown if m.value)
    v13 = pd.read_parquet(PROCESSED_DIR / "production_v13_daily.parquet")
    observed_date = pd.to_datetime(v13["date"]).max().date().isoformat()
    assert f"as of {observed_date}" in body


def test_signals_tab_never_shows_csv_missing_warning():
    at = _run()
    signals = at.tabs[4]
    warning_text = " ".join(w.value for w in signals.warning if w.value)
    assert "fred_wide" not in warning_text
    assert "not found" not in warning_text


def test_signals_tab_renders_evidence_without_fred_wide_csv_present(tmp_path, monkeypatch):
    """Even if fred_wide.csv is entirely absent (the real deployment
    condition), the Signals tab must still show effective/observed
    signals and dates -- no warning, no exception."""
    wide_path = PROCESSED_DIR / "fred_wide.csv"
    if not wide_path.exists():
        pytest.skip("fred_wide.csv not present locally -- nothing to hide for this test")
    backup = tmp_path / "fred_wide.csv.bak"
    wide_path.rename(backup)
    try:
        at = _run()
        assert not at.exception
        signals = at.tabs[4]
        body = " ".join(m.value for m in signals.markdown if m.value)
        warning_text = " ".join(w.value for w in signals.warning if w.value)
        assert "Effective signal" in body
        assert "Observed signal" in body
        assert "fred_wide" not in warning_text
        assert "not found" not in warning_text
    finally:
        backup.rename(wide_path)
