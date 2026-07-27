"""Streamlit research UI for macro-regime-taa.

Reads the artifacts produced by `python -m macro_regime.cli run-all`
(data/processed/*.csv, *.json). Locally, run `make fetch build-signals
evaluate` (or `make run-all`) before launching this app for the first
time. `data/processed/*` is git-ignored and never committed -- on a
fresh deploy (e.g. Streamlit Community Cloud) those files don't exist
yet, so this module bootstraps them automatically on first load; see
`_ensure_core_data` / `_ensure_backtest_data` below.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_regime.cli import (  # noqa: E402
    build_regime_output_cmd,
    build_signals,
    evaluate,
    fetch,
    run_backtest_cmd,
)
from macro_regime.config import MissingApiKeyError, load_config  # noqa: E402
from macro_regime.signals.regime import (  # noqa: E402
    Regime,
    regime_distribution,
    regime_durations,
    regime_transitions,
    transition_matrix,
)

PROCESSED_DIR = ROOT / "data" / "processed"
WIDE_PATH = PROCESSED_DIR / "fred_wide.csv"
SIGNALS_PATH = PROCESSED_DIR / "signals.csv"
REGIME_METADATA_PATH = PROCESSED_DIR / "regime_metadata.json"
EVAL_PATH = PROCESSED_DIR / "evaluation_report.json"
SERIES_METADATA_PATH = PROCESSED_DIR / "series_metadata.json"
REGIME_OUTPUT_PRIMARY_PATH = PROCESSED_DIR / "regime_output_primary.csv"
REGIME_OUTPUT_SECONDARY_PATH = PROCESSED_DIR / "regime_output_secondary.csv"
BACKTEST_SUMMARY_PATH = PROCESSED_DIR / "backtest_summary.csv"
BACKTEST_ANNUAL_RETURNS_PATH = PROCESSED_DIR / "backtest_annual_returns.csv"
BACKTEST_MONTHLY_RETURNS_PATH = PROCESSED_DIR / "backtest_monthly_returns.csv"
BACKTEST_ALLOCATIONS_PRIMARY_PATH = PROCESSED_DIR / "backtest_allocations_primary.csv"
BACKTEST_ALLOCATIONS_SECONDARY_PATH = PROCESSED_DIR / "backtest_allocations_secondary.csv"
BACKTEST_REGIME_ANALYSIS_PATH = PROCESSED_DIR / "backtest_regime_analysis.csv"
BACKTEST_OUTPUT_PATHS = [
    BACKTEST_SUMMARY_PATH,
    BACKTEST_ANNUAL_RETURNS_PATH,
    BACKTEST_MONTHLY_RETURNS_PATH,
    BACKTEST_ALLOCATIONS_PRIMARY_PATH,
    BACKTEST_ALLOCATIONS_SECONDARY_PATH,
    BACKTEST_REGIME_ANALYSIS_PATH,
]

REGIME_COLORS = {
    Regime.GOLDILOCKS.value: "#2E7D32",
    Regime.REFLATION.value: "#F9A825",
    Regime.STAGFLATION.value: "#C62828",
    Regime.CONTRACTION.value: "#455A64",
    Regime.UNKNOWN.value: "#9E9E9E",
}

st.set_page_config(page_title="macro-regime-taa", layout="wide")


def _bridge_streamlit_secrets_to_env() -> None:
    """Streamlit Community Cloud supplies secrets via `st.secrets`
    (configured in the app's Secrets settings), not a committed `.env`
    file -- `.env` is git-ignored and never deployed. Bridge
    `FRED_API_KEY` into the process environment once, so the existing
    `Settings`/`FredClient` (which only read `os.environ` / `.env`) pick
    it up unmodified. No-op locally where `FRED_API_KEY` is already set
    via `.env`, and degrades silently if no secrets store exists at all
    (e.g. no `.streamlit/secrets.toml` locally)."""
    if os.environ.get("FRED_API_KEY"):
        return
    try:
        key = st.secrets["FRED_API_KEY"]
    except Exception:
        return
    if key:
        os.environ["FRED_API_KEY"] = key


@st.cache_resource(show_spinner="Fetching FRED data and computing signals (first run only, ~1-2 min)...")
def _ensure_core_data() -> str | None:
    """Run fetch -> build-signals -> evaluate -> build-regime-output once
    per running server process if the core outputs don't exist yet --
    e.g. a fresh deploy, since `data/processed/*` is never committed.
    Returns an error message on failure, `None` on success (including
    "already had data" -- checked first, so this is cheap on every
    subsequent Streamlit rerun). `@st.cache_resource` scopes the actual
    pipeline run to once per process; it does not re-run on later calls
    or across user sessions sharing the same running app."""
    if SIGNALS_PATH.exists() and REGIME_OUTPUT_PRIMARY_PATH.exists():
        return None
    try:
        fetch(start_date=None, end_date=None, refresh_cache=False)
        build_signals(growth_model="all", inflation_model="all")
        evaluate()
        build_regime_output_cmd()
    except MissingApiKeyError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 -- surface any pipeline failure to the UI, never crash silently
        return f"Failed to build initial data: {exc}"
    return None


def _ensure_backtest_data() -> None:
    """Run the (slower, separately yfinance-backed) backtest pipeline if
    any of its output files are missing -- e.g. a fresh deploy, or a
    Streamlit Community Cloud reboot that wiped the ephemeral
    filesystem out from under an otherwise still-running process. Checks
    the filesystem directly on every call (deliberately not
    `@st.cache_resource`-memoized) so a reboot that clears the files is
    always noticed, instead of trusting a stale in-memory "already ran"
    flag. Only called when the Backtest page is actually visited -- no
    reason to pay this cost for users who never open that page.

    On success, triggers `st.rerun()` so the page re-executes from the
    top and reads the freshly written files -- reusing them in the same
    run would risk hitting a `st.cache_data`-wrapped loader that already
    cached `None` for that path from before the files existed. On
    failure, surfaces the real exception in the UI (`st.exception`) and
    halts (`st.stop()`) rather than swallowing it."""
    if all(p.exists() for p in BACKTEST_OUTPUT_PATHS):
        return

    if not REGIME_OUTPUT_PRIMARY_PATH.exists() or not REGIME_OUTPUT_SECONDARY_PATH.exists():
        core_error = _ensure_core_data()
        if core_error:
            st.error(f"Couldn't build the core regime output needed for the backtest: {core_error}")
            st.stop()

    st.info("Building backtest outputs. This may take 1-2 minutes...")
    try:
        with st.spinner("Fetching asset/FX prices and running the backtest..."):
            run_backtest_cmd(refresh_cache=False)
    except Exception as exc:  # noqa: BLE001 -- caught only to surface it via st.exception, never swallowed
        st.error("Failed to build backtest outputs.")
        st.exception(exc)
        st.stop()

    st.rerun()


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data
def _load_csv_plain(path: Path) -> pd.DataFrame | None:
    """Like `_load_csv`, but for CSVs with no date index (e.g. summary /
    long-format tables written with `index=False`)."""
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _missing_data_notice(needed: str) -> None:
    st.warning(
        f"{needed} not found. Run `make fetch`, `make build-signals`, and "
        f"`make evaluate` (or `make run-all` / `python -m macro_regime.cli run-all`) "
        f"before using this page."
    )


def page_overview(signals: pd.DataFrame | None, wide: pd.DataFrame | None, regime_meta: dict | None) -> None:
    st.title("Overview")
    if signals is None:
        _missing_data_notice("signals.csv")
        return

    growth_cols = [c for c in signals.columns if c.startswith("growth_model")]
    inflation_cols = [c for c in signals.columns if c.startswith("inflation_model")]
    regime_cols = [c for c in signals.columns if c.startswith("regime_")]

    # signals.csv's date index is the union of every underlying series'
    # resampled dates -- a still-in-progress month can already have a row
    # (e.g. weekly Initial Claims has a July observation) even though
    # monthly-frequency inputs (CLI, CPI, industrial production, ...)
    # haven't published a July value yet. That row would show up here as
    # NaN across most growth/inflation columns and UNKNOWN for every
    # regime. Use the most recent row where every growth/inflation model
    # actually produced a real classification instead of the literal last
    # row -- this only changes what the Overview page displays; signals.csv
    # itself and every other page are untouched.
    model_cols = growth_cols + inflation_cols
    complete_dates = signals.index[signals[model_cols].notna().all(axis=1)] if model_cols else signals.index
    raw_latest_date = signals.index.max()
    latest_date = complete_dates.max() if len(complete_dates) else raw_latest_date

    st.caption(f"Latest complete signal date: **{latest_date.date() if pd.notna(latest_date) else 'n/a'}**")
    if pd.notna(raw_latest_date) and pd.notna(latest_date) and raw_latest_date != latest_date:
        st.caption(
            f"signals.csv's most recent row ({raw_latest_date.date()}) is a still-in-progress month -- "
            "most monthly-frequency inputs haven't been published for it yet, so it's skipped here."
        )
    st.caption(
        "All figures below use current, revised FRED data (a 'revised-data backtest'), "
        "not ALFRED real-time vintages -- see the Methodology page."
    )

    latest = signals.loc[latest_date] if pd.notna(latest_date) else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Growth (latest)")
        for c in growth_cols:
            st.metric(c.replace("growth_model_", ""), str(latest[c]) if latest is not None else "n/a")
    with col2:
        st.subheader("Inflation (latest)")
        for c in inflation_cols:
            st.metric(c.replace("inflation_model_", ""), str(latest[c]) if latest is not None else "n/a")
    with col3:
        st.subheader("Regime (latest, by model pair)")
        for c in regime_cols[:6]:
            st.metric(c.replace("regime_", ""), str(latest[c]) if latest is not None else "n/a")

    st.divider()
    st.subheader("Model combinations in use")
    if regime_meta:
        st.dataframe(pd.DataFrame(regime_meta).T, use_container_width=True)
    else:
        st.info("Run `build-signals` to generate regime_metadata.json.")


def page_data_explorer(wide: pd.DataFrame | None, series_meta: dict | None) -> None:
    st.title("Data Explorer")
    if wide is None:
        _missing_data_notice("fred_wide.csv")
        return

    series = st.selectbox("Series", sorted(wide.columns))
    fig = px.line(wide[series].dropna(), title=series)
    st.plotly_chart(fig, use_container_width=True)

    if series_meta and series in series_meta:
        st.json(series_meta[series])


def page_growth_models(signals: pd.DataFrame | None) -> None:
    st.title("Growth Models")
    if signals is None:
        _missing_data_notice("signals.csv")
        return
    growth_cols = [c for c in signals.columns if c.startswith("growth_model")]
    if not growth_cols:
        st.info("No growth model output found. Some models may be disabled (missing external CSVs).")
        return

    fig = go.Figure()
    for c in growth_cols:
        numeric = signals[c].map({"Up": 1, "Down": -1, "Unknown": 0, "Neutral": 0})
        fig.add_trace(go.Scatter(x=signals.index, y=numeric, mode="lines", name=c))
    fig.update_layout(title="Growth direction by model (Up=1, Down=-1)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(signals[growth_cols].tail(24), use_container_width=True)


def page_inflation_models(signals: pd.DataFrame | None) -> None:
    st.title("Inflation Models")
    if signals is None:
        _missing_data_notice("signals.csv")
        return
    inflation_cols = [c for c in signals.columns if c.startswith("inflation_model")]
    if not inflation_cols:
        st.info("No inflation model output found.")
        return

    fig = go.Figure()
    for c in inflation_cols:
        numeric = signals[c].map({"Up": 1, "Down": -1, "Unknown": 0})
        fig.add_trace(go.Scatter(x=signals.index, y=numeric, mode="lines", name=c))
    fig.update_layout(title="Inflation direction by model (Up=1, Down=-1)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(signals[inflation_cols].tail(24), use_container_width=True)

    st.caption(
        "Model C (Cleveland Median CPI Momentum) is an institutional "
        "underlying-inflation trend signal, not the Cleveland Fed Inflation "
        "Nowcast. See the Methodology page."
    )


def page_regime_comparison(signals: pd.DataFrame | None, regime_meta: dict | None) -> None:
    st.title("Regime Comparison")
    if signals is None:
        _missing_data_notice("signals.csv")
        return
    regime_cols = [c for c in signals.columns if c.startswith("regime_")]
    if not regime_cols:
        st.info("No regime columns found.")
        return

    chosen = st.selectbox("Regime (growth x inflation model pair)", regime_cols)
    if regime_meta and chosen in regime_meta:
        st.caption(f"Growth model: `{regime_meta[chosen]['growth_model']}` | "
                   f"Inflation model: `{regime_meta[chosen]['inflation_model']}`")

    series = signals[chosen].dropna()
    order = ["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION", "UNKNOWN"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=[order.index(v) for v in series],
            mode="markers+lines",
            marker=dict(color=[REGIME_COLORS.get(v, "#000") for v in series]),
        )
    )
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=order)
    fig.update_layout(title=f"Regime timeline: {chosen}")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution")
        st.bar_chart(regime_distribution(series))
        st.metric("Number of transitions", regime_transitions(series))
    with col2:
        st.subheader("Transition matrix")
        st.dataframe(transition_matrix(series), use_container_width=True)

    st.subheader("Regime spells (duration)")
    st.dataframe(regime_durations(series), use_container_width=True)


def _render_regime_output_column(label: str, df: pd.DataFrame | None) -> None:
    st.subheader(label)
    if df is None or df.empty:
        st.info(f"{label} regime output not found. Run `build-regime-output` (or `run-all`).")
        return

    latest_date = df.index.max()
    latest = df.loc[latest_date]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest raw regime", str(latest["raw_regime"]))
    with col2:
        st.metric("Latest tradable regime", str(latest["tradable_regime"]))
    st.caption(f"As of {latest_date.date()}")

    st.markdown("**Last 24 months**")
    cols = [
        "growth_score",
        "growth_state",
        "inflation_score",
        "inflation_state",
        "raw_regime",
        "tradable_regime",
    ]
    st.dataframe(df[cols].tail(24), use_container_width=True)

    dist = regime_distribution(df["raw_regime"])
    durations = regime_durations(df["raw_regime"])
    avg_duration = (
        durations.groupby("regime")["length_months"].mean().reindex([r.value for r in Regime]).fillna(0.0)
        if not durations.empty
        else pd.Series(0.0, index=[r.value for r in Regime])
    )

    st.markdown("**Regime frequency (raw_regime) and average spell duration**")
    freq_table = pd.DataFrame({"frequency": dist, "avg_duration_months": avg_duration})
    st.dataframe(freq_table, use_container_width=True)


def page_regime_output(primary: pd.DataFrame | None, secondary: pd.DataFrame | None) -> None:
    st.title("Regime Output")
    st.caption(
        "Standard asset-allocation regime output for the two frozen core "
        "models (commit 40e43d7). Revised-data backtest -- see the "
        "Methodology page for tradable_regime semantics and why Core "
        "Inflation Momentum is treated as a current-environment classifier, "
        "not a strong short-term predictor."
    )
    col1, col2 = st.columns(2)
    with col1:
        _render_regime_output_column("Primary (US OECD CLI x Core Inflation Momentum)", primary)
    with col2:
        _render_regime_output_column("Secondary (AMTMNO + Claims x Core Inflation Momentum)", secondary)


def page_backtest(
    summary: pd.DataFrame | None,
    annual_returns: pd.DataFrame | None,
    monthly_returns: pd.DataFrame | None,
    allocations_primary: pd.DataFrame | None,
    allocations_secondary: pd.DataFrame | None,
    regime_analysis: pd.DataFrame | None,
) -> None:
    st.title("Backtest")
    if summary is None or monthly_returns is None:
        st.warning(
            "Backtest output not found. Run `python -m macro_regime.cli "
            "run-backtest` (after `build-regime-output`)."
        )
        return

    st.caption(
        "Fixed ex-ante regime allocation -- weights are not optimized or "
        "fit to data. Regime classification is a revised-FRED-data "
        "backtest; asset/FX prices are actual historical closes, but this "
        "is still not a real-time track record. 1-month "
        "portfolio-application lag via tradable_regime. Common sample "
        "starts 2009-05, driven by a real Yahoo Finance data gap in "
        "069500.KS (KODEX 200) from 2007-03 to 2009-03, not by BIL's "
        "~2007 inception -- 2008 (GFC) is therefore not covered, only "
        "2020 and 2022. See the Methodology page, 'Backtest', for full "
        "caveats."
    )

    st.subheader("Strategy performance (post-cost)")
    display_cols = [
        "strategy",
        "start_date",
        "end_date",
        "cagr_post_cost",
        "annualized_vol_post_cost",
        "sharpe_post_cost",
        "sortino_post_cost",
        "max_drawdown_post_cost",
        "calmar_post_cost",
        "monthly_win_rate",
        "annual_positive_year_ratio",
        "avg_annual_turnover",
        "final_value_post_cost",
    ]
    st.dataframe(summary[[c for c in display_cols if c in summary.columns]], use_container_width=True)
    with st.expander("Full summary (pre-cost vs. post-cost)"):
        st.dataframe(summary, use_container_width=True)

    cumulative = (1 + monthly_returns.fillna(0)).cumprod()
    st.subheader("Cumulative return (post-cost, starts at 1.0)")
    fig_cum = go.Figure()
    for col in cumulative.columns:
        fig_cum.add_trace(go.Scatter(x=cumulative.index, y=cumulative[col], mode="lines", name=col))
    st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Drawdown from running peak")
    drawdown = cumulative / cumulative.cummax() - 1.0
    fig_dd = go.Figure()
    for col in drawdown.columns:
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown[col], mode="lines", name=col))
    st.plotly_chart(fig_dd, use_container_width=True)

    if annual_returns is not None:
        st.subheader("Annual returns")
        st.dataframe(annual_returns, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primary: last 24 months")
        if allocations_primary is not None:
            st.dataframe(allocations_primary.tail(24), use_container_width=True)
    with col2:
        st.subheader("Secondary: last 24 months")
        if allocations_secondary is not None:
            st.dataframe(allocations_secondary.tail(24), use_container_width=True)

    if regime_analysis is not None:
        st.subheader("Primary vs. Secondary")
        agreement = regime_analysis[regime_analysis["analysis_type"] == "primary_secondary_agreement_rate"]
        if not agreement.empty:
            st.metric("tradable_regime agreement rate", f"{float(agreement['value'].iloc[0]):.1%}")
        with st.expander("Regime-average returns / crisis-year performance (full table)"):
            st.dataframe(regime_analysis, use_container_width=True)


def page_evaluation(eval_report: dict | None) -> None:
    st.title("Evaluation")
    if eval_report is None:
        _missing_data_notice("evaluation_report.json")
        return
    frame = pd.DataFrame(
        [
            {
                "model": ev["model_name"],
                "target": ev["target_name"],
                "horizon_months": ev["horizon_months"],
                "n_obs": ev["result"]["n_obs"],
                "accuracy": ev["result"]["accuracy"],
                "balanced_accuracy": ev["result"]["balanced_accuracy"],
                "up_recall": ev["result"]["up_recall"],
                "down_recall": ev["result"]["down_recall"],
                "down_precision": ev["result"]["down_precision"],
                "false_alarm_rate": ev["result"]["false_alarm_rate"],
                "always_up_baseline": ev["baseline_always_up"]["accuracy"],
                "always_down_baseline": ev["baseline_always_down"]["accuracy"],
                "avg_lead_months": ev["average_lead_months"],
            }
            for ev in eval_report["evaluations"]
        ]
    )
    st.dataframe(frame, use_container_width=True)
    st.caption(
        "Results are shown as-is, including cases where a model underperforms "
        "the naive always-Up / always-Down baseline. Thresholds are not tuned "
        "to hide weak performance."
    )


def page_methodology(config: dict) -> None:
    st.title("Methodology")
    methodology_path = ROOT / "docs" / "methodology.md"
    if methodology_path.exists():
        st.markdown(methodology_path.read_text(encoding="utf-8"))
    st.divider()
    st.subheader("Active configuration (config/default.yaml)")
    st.json(config)


def main() -> None:
    _bridge_streamlit_secrets_to_env()
    config = load_config()

    core_error = _ensure_core_data()
    if core_error:
        st.error(
            f"Couldn't build the initial data: {core_error}\n\n"
            "If this is a fresh deploy, add `FRED_API_KEY` under this app's "
            "Secrets settings (Streamlit Community Cloud) and reload."
        )

    wide = _load_csv(WIDE_PATH)
    signals = _load_csv(SIGNALS_PATH)
    regime_meta = _load_json(REGIME_METADATA_PATH)
    eval_report = _load_json(EVAL_PATH)
    series_meta = _load_json(SERIES_METADATA_PATH)
    regime_output_primary = _load_csv(REGIME_OUTPUT_PRIMARY_PATH)
    regime_output_secondary = _load_csv(REGIME_OUTPUT_SECONDARY_PATH)

    st.sidebar.title("macro-regime-taa")
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Data Explorer",
            "Growth Models",
            "Inflation Models",
            "Regime Comparison",
            "Regime Output",
            "Backtest",
            "Evaluation",
            "Methodology",
        ],
    )

    if page == "Overview":
        page_overview(signals, wide, regime_meta)
    elif page == "Data Explorer":
        page_data_explorer(wide, series_meta)
    elif page == "Growth Models":
        page_growth_models(signals)
    elif page == "Inflation Models":
        page_inflation_models(signals)
    elif page == "Regime Comparison":
        page_regime_comparison(signals, regime_meta)
    elif page == "Regime Output":
        page_regime_output(regime_output_primary, regime_output_secondary)
    elif page == "Backtest":
        _ensure_backtest_data()
        page_backtest(
            _load_csv_plain(BACKTEST_SUMMARY_PATH),
            _load_csv(BACKTEST_ANNUAL_RETURNS_PATH),
            _load_csv(BACKTEST_MONTHLY_RETURNS_PATH),
            _load_csv(BACKTEST_ALLOCATIONS_PRIMARY_PATH),
            _load_csv(BACKTEST_ALLOCATIONS_SECONDARY_PATH),
            _load_csv_plain(BACKTEST_REGIME_ANALYSIS_PATH),
        )
    elif page == "Evaluation":
        page_evaluation(eval_report)
    elif page == "Methodology":
        page_methodology(config)


if __name__ == "__main__":
    main()
