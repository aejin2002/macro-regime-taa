"""Macro Regime TAA -- strategy pitch + live operating dashboard.

Every number rendered here is read from the canonical parquet artifacts
built by `python -m macro_regime.cli update-all`
(`data/processed/production_v13_daily.parquet`,
`v1_2_regression_daily.parquet`, `benchmarks_daily.parquet`) -- this file
computes no strategy output, runs no backtest, and never calls
`AssetPriceClient`/`FredClient` directly. A Streamlit rerun (tab switch,
widget change) only re-reads already-computed data; `st.cache_data` is
keyed on each artifact's own mtime (see `ui/data_loader.py`) so a fresh
pipeline run is picked up automatically without a manual cache-clear.

Five tabs: Overview, Markets, Performance, Signals, Methodology. Overview
leads with current position/state (regime, risk overlays, allocation,
Growth/Inflation evidence) so a user sees "where the strategy stands
right now" before any historical performance chart -- the former
separate "Current Positioning" tab was folded into it; nothing that tab
rendered was dropped, only its own tab slot. Project 60/40
(`macro_regime.benchmarks.REGISTRY["project_6040"]`) is intentionally
never rendered on any of them -- it stays registered only for internal/
research reproducibility (`.analysis/*`, v1.0-v1.2 regression). US 60/40
is the only default-selected comparison; MALOX/SPY/AGG are opt-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from ui import charts, components  # noqa: E402
from ui.data_loader import (  # noqa: E402
    benchmarks_artifact_source_info,
    data_freshness_summary,
    get_dashboard_data,
    load_pipeline_status,
    load_v1_2_regression_daily,
    refresh_live_data_now,
    v1_3_artifact_source_info,
)
from ui.formatters import fmt_date, fmt_int, fmt_num, fmt_pct  # noqa: E402
from ui.theme import COLORS, inject_css  # noqa: E402

from macro_regime.backtest.metrics import average_annual_turnover, max_drawdown  # noqa: E402
from macro_regime.benchmarks import MAX_CONCURRENT_BENCHMARKS, list_ui_visible  # noqa: E402
from macro_regime.benchmarks import REGISTRY as BENCHMARK_REGISTRY  # noqa: E402
from macro_regime.config import load_config  # noqa: E402
from macro_regime.fast_crisis.metrics import (  # noqa: E402
    annualized_volatility_daily,
    cagr_daily,
    calmar_ratio_daily,
    month_end_value,
    sharpe_ratio_daily,
    sortino_ratio_daily,
)
from macro_regime.strategy_versions import DISPLAY_NAMES  # noqa: E402

st.set_page_config(page_title="Macro Regime TAA", page_icon="🩵", layout="wide")
inject_css()

ASSET_COLUMNS = [
    "spy",
    "kodex200_usd",
    "high_yield",
    "investment_grade",
    "intermediate_treasury",
    "long_treasury",
    "gold",
    "tbills",
    "commodities",
    "tips",
]
ASSET_DISPLAY = {
    "spy": "SPY",
    "kodex200_usd": "KODEX 200 (USD)",
    "high_yield": "HYG (High Yield)",
    "investment_grade": "LQD (IG Credit)",
    "intermediate_treasury": "IEF (7-10Y Treasury)",
    "long_treasury": "TLT (20+Y Treasury)",
    "gold": "GLD (Gold)",
    "tbills": "BIL (T-Bills)",
    "commodities": "DBC (Commodities)",
    "tips": "TIP (TIPS)",
}
MOVE_BY_ASSET_UNIVERSE = list(ASSET_COLUMNS)  # + agg, malox, vix handled separately (different data shape)
RANGE_OPTIONS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "ALL": None}
SIGNAL_RANGE_OPTIONS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "ALL": None}


# -- data bootstrap (read-only; never fetches, never backtests) -------------


@st.cache_data(show_spinner=False)
def _cached_config() -> dict:
    return load_config()


def _load_all():
    data = get_dashboard_data()
    v12 = load_v1_2_regression_daily()
    status = load_pipeline_status()
    freshness = data_freshness_summary(data["v1_3_df"], data["bench_df"], status)
    return data, v12, freshness, status


dashboard_data, v12_raw, freshness, pipeline_status = _load_all()
v13_raw = dashboard_data["v1_3_df"]
bench_raw = dashboard_data["bench_df"]
live_vix_series = dashboard_data["vix_series"]

if v13_raw is None or v13_raw.empty:
    st.error(
        "Could not load v1.3 production data -- the live production pipeline and the "
        f"`v1.3.0` GitHub Release fallback both failed.\n\nLast error: {dashboard_data.get('error')}\n\n"
        "This dashboard never shows fabricated data -- check FRED/Yahoo Finance connectivity, "
        "the `FRED_API_KEY` secret, or that the GitHub Release asset is reachable."
    )
    st.stop()

v13 = v13_raw.set_index("date").sort_index()
v13.index = pd.to_datetime(v13.index)


def bench_frame(benchmark_id: str) -> pd.DataFrame | None:
    """Returns a date-indexed frame with at least `daily_return`/`nav`
    for one benchmark, or None if that benchmark's data is unavailable
    (never silently substitutes a different one)."""
    if bench_raw is None:
        return None
    rows = bench_raw[bench_raw["benchmark_id"] == benchmark_id]
    if rows.empty or not bool(rows["available"].iloc[0]):
        return None
    out = rows.set_index("date").sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def bench_available(benchmark_id: str) -> bool:
    return bench_frame(benchmark_id) is not None


RISK_FREE = v13["asset_return_tbills"]
STRATEGY_RETURNS = v13["daily_return_net"]
STRATEGY_NAV = v13["strategy_nav"]


# -- shared metric computation (reused by Overview/Performance) -------------


def compute_metrics(returns: pd.Series, risk_free: pd.Series, *, turnover: pd.Series | None = None) -> dict:
    if returns is None or returns.empty:
        return {}
    value = (1 + returns).cumprod()
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    row = {
        "cagr": cagr_daily(value),
        "annualized_vol": annualized_volatility_daily(returns),
        "sharpe": sharpe_ratio_daily(returns, risk_free.reindex(returns.index)),
        "sortino": sortino_ratio_daily(returns, risk_free.reindex(returns.index)),
        "max_drawdown_daily": max_drawdown(value),
        "max_drawdown_month_end": max_drawdown(month_end_value(value)),
        "calmar": calmar_ratio_daily(value),
        "final_wealth": float(value.iloc[-1]),
        "positive_month_ratio": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "best_month": float(monthly.max()) if len(monthly) else float("nan"),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
    }
    if turnover is not None:
        row["avg_annual_turnover"] = average_annual_turnover(turnover)
    return row


def capture_ratios(returns: pd.Series, spy_returns: pd.Series) -> tuple[float, float]:
    monthly_port = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    monthly_spy = (1 + spy_returns).groupby(spy_returns.index.to_period("M")).prod() - 1
    aligned = pd.concat([monthly_port, monthly_spy], axis=1, join="inner")
    aligned.columns = ["port", "spy"]
    up, down = aligned[aligned["spy"] > 0], aligned[aligned["spy"] < 0]
    up_spy = float((1 + up["spy"]).prod() - 1) if len(up) else 0.0
    up_port = float((1 + up["port"]).prod() - 1) if len(up) else 0.0
    down_spy = float((1 + down["spy"]).prod() - 1) if len(down) else 0.0
    down_port = float((1 + down["port"]).prod() - 1) if len(down) else 0.0
    upside = up_port / up_spy if (len(up) and up_spy != 0) else float("nan")
    downside = down_port / down_spy if (len(down) and down_spy != 0) else float("nan")
    return upside, downside


# =============================================================================
# Sidebar -- shared "as of" summary, always visible
# =============================================================================

DATA_MODE_LABELS = {
    "live": ("🟢 Live", "Freshly fetched from FRED/Yahoo Finance and recomputed this refresh cycle."),
    "session_fallback": (
        "🟡 Cached live (stale)",
        "The latest live refresh attempt failed -- showing the last successful live result "
        "from this session instead of jumping straight to the frozen Release artifact.",
    ),
    "release_fallback": (
        "🔵 Release fallback",
        "Live refresh failed with no prior live result available -- showing the immutable "
        "`v1.3.0` GitHub Release artifact.",
    ),
}

with st.sidebar:
    st.markdown("### Macro Regime TAA v1.3")
    st.caption(DISPLAY_NAMES.get("v1_3", "v1.3"))

    mode = dashboard_data["mode"]
    mode_label, mode_note = DATA_MODE_LABELS.get(mode, ("Unknown", ""))
    st.markdown(f"**Production data mode**  \n{mode_label}")
    st.caption(mode_note)
    if dashboard_data.get("fetched_at") is not None:
        fetched_label = dashboard_data["fetched_at"].strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**Last successful live refresh**  \n{fetched_label}")
    if dashboard_data.get("error"):
        with st.expander("Refresh error detail"):
            st.code(dashboard_data["error"])

    if st.button("🔄 Refresh latest data", width="stretch"):
        refresh_live_data_now()
        st.rerun()

    st.markdown(
        f"**Dashboard generated**  \n{freshness['dashboard_generated_at'].strftime('%Y-%m-%d %H:%M')}"
    )
    st.markdown(f"**Strategy market data as of**  \n{fmt_date(freshness['strategy_market_data_as_of'])}")
    st.markdown(f"**US 60/40 as of**  \n{fmt_date(freshness['us_6040_as_of'])}")
    malox_label = fmt_date(freshness["malox_as_of"])
    if freshness.get("malox_is_stale"):
        malox_label += "  ⚠️ stale"
    st.markdown(f"**MALOX NAV as of**  \n{malox_label}")
    st.markdown(f"**Macro data available as of**  \n{fmt_date(freshness['macro_data_available_as_of'])}")
    st.markdown(
        f"**Current macro allocation effective from**  \n"
        f"{fmt_date(freshness['current_macro_allocation_effective_from'])}"
    )

    if mode == "release_fallback":
        if pipeline_status:
            st.caption(f"Pipeline last completed: {pipeline_status.get('completed_at', 'n/a')[:19]}")

        artifact_info = v1_3_artifact_source_info()
        if artifact_info:
            source_label = (
                "local build"
                if artifact_info["source"] == "local"
                else f"GitHub Release {artifact_info['tag']}"
            )
            st.caption(f"v1.3 artifact source: {source_label}  \nSHA256: `{artifact_info['sha256'][:12]}...`")

        bench_artifact_info = benchmarks_artifact_source_info()
        if bench_artifact_info:
            bench_source_label = (
                "local build"
                if bench_artifact_info["source"] == "local"
                else f"GitHub Release {bench_artifact_info['tag']}"
            )
            st.caption(
                f"Benchmarks artifact source: {bench_source_label}  \n"
                f"SHA256: `{bench_artifact_info['sha256'][:12]}...`"
            )

        # Stale-cache warning: AssetPriceClient's on-disk cache has no TTL
        # of its own, so this checks its age explicitly rather than
        # trusting it's fresh just because a file exists. Only meaningful
        # in fallback mode -- live mode always fetches with
        # refresh_cache=True, bypassing this cache entirely.
        try:
            from macro_regime.data.asset_prices import AssetPriceClient as _APC

            _cache_status = _APC().get_cache_status("SPY", "2008-01-01", ttl_seconds=24 * 3600)
        except Exception:  # noqa: BLE001
            _cache_status = None
        if _cache_status and _cache_status.get("is_stale"):
            age_h = (_cache_status["age_seconds"] or 0) / 3600
            st.warning(
                f"⚠️ Asset-price cache last refreshed {age_h:.1f}h ago (>24h) -- "
                f"run `update-all` to refresh before trusting the latest date."
            )

tab_overview, tab_markets, tab_performance, tab_signals, tab_methodology = st.tabs(
    ["Overview", "Markets", "Performance", "Signals", "Methodology"]
)


# =============================================================================
# Tab 1 -- Overview (pitch + at-a-glance operating state)
# =============================================================================

with tab_overview:
    latest = v13.iloc[-1]
    components.render_hero(
        "Macro Regime TAA v1.3",
        "Adaptive allocation. Explicit risk control.",
        {
            "As of": fmt_date(v13.index.max()),
            "Regime": str(latest["macro_regime"]),
            "Fast Crisis": str(latest["fast_crisis_tradable"]),
        },
    )
    st.markdown(
        "성장과 물가 환경에 따라 기본 자산배분을 전환하고, "
        "금리 위험과 금융시장 위기를 별도 레이어로 관리합니다."
    )
    if bool(latest["partial_month"]):
        st.caption(
            f"Partial month through {fmt_date(v13.index.max())} -- current month's macro target "
            "holds last month's decision forward."
        )

    # -- Current Positioning (folded in from the former separate tab) ------
    # Every value below is read from `latest`/`freshness`/`config`, already
    # computed by the pipeline -- nothing here recomputes a signal,
    # allocation, or performance figure.
    st.markdown("#### Current Risk State")
    col1, col2, col3 = st.columns(3)
    with col1:
        components.render_status_card(
            "Macro Regime",
            str(latest["macro_regime"]),
            None,
            f"Effective since {fmt_date(freshness['current_macro_allocation_effective_from'])}",
            components.regime_tone_for(str(latest["macro_regime"])),
            compact=True,
        )
    with col2:
        tone = "risk" if str(latest["fast_crisis_tradable"]) == "ON" else "good"
        components.render_status_card(
            "Fast Crisis",
            str(latest["fast_crisis_tradable"]),
            str(latest["fast_crisis_tradable"]),
            f"Votes: {fmt_int(latest['fast_crisis_votes'])}/3",
            tone,
            compact=True,
        )
    with col3:
        tone = "risk" if str(latest["bei_gate_tradable"]) == "ON" else "good"
        components.render_status_card(
            "BEI Duration Gate",
            str(latest["bei_gate_tradable"]),
            str(latest["bei_gate_tradable"]),
            "Only tilts duration sleeve in Contraction",
            tone,
            compact=True,
        )

    st.markdown("#### Last Rebalance / Next Expected Macro Update")
    rebalance_days = v13[v13["rebalance_event"]]
    last_rebalance = rebalance_days.index.max() if len(rebalance_days) else None
    # Anchored to the CURRENT OPEN month (the latest artifact date), not the
    # already-effective closed month -- `effective_date + 1 month` is the
    # rebalance that already happened (== last_rebalance); the next one is
    # one month further out, when the current open month itself closes.
    current_open_month_start = v13.index.max().replace(day=1)
    next_expected_macro_update = (
        (current_open_month_start + pd.DateOffset(months=1)).replace(day=1)
        if freshness["current_macro_allocation_effective_from"] is not None
        else None
    )
    col_r1, col_r2 = st.columns(2)
    col_r1.metric("Last rebalance event", fmt_date(last_rebalance))
    col_r2.metric("Next expected macro update", fmt_date(next_expected_macro_update))

    st.markdown("#### Current Partial-Month Performance")
    if bool(latest["partial_month"]):
        partial = v13.loc[
            v13.index >= v13.index[v13.index.to_period("M") == v13.index[-1].to_period("M")].min(),
            "daily_return_net",
        ]
        partial_cum = float((1 + partial).prod() - 1)
        st.metric(f"Partial month through {fmt_date(v13.index.max())}", fmt_pct(partial_cum, signed=True))
    else:
        st.caption("Current month is complete; no partial-month figure to show.")

    st.markdown("#### Growth Signal Evidence")
    config = _cached_config()
    g_conf = config["growth_models"]["model_a_cli"]
    lag_months = config["regime_output"]["tradable_lag_months"]
    effective_date = freshness["current_macro_allocation_effective_from"]
    observed_date = v13.index.max()
    components.render_evidence_card(
        title=f"Why Growth is {latest['growth_state']} (currently effective)",
        effective_value=str(latest["growth_state"]),
        effective_since=effective_date,
        observed_value=str(latest["growth_state_observed"]),
        observed_as_of=observed_date,
        expected_effective=next_expected_macro_update,
        lag_months=lag_months,
        score_label=f"Score {fmt_num(latest['growth_score'], 4)} -- {g_conf['us_series']} (OECD CLI, US)",
    )

    st.markdown("#### Inflation Signal Evidence")
    i_conf = config["inflation_models"]["model_a_realized_core"]
    components.render_evidence_card(
        title=f"Why Inflation is {latest['inflation_state']} (currently effective)",
        effective_value=str(latest["inflation_state"]),
        effective_since=effective_date,
        observed_value=str(latest["inflation_state_observed"]),
        observed_as_of=observed_date,
        expected_effective=next_expected_macro_update,
        lag_months=lag_months,
        score_label=f"Score {fmt_pct(latest['inflation_score'])} -- {i_conf['core_series']} (Core CPI, US)",
    )

    st.markdown("#### Current Allocation")
    target_weights = {ASSET_DISPLAY[c]: latest[f"target_weight_{c}"] for c in ASSET_COLUMNS}
    drifted_weights_ov = {ASSET_DISPLAY[c]: latest[f"drifted_weight_{c}"] for c in ASSET_COLUMNS}
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            charts.allocation_donut(target_weights, "Target Allocation"), width="stretch", key="ov_donut"
        )
    with col_b:
        st.plotly_chart(
            charts.allocation_bar(drifted_weights_ov, "Drifted Allocation (actual, bar)"),
            width="stretch", key="ov_drifted_bar",
        )
    st.caption(
        "Target = the last rebalance decision. Drifted = actual current weights after price "
        "movement since then."
    )

    st.markdown("#### Latest Contribution by Asset")
    contrib = {ASSET_DISPLAY[c]: latest[f"asset_contribution_{c}"] for c in ASSET_COLUMNS}
    st.plotly_chart(
        charts.allocation_bar(contrib, "Latest-day contribution by asset"), width="stretch", key="pos_contrib"
    )

    # -- Rest of the existing Overview detail --------------------------------
    m_strategy = compute_metrics(STRATEGY_RETURNS, RISK_FREE, turnover=v13["daily_turnover"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", fmt_pct(m_strategy.get("cagr")), border=True)
    c2.metric("Sharpe", fmt_num(m_strategy.get("sharpe")), border=True)
    c3.metric("Daily MDD", fmt_pct(m_strategy.get("max_drawdown_daily")), border=True)
    c4.metric("Final Wealth", f"{m_strategy.get('final_wealth', float('nan')):.2f}x", border=True)

    st.markdown("#### Move by Asset (preview)")
    preview_cols = st.columns(5)
    for i, asset in enumerate(ASSET_COLUMNS[:5]):
        ret_1d = v13[f"asset_return_{asset}"].iloc[-1]
        with preview_cols[i]:
            st.metric(ASSET_DISPLAY[asset], fmt_pct(ret_1d, signed=True), border=True)
    st.caption("See the Markets tab for the full grid with 1M/3M/6M/1Y/2Y/ALL history.")

    st.markdown("#### Recent Signal Events")
    recent_events = v13[v13["signal_event"]].tail(5)[
        ["macro_regime", "fast_crisis_tradable", "bei_gate_tradable"]
    ]
    if len(recent_events):
        st.dataframe(recent_events, width="stretch")
    else:
        st.caption("No signal events in range.")

    if str(latest["fast_crisis_tradable"]) == "ON":
        st.info(
            "Fast Crisis Overlay is currently **ON** -- growth/high-yield/commodity exposure "
            "is de-risked into T-Bills."
        )
    else:
        st.success("Fast Crisis Overlay is currently OFF -- no crisis de-risking active.")


# =============================================================================
# Tab 2 -- Markets (Move by Asset)
# =============================================================================


# AGG/MALOX come from the deployment-safe `benchmarks_daily.parquet`
# (local-first, GitHub Release fallback -- see `bench_frame`). VIX exists
# in neither `benchmarks_daily.parquet` nor `production_v13_daily.parquet`
# -- the only two artifacts this dashboard reads -- so it is deliberately
# NOT wired to any raw CSV or live API here; selecting it shows a clear
# "unavailable" message instead of fabricated data (see the final report
# for the minimal artifact change that would add it).
MARKET_SERIES_LABELS = {
    "AGG — US Aggregate Bond ETF": "agg",
    "MALOX — Allocation Fund NAV": "malox",
    "VIX — CBOE Volatility Index": "vix",
}
MARKET_SERIES_UNIT = {"agg": "NAV index", "malox": "NAV index", "vix": "index level"}


def _period_return(series: pd.Series, n_days: int | None) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    window = s if n_days is None else s.iloc[-min(n_days + 1, len(s)) :]
    if len(window) < 2:
        return None
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


with tab_markets:
    st.markdown("#### Move by Asset")
    range_label = st.radio("Range", list(RANGE_OPTIONS.keys()), index=1, horizontal=True, key="markets_range")
    n_days = RANGE_OPTIONS[range_label]

    close_proxies = {}  # reconstructed price-level proxy per asset (NAV from returns), for mini-charts
    for c in ASSET_COLUMNS:
        close_proxies[c] = (1 + v13[f"asset_return_{c}"]).cumprod()

    grid_cols = st.columns(4)
    for i, asset in enumerate(ASSET_COLUMNS):
        ret_series = v13[f"asset_return_{asset}"]
        proxy = close_proxies[asset]
        with grid_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{ASSET_DISPLAY[asset]}**")
                st.caption(f"as of {fmt_date(v13.index.max())}")
                st.markdown(
                    f"1D: {fmt_pct(ret_series.iloc[-1], signed=True)}  ·  "
                    f"5D: {fmt_pct(_period_return(proxy, 5), signed=True)}"
                )
                st.markdown(
                    f"1M: {fmt_pct(_period_return(proxy, 21), signed=True)}  ·  "
                    f"3M: {fmt_pct(_period_return(proxy, 63), signed=True)}"
                )
                ma20 = proxy.rolling(20).mean().iloc[-1]
                ma200 = proxy.rolling(200).mean().iloc[-1] if len(proxy) >= 200 else None
                dist20 = (proxy.iloc[-1] / ma20 - 1.0) if ma20 else None
                dist200 = (proxy.iloc[-1] / ma200 - 1.0) if ma200 else None
                st.caption(
                    f"vs 20D MA: {fmt_pct(dist20, signed=True)}  ·  "
                    f"vs 200D MA: {fmt_pct(dist200, signed=True)}"
                )
                window = proxy if n_days is None else proxy.iloc[-n_days:]
                mini = go.Figure(
                    go.Scatter(
                        x=window.index,
                        y=window.values,
                        mode="lines",
                        line=dict(color=COLORS["accent"], width=1.5),
                    )
                )
                mini.update_layout(
                    height=90,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    paper_bgcolor=COLORS["card_bg"],
                    plot_bgcolor=COLORS["card_bg"],
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(mini, width="stretch", key=f"mini_{asset}")

    st.markdown("#### Benchmark & Volatility Charts")
    selected_market_series = st.multiselect(
        "Series",
        list(MARKET_SERIES_LABELS.keys()),
        default=["AGG — US Aggregate Bond ETF", "MALOX — Allocation Fund NAV"],
        key="markets_bench_series",
    )
    if not selected_market_series:
        st.caption("Select a series above to see its chart.")
    # One independent figure per selected series -- never overlaid, never
    # normalized against each other (different units: NAV index vs. a
    # volatility level).
    for label in selected_market_series:
        series_key = MARKET_SERIES_LABELS[label]
        unit_label = MARKET_SERIES_UNIT[series_key]
        with st.container(border=True):
            if series_key == "vix":
                if live_vix_series is None or live_vix_series.empty:
                    st.markdown(f"**{label}**")
                    st.caption(
                        "Data unavailable -- this dashboard is currently showing the Release "
                        "fallback (not a live refresh), and VIX is not included in the frozen "
                        "production/benchmarks artifact. No raw CSV or non-FRED live fallback "
                        "is used here; VIX resolves automatically once a live refresh succeeds "
                        "(it is fetched from FRED as part of the normal macro data pull)."
                    )
                    continue
                st.caption(f"as of {fmt_date(live_vix_series.index.max())}  ·  {unit_label}")
                vix_window = live_vix_series if n_days is None else live_vix_series.iloc[-n_days:]
                st.plotly_chart(
                    charts.single_series_chart(vix_window, f"{label} -- {unit_label}"),
                    width="stretch",
                    key=f"mkt_series_{series_key}",
                )
                continue
            frame = bench_frame(series_key)
            if frame is None:
                st.markdown(f"**{label}**")
                st.caption("unavailable")
                continue
            stale = (
                bool(frame["is_stale"].iloc[-1])
                if series_key == "malox" and "is_stale" in frame.columns
                else False
            )
            as_of_label = fmt_date(frame.index.max()) + (" ⚠️ stale" if stale else "")
            st.caption(f"as of {as_of_label}  ·  {unit_label}")
            window = frame if n_days is None else frame.iloc[-n_days:]
            st.plotly_chart(
                charts.single_series_chart(window["nav"], f"{label} -- {unit_label}"),
                width="stretch",
                key=f"mkt_series_{series_key}",
            )

    st.markdown("#### Normalized Performance (selected assets, rebased to 100)")
    selected_assets = st.multiselect(
        "Assets",
        [ASSET_DISPLAY[c] for c in ASSET_COLUMNS],
        default=[ASSET_DISPLAY[c] for c in ["spy", "gold", "tbills"]],
        key="markets_normalized",
    )
    mode = st.radio(
        "Mode", ["Normalized (rebased to 100)", "Absolute (NAV proxy)"], horizontal=True, key="markets_mode"
    )
    if selected_assets:
        reverse_display = {v: k for k, v in ASSET_DISPLAY.items()}
        cols = {}
        for disp in selected_assets:
            key = reverse_display[disp]
            series = close_proxies[key]
            window = series if n_days is None else series.iloc[-n_days:]
            cols[disp] = (window / window.iloc[0] * 100) if mode.startswith("Normalized") else window
        norm_df = pd.DataFrame(cols)
        st.plotly_chart(
            charts.nav_chart(norm_df, "Selected Assets"), width="stretch", key="markets_norm_chart"
        )

    st.markdown("#### Return Matrix")
    matrix_rows = []
    for asset in ASSET_COLUMNS:
        proxy = close_proxies[asset]
        matrix_rows.append(
            {
                "Asset": ASSET_DISPLAY[asset],
                "1D": v13[f"asset_return_{asset}"].iloc[-1],
                "1M": _period_return(proxy, 21),
                "3M": _period_return(proxy, 63),
                "6M": _period_return(proxy, 126),
                "1Y": _period_return(proxy, 252),
            }
        )
    matrix_df = pd.DataFrame(matrix_rows).set_index("Asset")
    st.dataframe(matrix_df.style.format("{:.2%}"), width="stretch")


# =============================================================================
# Tab 3 -- Performance
# =============================================================================


def _relative_drawdown_stats(strategy_nav: pd.Series, bench_nav: pd.Series) -> dict:
    common = strategy_nav.index.intersection(bench_nav.index)
    if len(common) < 2:
        return {}
    rel = strategy_nav.reindex(common) / bench_nav.reindex(common)
    running_max = rel.cummax()
    dd = rel / running_max - 1.0
    trough = dd.idxmin()
    at_peak = rel.loc[:trough] == running_max.loc[:trough]
    peak = rel.loc[:trough][at_peak].index[-1] if at_peak.any() else common[0]
    peak_value = rel.loc[peak]
    after = rel.loc[trough:]
    recovered = after[after >= peak_value]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "current_relative_nav": float(rel.iloc[-1]),
        "max_relative_drawdown": float(dd.min()),
        "current_relative_drawdown": float(dd.iloc[-1]),
        "relative_peak_date": peak,
        "relative_trough_date": trough,
        "recovery_date": recovery,
        "ongoing": recovery is None,
    }


with tab_performance:
    st.markdown("#### Benchmark Selection")
    perf_cols = st.columns(4)
    selections = {}
    with perf_cols[0]:
        selections["us_60_40"] = st.checkbox("US 60/40", value=True, key="perf_us6040")
    with perf_cols[1]:
        selections["malox"] = st.checkbox(
            "MALOX", value=False, disabled=not bench_available("malox"), key="perf_malox"
        )
    with perf_cols[2]:
        selections["spy"] = st.checkbox(
            "SPY", value=False, disabled=not bench_available("spy"), key="perf_spy"
        )
    with perf_cols[3]:
        selections["agg"] = st.checkbox(
            "AGG", value=False, disabled=not bench_available("agg"), key="perf_agg"
        )

    active = [k for k, v in selections.items() if v]
    if len(active) > MAX_CONCURRENT_BENCHMARKS:
        st.warning(
            f"At most {MAX_CONCURRENT_BENCHMARKS} benchmarks can be shown at once -- "
            f"showing the first {MAX_CONCURRENT_BENCHMARKS}."
        )
        active = active[:MAX_CONCURRENT_BENCHMARKS]

    bench_series_map = {bid: bench_frame(bid) for bid in active}
    bench_series_map = {k: v for k, v in bench_series_map.items() if v is not None}

    nav_cols = {"Strategy v1.3": STRATEGY_NAV}
    common_start_note = None
    for bid, frame in bench_series_map.items():
        label = BENCHMARK_REGISTRY[bid].display_name
        nav_cols[label] = frame["nav"].reindex(STRATEGY_NAV.index)
        if bid == "malox" and frame.index.min() > STRATEGY_NAV.index.min():
            common_start_note = frame.index.min()
    title = "Cumulative NAV"
    if common_start_note is not None:
        title += (
            f" (MALOX only from {fmt_date(common_start_note)} -- full-period "
            "Strategy/US60-40 figures are NOT restricted to this window)"
        )
    st.plotly_chart(
        charts.nav_chart(pd.DataFrame(nav_cols), title, initial_days=365 * 5), width="stretch", key="perf_nav"
    )

    dd_cols = {"Strategy v1.3": STRATEGY_NAV / STRATEGY_NAV.cummax() - 1.0}
    for bid, frame in bench_series_map.items():
        label = BENCHMARK_REGISTRY[bid].display_name
        nav = frame["nav"].reindex(STRATEGY_NAV.index)
        dd_cols[label] = nav / nav.cummax() - 1.0
    st.plotly_chart(
        charts.drawdown_chart(pd.DataFrame(dd_cols), initial_days=365 * 5), width="stretch", key="perf_dd"
    )

    st.markdown("#### Relative Performance")
    for bid, frame in bench_series_map.items():
        label = BENCHMARK_REGISTRY[bid].display_name
        stats = _relative_drawdown_stats(STRATEGY_NAV, frame["nav"])
        if not stats:
            continue
        st.markdown(f"**Strategy v1.3 / {label}**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Current relative NAV", fmt_num(stats["current_relative_nav"]))
        rc2.metric("Max relative drawdown", fmt_pct(stats["max_relative_drawdown"]))
        rc3.metric("Current relative drawdown", fmt_pct(stats["current_relative_drawdown"]))
        rc4.metric(
            "Status", "Ongoing" if stats["ongoing"] else f"Recovered {fmt_date(stats['recovery_date'])}"
        )
        st.caption(
            f"Relative peak: {fmt_date(stats['relative_peak_date'])}  ·  "
            f"Relative trough: {fmt_date(stats['relative_trough_date'])}"
        )

    st.markdown("#### Annual Returns")
    annual_cols = {"Strategy v1.3": (1 + STRATEGY_RETURNS).groupby(STRATEGY_RETURNS.index.year).prod() - 1}
    for bid, frame in bench_series_map.items():
        label = BENCHMARK_REGISTRY[bid].display_name
        r = frame["daily_return"]
        annual_cols[label] = (1 + r).groupby(r.index.year).prod() - 1
    annual_df = pd.DataFrame(annual_cols)
    st.plotly_chart(charts.annual_returns_chart(annual_df), width="stretch", key="perf_annual")

    st.markdown("#### Rolling 12-Month Return")
    rolling_cols = {
        "Strategy v1.3": (1 + STRATEGY_RETURNS).rolling(252).apply(lambda x: x.prod() - 1, raw=True)
    }
    for bid, frame in bench_series_map.items():
        label = BENCHMARK_REGISTRY[bid].display_name
        r = frame["daily_return"].reindex(STRATEGY_RETURNS.index)
        rolling_cols[label] = (1 + r).rolling(252).apply(lambda x: x.prod() - 1, raw=True)
    rolling_df = pd.DataFrame(rolling_cols).dropna(how="all")
    if len(rolling_df):
        rolling_fig = charts.nav_chart(rolling_df, "Rolling 12-Month Return", initial_days=365 * 5)
        rolling_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(rolling_fig, width="stretch", key="perf_rolling")
    st.caption("Rolling 252-trading-day (~12-month) compounded return, trailing window.")

    st.markdown("#### Benchmark Metric Table")
    rows = []
    m = compute_metrics(STRATEGY_RETURNS, RISK_FREE, turnover=v13["daily_turnover"])
    upside, downside = capture_ratios(STRATEGY_RETURNS, v13["asset_return_spy"])
    rows.append(
        {
            "Strategy/Benchmark": "Strategy v1.3",
            "CAGR": m.get("cagr"),
            "Sharpe": m.get("sharpe"),
            "Sortino": m.get("sortino"),
            "Daily MDD": m.get("max_drawdown_daily"),
            "Month-end MDD": m.get("max_drawdown_month_end"),
            "Calmar": m.get("calmar"),
            "Final Wealth": m.get("final_wealth"),
            "Upside Capture": upside,
            "Downside Capture": downside,
            "Common Start": str(STRATEGY_NAV.index.min().date()),
            "Common End": str(STRATEGY_NAV.index.max().date()),
            "Observations": len(STRATEGY_RETURNS),
        }
    )
    for bid, frame in bench_series_map.items():
        label = BENCHMARK_REGISTRY[bid].display_name
        r = frame["daily_return"]
        mb = compute_metrics(r, RISK_FREE)
        up_b, down_b = capture_ratios(r, v13["asset_return_spy"].reindex(r.index))
        rows.append(
            {
                "Strategy/Benchmark": label,
                "CAGR": mb.get("cagr"),
                "Sharpe": mb.get("sharpe"),
                "Sortino": mb.get("sortino"),
                "Daily MDD": mb.get("max_drawdown_daily"),
                "Month-end MDD": mb.get("max_drawdown_month_end"),
                "Calmar": mb.get("calmar"),
                "Final Wealth": mb.get("final_wealth"),
                "Upside Capture": up_b,
                "Downside Capture": down_b,
                "Common Start": str(r.index.min().date()),
                "Common End": str(r.index.max().date()),
                "Observations": len(r),
            }
        )
    metric_table = pd.DataFrame(rows).set_index("Strategy/Benchmark")
    st.dataframe(metric_table, width="stretch")
    st.caption(
        "Rows use each strategy/benchmark's OWN observation window -- Strategy's inception-period "
        "figures are never mixed with a shorter-history benchmark's common-period figures."
    )


# =============================================================================
# Tab 4 -- Signals
# =============================================================================

EVENT_TYPES = [
    "Macro regime change",
    "Growth state change",
    "Inflation state change",
    "BEI Gate ON",
    "BEI Gate OFF",
    "Fast Crisis entry",
    "Fast Crisis extension",
    "Fast Crisis exit",
    "Scheduled rebalance",
    "Data freshness warning",
]


def _classify_event(row: pd.Series, prev_row: pd.Series | None) -> str:
    if prev_row is None:
        return "Scheduled rebalance"
    if row["macro_regime"] != prev_row["macro_regime"]:
        return "Macro regime change"
    if row["growth_state"] != prev_row["growth_state"]:
        return "Growth state change"
    if row["inflation_state"] != prev_row["inflation_state"]:
        return "Inflation state change"
    if row["bei_gate_tradable"] != prev_row["bei_gate_tradable"]:
        return "BEI Gate ON" if row["bei_gate_tradable"] == "ON" else "BEI Gate OFF"
    if row["fast_crisis_tradable"] != prev_row["fast_crisis_tradable"]:
        return "Fast Crisis entry" if row["fast_crisis_tradable"] == "ON" else "Fast Crisis exit"
    if bool(row["rebalance_event"]):
        return "Scheduled rebalance"
    return "Fast Crisis extension" if row["fast_crisis_tradable"] == "ON" else "Scheduled rebalance"


with tab_signals:
    st.markdown("#### Growth Signal Evidence")
    config = _cached_config()
    growth_row = v13.iloc[-1]
    g_conf = config["growth_models"]["model_a_cli"]
    lag_months = config["regime_output"]["tradable_lag_months"]
    effective_date = freshness["current_macro_allocation_effective_from"]
    observed_date = v13.index.max()
    with st.container(border=True):
        st.markdown(f"**Why Growth is {growth_row['growth_state']}** (currently effective)")
        st.markdown(
            f"- Model: `{g_conf['us_series']}` (OECD Composite Leading Indicator, US)\n"
            f"- Effective signal (used for the CURRENT regime, lagged {lag_months} month(s)): "
            f"**{growth_row['growth_state']}** as of {fmt_date(effective_date)} "
            f"(score {fmt_num(growth_row['growth_score'], 4)})\n"
            f"- Observed signal (most recent reading, not yet effective): "
            f"**{growth_row['growth_state_observed']}** as of {fmt_date(observed_date)}"
        )
        st.caption(
            "Observed becomes effective next month under the "
            f"{lag_months}-month lag. Values above are read directly from the production "
            "artifact -- this dashboard never recomputes a signal."
        )

    st.markdown("#### Inflation Signal Evidence")
    i_conf = config["inflation_models"]["model_a_realized_core"]
    with st.container(border=True):
        st.markdown(f"**Why Inflation is {growth_row['inflation_state']}** (currently effective)")
        st.markdown(
            f"- Model: `{i_conf['core_series']}` (Core CPI, US)\n"
            f"- Effective signal (used for the CURRENT regime, lagged {lag_months} month(s)): "
            f"**{growth_row['inflation_state']}** as of {fmt_date(effective_date)} "
            f"(score {fmt_pct(growth_row['inflation_score'])})\n"
            f"- Observed signal (most recent reading, not yet effective): "
            f"**{growth_row['inflation_state_observed']}** as of {fmt_date(observed_date)}"
        )
        st.caption(
            "Observed becomes effective next month under the "
            f"{lag_months}-month lag. Values above are read directly from the production "
            "artifact -- this dashboard never recomputes a signal."
        )

    st.markdown("#### BEI Duration Gate")
    bc1, bc2 = st.columns(2)
    bc1.metric("Raw", str(growth_row["bei_gate_raw"]))
    bc2.metric("Tradable", str(growth_row["bei_gate_tradable"]))

    st.markdown("#### Fast Crisis Conditions")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Raw trigger", str(growth_row["fast_crisis_raw"]))
    fc2.metric("Tradable", str(growth_row["fast_crisis_tradable"]))
    fc3.metric("Votes", fmt_int(growth_row["fast_crisis_votes"]))

    st.markdown("#### Recent Signal Table")
    range_label_sig = st.radio(
        "Range", list(SIGNAL_RANGE_OPTIONS.keys()), index=2, horizontal=True, key="sig_range"
    )
    n_days_sig = SIGNAL_RANGE_OPTIONS[range_label_sig]
    show_all = st.checkbox("Show all daily states (not just events)", value=False, key="sig_show_all")
    event_filter = st.multiselect("Event type filter", EVENT_TYPES, default=[], key="sig_event_filter")

    window = v13 if n_days_sig is None else v13.iloc[-n_days_sig:]
    events = []
    prev = None
    for date, row in window.iterrows():
        etype = _classify_event(row, prev)
        if show_all or bool(row["signal_event"]) or bool(row["rebalance_event"]):
            events.append(
                {
                    "Decision date": date,
                    "Effective date": date,
                    "Event type": etype,
                    "Macro regime": row["macro_regime"],
                    "Growth": row["growth_state"],
                    "Inflation": row["inflation_state"],
                    "BEI Gate": row["bei_gate_tradable"],
                    "Fast Crisis": row["fast_crisis_tradable"],
                    "Votes": row["fast_crisis_votes"],
                    "Turnover": row["daily_turnover"],
                    "Cost": row["transaction_cost"],
                }
            )
        prev = row
    events_df = pd.DataFrame(events).sort_values("Decision date", ascending=False)
    if event_filter:
        events_df = events_df[events_df["Event type"].isin(event_filter)]
    st.dataframe(events_df, width="stretch")
    st.download_button(
        "Download CSV", events_df.to_csv(index=False), file_name="recent_signal_table.csv", key="sig_download"
    )


# =============================================================================
# Tab 5 -- Methodology
# =============================================================================

with tab_methodology:
    st.markdown("#### v1.3 Allocations")
    config = _cached_config()
    v13_overrides = config["strategy_versions"]["v1_3"]["regime_allocation_overrides"]
    for regime, weights in v13_overrides.items():
        with st.expander(f"{regime} (v1.3)"):
            st.table(pd.Series(weights, name="weight").to_frame())
    with st.expander("CONTRACTION / STAGFLATION / UNKNOWN (unchanged from v1.2)"):
        for regime in ["CONTRACTION", "STAGFLATION", "UNKNOWN"]:
            st.markdown(f"**{regime}**")
            st.table(pd.Series(config["backtest"]["regime_allocations"][regime], name="weight").to_frame())

    st.markdown("#### Exact Rules (unchanged from v1.2)")
    st.markdown(
        "- One-month implementation lag on the macro signal\n"
        "- BEI Duration Risk Gate (Contraction-only duration-sleeve tilt)\n"
        "- Fast Crisis 2-of-3 (VIX shock / SPY 5D shock / HYG 5D shock), t+1 execution\n"
        "- Minimum crisis hold: 10 trading days; 5 consecutive OFF days required to exit\n"
        f"- Transaction cost: {config['backtest']['transaction_cost_bps']}bp per unit one-way turnover\n"
        "- Fast Crisis ON: Growth Basket/HYG/DBC -> 0, moved to BIL; GLD/TIP/LQD/IEF/TLT unchanged"
    )

    st.markdown("#### Version History")
    st.markdown(
        "**Macro Regime TAA v1.2** -- baseline regime allocation table, frozen and immutable "
        "(`config/default.yaml`, `backtest.regime_allocations`).\n\n"
        "**Macro Regime TAA v1.3 -- Growth Participation** (adopted) -- Goldilocks/Reflation Growth Basket "
        "+5pp, funded entirely from BIL; Contraction/Stagflation/Unknown unchanged. Adopted after the "
        'Candidate "Both +5" robustness study '
        "(`.analysis/growth_participation_robustness/report_ko.md`):\n\n"
        "| Metric | v1.2 | v1.3 (approx., see live tables above for exact) |\n|---|---|---|\n"
        "| Full CAGR | 10.80% | ~11.41% |\n"
        "| Sharpe | 1.165 | ~1.184 |\n"
        "| Daily MDD | -16.64% | -16.64% |\n"
        "| Last 3Y CAGR | 17.75% | ~18.93% |\n"
        "| Last 5Y CAGR | 11.03% | ~11.70% |\n"
        "| COVID MDD | -7.81% | ~-8.34% |\n"
        "| 2022 performance | -- | unchanged |\n"
        "| Rolling 36m CAGR win rate | -- | ~99.7% |\n"
        "| Rolling 60m CAGR win rate | -- | 100% |\n"
        "| Calm-market upside capture | -- | 79.7% (vs 80% pre-registered target) |\n\n"
        "> The candidate narrowly missed the pre-specified 80% calm-market upside-capture threshold, "
        "but was adopted based on its higher CAGR, improved Sharpe, unchanged full-period drawdown, "
        "strong rolling consistency and limited degradation during crisis periods.\n\n"
        "*Figures above are recorded from the robustness study for context; every number elsewhere in "
        "this dashboard is computed live from `production_v13_daily.parquet`, never hardcoded from this "
        "table.*"
    )

    st.markdown("#### Benchmark Definitions")
    for b in list_ui_visible():
        with st.expander(b.display_name):
            cost_display = b.transaction_cost_bps if b.transaction_cost_bps is not None else "n/a"
            st.markdown(
                f"- **Category**: {b.category}\n- **Method**: {b.calculation_method}\n"
                f"- **Rebalance**: {b.rebalance_rule}\n- **Transaction cost**: {cost_display}\n"
                f"- **Calendar**: {b.calendar_rule}\n"
                f"- **Total return treatment**: {b.total_return_treatment}\n"
                f"- **Notes**: {b.notes or '-'}"
            )

    st.markdown("#### MALOX Data Treatment")
    st.markdown(
        "- Ticker resolves to **BlackRock Global Allocation Fund, Institutional Shares** (verified via "
        "provider metadata: `quoteType=MUTUALFUND`, `longName='BlackRock Global Allocation Fund'`).\n"
        "- Mutual fund: one NAV per trading day, never intraday; a NAV may post up to a day later than the "
        "strategy's own market date.\n"
        "- Adjusted series reflects distributions -- verified against an actual distribution event "
        "($0.90/share on 2025-12-16): the pre-distribution adjusted price is correctly reduced to preserve "
        "total-return continuity.\n"
        "- Never backward-filled; a day with no new MALOX NAV holds the last real NAV forward for valuation "
        "but reports exactly 0% return that day (never a fabricated/interpolated one), "
        "and is flagged `is_stale`.\n"
        "- No synthetic rebalancing transaction cost is added -- the fund's own reported NAV/adjusted return "
        "is used as-is; internal fund expenses are already embedded in NAV and are not the same as an "
        "external investor's own sales-load, which this benchmark does not model."
    )

    st.markdown("#### Data Freshness Rules")
    st.markdown(
        "- Portfolio NAV and every benchmark extend through the latest common trading day the underlying "
        "price data actually covers -- never capped at the monthly macro signal's own last closed month.\n"
        "- The current, still-open calendar month is marked `partial_month=True` and holds the prior month's "
        "already-decided macro target forward (no lookahead).\n"
        "- `python -m macro_regime.cli update-all` always fetches fresh data (`--refresh-cache` by default) "
        "and fails loudly (nonzero exit, no artifact overwritten with stale-labeled-as-fresh data) if the "
        "resulting US 60/40 benchmark date falls materially behind the strategy's own market date.\n"
        "- A MALOX-specific fetch failure is isolated: it does not fail the strategy or US 60/40 build, and "
        "renders as `unavailable`/`stale` in the UI rather than reusing an old value silently."
    )

    st.markdown("#### Limitations")
    st.markdown(
        "- Regime classification is on revised FRED data, not the real-time-as-published vintage.\n"
        "- MALOX's fee/distribution-adjustment treatment could not be independently cross-checked against a "
        "second data source in this session; the adjusted series is trusted based on the single distribution-"
        "event verification described above.\n"
        "- Past performance shown anywhere in this dashboard does not represent a live/real-time "
        "track record."
    )
    st.caption("Regime mapping matrix has moved here from the main flow -- see the expanders above.")
