"""Streamlit research UI for macro-regime-taa.

Reads the artifacts produced by `python -m macro_regime.cli run-all`
(data/processed/*.csv, *.json). Run `make fetch build-signals evaluate`
(or `make run-all`) before launching this app for the first time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_regime.config import load_config  # noqa: E402
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

REGIME_COLORS = {
    Regime.GOLDILOCKS.value: "#2E7D32",
    Regime.REFLATION.value: "#F9A825",
    Regime.STAGFLATION.value: "#C62828",
    Regime.CONTRACTION.value: "#455A64",
    Regime.UNKNOWN.value: "#9E9E9E",
}

st.set_page_config(page_title="macro-regime-taa", layout="wide")


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


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

    latest_date = signals.index.max()
    st.caption(f"Latest signal date: **{latest_date.date() if pd.notna(latest_date) else 'n/a'}**")
    st.caption(
        "All figures below use current, revised FRED data (a 'revised-data backtest'), "
        "not ALFRED real-time vintages -- see the Methodology page."
    )

    growth_cols = [c for c in signals.columns if c.startswith("growth_model")]
    inflation_cols = [c for c in signals.columns if c.startswith("inflation_model")]
    regime_cols = [c for c in signals.columns if c.startswith("regime_")]

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
    config = load_config()
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
    elif page == "Evaluation":
        page_evaluation(eval_report)
    elif page == "Methodology":
        page_methodology(config)


if __name__ == "__main__":
    main()
