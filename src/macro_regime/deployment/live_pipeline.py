"""Live production pipeline runner for the Streamlit dashboard's
"live refresh" mode.

Calls the EXACT SAME functions `cli.update_all()` calls, in the same
order, with zero reimplementation of fetch/signal/regime/Fast Crisis/
allocation logic -- this module only orchestrates and returns in-memory
DataFrames instead of writing the two Release-pinned parquet artifacts.
`production_v13_daily.parquet` and `benchmarks_daily.parquet` are never
written by this module; only the intermediate scratch files `fetch()`/
`build_signals()`/`build_regime_output_cmd()` themselves already write
(`fred_wide.csv`, `signals.csv`, `regime_output_primary.csv`, etc.) are
touched, as an unavoidable side effect of reusing those functions
unmodified rather than forking them to accept an in-memory-only mode.

v1.2 is intentionally NOT rebuilt here (the dashboard never renders it --
see `app/streamlit_app.py`'s `v12_raw`, loaded but unused) to keep the
live run to one `run_fast_crisis_backtest` pass instead of two.

VIX: this pipeline's `fetch()` step downloads FRED's `VIXCLS` series
(one of the 19 already-configured series) as a side effect of the
normal macro fetch -- no extra network call. `run_live_production_pipeline`
surfaces it as `vix_series` so Markets can show a real VIX chart in live
mode, while the frozen Release artifacts (which don't carry VIX) keep
showing "unavailable" in fallback mode -- see `app/streamlit_app.py`.
"""

from __future__ import annotations

import os

import pandas as pd

from macro_regime.cli import (
    PROCESSED_DIR,
    WIDE_PATH,
    build_regime_output_cmd,
    build_signals,
    evaluate,
    fetch,
)
from macro_regime.config import load_config
from macro_regime.data.asset_prices import AssetPriceClient
from macro_regime.production import build_benchmark_daily_artifact, build_v1_3_daily_artifact


class LivePipelineError(RuntimeError):
    """Any failure fetching data or running the production pipeline
    live. Callers must never show this (or its traceback) directly to
    end users -- catch it and fall back to the last known-good live
    result or the Release artifact."""


def bridge_fred_api_key_from_streamlit_secrets() -> None:
    """On Streamlit Community Cloud, `FRED_API_KEY` is set as an app
    secret (`st.secrets`), not a process env var -- `Settings`/
    `FredClient` only ever read `os.environ`/.env (config.py), so this
    copies it across once if present and not already set. A no-op
    locally (`.env` already sets the env var directly) and a no-op if
    no secrets.toml exists at all (never raises)."""
    if os.environ.get("FRED_API_KEY"):
        return
    try:
        import streamlit as st

        key = st.secrets.get("FRED_API_KEY")
    except Exception:  # noqa: BLE001
        return
    if key:
        os.environ["FRED_API_KEY"] = key


def run_live_production_pipeline(*, refresh_cache: bool = True) -> dict:
    """Runs fetch -> build-signals -> evaluate -> build-regime-output ->
    v1.3 daily artifact -> benchmark artifact, reusing those functions
    exactly as `cli.update_all()` does. Returns a dict:
    `v1_3_df`, `bench_df`, `vix_series`, `fetched_at`. As-of/freshness
    dates are deliberately NOT computed here -- the caller (Streamlit)
    derives them from `v1_3_df`/`bench_df` via the existing
    `data_freshness_summary()` rules (partial-month handling etc.),
    the same way it already does for the Release-artifact path, so
    "live" and "fallback" modes are never subject to two different
    freshness definitions.
    Raises `LivePipelineError` on any failure -- never returns a partial
    result, matching `update_all()`'s own "no partial artifact on
    failure" contract."""
    bridge_fred_api_key_from_streamlit_secrets()
    try:
        fetch(refresh_cache=refresh_cache)
        build_signals(growth_model="all", inflation_model="all")
        evaluate()
        build_regime_output_cmd()

        config = load_config()
        wide = pd.read_csv(WIDE_PATH, index_col=0, parse_dates=True)
        primary_df = pd.read_csv(
            PROCESSED_DIR / "regime_output_primary.csv", index_col=0, parse_dates=True
        )
        client = AssetPriceClient()
        as_of = pd.Timestamp.now()

        v1_3_df, bt = build_v1_3_daily_artifact(
            config,
            primary_df["tradable_regime"],
            wide,
            primary_df,
            as_of=as_of,
            client=client,
            refresh_cache=refresh_cache,
        )
        calendar = pd.DatetimeIndex(v1_3_df["date"])
        bench_df = build_benchmark_daily_artifact(
            bt, calendar, as_of, client=client, refresh_cache=refresh_cache
        )
    except Exception as exc:  # noqa: BLE001
        raise LivePipelineError(f"Live production pipeline failed: {exc}") from exc

    return {
        "v1_3_df": v1_3_df,
        "bench_df": bench_df,
        "vix_series": wide["VIXCLS"].dropna() if "VIXCLS" in wide.columns else None,
        "fetched_at": pd.Timestamp.now(),
    }
