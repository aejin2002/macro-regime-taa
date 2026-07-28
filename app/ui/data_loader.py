"""Read-only Streamlit data access -- loads the canonical parquet
artifacts built by `python -m macro_regime.cli update-all`. Streamlit
itself never fetches data or runs a backtest; it only renders what the
pipeline already computed. Cache keys include each file's own mtime, so
a stale `st.cache_data` entry is automatically invalidated the moment a
fresh pipeline run rewrites the artifact -- no manual "clear cache"
button needed, and a rerun can never keep showing yesterday's data just
because the process didn't restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

V1_3_PATH = PROCESSED_DIR / "production_v13_daily.parquet"
V1_2_PATH = PROCESSED_DIR / "v1_2_regression_daily.parquet"
BENCHMARKS_PATH = PROCESSED_DIR / "benchmarks_daily.parquet"
STATUS_PATH = PROCESSED_DIR / "update_all_status.json"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_parquet(path_str: str, _mtime_key: float) -> pd.DataFrame:
    df = pd.read_parquet(path_str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_v1_3_daily() -> pd.DataFrame | None:
    if not V1_3_PATH.exists():
        return None
    return _read_parquet(str(V1_3_PATH), _mtime(V1_3_PATH))


def load_v1_2_regression_daily() -> pd.DataFrame | None:
    if not V1_2_PATH.exists():
        return None
    return _read_parquet(str(V1_2_PATH), _mtime(V1_2_PATH))


def load_benchmarks_daily() -> pd.DataFrame | None:
    if not BENCHMARKS_PATH.exists():
        return None
    return _read_parquet(str(BENCHMARKS_PATH), _mtime(BENCHMARKS_PATH))


def load_pipeline_status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def data_freshness_summary() -> dict:
    """Every distinct 'as of' date the UI needs to show separately (see
    docs/methodology.md, 'Date concepts'). Never backfills a missing
    artifact with a guess -- a `None` here must render as 'unavailable',
    not as today's date."""
    v1_3 = load_v1_3_daily()
    bench = load_benchmarks_daily()
    status = load_pipeline_status()

    market_date = v1_3["date"].max() if v1_3 is not None and len(v1_3) else None
    macro_month = None
    if v1_3 is not None and len(v1_3):
        closed = v1_3.loc[~v1_3["partial_month"], "date"]
        macro_month = closed.max() if len(closed) else None

    us_6040_date = None
    malox_date = None
    malox_stale = None
    if bench is not None:
        us_row = bench[bench["benchmark_id"] == "us_60_40"]
        if len(us_row) and bool(us_row["available"].iloc[0]):
            us_6040_date = us_row["date"].max()
        malox_row = bench[bench["benchmark_id"] == "malox"]
        if len(malox_row) and bool(malox_row["available"].iloc[0]):
            malox_date = malox_row["date"].max()
            if "is_stale" in malox_row.columns:
                last_row = malox_row.sort_values("date").iloc[-1]
                malox_stale = bool(last_row.get("is_stale", False))

    return {
        "dashboard_generated_at": pd.Timestamp.now(),
        "strategy_market_data_as_of": market_date,
        "portfolio_nav_as_of": market_date,
        "us_6040_as_of": us_6040_date,
        "malox_as_of": malox_date,
        "malox_is_stale": malox_stale,
        "macro_data_available_as_of": macro_month,
        "current_macro_allocation_effective_from": macro_month,
        "pipeline_status": status,
    }
