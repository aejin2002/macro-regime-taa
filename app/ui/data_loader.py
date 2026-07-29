"""Streamlit data access -- two layers.

1. Live: `get_dashboard_data()` runs the real production pipeline
   (`macro_regime.deployment.live_pipeline`) against fresh FRED/Yahoo
   Finance data, cached up to `LIVE_REFRESH_TTL_SECONDS` so a normal
   widget rerun/tab switch never re-triggers a fetch. This is the
   PRIMARY path -- the dashboard is a live production view, not a
   static artifact viewer.
2. Fallback: `load_v1_3_daily()`/`load_benchmarks_daily()` -- the
   original local-first/GitHub-Release-fallback static loader,
   unchanged. `get_dashboard_data()` falls back to this tier only when
   a live attempt fails and no prior-in-process live result exists to
   reuse. The Release artifact remains the reproducibility/disaster-
   recovery anchor -- see `docs/methodology.md`.

Cache keys for the fallback tier include each file's own mtime, so a
stale `st.cache_data` entry is automatically invalidated the moment a
fresh local pipeline run rewrites the artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from macro_regime.deployment import (  # noqa: E402
    ArtifactDownloadError,
    ArtifactValidationError,
    LivePipelineError,
    resolve_artifact_path,
    resolve_benchmarks_artifact_path,
    run_live_production_pipeline,
)

# FRED is mostly monthly-frequency and Yahoo Finance closes once/day, so
# refetching more often than this buys negligible freshness for a
# ~30-70s full pipeline run (see the live-pipeline architecture report).
# TTL expiry and the sidebar's "Refresh latest data" button are the only
# two triggers -- an ordinary rerun (tab switch, widget change) within
# this window reuses the cached result, never re-fetching/recomputing.
LIVE_REFRESH_TTL_SECONDS = 3600

V1_2_PATH = PROCESSED_DIR / "v1_2_regression_daily.parquet"
STATUS_PATH = PROCESSED_DIR / "update_all_status.json"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_parquet(path_str: str, _mtime_key: float) -> pd.DataFrame:
    df = pd.read_parquet(path_str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner="Resolving v1.3 production artifact...")
def _resolved_v1_3_artifact():
    """Resolved once per process (not per rerun) via `st.cache_resource`
    -- a local-path check or a GitHub Release download, never a backtest.
    Returns None (not an exception) on failure so the caller can render a
    clear in-app error instead of crashing the whole script."""
    try:
        return resolve_artifact_path()
    except (ArtifactDownloadError, ArtifactValidationError) as exc:
        return exc


def load_v1_3_daily() -> pd.DataFrame | None:
    resolved = _resolved_v1_3_artifact()
    if isinstance(resolved, Exception):
        st.error(
            f"Could not load the v1.3 production artifact: {resolved}\n\n"
            "This dashboard never computes a backtest itself -- run "
            "`python -m macro_regime.cli update-all` locally, or check that the "
            "`v1.3.0` GitHub Release asset is reachable."
        )
        return None
    return _read_parquet(str(resolved.path), _mtime(resolved.path))


def v1_3_artifact_source_info() -> dict | None:
    """Diagnostics for the Methodology/sidebar: where the artifact came
    from (local vs. downloaded), which release tag, and its checksum."""
    resolved = _resolved_v1_3_artifact()
    if isinstance(resolved, Exception):
        return None
    return {
        "source": resolved.source, "tag": resolved.tag, "sha256": resolved.sha256, "path": str(resolved.path)
    }


def load_v1_2_regression_daily() -> pd.DataFrame | None:
    if not V1_2_PATH.exists():
        return None
    return _read_parquet(str(V1_2_PATH), _mtime(V1_2_PATH))


@st.cache_resource(show_spinner="Resolving benchmarks artifact...")
def _resolved_benchmarks_artifact():
    """Same pattern as `_resolved_v1_3_artifact`, independently: a local-
    path check or a GitHub Release download for `benchmarks_daily.parquet`.
    A benchmarks-resolution failure is isolated -- it never affects v1.3
    strategy loading -- and is surfaced as `None` (unavailable) rather
    than crashing the dashboard, matching the existing MALOX-isolation
    design elsewhere in this app."""
    try:
        return resolve_benchmarks_artifact_path()
    except (ArtifactDownloadError, ArtifactValidationError) as exc:
        return exc


def load_benchmarks_daily() -> pd.DataFrame | None:
    resolved = _resolved_benchmarks_artifact()
    if isinstance(resolved, Exception):
        return None
    return _read_parquet(str(resolved.path), _mtime(resolved.path))


def benchmarks_artifact_source_info() -> dict | None:
    """Diagnostics for the sidebar: where benchmarks_daily.parquet came
    from (local vs. downloaded), which release tag, and its checksum."""
    resolved = _resolved_benchmarks_artifact()
    if isinstance(resolved, Exception):
        return None
    return {
        "source": resolved.source, "tag": resolved.tag, "sha256": resolved.sha256, "path": str(resolved.path)
    }


def load_pipeline_status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def data_freshness_summary(
    v1_3: pd.DataFrame | None = None, bench: pd.DataFrame | None = None, status: dict | None = None
) -> dict:
    """Every distinct 'as of' date the UI needs to show separately (see
    docs/methodology.md, 'Date concepts'). Never backfills a missing
    artifact with a guess -- a `None` here must render as 'unavailable',
    not as today's date.

    Takes `v1_3`/`bench` as parameters (rather than loading them itself)
    so the SAME freshness rules apply whether the data came from a live
    pipeline run or the Release-artifact fallback -- callers pass
    `get_dashboard_data()`'s DataFrames. Falls back to the static loader
    only if a caller omits them (keeps this callable standalone, e.g.
    from a REPL/test)."""
    if v1_3 is None:
        v1_3 = load_v1_3_daily()
    if bench is None:
        bench = load_benchmarks_daily()
    if status is None:
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


# =============================================================================
# Live production data -- primary path
#
# Freshness contract (never violate): a Release artifact must never be
# labeled "Live" or "Cached live", a live result's `fetched_at` must
# never be backfilled from an artifact-download timestamp, and a
# "Cached live" result must carry explicit proof it came from a real
# live run (`source_mode == "live_pipeline"`), never inferred merely
# from a DataFrame being present. See tests in test_streamlit_app.py's
# "Live production data: freshness contract" section.
# =============================================================================

# Recommended fallback copy (kept as one named constant so the UI and
# tests can't drift apart on the exact wording).
RELEASE_FALLBACK_MESSAGE = (
    "Live production data is unavailable. Showing the last validated release snapshot as of {as_of}."
)

# A cached-live (or even freshly-live) result whose own strategy market
# date is more than this many CALENDAR days behind today is flagged
# stale in the UI -- deliberately not calendar/holiday-aware (no new
# trading-calendar logic invented for this), just wide enough (a long
# weekend is 3 days) not to false-positive on an ordinary Monday.
STALE_MARKET_DATE_THRESHOLD_DAYS = 5


@st.cache_resource
def _live_result_holder() -> dict:
    """Process-wide (not per-browser-session) mutable holder for the
    last successful live pipeline result -- deliberately outlives that
    result's own TTL cache entry, so a LATER failed refresh attempt
    still has something better than jumping straight to the frozen
    Release artifact. Cleared only by a full process restart."""
    return {}


@st.cache_resource
def _live_pipeline_lock() -> threading.Lock:
    """One lock per process, shared across all sessions -- prevents two
    concurrent refresh attempts (e.g. two users' sessions both missing
    the TTL cache at once) from running the ~30-70s pipeline
    redundantly in parallel and racing on the same scratch files
    (`fred_wide.csv` etc.). `st.cache_data` already de-duplicates
    concurrent calls to the same cache key internally, but this lock
    makes that guarantee explicit and independent of that internal
    behavior."""
    return threading.Lock()


def _config_fingerprint() -> str:
    """A short hash of `config/default.yaml`'s content -- passed into
    `_run_live_pipeline_cached` so its `st.cache_data` cache key is tied
    to source configuration, not just wall-clock TTL: editing the
    config (a real deployment update) invalidates any in-flight cached
    live result instead of silently continuing to serve a result built
    from the previous configuration."""
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    try:
        return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "unknown"


@st.cache_data(
    ttl=LIVE_REFRESH_TTL_SECONDS,
    show_spinner="Refreshing live production data (FRED + Yahoo Finance)...",
)
def _run_live_pipeline_cached(_config_fingerprint_key: str) -> dict:
    lock = _live_pipeline_lock()
    with lock:
        return run_live_production_pipeline(refresh_cache=True)


def refresh_live_data_now() -> None:
    """Sidebar 'Refresh latest data' button handler -- forces the next
    `get_dashboard_data()` call to bypass the TTL cache and re-run the
    live pipeline immediately."""
    _run_live_pipeline_cached.clear()


def _is_live_provenance_valid(raw: dict) -> bool:
    """Defense-in-depth: a session-cached result is only ever trusted as
    "Cached live" if it explicitly carries proof of a real live run --
    never inferred just because `v1_3_df`/`bench_df` happen to be
    present in the holder dict."""
    return isinstance(raw, dict) and raw.get("source_mode") == "live_pipeline" and bool(raw.get("fetched_at"))


def _is_market_date_stale(v1_3_df: pd.DataFrame | None) -> bool:
    if v1_3_df is None or v1_3_df.empty or "date" not in v1_3_df.columns:
        return False
    market_date = pd.Timestamp(v1_3_df["date"].max())
    age_days = (pd.Timestamp.now().normalize() - market_date.normalize()).days
    return age_days > STALE_MARKET_DATE_THRESHOLD_DAYS


def get_dashboard_data() -> dict:
    """The single entry point the app uses for v1.3/benchmark data.
    Three tiers, in order, never raising:

    1. "live" -- a fresh (or TTL-cached) live pipeline run.
    2. "session_fallback" -- a live attempt just failed, but a PRIOR
       live run in this process succeeded (validated via
       `_is_live_provenance_valid`); that result is reused (marked
       stale) rather than jumping straight to tier 3.
    3. "release_fallback" -- the unchanged local-first/GitHub-Release
       static loader, used only when both live tiers are unavailable.
       `fetched_at` is ALWAYS None here -- the artifact's own download/
       resolution time is never used as a live-refresh timestamp.

    Returns a dict with `v1_3_df`, `bench_df`, `vix_series`,
    `fetched_at`, `mode`, `error`, `is_stale` -- `mode` lets the caller
    render an accurate status rather than silently mixing tiers."""
    holder = _live_result_holder()
    try:
        raw = _run_live_pipeline_cached(_config_fingerprint())
        if not _is_live_provenance_valid(raw):
            # Should be unreachable (run_live_production_pipeline always
            # stamps this) -- treat as a failure rather than trust it.
            raise LivePipelineError("live result missing provenance metadata")  # noqa: TRY301
        holder["last_good"] = raw
        return {**raw, "mode": "live", "error": None, "is_stale": _is_market_date_stale(raw.get("v1_3_df"))}
    except Exception as exc:  # noqa: BLE001
        last_good = holder.get("last_good")
        if _is_live_provenance_valid(last_good):
            return {
                **last_good,
                "mode": "session_fallback",
                "error": str(exc),
                "is_stale": True,  # a live attempt JUST failed -- always stale by definition
            }
        v1_3_df = load_v1_3_daily()
        bench_df = load_benchmarks_daily()
        if v1_3_df is None or v1_3_df.empty:
            return {
                "v1_3_df": None, "bench_df": None, "vix_series": None, "fetched_at": None,
                "mode": "error", "error": str(exc), "is_stale": True,
            }
        return {
            "v1_3_df": v1_3_df,
            "bench_df": bench_df,
            "vix_series": None,
            "fetched_at": None,
            "mode": "release_fallback",
            "error": str(exc),
            "is_stale": True,
        }
