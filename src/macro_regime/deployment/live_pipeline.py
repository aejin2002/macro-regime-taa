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

Two independent VIX sources, deliberately kept separate (see
`fetch_market_vix_series`'s docstring for why the Fast Crisis production
signal input is NOT touched):
- Fast Crisis's OWN input stays FRED `VIXCLS` (`wide["VIXCLS"]`, fetched
  as part of the normal 19-series macro pull) -- unchanged from before
  this module existed.
- The Markets tab's VIX chart uses Yahoo `^VIX` instead, fetched
  independently and returned as `vix_series` with `vix_source` set to
  disambiguate it in any diagnostics/tests.

Yahoo fetch performance (see the accompanying report for full timing/
parity detail): the 11 daily tickers `run_fast_crisis_backtest` needs
(9 US ETFs + KODEX200 + FX, all queried from the SAME `fetch_start`
date -- confirmed by reading `fast_crisis.daily_data.load_daily_prices`'s
default and its two call sites, neither of which override it) are
prefetched CONCURRENTLY (bounded ThreadPoolExecutor) into
`AssetPriceClient`'s own on-disk cache; the v1.3 artifact build then
runs with `refresh_cache=False` so it reads that just-warmed,
same-run-fresh cache instead of re-fetching serially. The benchmark
builder's own tickers (SPY/AGG/MALOX) use a DIFFERENT start date
("2000-01-01" vs "2008-01-01") than the prefetch batch, so they are
NOT covered by it and still fetch live (`refresh_cache=True`, unchanged)
-- deliberately not unified, since changing either module's literal
start-date constant to make them match would be a production-logic
change this task does not make. A request-scoped memoization wrapper
around the shared `AssetPriceClient` instance still dedupes the
benchmark builder's OWN internal repeats (SPY and AGG are each fetched
twice there -- once standalone, once as a blend component).
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from macro_regime.fast_crisis.daily_data import US_TICKER_COLUMNS, load_daily_prices
from macro_regime.production import build_benchmark_daily_artifact, build_v1_3_daily_artifact

MARKET_VIX_TICKER = "^VIX"
PREFETCH_MAX_WORKERS = 4


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


def _memoize_client_for_one_run(client: AssetPriceClient) -> AssetPriceClient:
    """Wraps `client.get_daily_close` so that, within ONE live pipeline
    run, the exact same (ticker, start, refresh_cache) call only ever
    hits Yahoo Finance once. Needed because the benchmark builder calls
    `get_daily_close("SPY", "2000-01-01", ...)` twice on its own (once
    standalone, once as the US 60/40 blend's SPY leg) and the same for
    AGG -- purely an in-memory, run-scoped layer; never touches
    `AssetPriceClient`'s own on-disk FileCache/TTL semantics, and is
    discarded when this function returns (a fresh client each run)."""
    lock = threading.Lock()
    memo: dict[tuple, pd.Series] = {}
    original = client.get_daily_close

    @functools.wraps(original)
    def wrapper(ticker: str, start: str, *, refresh_cache: bool = False) -> pd.Series:
        key = (ticker, start, refresh_cache)
        with lock:
            cached = memo.get(key)
        if cached is not None:
            return cached.copy()
        result = original(ticker, start, refresh_cache=refresh_cache)
        with lock:
            memo[key] = result
        return result.copy()

    client.get_daily_close = wrapper  # type: ignore[method-assign]
    return client


def _fast_crisis_fetch_start() -> str:
    """Reads `load_daily_prices`'s own default `fetch_start` value
    (currently "2008-01-01") via introspection rather than duplicating
    the literal -- if that default ever changes, this prefetch step
    changes with it instead of silently going stale."""
    default = inspect.signature(load_daily_prices).parameters["fetch_start"].default
    return str(default)


def _prefetch_tickers_concurrently(
    client: AssetPriceClient, tickers: list[str], start: str, *, max_workers: int = PREFETCH_MAX_WORKERS
) -> dict[str, str]:
    """Fetches `tickers` at `start` concurrently (bounded ThreadPoolExecutor,
    `refresh_cache=True` -- a genuine live fetch, not a cache read) so
    they land, freshly, in `client`'s own on-disk FileCache before the
    sequential production functions ask for them one at a time. Returns
    a dict of {ticker: error_message} for any ticker that failed --
    empty if everything succeeded. Does NOT raise on a per-ticker
    failure; the caller decides whether a partial prefetch is fatal
    (currently: it always is, since the downstream v1.3 build will
    itself fail loudly re-fetching that same ticker with a normal,
    single, clear error -- this prefetch step's only job is speed, not
    a second, silent success path)."""
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers)) or 1) as executor:
        futures = {
            executor.submit(client.get_daily_close, ticker, start, refresh_cache=True): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                errors[ticker] = str(exc)
    return errors


def fetch_market_vix_series(client: AssetPriceClient) -> pd.Series | None:
    """VIX for the Markets tab chart ONLY -- Yahoo `^VIX`, completely
    independent of Fast Crisis's own FRED `VIXCLS` input. Deliberately
    NOT wired into `run_fast_crisis_backtest`/`load_daily_prices`: that
    function's `vix_daily` parameter is a validated production signal
    input, and swapping its source would be a strategy-input change,
    not a UI change -- out of scope unless full same-cutoff parity
    against FRED VIXCLS were separately proven and reported, which this
    task does not attempt. Returns None (never raises) on failure --
    a missing Markets-only chart series must never fail the whole live
    refresh."""
    try:
        start = _fast_crisis_fetch_start()
        series = client.get_daily_close(MARKET_VIX_TICKER, start, refresh_cache=True)
    except Exception:  # noqa: BLE001
        return None
    return series.dropna() if series is not None else None


def run_live_production_pipeline(*, refresh_cache: bool = True) -> dict:
    """Runs fetch -> build-signals -> evaluate -> build-regime-output ->
    v1.3 daily artifact -> benchmark artifact, reusing those functions
    exactly as `cli.update_all()` does. Returns a dict:
    `v1_3_df`, `bench_df`, `vix_series`, `vix_source`, `fetched_at`,
    `pipeline_run_id`, `source_mode`. `pipeline_run_id`/`source_mode`
    are explicit provenance markers -- a caller must never infer "this
    came from a real live run" just because a DataFrame happens to be
    present; see `app/ui/data_loader.py`'s freshness-contract tests.
    As-of/freshness dates are deliberately NOT computed here -- the
    caller (Streamlit) derives them from `v1_3_df`/`bench_df` via the
    existing `data_freshness_summary()` rules (partial-month handling
    etc.), the same way it already does for the Release-artifact path,
    so "live" and "fallback" modes are never subject to two different
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
        client = _memoize_client_for_one_run(AssetPriceClient())
        as_of = pd.Timestamp.now()

        fast_crisis_tickers = list(US_TICKER_COLUMNS.values()) + [
            config["growth_basket"]["kodex200_ticker"],
            config["growth_basket"]["fx_ticker"],
        ]
        prefetch_errors = _prefetch_tickers_concurrently(
            client, fast_crisis_tickers, _fast_crisis_fetch_start()
        )
        if prefetch_errors:
            # Surface exactly which ticker(s) failed rather than a
            # generic message -- and let the downstream build attempt
            # its own (serial) fetch for a clean, single-source error
            # instead of raising here on a prefetch-only failure.
            v1_3_refresh_cache = refresh_cache
        else:
            v1_3_refresh_cache = False  # just warmed, same run -- safe to read back

        v1_3_df, bt = build_v1_3_daily_artifact(
            config,
            primary_df["tradable_regime"],
            wide,
            primary_df,
            as_of=as_of,
            client=client,
            refresh_cache=v1_3_refresh_cache,
        )
        calendar = pd.DatetimeIndex(v1_3_df["date"])
        # NOT prefetched (different start date than the batch above --
        # see module docstring) -- stays a genuine live fetch.
        bench_df = build_benchmark_daily_artifact(
            bt, calendar, as_of, client=client, refresh_cache=refresh_cache
        )
        market_vix = fetch_market_vix_series(client)
    except Exception as exc:  # noqa: BLE001
        raise LivePipelineError(f"Live production pipeline failed: {exc}") from exc

    return {
        "v1_3_df": v1_3_df,
        "bench_df": bench_df,
        "vix_series": market_vix,
        "vix_source": "yahoo_^VIX",
        "fetched_at": pd.Timestamp.now(),
        "pipeline_run_id": f"{as_of.isoformat()}-{id(client)}",
        "source_mode": "live_pipeline",
    }
