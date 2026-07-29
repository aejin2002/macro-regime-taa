"""Deterministic parity: baseline (commit 9da5917, pre-Yahoo-optimization)
vs current (this branch, post-optimization) `run_live_production_pipeline`,
run against an IDENTICAL, FIXED raw input snapshot.

No live network call is ever made here. The snapshot is real production
data already sitting in this machine's on-disk FileCache (`data/raw/cache`
for FRED, `data/raw/asset_cache` for Yahoo) from prior sessions' live
runs -- loaded once, frozen into memory, and served to BOTH pipeline
versions via mocks at the exact same two injection points every commit
between 9da5917 and HEAD shares unchanged:
  1. `fetch()` (writes `fred_wide.csv`) -- mocked to write a frozen
     snapshot instead of hitting FRED.
  2. `yfinance.Ticker(ticker).history(...)` -- mocked to return frozen
     per-(ticker, cutoff) OHLC data instead of hitting Yahoo Finance.

Only `live_pipeline.py` differs in a way that matters for this
comparison (the Yahoo-fetch orchestration: prefetch/concurrency/
memoization/refresh_cache flagging). `asset_prices.py` gained a
thread-safety lock (see test_asset_prices.py) that does not change any
output value. Every other module `live_pipeline.py` calls into
(`cli.py`, `production.py`, `fast_crisis/*`, `signals/*`, `benchmarks/*`)
is BYTE-IDENTICAL between the two commits (verified: `git diff
9da5917...HEAD --stat` touches only `live_pipeline.py`, `asset_prices.py`,
`data_loader.py`, `streamlit_app.py`, and test files) -- so the baseline
comparison only needs a second COPY of `live_pipeline.py` itself,
extracted via `git show` and dynamically loaded under its own module
name; everything it imports resolves against the SAME installed
`macro_regime` package current code already uses.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "9da5917"

FRED_SERIES_IDS = [
    "INDPRO", "CFNAI", "CFNAIMA3", "ICSA", "PERMIT", "USPHCI", "USALOLITOAASTSAM", "AMTMNO",
    "CPILFESL", "PCEPILFE", "CPIAUCSL", "MEDCPIM158SFRBCLE", "PALLFNFINDEXM",
    "T10Y3M", "T10Y2Y", "T5YIE", "DGS10", "T10YIE", "VIXCLS",
]
# The 11 US-ETF+KODEX200+FX tickers are fetched at TWO different start
# dates by two independent, byte-identical-across-commits production
# call sites: `load_daily_prices` (fetch_start default "2008-01-01", for
# the daily Fast Crisis/v1.3 build) AND `backtest.assets.build_monthly_return_matrix`
# (start default "2000-01-01", called unconditionally from inside
# `run_fast_crisis_backtest` for the legacy monthly project_6040
# benchmark strategy -- NOT gated behind any flag, so every pipeline run
# fetches this same 11-ticker set at BOTH start dates). AGG/MALOX are
# only ever needed at "2000-01-01" (`benchmarks/engine.py`'s TICKER_MAP,
# a separate/independent "2000-01-01" caller).
TICKERS_2008 = ["SPY", "HYG", "LQD", "IEF", "TLT", "GLD", "BIL", "DBC", "TIP", "069500.KS", "KRW=X"]
TICKERS_2000 = TICKERS_2008 + ["AGG", "MALOX"]
MARKET_VIX_TICKER = "^VIX"

# Representative cutoffs: latest, two ordinary periods, a Fast Crisis ON
# period (COVID), and the trading days immediately before/after a real
# Macro Regime transition -- all found by inspecting the existing
# production_v13_daily.parquet (GOLDILOCKS -> CONTRACTION on 2010-07-01).
CUTOFFS = {
    "latest": "2026-07-28",
    "normal_2015": "2015-06-30",
    "normal_2019": "2019-03-29",
    "crisis_covid": "2020-03-31",
    "regime_transition_before": "2010-06-30",
    "regime_transition_after": "2010-07-02",
}


def _require_real_cache_available() -> bool:
    asset_cache = PROJECT_ROOT / "data" / "raw" / "asset_cache"
    fred_wide = PROJECT_ROOT / "data" / "processed" / "fred_wide.csv"
    return asset_cache.exists() and fred_wide.exists()


pytestmark = pytest.mark.skipif(
    not _require_real_cache_available(),
    reason="requires the on-disk FRED/asset cache and fred_wide.csv populated by a prior live run "
    "(this file never makes a live network call itself)",
)


# =============================================================================
# Frozen snapshot: loaded once per test session from on-disk caches.
# =============================================================================


@pytest.fixture(scope="session")
def fred_wide_snapshot() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fred_wide.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="session")
def yahoo_snapshot() -> dict[tuple[str, str], list[dict]]:
    """{(ticker, start): [{"date": ..., "close": ...}, ...]} -- read
    directly via the real FileCache API against the real on-disk
    asset_cache, so the exact same (ticker, start) keys the production
    code uses are the ones served back."""
    from macro_regime.data.cache import FileCache

    cache = FileCache(PROJECT_ROOT / "data" / "raw" / "asset_cache")
    snapshot: dict[tuple[str, str], list[dict]] = {}
    for ticker in TICKERS_2008:
        cached = cache.get(ticker, {"start": "2008-01-01"})
        assert cached is not None, f"missing real cache entry for {ticker}@2008-01-01 -- run update-all first"
        snapshot[(ticker, "2008-01-01")] = cached["records"]
    for ticker in TICKERS_2000:
        cached = cache.get(ticker, {"start": "2000-01-01"})
        assert cached is not None, f"missing real cache entry for {ticker}@2000-01-01 -- run update-all first"
        snapshot[(ticker, "2000-01-01")] = cached["records"]
    cached = cache.get(MARKET_VIX_TICKER, {"start": "2008-01-01"})
    if cached is not None:
        snapshot[(MARKET_VIX_TICKER, "2008-01-01")] = cached["records"]
    return snapshot


class _FrozenHistoryTicker:
    """Stands in for `yfinance.Ticker` -- `.history(start=..., auto_adjust=True)`
    returns the frozen snapshot's records for (self.ticker, start),
    truncated to `cutoff` if one is set. Deterministic: same input,
    same output, every call, regardless of call order."""

    def __init__(self, ticker: str, *, snapshot: dict, cutoff: str | None) -> None:
        self.ticker = ticker
        self._snapshot = snapshot
        self._cutoff = pd.Timestamp(cutoff) if cutoff else None

    def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
        records = self._snapshot.get((self.ticker, start))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if self._cutoff is not None:
            df = df.loc[df.index <= self._cutoff]
        df = df.rename(columns={"close": "Close"})
        return df


def _make_fetch_mock(fred_wide: pd.DataFrame, wide_path: Path, cutoff: str | None):
    """Replaces `fetch()` -- writes the frozen FRED snapshot (truncated
    to `cutoff`) to `wide_path` instead of calling the real FRED API."""

    def _fake_fetch(*, start_date=None, end_date=None, refresh_cache: bool = False) -> None:
        df = fred_wide
        if cutoff is not None:
            df = df.loc[df.index <= pd.Timestamp(cutoff)]
        wide_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(wide_path)

    return _fake_fetch


# =============================================================================
# Dynamic baseline module loading
# =============================================================================


@pytest.fixture(scope="session")
def baseline_source(tmp_path_factory) -> str:
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:src/macro_regime/deployment/live_pipeline.py"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _load_pipeline_module(name: str, source: str, tmp_path: Path) -> types.ModuleType:
    """Loads `source` as a fresh module under `name`, with its own
    isolated PROCESSED_DIR/WIDE_PATH -- everything it `from macro_regime...
    import`s resolves against the currently-installed package (identical
    between the two commits for every module except live_pipeline.py
    itself)."""
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    processed_dir = tmp_path / f"{name}_processed"
    processed_dir.mkdir(exist_ok=True)
    module.PROCESSED_DIR = processed_dir
    module.WIDE_PATH = processed_dir / "fred_wide.csv"
    return module


def _run_pipeline(
    module: types.ModuleType,
    *,
    fred_wide: pd.DataFrame,
    yahoo_snapshot: dict,
    cutoff: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shuffle_delay_seed: int | None = None,
    completion_log: list[str] | None = None,
) -> dict:
    """Runs `module.run_live_production_pipeline()` with fetch() and
    yfinance.Ticker mocked to the frozen snapshot at `cutoff`. If
    `shuffle_delay_seed` is given, each ticker fetch sleeps a small,
    seed-derived, per-ticker delay -- used to prove concurrent-fetch
    completion ORDER never affects the result. If `completion_log` is
    given, each ticker fetch's completion (post-sleep, i.e. the moment
    its data becomes available to the executor) is appended to it under
    a lock -- lets a caller directly verify two runs actually raced in a
    DIFFERENT real order, not just assume the delays make them differ."""
    import random
    import threading
    import time

    _log_lock = threading.Lock()

    monkeypatch.setattr(module, "fetch", _make_fetch_mock(fred_wide, module.WIDE_PATH, cutoff))
    monkeypatch.setattr(module, "bridge_fred_api_key_from_streamlit_secrets", lambda: None)

    # build_signals/evaluate/build_regime_output_cmd are the REAL,
    # unmocked functions -- they only read/write within `module.PROCESSED_DIR`
    # (already isolated above) and never touch the network.
    import macro_regime.cli as cli_module

    monkeypatch.setattr(cli_module, "PROCESSED_DIR", module.PROCESSED_DIR)
    monkeypatch.setattr(cli_module, "WIDE_PATH", module.WIDE_PATH)
    monkeypatch.setattr(cli_module, "SIGNALS_PATH", module.PROCESSED_DIR / "signals.csv")
    monkeypatch.setattr(cli_module, "REGIME_METADATA_PATH", module.PROCESSED_DIR / "regime_metadata.json")
    monkeypatch.setattr(cli_module, "EVAL_PATH", module.PROCESSED_DIR / "evaluation_report.json")
    monkeypatch.setattr(
        cli_module, "REGIME_OUTPUT_PRIMARY_PATH", module.PROCESSED_DIR / "regime_output_primary.csv"
    )
    monkeypatch.setattr(
        cli_module, "REGIME_OUTPUT_SECONDARY_PATH", module.PROCESSED_DIR / "regime_output_secondary.csv"
    )

    class _SeededFrozenTicker(_FrozenHistoryTicker):
        def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
            if shuffle_delay_seed is not None:
                rng = random.Random(f"{shuffle_delay_seed}-{self.ticker}-{start}")
                time.sleep(rng.uniform(0.0, 0.03))
            if completion_log is not None:
                with _log_lock:
                    completion_log.append(f"{self.ticker}@{start}")
            return super().history(start, auto_adjust)

    def _ticker_factory(ticker: str):
        return _SeededFrozenTicker(ticker, snapshot=yahoo_snapshot, cutoff=cutoff)

    monkeypatch.setattr("macro_regime.data.asset_prices.yf.Ticker", _ticker_factory)

    def _fresh_client_factory():
        from macro_regime.data.asset_prices import AssetPriceClient
        from macro_regime.data.cache import FileCache

        cache_dir = tmp_path / f"asset_cache_{id(object())}"
        return AssetPriceClient(cache=FileCache(cache_dir), min_request_interval_seconds=0.0)

    monkeypatch.setattr(module, "AssetPriceClient", _fresh_client_factory)

    return module.run_live_production_pipeline(refresh_cache=True)


# =============================================================================
# Comparison helpers
# =============================================================================


def _compare_dataframes(baseline: pd.DataFrame, current: pd.DataFrame, *, label: str) -> dict:
    """Returns a report dict; raises AssertionError with the FIRST
    mismatched date/column/values on any difference, never silently
    widening a tolerance."""
    report = {"label": label, "rows_baseline": len(baseline), "rows_current": len(current)}
    assert list(baseline.columns) == list(current.columns), (
        f"[{label}] column set/order differs: {list(baseline.columns)} vs {list(current.columns)}"
    )
    assert len(baseline) == len(current), f"[{label}] row count differs: {len(baseline)} vs {len(current)}"
    assert list(baseline["date"]) == list(current["date"]), f"[{label}] date index differs"

    compared_columns = 0
    exact_match_columns = 0
    for col in baseline.columns:
        compared_columns += 1
        b, c = baseline[col], current[col]
        if pd.api.types.is_float_dtype(b) or pd.api.types.is_float_dtype(c):
            diff = (b.astype(float) - c.astype(float)).abs()
            first_bad = diff[diff > 0]
            if len(first_bad) == 0:
                exact_match_columns += 1
                continue
            idx = first_bad.index[0]
            raise AssertionError(
                f"[{label}] column '{col}' first differs at row {idx} "
                f"(date={baseline['date'].iloc[idx]}): baseline={b.iloc[idx]!r} current={c.iloc[idx]!r} "
                f"abs_diff={first_bad.iloc[0]!r}"
            )
        else:
            if b.equals(c):
                exact_match_columns += 1
                continue
            mismatch = b != c
            idx = mismatch[mismatch].index[0]
            raise AssertionError(
                f"[{label}] column '{col}' first differs at row {idx} "
                f"(date={baseline['date'].iloc[idx]}): baseline={b.iloc[idx]!r} current={c.iloc[idx]!r}"
            )
    report["compared_columns"] = compared_columns
    report["exact_match_columns"] = exact_match_columns
    return report


# =============================================================================
# Parity tests, one per representative cutoff
# =============================================================================


@pytest.fixture
def loaded_modules(baseline_source, tmp_path_factory):
    import macro_regime.deployment.live_pipeline as current_module

    tmp_path = tmp_path_factory.mktemp("parity_modules")
    baseline_module = _load_pipeline_module("baseline_live_pipeline", baseline_source, tmp_path)
    return baseline_module, current_module


@pytest.mark.parametrize("cutoff_name", list(CUTOFFS.keys()))
def test_parity_at_cutoff(
    cutoff_name, loaded_modules, fred_wide_snapshot, yahoo_snapshot, monkeypatch, tmp_path
):
    baseline_module, current_module = loaded_modules
    cutoff = CUTOFFS[cutoff_name]

    baseline_result = _run_pipeline(
        baseline_module, fred_wide=fred_wide_snapshot, yahoo_snapshot=yahoo_snapshot,
        cutoff=cutoff, monkeypatch=monkeypatch, tmp_path=tmp_path / "baseline",
    )
    current_result = _run_pipeline(
        current_module, fred_wide=fred_wide_snapshot, yahoo_snapshot=yahoo_snapshot,
        cutoff=cutoff, monkeypatch=monkeypatch, tmp_path=tmp_path / "current",
    )

    v13_report = _compare_dataframes(
        baseline_result["v1_3_df"], current_result["v1_3_df"], label=f"v1_3_df@{cutoff_name}"
    )
    bench_report = _compare_dataframes(
        baseline_result["bench_df"], current_result["bench_df"], label=f"bench_df@{cutoff_name}"
    )
    assert v13_report["exact_match_columns"] == v13_report["compared_columns"]
    assert bench_report["exact_match_columns"] == bench_report["compared_columns"]


# =============================================================================
# Concurrency determinism: completion order must never affect the result.
# =============================================================================


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_concurrent_completion_order_does_not_affect_output(
    seed, loaded_modules, fred_wide_snapshot, yahoo_snapshot, monkeypatch, tmp_path
):
    """Runs the CURRENT (optimized, concurrent-prefetch) pipeline twice
    at the SAME cutoff with DIFFERENT per-ticker artificial delays (so
    the ThreadPoolExecutor's actual completion order differs between
    runs) and asserts byte-identical output both times."""
    _, current_module = loaded_modules
    cutoff = CUTOFFS["latest"]

    log_a: list[str] = []
    log_b: list[str] = []
    run_a = _run_pipeline(
        current_module, fred_wide=fred_wide_snapshot, yahoo_snapshot=yahoo_snapshot,
        cutoff=cutoff, monkeypatch=monkeypatch, tmp_path=tmp_path / f"order_a_{seed}",
        shuffle_delay_seed=seed, completion_log=log_a,
    )
    run_b = _run_pipeline(
        current_module, fred_wide=fred_wide_snapshot, yahoo_snapshot=yahoo_snapshot,
        cutoff=cutoff, monkeypatch=monkeypatch, tmp_path=tmp_path / f"order_b_{seed + 100}",
        shuffle_delay_seed=seed + 100, completion_log=log_b,  # different delay pattern -> different order
    )

    # The 11-ticker "@2008-01-01" batch is the one actually run through
    # `ThreadPoolExecutor` (`_prefetch_tickers_concurrently`) -- confirm
    # completion order genuinely differed between the two runs, so a
    # pass below is proof of order-independence, not a coincidence where
    # both runs happened to race in the same order.
    order_a = [entry for entry in log_a if entry.endswith("@2008-01-01")]
    order_b = [entry for entry in log_b if entry.endswith("@2008-01-01")]
    assert sorted(order_a) == sorted(order_b), "same ticker set must be fetched in both runs"
    assert order_a != order_b, (
        f"completion order did not actually differ between the two runs (seed={seed} vs {seed + 100}) "
        f"-- both raced to: {order_a}. This test only proves order-independence when the orders "
        f"genuinely differ; increase delay variance or pick different seeds."
    )

    report = _compare_dataframes(run_a["v1_3_df"], run_b["v1_3_df"], label=f"concurrency_order_seed{seed}")
    assert report["exact_match_columns"] == report["compared_columns"]
    bench_report = _compare_dataframes(
        run_a["bench_df"], run_b["bench_df"], label=f"concurrency_order_bench_seed{seed}"
    )
    assert bench_report["exact_match_columns"] == bench_report["compared_columns"]
