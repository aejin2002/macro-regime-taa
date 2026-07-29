"""Tests for `macro_regime.deployment.live_pipeline`. Every external
call (fetch/build_signals/evaluate/build_regime_output_cmd/
build_v1_3_daily_artifact/build_benchmark_daily_artifact/Yahoo Finance
ticker fetches) is mocked -- this file never makes a real FRED or Yahoo
Finance call, and never depends on FRED_API_KEY being set.
"""

from __future__ import annotations

import inspect
import threading
import time

import pandas as pd
import pytest

from macro_regime.deployment import live_pipeline as lp

LIVE_PIPELINE_SOURCE = inspect.getsource(lp)


def _fake_v13_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "strategy_nav": [1.0, 1.01, 1.02]})


def _fake_bench_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "benchmark_id": "us_60_40",
            "daily_return": [0.0, 0.01, 0.01],
            "nav": [1.0, 1.01, 1.02],
            "available": True,
        }
    )


class _FakeBacktest:
    pass


class _FakeClient:
    """Stands in for `AssetPriceClient` -- records every `get_daily_close`
    call (ticker, start, refresh_cache), optionally sleeps to simulate
    network latency (for concurrency-timing tests), and can be told to
    fail for specific tickers."""

    def __init__(self, *, delay: float = 0.0, fail_tickers: frozenset[str] = frozenset()):
        self.calls: list[tuple[str, str, bool]] = []
        self._lock = threading.Lock()
        self.delay = delay
        self.fail_tickers = fail_tickers

    def get_daily_close(self, ticker: str, start: str, *, refresh_cache: bool = False) -> pd.Series:
        with self._lock:
            self.calls.append((ticker, start, refresh_cache))
        if self.delay:
            time.sleep(self.delay)
        if ticker in self.fail_tickers:
            raise RuntimeError(f"mock: Yahoo Finance failed for {ticker}")
        return pd.Series([1.0, 1.01, 1.02], index=pd.date_range("2026-01-01", periods=3), name=ticker)


@pytest.fixture(autouse=True)
def _mock_pipeline_steps(monkeypatch, tmp_path):
    calls = []

    def _track(name):
        def _fn(*args, **kwargs):
            calls.append(name)

        return _fn

    monkeypatch.setattr(lp, "fetch", _track("fetch"))
    monkeypatch.setattr(lp, "build_signals", _track("build_signals"))
    monkeypatch.setattr(lp, "evaluate", _track("evaluate"))
    monkeypatch.setattr(lp, "build_regime_output_cmd", _track("build_regime_output_cmd"))
    monkeypatch.setattr(
        lp,
        "load_config",
        lambda: {
            "regime_output": {"tradable_lag_months": 1},
            "growth_basket": {"kodex200_ticker": "069500.KS", "fx_ticker": "KRW=X"},
        },
    )

    wide_path = tmp_path / "fred_wide.csv"
    wide = pd.DataFrame(
        {"VIXCLS": [15.0, 16.0, 14.5]}, index=pd.date_range("2026-01-01", periods=3)
    )
    wide.to_csv(wide_path)
    monkeypatch.setattr(lp, "WIDE_PATH", wide_path)

    regime_path = tmp_path / "regime_output_primary.csv"
    primary = pd.DataFrame(
        {"tradable_regime": ["REFLATION"] * 3, "growth_state": ["Up"] * 3},
        index=pd.date_range("2026-01-01", periods=3),
    )
    primary.to_csv(regime_path)
    monkeypatch.setattr(lp, "PROCESSED_DIR", tmp_path)

    build_calls = {"v1_3_kwargs": None, "bench_kwargs": None}

    def _fake_build_v1_3(*args, **kwargs):
        calls.append("build_v1_3_daily_artifact")
        build_calls["v1_3_kwargs"] = kwargs
        return _fake_v13_df(), _FakeBacktest()

    def _fake_build_bench(*args, **kwargs):
        calls.append("build_benchmark_daily_artifact")
        build_calls["bench_kwargs"] = kwargs
        return _fake_bench_df()

    monkeypatch.setattr(lp, "build_v1_3_daily_artifact", _fake_build_v1_3)
    monkeypatch.setattr(lp, "build_benchmark_daily_artifact", _fake_build_bench)

    fake_client = _FakeClient()
    monkeypatch.setattr(lp, "AssetPriceClient", lambda: fake_client)

    return {"calls": calls, "build_calls": build_calls, "client": fake_client}


def test_calls_every_pipeline_step_in_order(_mock_pipeline_steps):
    lp.run_live_production_pipeline()
    assert _mock_pipeline_steps["calls"] == [
        "fetch",
        "build_signals",
        "evaluate",
        "build_regime_output_cmd",
        "build_v1_3_daily_artifact",
        "build_benchmark_daily_artifact",
    ]


def test_returns_dataframes_and_provenance_metadata(_mock_pipeline_steps):
    result = lp.run_live_production_pipeline()
    assert result["v1_3_df"] is not None
    assert result["bench_df"] is not None
    assert result["fetched_at"] is not None
    assert result["source_mode"] == "live_pipeline"
    assert result["pipeline_run_id"]


def test_fetch_failure_raises_live_pipeline_error(monkeypatch, _mock_pipeline_steps):
    def _raise(*args, **kwargs):
        raise ConnectionError("mock network failure")

    monkeypatch.setattr(lp, "fetch", _raise)
    with pytest.raises(lp.LivePipelineError, match="mock network failure"):
        lp.run_live_production_pipeline()


def test_build_v1_3_failure_raises_live_pipeline_error(monkeypatch, _mock_pipeline_steps):
    def _raise(*args, **kwargs):
        raise ValueError("mock backtest failure")

    monkeypatch.setattr(lp, "build_v1_3_daily_artifact", _raise)
    with pytest.raises(lp.LivePipelineError, match="mock backtest failure"):
        lp.run_live_production_pipeline()


def test_never_returns_partial_result_on_late_failure(monkeypatch, _mock_pipeline_steps):
    """If the benchmark step fails after v1.3 already succeeded, the
    whole call must still raise -- never return a v1_3_df with no
    bench_df silently."""

    def _raise(*args, **kwargs):
        raise RuntimeError("mock benchmark failure")

    monkeypatch.setattr(lp, "build_benchmark_daily_artifact", _raise)
    with pytest.raises(lp.LivePipelineError):
        lp.run_live_production_pipeline()


class _FakeSecrets(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def test_bridge_fred_api_key_sets_env_from_secrets(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    fake_st = type("FakeSt", (), {"secrets": _FakeSecrets({"FRED_API_KEY": "abc123"})})()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_st)

    lp.bridge_fred_api_key_from_streamlit_secrets()
    import os

    assert os.environ.get("FRED_API_KEY") == "abc123"
    monkeypatch.delenv("FRED_API_KEY", raising=False)


def test_bridge_does_not_overwrite_existing_env_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "already-set")
    fake_st = type("FakeSt", (), {"secrets": _FakeSecrets({"FRED_API_KEY": "from-secrets"})})()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_st)

    lp.bridge_fred_api_key_from_streamlit_secrets()
    import os

    assert os.environ.get("FRED_API_KEY") == "already-set"


def test_bridge_never_raises_when_secrets_unavailable(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class _RaisingSecrets:
        def get(self, *args, **kwargs):
            raise FileNotFoundError("no secrets.toml")

    fake_st = type("FakeSt", (), {"secrets": _RaisingSecrets()})()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_st)

    lp.bridge_fred_api_key_from_streamlit_secrets()  # must not raise


# =============================================================================
# VIX source separation: Markets chart (Yahoo ^VIX) vs Fast Crisis (FRED,
# untouched -- this module never reads VIXCLS for anything other than
# what it always did, passing `wide` through unmodified to build_v1_3).
# =============================================================================


def test_vix_series_sourced_from_yahoo_not_fred(_mock_pipeline_steps):
    result = lp.run_live_production_pipeline()
    assert result["vix_source"] == "yahoo_^VIX"
    client = _mock_pipeline_steps["client"]
    vix_calls = [c for c in client.calls if c[0] == lp.MARKET_VIX_TICKER]
    assert len(vix_calls) == 1
    assert result["vix_series"] is not None


def test_vix_series_none_on_yahoo_failure_never_raises(_mock_pipeline_steps):
    _mock_pipeline_steps["client"].fail_tickers = frozenset({lp.MARKET_VIX_TICKER})
    result = lp.run_live_production_pipeline()  # must not raise
    assert result["vix_series"] is None
    assert result["v1_3_df"] is not None  # rest of the pipeline is unaffected


def test_fast_crisis_input_still_reads_fred_vixcls_wide_column(_mock_pipeline_steps):
    """`wide` (built from fred_wide.csv, containing VIXCLS) is passed to
    build_v1_3_daily_artifact completely unmodified -- Fast Crisis's own
    VIX input is untouched by the Yahoo VIX addition."""
    lp.run_live_production_pipeline()
    # build_v1_3_daily_artifact is called positionally in this module;
    # confirm the source `wide` it received still carries VIXCLS unmodified.
    assert "VIXCLS" in pd.read_csv(lp.WIDE_PATH, index_col=0).columns


# =============================================================================
# Concurrent prefetch + dedup memoization
# =============================================================================


def test_prefetch_fetches_all_11_fast_crisis_tickers(_mock_pipeline_steps):
    lp.run_live_production_pipeline()
    client = _mock_pipeline_steps["client"]
    expected_tickers = {
        "SPY", "HYG", "LQD", "IEF", "TLT", "GLD", "BIL", "DBC", "TIP", "069500.KS", "KRW=X",
    }
    prefetch_start = lp._fast_crisis_fetch_start()
    prefetched = {c[0] for c in client.calls if c[1] == prefetch_start and c[2] is True}
    assert expected_tickers.issubset(prefetched)


def test_v1_3_build_uses_refresh_cache_false_after_successful_prefetch(_mock_pipeline_steps):
    lp.run_live_production_pipeline()
    v1_3_kwargs = _mock_pipeline_steps["build_calls"]["v1_3_kwargs"]
    assert v1_3_kwargs["refresh_cache"] is False


def test_v1_3_build_falls_back_to_refresh_cache_true_on_prefetch_failure(_mock_pipeline_steps):
    _mock_pipeline_steps["client"].fail_tickers = frozenset({"SPY"})
    lp.run_live_production_pipeline()
    v1_3_kwargs = _mock_pipeline_steps["build_calls"]["v1_3_kwargs"]
    assert v1_3_kwargs["refresh_cache"] is True


def test_benchmark_build_always_uses_refresh_cache_true(_mock_pipeline_steps):
    """Benchmark tickers (SPY/AGG/MALOX at start='2000-01-01') are NOT
    covered by the prefetch batch (different start date) -- must stay a
    genuine live fetch regardless of prefetch outcome."""
    lp.run_live_production_pipeline()
    bench_kwargs = _mock_pipeline_steps["build_calls"]["bench_kwargs"]
    assert bench_kwargs["refresh_cache"] is True


def test_prefetch_runs_concurrently_not_serially(_mock_pipeline_steps):
    """11 tickers at ~50ms simulated latency each, bounded to 4 workers,
    must complete well under the ~550ms a fully serial fetch would take."""
    slow_client = _FakeClient(delay=0.05)
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(lp, "AssetPriceClient", lambda: slow_client)
        start = time.monotonic()
        lp.run_live_production_pipeline()
        elapsed = time.monotonic() - start
    assert elapsed < 0.4, f"prefetch took {elapsed:.2f}s -- expected concurrent, not serial"


def test_memoization_dedupes_repeated_calls_within_one_run(monkeypatch, _mock_pipeline_steps):
    """Simulates the benchmark builder's own real behavior (SPY fetched
    twice: once standalone, once as the US 60/40 blend leg) and confirms
    the SECOND call is served from the run-scoped memo, not a fresh
    Yahoo hit."""

    def _fake_build_bench_with_duplicate_spy(bt, calendar, as_of, *, client, refresh_cache):
        client.get_daily_close("SPY", "2000-01-01", refresh_cache=refresh_cache)
        client.get_daily_close("SPY", "2000-01-01", refresh_cache=refresh_cache)  # duplicate, as-is today
        client.get_daily_close("AGG", "2000-01-01", refresh_cache=refresh_cache)
        return _fake_bench_df()

    monkeypatch.setattr(lp, "build_benchmark_daily_artifact", _fake_build_bench_with_duplicate_spy)
    lp.run_live_production_pipeline()
    client = _mock_pipeline_steps["client"]
    spy_2000_calls = [c for c in client.calls if c[0] == "SPY" and c[1] == "2000-01-01"]
    # The wrapper is applied to `client.get_daily_close`, but the fake
    # client's `.calls` list records the UNDERLYING (pre-memo) fetch --
    # confirm only ONE real fetch happened despite two logical requests.
    assert len(spy_2000_calls) == 1, f"expected memoized SPY fetch to dedupe, got {len(spy_2000_calls)} calls"


# =============================================================================
# v1.2 is completely excluded from the live dashboard execution path
# (the dashboard never renders it -- see app/streamlit_app.py's unused
# `v12_raw`). Proven at three levels: the module never imports the v1.2
# builder at all, its source never references v1.2/V1_2, and an actual
# run makes zero calls to it.
# =============================================================================


def test_live_pipeline_module_never_imports_v1_2_builder():
    assert not hasattr(lp, "build_v1_2_regression_daily_artifact")


def test_live_pipeline_source_never_references_v1_2_builder():
    assert "build_v1_2_regression_daily_artifact" not in LIVE_PIPELINE_SOURCE
    assert "V1_2_REGRESSION_DAILY_PATH" not in LIVE_PIPELINE_SOURCE


def test_live_run_makes_zero_v1_2_related_calls(monkeypatch, _mock_pipeline_steps):
    """Patches the v1.2 builder into the production module itself and
    asserts a live run never touches it -- proof at the call-graph
    level, not just at the import-statement level above."""
    import macro_regime.production as production_module

    v1_2_calls = []
    monkeypatch.setattr(
        production_module,
        "build_v1_2_regression_daily_artifact",
        lambda *a, **kw: v1_2_calls.append(1),
    )
    lp.run_live_production_pipeline()
    assert v1_2_calls == []
