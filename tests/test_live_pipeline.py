"""Tests for `macro_regime.deployment.live_pipeline`. Every external
call (fetch/build_signals/evaluate/build_regime_output_cmd/
build_v1_3_daily_artifact/build_benchmark_daily_artifact) is mocked --
this file never makes a real FRED or Yahoo Finance call, and never
depends on FRED_API_KEY being set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from macro_regime.deployment import live_pipeline as lp


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
    monkeypatch.setattr(lp, "load_config", lambda: {"regime_output": {"tradable_lag_months": 1}})

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

    def _fake_build_v1_3(*args, **kwargs):
        calls.append("build_v1_3_daily_artifact")
        return _fake_v13_df(), _FakeBacktest()

    def _fake_build_bench(*args, **kwargs):
        calls.append("build_benchmark_daily_artifact")
        return _fake_bench_df()

    monkeypatch.setattr(lp, "build_v1_3_daily_artifact", _fake_build_v1_3)
    monkeypatch.setattr(lp, "build_benchmark_daily_artifact", _fake_build_bench)
    monkeypatch.setattr(lp, "AssetPriceClient", lambda: object())

    return calls


def test_calls_every_pipeline_step_in_order(_mock_pipeline_steps):
    lp.run_live_production_pipeline()
    assert _mock_pipeline_steps == [
        "fetch",
        "build_signals",
        "evaluate",
        "build_regime_output_cmd",
        "build_v1_3_daily_artifact",
        "build_benchmark_daily_artifact",
    ]


def test_returns_dataframes_and_vix_series(_mock_pipeline_steps):
    result = lp.run_live_production_pipeline()
    assert result["v1_3_df"] is not None
    assert result["bench_df"] is not None
    assert result["vix_series"] is not None
    assert list(result["vix_series"].values) == [15.0, 16.0, 14.5]
    assert result["fetched_at"] is not None


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


def test_missing_vixcls_column_returns_none_vix_series(monkeypatch, tmp_path, _mock_pipeline_steps):
    wide_path = tmp_path / "fred_wide_no_vix.csv"
    pd.DataFrame({"OTHER": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2)).to_csv(wide_path)
    monkeypatch.setattr(lp, "WIDE_PATH", wide_path)
    result = lp.run_live_production_pipeline()
    assert result["vix_series"] is None


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
