import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from macro_regime.data.asset_prices import AssetPriceApiError, AssetPriceClient
from macro_regime.data.cache import FileCache


class _FakeTicker:
    """Stand-in for `yfinance.Ticker`: `.history(...)` raises
    `YFRateLimitError` for the first `fail_times` calls (tracked via the
    shared `calls` dict, since yfinance instantiates a new `Ticker` object
    per call), then returns a small fixed price history."""

    def __init__(self, ticker: str, *, calls: dict, fail_times: int) -> None:
        self.ticker = ticker
        self._calls = calls
        self._fail_times = fail_times

    def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
        self._calls["n"] += 1
        if self._calls["n"] <= self._fail_times:
            raise YFRateLimitError()
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)


def _client(tmp_path, **kwargs) -> AssetPriceClient:
    return AssetPriceClient(
        cache=FileCache(tmp_path),
        max_retries=kwargs.pop("max_retries", 4),
        backoff_seconds=kwargs.pop("backoff_seconds", 1.0),
        min_request_interval_seconds=kwargs.pop("min_request_interval_seconds", 0.0),
        **kwargs,
    )


def test_get_daily_close_retries_on_rate_limit_then_succeeds(monkeypatch, tmp_path):
    calls = {"n": 0}
    monkeypatch.setattr(
        "macro_regime.data.asset_prices.yf.Ticker",
        lambda ticker: _FakeTicker(ticker, calls=calls, fail_times=2),
    )
    sleeps: list[float] = []
    monkeypatch.setattr("macro_regime.data.asset_prices.time.sleep", lambda s: sleeps.append(s))

    client = _client(tmp_path, max_retries=5)
    series = client.get_daily_close("FAKE", "2020-01-01")

    assert calls["n"] == 3  # two rate-limited attempts, third succeeds
    assert list(series.values) == [100.0, 101.0]
    assert len(sleeps) == 2  # one backoff sleep per failed attempt before success


def test_get_daily_close_raises_after_exhausting_retries(monkeypatch, tmp_path):
    calls = {"n": 0}
    monkeypatch.setattr(
        "macro_regime.data.asset_prices.yf.Ticker",
        lambda ticker: _FakeTicker(ticker, calls=calls, fail_times=999),
    )
    monkeypatch.setattr("macro_regime.data.asset_prices.time.sleep", lambda s: None)

    client = _client(tmp_path, max_retries=3)
    with pytest.raises(AssetPriceApiError, match="rate-limited"):
        client.get_daily_close("FAKE", "2020-01-01")

    assert calls["n"] == 3  # tried exactly max_retries times, no more


def test_get_daily_close_does_not_retry_non_rate_limit_errors(monkeypatch, tmp_path):
    calls = {"n": 0}

    class _FakeTickerOtherError:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
            calls["n"] += 1
            raise ValueError("some unrelated failure")

    monkeypatch.setattr("macro_regime.data.asset_prices.yf.Ticker", _FakeTickerOtherError)
    monkeypatch.setattr("macro_regime.data.asset_prices.time.sleep", lambda s: None)

    client = _client(tmp_path, max_retries=5)
    with pytest.raises(ValueError, match="some unrelated failure"):
        client.get_daily_close("FAKE", "2020-01-01")

    assert calls["n"] == 1  # not retried


def test_get_daily_close_uses_cache_without_touching_network(monkeypatch, tmp_path):
    calls = {"n": 0}
    monkeypatch.setattr(
        "macro_regime.data.asset_prices.yf.Ticker",
        lambda ticker: _FakeTicker(ticker, calls=calls, fail_times=0),
    )
    monkeypatch.setattr("macro_regime.data.asset_prices.time.sleep", lambda s: None)

    client = _client(tmp_path)
    first = client.get_daily_close("FAKE", "2020-01-01")
    second = client.get_daily_close("FAKE", "2020-01-01")

    pd.testing.assert_series_equal(first, second)
    assert calls["n"] == 1  # second call served from cache, no new network call
