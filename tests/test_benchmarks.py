from __future__ import annotations

import numpy as np
import pandas as pd

from macro_regime.benchmarks import (
    MAX_CONCURRENT_BENCHMARKS,
    REGISTRY,
    compute_benchmark_metrics,
    compute_benchmark_series,
    list_ui_visible,
)


class _FakeAssetPriceClient:
    """No-network stand-in, same pattern as test_backtest.py's."""

    def __init__(self, series_by_ticker: dict[str, pd.Series]):
        self._series_by_ticker = series_by_ticker

    def get_daily_close(self, ticker: str, start: str, *, refresh_cache: bool = False) -> pd.Series:
        return self._series_by_ticker[ticker]


def _calendar(n: int, start="2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _price_series(calendar: pd.DatetimeIndex, start_price: float, daily_ret: float) -> pd.Series:
    values = start_price * (1 + daily_ret) ** np.arange(len(calendar))
    return pd.Series(values, index=calendar)


def test_registry_excludes_project_6040_from_ui():
    visible_ids = {b.id for b in list_ui_visible()}
    assert "project_6040" not in visible_ids
    assert "project_6040" in REGISTRY  # still registered internally


def test_registry_ui_visible_set_matches_spec():
    visible_ids = {b.id for b in list_ui_visible()}
    assert visible_ids == {"us_60_40", "malox", "spy", "agg"}


def test_us_60_40_is_default_selected():
    reg = REGISTRY["us_60_40"]
    assert reg.default_selected is True
    assert REGISTRY["malox"].default_selected is False


def test_max_concurrent_benchmarks_is_reasonable():
    assert 1 <= MAX_CONCURRENT_BENCHMARKS <= 5


def test_us_60_40_blended_series_monthly_rebalance():
    cal = _calendar(60)
    client = _FakeAssetPriceClient(
        {
            "SPY": _price_series(cal, 100.0, 0.001),
            "AGG": _price_series(cal, 100.0, 0.0001),
        }
    )
    series = compute_benchmark_series("us_60_40", strategy_calendar=cal, as_of=cal.max(), client=client)
    assert series.status.available
    assert len(series.returns) > 0
    assert series.turnover is not None
    assert series.turnover.iloc[0] > 0  # initial allocation counts as turnover


def test_single_asset_spy_series():
    cal = _calendar(30)
    client = _FakeAssetPriceClient({"SPY": _price_series(cal, 100.0, 0.002)})
    series = compute_benchmark_series("spy", strategy_calendar=cal, as_of=cal.max(), client=client)
    assert series.status.available
    assert series.turnover is None
    np.testing.assert_allclose(series.returns.dropna().to_numpy(), 0.002, atol=1e-9)


def test_malox_no_future_fill_and_stale_days_flagged():
    cal = _calendar(20)
    # MALOX only "publishes" on even-indexed days -- odd days are stale.
    malox_native = _price_series(cal, 20.0, 0.001)[::2]
    client = _FakeAssetPriceClient({"MALOX": malox_native})
    series = compute_benchmark_series("malox", strategy_calendar=cal, as_of=cal.max(), client=client)
    assert series.status.available
    assert series.is_stale is not None
    assert series.is_stale.sum() > 0
    # a stale (held-forward) day must show exactly 0.0 return, never a
    # fabricated/interpolated one
    stale_days = series.is_stale[series.is_stale].index
    assert (series.returns.reindex(stale_days).dropna() == 0.0).all()


def test_malox_never_backward_filled():
    cal = _calendar(10, start="2024-01-02")
    # MALOX data starts LATER than the strategy calendar.
    malox_native = _price_series(cal[3:], 20.0, 0.0)
    client = _FakeAssetPriceClient({"MALOX": malox_native})
    series = compute_benchmark_series("malox", strategy_calendar=cal, as_of=cal.max(), client=client)
    assert series.status.available
    assert (
        series.returns.index.min() >= malox_native.index.min()
    )  # never extends before MALOX's own first date


def test_unavailable_benchmark_does_not_raise():
    from macro_regime.data.asset_prices import AssetPriceApiError

    class _FailingClient:
        def get_daily_close(self, ticker, start, *, refresh_cache=False):
            raise AssetPriceApiError(f"no data for {ticker}")

    cal = _calendar(10)
    series = compute_benchmark_series(
        "malox", strategy_calendar=cal, as_of=cal.max(), client=_FailingClient()
    )
    assert series.status.available is False
    assert series.status.error is not None
    assert series.returns.empty


def test_compute_benchmark_metrics_on_empty_series_does_not_raise():
    result = compute_benchmark_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    assert result["n_observations"] == 0


def test_compute_benchmark_metrics_basic():
    cal = _calendar(300)
    returns = pd.Series(0.0005, index=cal)
    risk_free = pd.Series(0.0001, index=cal)
    m = compute_benchmark_metrics(returns, risk_free)
    assert m["n_observations"] == 300
    assert m["cagr"] > 0
    assert m["max_drawdown_daily_close"] <= 0
