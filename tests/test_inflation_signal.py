import numpy as np
import pandas as pd
import pytest

from macro_regime.signals.inflation import (
    cleveland_median_cpi_momentum,
    commodity_core_composite,
    core_inflation_momentum,
)


def test_core_cpi_3m_annualized_calculation():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    cpi = pd.Series([100.0, 100.3, 100.6, 101.0], index=idx)
    df = core_inflation_momentum(cpi, short_window_months=3, long_window_months=3)
    expected = (101.0 / 100.0) ** 4 - 1
    assert df["core_short_annualized"].iloc[-1] == pytest.approx(expected)


def test_unknown_label_when_insufficient_history():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    cpi = pd.Series([100.0, 100.5], index=idx)
    df = core_inflation_momentum(cpi, short_window_months=3, long_window_months=12)
    assert df["inflation_label"].iloc[-1] == "Unknown"


def test_cleveland_median_cpi_up_when_ma_rising():
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    median_cpi = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], index=idx)
    df = cleveland_median_cpi_momentum(median_cpi, ma_window_months=3, lag_months=3)
    assert df["inflation_label"].iloc[-1] == "Up"


def test_cleveland_median_cpi_down_when_ma_falling():
    idx = pd.date_range("2020-01-31", periods=10, freq="ME")
    median_cpi = pd.Series([4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0], index=idx)
    df = cleveland_median_cpi_momentum(median_cpi, ma_window_months=3, lag_months=3)
    assert df["inflation_label"].iloc[-1] == "Down"


def test_cleveland_median_cpi_unknown_on_exact_tie():
    # A perfectly flat series makes the 3-month MA identical to itself
    # 3 months ago -- change == 0, which must label Unknown, not Down/Up.
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    median_cpi = pd.Series([5.0] * 8, index=idx)
    df = cleveland_median_cpi_momentum(median_cpi, ma_window_months=3, lag_months=3)
    assert df["inflation_label"].iloc[-1] == "Unknown"


def test_cleveland_median_cpi_unknown_when_insufficient_history():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    median_cpi = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    df = cleveland_median_cpi_momentum(median_cpi, ma_window_months=3, lag_months=3)
    assert df["inflation_label"].iloc[-1] == "Unknown"


def _synthetic_monthly(seed: int, periods: int = 90) -> pd.Series:
    idx = pd.date_range("2015-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(seed)
    return idx, rng


def test_commodity_core_composite_is_2signal_only():
    # commodity_core_composite has no ISM parameter at all -- it cannot
    # produce or claim a 3-signal result under any circumstance.
    idx, rng = _synthetic_monthly(1)
    core = pd.Series(200 + np.arange(len(idx)) * 0.3 + rng.normal(scale=0.3, size=len(idx)), index=idx)
    commodity = pd.Series(100 + np.arange(len(idx)) * 0.2 + rng.normal(scale=1.0, size=len(idx)), index=idx)

    df = commodity_core_composite(core, commodity, zscore_window=24, zscore_min_periods=6)
    assert "ism_component" not in df.columns
    assert df["inflation_score"].notna().any()
    assert df["inflation_label"].isin(["Up", "Down", "Unknown"]).all()
