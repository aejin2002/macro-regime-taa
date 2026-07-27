import pandas as pd
import pytest

from macro_regime.signals.inflation import core_inflation_momentum


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
