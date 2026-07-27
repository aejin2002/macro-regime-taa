import pandas as pd
import pytest

from macro_regime.signals.growth import (
    cli_diagnostics,
    growth_model_b_fred_minimal,
    growth_model_c_simple_two_signal,
)


def test_cli_6m_change_calculation():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    cli = pd.Series([100, 101, 102, 103, 104, 103, 102, 101], index=idx, dtype=float)
    diag = cli_diagnostics(cli)
    assert diag["chg_6m"].iloc[6] == pytest.approx(102 - 100)


def test_claims_component_sign_is_flipped():
    idx = pd.date_range("2015-01-04", periods=200, freq="W")
    # steadily falling claims => improving labor market => should push growth up
    claims = pd.Series(range(200, 0, -1), index=idx, dtype=float)
    permits_idx = pd.date_range("2015-01-31", periods=90, freq="ME")
    permits = pd.Series(100.0, index=permits_idx)
    cfnai = pd.Series(0.0, index=permits_idx)

    df = growth_model_b_fred_minimal(claims, permits, cfnai, zscore_window=24, zscore_min_periods=6)
    assert df["claims_component"].dropna().iloc[-1] > 0


def test_growth_model_c_disabled_without_ism_csv():
    idx = pd.date_range("2015-01-04", periods=50, freq="W")
    claims = pd.Series(1.0, index=idx)
    result = growth_model_c_simple_two_signal(None, claims)
    assert result is None
