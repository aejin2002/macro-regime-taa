import numpy as np
import pandas as pd
import pytest

from macro_regime.signals.growth import (
    cli_diagnostics,
    growth_model_b_fred_minimal,
    growth_model_c_simple_two_signal,
    growth_model_d_amtmno_claims,
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


def test_growth_model_b_aligns_first_of_month_and_month_end_conventions():
    # FRED stamps PERMIT/CFNAIMA3 first-of-month (e.g. "2015-01-01"), while
    # resampling weekly ICSA to monthly produces month-end labels
    # (e.g. "2015-01-31"). Without normalizing both to the same convention,
    # the three components never share an index and growth_score is all-NaN.
    idx = pd.date_range("2015-01-04", periods=400, freq="W")
    claims = pd.Series(range(400, 0, -1), index=idx, dtype=float)
    permits_idx = pd.date_range("2015-01-01", periods=90, freq="MS")
    rng = np.random.default_rng(0)
    permits = pd.Series(100.0 + rng.normal(size=len(permits_idx)), index=permits_idx)
    cfnai = pd.Series(rng.normal(size=len(permits_idx)), index=permits_idx)

    df = growth_model_b_fred_minimal(claims, permits, cfnai, zscore_window=24, zscore_min_periods=6)
    assert df["growth_score"].notna().sum() > 0


def test_amtmno_claims_claims_component_sign_is_flipped():
    idx = pd.date_range("2015-01-04", periods=200, freq="W")
    # steadily falling claims => improving labor market => should push growth up
    claims = pd.Series(range(200, 0, -1), index=idx, dtype=float)
    amtmno_idx = pd.date_range("2015-01-31", periods=90, freq="ME")
    amtmno = pd.Series(100.0, index=amtmno_idx)

    df = growth_model_d_amtmno_claims(amtmno, claims, zscore_window=24, zscore_min_periods=6)
    assert df["claims_component"].dropna().iloc[-1] > 0


def test_amtmno_claims_aligns_first_of_month_and_month_end_conventions():
    # Same class of bug as growth_model_b_fred_minimal: AMTMNO is FRED-native
    # first-of-month, while claims resampled from weekly lands on month-end.
    # Without normalizing both, the two components never share an index and
    # growth_score is all-NaN.
    idx = pd.date_range("2015-01-04", periods=400, freq="W")
    claims = pd.Series(range(400, 0, -1), index=idx, dtype=float)
    amtmno_idx = pd.date_range("2015-01-01", periods=90, freq="MS")
    rng = np.random.default_rng(0)
    amtmno = pd.Series(1000.0 + rng.normal(size=len(amtmno_idx)), index=amtmno_idx)

    df = growth_model_d_amtmno_claims(amtmno, claims, zscore_window=24, zscore_min_periods=6)
    assert df["growth_score"].notna().sum() > 0
