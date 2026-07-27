import numpy as np
import pandas as pd

from macro_regime.config import DEFAULT_CONFIG_PATH, load_config
from macro_regime.data.availability import add_monthly_availability
from macro_regime.evaluation.classification import naive_baseline
from macro_regime.evaluation.lead_lag import growth_forward_target
from macro_regime.utils.stats import rolling_zscore


def test_rolling_zscore_has_no_lookahead():
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    values = np.random.default_rng(0).normal(size=200)
    series = pd.Series(values, index=idx)

    full = rolling_zscore(series, window=120, min_periods=60)
    truncated = rolling_zscore(series.iloc[:150], window=120, min_periods=60)

    pd.testing.assert_series_equal(full.iloc[:150], truncated, check_names=False)


def test_release_lag_applied_when_release_date_unknown():
    config = load_config(DEFAULT_CONFIG_PATH)
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"])})
    out = add_monthly_availability(df, "date", config)
    lag = config["lookahead"]["monthly_macro_lag_months"]
    expected = pd.Timestamp("2020-01-31") + pd.DateOffset(months=lag)
    assert out["availability_date"].iloc[0] == expected


def test_effective_date_is_first_trading_day_of_following_month():
    config = load_config(DEFAULT_CONFIG_PATH)
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"])})
    out = add_monthly_availability(df, "date", config)
    assert out["effective_date"].iloc[0] == pd.Timestamp("2020-03-01")


def test_forward_target_creation_has_no_alignment_errors():
    idx = pd.date_range("2000-01-31", periods=20, freq="ME")
    indpro = pd.Series(range(100, 120), index=idx, dtype=float)
    target = growth_forward_target(indpro, horizon_months=3)
    assert target.index.equals(indpro.index)
    assert target["actual_label"].iloc[0] == "Up"
    assert target["actual_label"].iloc[-1] == "Unknown"


def test_model_compared_against_always_up_baseline():
    idx = pd.date_range("2000-01-31", periods=10, freq="ME")
    actual = pd.Series(["Up"] * 7 + ["Down"] * 3, index=idx)
    baseline = naive_baseline(actual, "always_up")
    assert baseline.accuracy == 0.7
