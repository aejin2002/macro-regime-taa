import numpy as np
import pandas as pd

from macro_regime import cli as cli_module


def _config() -> dict:
    return {
        "zscore": {"rolling_window_months": 24, "min_periods_months": 6},
        "growth_models": {
            "model_a_cli": {"us_series": "USALOLITOAASTSAM", "primary_window_months": 3},
            "model_b_fred_minimal": {
                "claims_series": "ICSA",
                "claims_change_months": 3,
                "permits_series": "PERMIT",
                "permits_change_months": 6,
                "activity_series": "CFNAIMA3",
            },
            "model_c_simple_two_signal": {
                "claims_series": "ICSA",
                "ism_new_orders_change_months": 3,
            },
            "model_d_amtmno_claims": {
                "amtmno_series": "AMTMNO",
                "amtmno_change_months": 3,
                "claims_series": "ICSA",
                "claims_change_months": 3,
            },
        },
        "inflation_models": {
            "model_a_realized_core": {
                "core_series": "CPILFESL",
                "short_window_months": 3,
                "long_window_months": 12,
            },
            "model_b_leading_composite": {
                "breakeven_series": "T5YIE",
                "breakeven_change_months": 3,
                "core_series": "CPILFESL",
            },
            "model_c_cleveland_median_cpi": {
                "median_cpi_series": "MEDCPIM158SFRBCLE",
                "ma_window_months": 3,
                "lag_months": 3,
            },
            "model_d_commodity_core_aux": {
                "commodity_series": "PALLFNFINDEXM",
                "commodity_change_months": 3,
                "core_series": "CPILFESL",
            },
        },
        "ism_new_orders_csv": "data/external/does_not_exist_new_orders.csv",
        "ism_prices_paid_csv": "data/external/does_not_exist_prices_paid.csv",
    }


def _synthetic_wide() -> pd.DataFrame:
    # Mirrors real fred_wide.csv conventions: native monthly FRED series are
    # stamped first-of-month, ICSA is weekly, T5YIE is a daily business-day
    # series -- each with its own index, combined into one wide frame.
    months = pd.date_range("2010-01-01", periods=90, freq="MS")
    n = pd.RangeIndex(len(months)).to_numpy().astype(float)
    weekly = pd.date_range("2010-01-01", periods=400, freq="W")
    daily = pd.date_range("2010-01-01", periods=1800, freq="B")

    # A little noise is required: growth_model_b_fred_minimal's components
    # are rolling z-scores, which the codebase deliberately returns as NaN
    # when the rolling window has zero variance (see docs/methodology.md,
    # "Standardization") -- a perfectly linear/deterministic series would
    # make PERMIT's z-score NaN everywhere and defeat this test.
    rng = np.random.default_rng(0)
    series = {
        "USALOLITOAASTSAM": pd.Series(100 + n * 0.05, index=months),
        "AMTMNO": pd.Series(50000 + n * 100.0 + rng.normal(scale=200.0, size=len(months)), index=months),
        "PERMIT": pd.Series(1000 + n * 2.0 + rng.normal(scale=5.0, size=len(months)), index=months),
        "CFNAIMA3": pd.Series(((n.astype(int) % 5) - 2).astype(float), index=months),
        "CPILFESL": pd.Series(200 + n * 0.3 + rng.normal(scale=0.5, size=len(months)), index=months),
        "ICSA": pd.Series(range(len(weekly)), index=weekly, dtype=float),
        "T5YIE": pd.Series(
            np.arange(len(daily)) * 0.001 + 2.0 + rng.normal(scale=0.05, size=len(daily)), index=daily
        ),
    }
    return pd.concat(series, axis=1)


def test_kr_cli_growth_model_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "PROCESSED_DIR", tmp_path)
    growth = cli_module._growth_signals(_synthetic_wide(), _config(), "all")
    assert "growth_model_kr_cli" not in growth
    assert set(growth) == {
        "growth_model_us_cli",
        "growth_model_fred_minimal",
        "growth_model_amtmno_claims",
    }


def test_growth_and_inflation_outputs_share_one_row_per_month(monkeypatch, tmp_path):
    # Regression test for the pipeline-assembly-level version of the
    # month-end/first-of-month bug: growth_model_us_cli is a single-series
    # model that never gets internally normalized, while
    # growth_model_fred_minimal joins multiple sources and is normalized to
    # month-end inside growth_model_b_fred_minimal. Combining the two
    # without normalizing every model's *output* the same way silently
    # doubles the row count and makes evaluation see n_obs=0.
    monkeypatch.setattr(cli_module, "PROCESSED_DIR", tmp_path)
    wide = _synthetic_wide()
    config = _config()

    growth = cli_module._growth_signals(wide, config, "all")
    inflation = cli_module._inflation_signals(wide, config, "all")
    combined = pd.concat({**growth, **inflation}, axis=1)

    month_periods = combined.index.to_period("M")
    assert month_periods.is_unique, "each calendar month must appear as a single row"

    both_classified = (combined["growth_model_us_cli"] != "Unknown") & (
        combined["growth_model_fred_minimal"] != "Unknown"
    )
    assert both_classified.sum() > 0, "US CLI and FRED-minimal must overlap on real dates"
