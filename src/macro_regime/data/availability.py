"""Look-ahead-bias bookkeeping.

Every signal frame carries five date columns so that "when could this
number actually have been used" is always explicit and auditable:

    observation_date  -- the period the data describes (e.g. month-end)
    release_date      -- when the statistical agency actually published it,
                          if known; otherwise NaT
    availability_date -- release_date if known, else observation_date plus
                          the configured conservative lag
    signal_date       -- the date the signal is attributed to (== observation_date)
    effective_date     -- the first date the signal may be used in a backtest
                          (first trading day of the month after availability_date)

Series without a real release calendar (most FRED macro series in this
build) fall back to `lookahead.monthly_macro_lag_months` from
config/default.yaml. Market/daily series use `lookahead.daily_market_lag_days`.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.utils.dates import next_trading_day_of_month


def add_monthly_availability(
    df: pd.DataFrame,
    date_col: str,
    config: dict,
    *,
    release_date_col: str | None = None,
) -> pd.DataFrame:
    """Add observation/release/availability/signal/effective date columns
    to a monthly-frequency DataFrame.

    `release_date_col`, if present in `df`, is used as the authoritative
    release date wherever it is not null; the configured lag fills the rest.
    """
    out = df.copy()
    lag_months = config["lookahead"]["monthly_macro_lag_months"]

    observation_date = pd.to_datetime(out[date_col])
    out["observation_date"] = observation_date
    out["signal_date"] = observation_date

    fallback_availability = observation_date + pd.DateOffset(months=lag_months)
    if release_date_col and release_date_col in out.columns:
        release_date = pd.to_datetime(out[release_date_col])
        out["release_date"] = release_date
        out["availability_date"] = release_date.fillna(fallback_availability)
    else:
        out["release_date"] = pd.NaT
        out["availability_date"] = fallback_availability

    out["effective_date"] = out["availability_date"].apply(
        lambda ts: next_trading_day_of_month(pd.Timestamp(ts).to_period("M").to_timestamp("M"))
        if pd.notna(ts)
        else pd.NaT
    )
    return out


def add_daily_availability(df: pd.DataFrame, date_col: str, config: dict) -> pd.DataFrame:
    """Add availability columns to daily/market data using the configured
    daily market lag (no release-date ambiguity for market data)."""
    out = df.copy()
    lag_days = config["lookahead"]["daily_market_lag_days"]

    observation_date = pd.to_datetime(out[date_col])
    out["observation_date"] = observation_date
    out["signal_date"] = observation_date
    out["release_date"] = observation_date
    out["availability_date"] = observation_date
    out["effective_date"] = observation_date + pd.Timedelta(days=lag_days)
    return out
