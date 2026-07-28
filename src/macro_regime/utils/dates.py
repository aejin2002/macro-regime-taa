"""Date helpers shared across data, signal, and evaluation code.

All series are normalized to month-end timestamps so that growth and
inflation signals -- which are fundamentally monthly -- can be joined
without ambiguity, regardless of the native frequency (daily, weekly,
monthly) of the underlying FRED series.
"""

from __future__ import annotations

import pandas as pd


def to_month_end(dates: pd.Series) -> pd.Series:
    """Floor timestamps to their calendar month, then snap to month end."""
    periods = pd.to_datetime(dates).dt.to_period("M")
    return pd.Series(periods.dt.to_timestamp("M"), index=dates.index)


def resample_to_monthly(series: pd.Series, how: str = "last") -> pd.Series:
    """Resample a date-indexed series (daily/weekly/monthly) to month-end.

    how="last" takes the last observation in the month (appropriate for
    stock-like series such as indices or rates). how="mean" averages
    (appropriate for series like weekly claims when a monthly value is
    wanted without discarding intra-month information).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("resample_to_monthly requires a DatetimeIndex")
    if how == "last":
        return series.resample("ME").last()
    if how == "mean":
        return series.resample("ME").mean()
    raise ValueError(f"Unsupported resample method: {how}")


def normalize_month_end_index(series: pd.Series) -> pd.Series:
    """Re-stamp a monthly series' index to month-end, regardless of whether
    the source used first-of-month (FRED's usual convention for series like
    PERMIT/CFNAIMA3) or already-month-end labels (e.g. after resampling).

    Combining monthly series from different sources without this raises a
    silent bug: two economically-identical months (e.g. "2020-01-01" from
    one series and "2020-01-31" from another) are treated as different
    index labels, so joins/`pd.DataFrame({...})` produce all-NaN rows
    instead of erroring.
    """
    idx = pd.to_datetime(series.index).to_period("M").to_timestamp("M")
    return series.set_axis(idx)


def next_trading_day_of_month(month_end: pd.Timestamp) -> pd.Timestamp:
    """Approximate the first calendar day of the month after `month_end`.

    This is a calendar-day approximation (not an exchange trading calendar).
    It is used to compute conservative effective_date values for signals
    that must not be used before they could plausibly have been observed.
    """
    first_of_next_month = (month_end + pd.offsets.MonthBegin(1)).normalize()
    return first_of_next_month


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def drop_incomplete_trailing_month(series: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Drop a still-in-progress trailing month from a month-end-indexed
    series. `resample("ME")` (via `resample_to_monthly`) labels a
    still-in-progress month with a *future* month-end date, using
    whatever partial data exists so far -- e.g. if `as_of` is
    2026-07-27, a July row stamped "2026-07-31" reflects an incomplete
    month, not a real completed one, and must be excluded rather than
    mistaken for a finished month. Only ever trims the single most
    recent row when it does."""
    return series.loc[series.index <= as_of]
