"""Tests for the daily/monthly separation fix in `fast_crisis.backtest`:
daily portfolio valuation must extend past the monthly macro signal's
last CLOSED month (holding that month's target forward), never silently
truncate the whole daily backtest at the monthly signal's own boundary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from macro_regime.fast_crisis.backtest import _broadcast_monthly_onto_daily


def _month_ends(labels: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(x) for x in labels])


def test_exact_month_lookup_unchanged_for_closed_months():
    monthly = pd.Series([1.0, 2.0, 3.0], index=_month_ends(["2026-04-30", "2026-05-31", "2026-06-30"]))
    daily_index = pd.bdate_range("2026-04-01", "2026-06-30")
    result = _broadcast_monthly_onto_daily(monthly, daily_index)
    assert (result.loc["2026-04-01":"2026-04-30"] == 1.0).all()
    assert (result.loc["2026-05-01":"2026-05-29"] == 2.0).all()
    assert (result.loc["2026-06-01":"2026-06-30"] == 3.0).all()


def test_partial_current_month_holds_last_closed_month_forward():
    monthly = pd.Series([1.0, 2.0, 3.0], index=_month_ends(["2026-04-30", "2026-05-31", "2026-06-30"]))
    daily_index = pd.bdate_range("2026-04-01", "2026-07-27")  # July is NOT in `monthly`
    result = _broadcast_monthly_onto_daily(monthly, daily_index)
    july_days = result.loc["2026-07-01":"2026-07-27"]
    assert (july_days == 3.0).all()  # June's (last closed month) value held forward


def test_partial_month_never_uses_a_value_from_the_future():
    """The held-forward value must be the LAST already-decided one, never
    interpolated or blended with anything from the open month."""
    monthly = pd.DataFrame({"weight": [0.1, 0.9]}, index=_month_ends(["2026-05-31", "2026-06-30"]))
    daily_index = pd.bdate_range("2026-06-01", "2026-07-15")
    result = _broadcast_monthly_onto_daily(monthly, daily_index)
    assert (result.loc["2026-07-01":"2026-07-15", "weight"] == 0.9).all()
    assert 0.9 in monthly["weight"].to_numpy()  # confirms 0.9 is June's real, already-published value


def test_dataframe_broadcast_holds_forward_too():
    monthly = pd.DataFrame({"a": [10, 20], "b": [100, 200]}, index=_month_ends(["2026-05-31", "2026-06-30"]))
    daily_index = pd.bdate_range("2026-06-01", "2026-07-10")
    result = _broadcast_monthly_onto_daily(monthly, daily_index)
    assert (result.loc["2026-07-01":"2026-07-10", "a"] == 20).all()
    assert (result.loc["2026-07-01":"2026-07-10", "b"] == 200).all()


def test_gap_before_monthly_series_start_raises():
    monthly = pd.Series([5.0], index=_month_ends(["2026-06-30"]))
    daily_index = pd.bdate_range("2026-05-01", "2026-06-30")  # May has no monthly value at all
    with pytest.raises(ValueError, match="No monthly value available"):
        _broadcast_monthly_onto_daily(monthly, daily_index)
