"""Inflation direction models.

Model C (Cleveland Fed Inflation Nowcast) is a stub: no official,
reproducible historical vintage file was located for this build (see
docs/methodology.md), so it produces no backtest output. It is included
only so the input schema and activation path are documented and ready.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.utils.dates import normalize_month_end_index, resample_to_monthly
from macro_regime.utils.stats import (
    annualize_from_periods,
    diff_n,
    pct_change_n,
    rolling_zscore,
    sign_label,
)

UP, DOWN, UNKNOWN = "Up", "Down", "Unknown"


# ---------------------------------------------------------------------------
# Model A: Realized Core Inflation Momentum
# ---------------------------------------------------------------------------


def core_inflation_momentum(
    core_index_monthly: pd.Series,
    *,
    short_window_months: int = 3,
    long_window_months: int = 12,
) -> pd.DataFrame:
    """core_3m_annualized - core_12m, per the project's Inflation Model A definition:

        core_3m_annualized = (core_t / core_t-3) ** 4 - 1
        core_12m            = core_t / core_t-12 - 1
        signal_raw           = core_3m_annualized - core_12m
    """
    short_growth = pct_change_n(core_index_monthly, short_window_months)
    periods_per_year = 12 / short_window_months
    core_short_annualized = annualize_from_periods(short_growth, periods_per_year)
    core_long = pct_change_n(core_index_monthly, long_window_months)

    out = pd.DataFrame(index=core_index_monthly.index)
    out["core_short_annualized"] = core_short_annualized
    out["core_long"] = core_long
    out["signal_raw"] = core_short_annualized - core_long
    out["inflation_label"] = out["signal_raw"].apply(
        lambda x: sign_label(x, up=UP, down=DOWN, unknown=UNKNOWN)
    )
    return out


# ---------------------------------------------------------------------------
# Model B: Leading Inflation Composite
# ---------------------------------------------------------------------------


def leading_inflation_composite(
    core_index_monthly: pd.Series,
    breakeven_5y_daily_or_monthly: pd.Series,
    ism_prices_paid_monthly: pd.Series | None,
    *,
    breakeven_change_months: int = 3,
    core_short_window_months: int = 3,
    core_long_window_months: int = 12,
    zscore_window: int = 120,
    zscore_min_periods: int = 60,
) -> pd.DataFrame:
    """mean(z(change_3m(ISM Prices Paid)), z(change_3m(T5YIE)), z(core momentum)).

    Falls back to the 2-signal `inflation_score_no_ism` variant (breakeven +
    core momentum only) when ISM Prices Paid is unavailable; both columns
    are always present in the output but the ISM-inclusive one is all-NaN
    if ISM data was not supplied.
    """
    core_index_monthly = normalize_month_end_index(core_index_monthly)
    momentum = core_inflation_momentum(
        core_index_monthly,
        short_window_months=core_short_window_months,
        long_window_months=core_long_window_months,
    )
    core_component = rolling_zscore(momentum["signal_raw"], zscore_window, zscore_min_periods)

    # T5YIE is a daily series; it must be resampled to one observation per
    # month before diff_n's "periods" argument means months rather than
    # calendar/trading days. The resulting index is then normalized to
    # month-end to line up with core_index_monthly (which FRED stamps
    # first-of-month) when the two are combined below.
    breakeven_monthly = normalize_month_end_index(
        resample_to_monthly(breakeven_5y_daily_or_monthly, how="last")
    )
    breakeven_change = diff_n(breakeven_monthly, breakeven_change_months)
    breakeven_component = rolling_zscore(breakeven_change, zscore_window, zscore_min_periods)

    out = pd.DataFrame(index=core_index_monthly.index)
    out["core_component"] = core_component
    out["breakeven_component"] = breakeven_component.reindex(out.index)

    out["inflation_score_no_ism"] = out[["breakeven_component", "core_component"]].mean(
        axis=1, skipna=False
    )

    if ism_prices_paid_monthly is not None:
        ism_prices_paid_monthly = normalize_month_end_index(ism_prices_paid_monthly)
        ism_change = diff_n(ism_prices_paid_monthly, breakeven_change_months)
        ism_component = rolling_zscore(ism_change, zscore_window, zscore_min_periods)
        out["ism_component"] = ism_component.reindex(out.index)
        out["inflation_score"] = out[
            ["ism_component", "breakeven_component", "core_component"]
        ].mean(axis=1, skipna=False)
        active_score_col = "inflation_score"
    else:
        out["ism_component"] = float("nan")
        out["inflation_score"] = float("nan")
        active_score_col = "inflation_score_no_ism"

    out["inflation_label"] = out[active_score_col].apply(
        lambda x: sign_label(x, up=UP, down=DOWN, unknown=UNKNOWN)
    )
    out.attrs["active_score_column"] = active_score_col
    return out


# ---------------------------------------------------------------------------
# Model C: Cleveland Fed Inflation Nowcast (stub)
# ---------------------------------------------------------------------------


def cleveland_nowcast_signal(nowcast_history: pd.DataFrame | None) -> pd.DataFrame | None:
    """Returns None unless a reproducible historical vintage file has been
    supplied via data/external/cleveland_fed_inflation_nowcast.csv.

    Even when a file is present, this function deliberately does not
    attempt to guess a direction rule -- that must be defined once real
    vintage data is available and validated against the disclosed
    methodology of the nowcast itself.
    """
    if nowcast_history is None:
        return None
    raise NotImplementedError(
        "Cleveland Nowcast history was supplied, but Inflation Model C's "
        "classification rule has not been implemented/validated in this "
        "build. See docs/methodology.md before enabling this model."
    )
