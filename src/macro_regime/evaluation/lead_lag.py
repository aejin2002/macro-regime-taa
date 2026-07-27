"""Forward targets (what a signal is trying to predict) and a simple
lead-time diagnostic.

Target construction and alignment
----------------------------------
Forward targets are built by shifting the *target* series backward
(`.shift(-horizon)`) so that each row `t` carries the realized outcome
between `t` and `t+horizon`, while the signal itself stays anchored at
`t`. No row's target ever depends on the signal's own future value, and
merges are always done on the shared date index (an inner/outer join on
`date`), so misaligned indices raise/produce NaN rather than silently
shifting rows relative to each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UP, DOWN, UNKNOWN = "Up", "Down", "Unknown"


def growth_forward_target(indpro_monthly: pd.Series, horizon_months: int) -> pd.DataFrame:
    """ip_forward_Nm = log(INDPRO_t+N) - log(INDPRO_t); Up if > 0 else Down."""
    log_level = np.log(indpro_monthly)
    forward_log_change = log_level.shift(-horizon_months) - log_level
    label = forward_log_change.apply(
        lambda x: UNKNOWN if pd.isna(x) else (UP if x > 0 else (DOWN if x < 0 else UNKNOWN))
    )
    return pd.DataFrame({"forward_log_change": forward_log_change, "actual_label": label})


def inflation_forward_target(
    core_index_monthly: pd.Series,
    horizon_months: int,
    *,
    compare_to: pd.Series | None = None,
) -> pd.DataFrame:
    """future_core_inflation_Nm = annualized inflation from t to t+N.

    Direction is Up/Down relative to `compare_to` (typically the trailing
    12-month core inflation rate at t) when supplied, otherwise relative to
    zero. This comparison basis must be documented wherever the target is
    used -- see README / docs/methodology.md.
    """
    periods_per_year = 12 / horizon_months
    forward_growth = core_index_monthly.shift(-horizon_months) / core_index_monthly - 1.0
    forward_annualized = (1.0 + forward_growth) ** periods_per_year - 1.0

    if compare_to is not None:
        baseline = compare_to
    else:
        baseline = pd.Series(0.0, index=core_index_monthly.index)
    diff = forward_annualized - baseline.reindex(forward_annualized.index)

    label = diff.apply(
        lambda x: UNKNOWN if pd.isna(x) else (UP if x > 0 else (DOWN if x < 0 else UNKNOWN))
    )
    return pd.DataFrame(
        {"forward_annualized": forward_annualized, "vs_baseline": diff, "actual_label": label}
    )


def _turning_points(series: pd.Series) -> pd.DatetimeIndex:
    """Dates where a numeric series changes sign of month-over-month change
    (a crude proxy for local turning points)."""
    diff_sign = np.sign(series.diff())
    turn = (diff_sign != diff_sign.shift(1)) & diff_sign.notna() & diff_sign.shift(1).notna()
    return series.index[turn]


def average_lead_time_months(
    signal_score: pd.Series,
    target_level: pd.Series,
    *,
    max_lag_months: int = 12,
) -> float:
    """Estimate the average lead time (in months) of `signal_score` over
    `target_level` by finding the lag (signal shifted forward by k months)
    that maximizes their Pearson correlation, searching k in
    [0, max_lag_months]. A positive result means the signal tends to lead
    the target by that many months. This is a simple cross-correlation
    diagnostic, not a formal Granger-causality test.
    """
    target_change = target_level.diff()
    best_lag = 0
    best_corr = -np.inf
    for lag in range(0, max_lag_months + 1):
        shifted = signal_score.shift(lag)
        aligned = pd.concat({"s": shifted, "t": target_change}, axis=1).dropna()
        if len(aligned) < 24:
            continue
        corr = aligned["s"].corr(aligned["t"])
        if pd.notna(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_lag)
