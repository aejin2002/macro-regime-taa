"""Statistics helpers shared by growth, inflation, and evaluation code.

`rolling_zscore` is the single standardization function used everywhere in
this project specifically because pandas' `.rolling()` only ever looks
backward from each timestamp -- it cannot see future observations, so it
cannot introduce look-ahead bias by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    series: pd.Series,
    window: int = 120,
    min_periods: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """Point-in-time rolling z-score: at each t, uses only observations in
    (t - window, t]. Returns NaN wherever fewer than `min_periods`
    observations are available, or where the rolling std is exactly 0.
    """
    rolling = series.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=ddof)
    z = (series - mean) / std
    return z.where(std != 0)


def pct_change_n(series: pd.Series, periods: int) -> pd.Series:
    """t vs t-`periods` percent change, using only current/past data."""
    return series / series.shift(periods) - 1.0


def diff_n(series: pd.Series, periods: int) -> pd.Series:
    return series - series.shift(periods)


def annualize_from_periods(growth: pd.Series, periods_per_year_fraction: float) -> pd.Series:
    """Annualize a period-over-period growth rate, e.g. (x_t/x_t-3)**4 - 1
    for a 3-month change annualized (periods_per_year_fraction = 4)."""
    return (1.0 + growth) ** periods_per_year_fraction - 1.0


def sign_label(x: float, *, up: str = "Up", down: str = "Down", unknown: str = "Unknown") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return unknown
    if x > 0:
        return up
    if x < 0:
        return down
    return unknown
