"""Growth direction models.

Three independent models, each producing an "Up" / "Down" / "Unknown"
monthly label plus a continuous score where applicable. Models are kept
independent (not blended) so their individual predictive power can be
evaluated separately -- see evaluation/classification.py.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.utils.dates import resample_to_monthly
from macro_regime.utils.stats import diff_n, rolling_zscore, sign_label

UP, DOWN, UNKNOWN, NEUTRAL = "Up", "Down", "Unknown", "Neutral"


# ---------------------------------------------------------------------------
# Model A: OECD Composite Leading Indicator (CLI)
# ---------------------------------------------------------------------------


def cli_diagnostics(cli: pd.Series, windows: tuple[int, int, int] = (1, 3, 6)) -> pd.DataFrame:
    """Level and 1/3/6-month change diagnostics for a monthly CLI series."""
    m1, m3, m6 = windows
    out = pd.DataFrame(index=cli.index)
    out["level"] = cli
    out["above_100"] = cli >= 100
    out[f"chg_{m1}m"] = diff_n(cli, m1)
    out[f"chg_{m3}m"] = diff_n(cli, m3)
    out[f"chg_{m6}m"] = diff_n(cli, m6)
    out["agree_3m_6m"] = (out[f"chg_{m3}m"] > 0) == (out[f"chg_{m6}m"] > 0)
    return out


def classify_growth_model_a(
    cli: pd.Series,
    *,
    primary_window: int = 3,
    stabilized: bool = False,
    stabilization_windows: tuple[int, int] = (3, 6),
) -> pd.Series:
    """Primary rule: sign of cli_change_3m (CLI_t - CLI_t-primary_window).

    If `stabilized`, use the alternative rule instead: Up when both the
    3-month and 6-month changes are positive, Down when both are negative,
    and otherwise carry the previous state forward (falling back to
    Neutral at the very start of the series).
    """
    if not stabilized:
        change = diff_n(cli, primary_window)
        return change.apply(lambda x: sign_label(x, up=UP, down=DOWN, unknown=UNKNOWN))

    w_short, w_long = stabilization_windows
    chg_short = diff_n(cli, w_short)
    chg_long = diff_n(cli, w_long)

    labels: list[str] = []
    previous = NEUTRAL
    for short, long in zip(chg_short, chg_long, strict=True):
        if pd.isna(short) or pd.isna(long):
            label = UNKNOWN
        elif short > 0 and long > 0:
            label = UP
        elif short < 0 and long < 0:
            label = DOWN
        else:
            label = previous
        labels.append(label)
        if label not in (UNKNOWN,):
            previous = label
    return pd.Series(labels, index=cli.index)


# ---------------------------------------------------------------------------
# Model B: FRED Minimal (Initial Claims, Building Permits, CFNAI-MA3)
# ---------------------------------------------------------------------------


def growth_model_b_fred_minimal(
    claims_weekly: pd.Series,
    permits_monthly: pd.Series,
    cfnai_ma3_monthly: pd.Series,
    *,
    claims_change_months: int = 3,
    permits_change_months: int = 6,
    zscore_window: int = 120,
    zscore_min_periods: int = 60,
) -> pd.DataFrame:
    claims_monthly = resample_to_monthly(claims_weekly, how="mean")
    claims_change = diff_n(claims_monthly, claims_change_months)
    claims_component = -rolling_zscore(claims_change, zscore_window, zscore_min_periods)

    permits_change = diff_n(permits_monthly, permits_change_months)
    permits_component = rolling_zscore(permits_change, zscore_window, zscore_min_periods)

    activity_component = rolling_zscore(cfnai_ma3_monthly, zscore_window, zscore_min_periods)

    df = pd.DataFrame(
        {
            "claims_component": claims_component,
            "permits_component": permits_component,
            "activity_component": activity_component,
        }
    )
    df["growth_score"] = df.mean(axis=1, skipna=False)
    df["growth_label"] = df["growth_score"].apply(
        lambda x: sign_label(x, up=UP, down=DOWN, unknown=UNKNOWN)
    )
    return df


# ---------------------------------------------------------------------------
# Model C: Simple Two-Signal (ISM New Orders vs. Initial Claims)
# ---------------------------------------------------------------------------


def growth_model_c_simple_two_signal(
    ism_new_orders_monthly: pd.Series | None,
    claims_weekly: pd.Series,
    *,
    change_months: int = 3,
    zscore_window: int = 120,
    zscore_min_periods: int = 60,
) -> pd.DataFrame | None:
    """Returns None (model disabled) when ISM New Orders data is unavailable."""
    if ism_new_orders_monthly is None:
        return None

    ism_change = diff_n(ism_new_orders_monthly, change_months)
    ism_z = rolling_zscore(ism_change, zscore_window, zscore_min_periods)

    claims_monthly = resample_to_monthly(claims_weekly, how="mean")
    claims_change = diff_n(claims_monthly, change_months)
    claims_z = rolling_zscore(claims_change, zscore_window, zscore_min_periods)

    df = pd.DataFrame({"ism_component": ism_z, "claims_component": claims_z}).dropna(how="all")
    df["growth_score"] = (df["ism_component"] - df["claims_component"]) / 2
    df["growth_label"] = df["growth_score"].apply(
        lambda x: sign_label(x, up=UP, down=DOWN, unknown=UNKNOWN)
    )
    return df
