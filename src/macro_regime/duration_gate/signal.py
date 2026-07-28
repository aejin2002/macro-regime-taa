"""BEI Duration Risk Gate (v1.1) -- signal computation.

Not a rate-level forecasting model. A single ON/OFF/UNKNOWN gate that
confirms a long-end nominal Treasury sell-off (DGS10 up, TLT price down)
that is being driven by rising market inflation compensation (T10YIE up
on both a 1-month and 3-month window). Used only to tilt the *duration*
sleeve (intermediate_treasury + long_treasury) of Primary v1.0's
regime-driven allocation -- see `duration_gate/allocation.py`. It never
changes the macro regime classification or any other asset.

T10YIE ("10-Year Breakeven Inflation Rate") is the market's inflation
*compensation* embedded in nominal vs. TIPS yields -- it is not a pure
survey- or model-based expected-inflation measure, and is influenced by
an inflation risk premium and by TIPS-market liquidity conditions as
well as by inflation expectations themselves. It is always referred to
as "market inflation compensation" in this module's documentation and
in the product UI, never as "expected inflation."

All four raw-gate conditions, the 1-month/3-month windows, and the
30%/70% IEF/BIL split in `allocation.py` are ex-ante heuristics fixed
before running any backtest -- none of them is fit, tuned, or chosen
after seeing performance results.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.utils.dates import drop_incomplete_trailing_month, resample_to_monthly

ON = "ON"
OFF = "OFF"
UNKNOWN = "UNKNOWN"


def monthly_rate_level(daily_series: pd.Series, *, as_of: pd.Timestamp | None = None) -> pd.Series:
    """Month-end-labeled monthly average of a daily FRED rate series
    (DGS10 or T10YIE), completed months only. No forward fill, no
    backfilling before the series' own start, no partial/in-progress
    trailing month."""
    as_of = as_of or pd.Timestamp.now()
    monthly = resample_to_monthly(daily_series.dropna(), how="mean")
    return drop_incomplete_trailing_month(monthly, as_of)


def classify_raw_bei_duration_gate(
    dgs10_change_1m: pd.Series,
    bei_change_1m: pd.Series,
    bei_change_3m: pd.Series,
    tlt_return_1m: pd.Series,
) -> pd.Series:
    """raw_bei_duration_gate[t] = ON iff all four hold:
    1. dgs10_change_1m > 0  (long-end nominal rate actually rose)
    2. tlt_return_1m < 0    (TLT's own price actually fell)
    3. bei_change_1m > 0    (market inflation compensation rising, 1m)
    4. bei_change_3m > 0    (market inflation compensation rising, 3m)

    OFF if every one of the four values is present but the AND fails.
    UNKNOWN if any of the four values needed is missing (e.g. warm-up
    period, or a gap in one of the underlying series) -- never guessed
    or defaulted to OFF."""
    idx = dgs10_change_1m.index
    bei_1m = bei_change_1m.reindex(idx)
    bei_3m = bei_change_3m.reindex(idx)
    tlt_1m = tlt_return_1m.reindex(idx)

    states = []
    for d10, b1, b3, tlt in zip(dgs10_change_1m, bei_1m, bei_3m, tlt_1m, strict=True):
        if pd.isna(d10) or pd.isna(b1) or pd.isna(b3) or pd.isna(tlt):
            states.append(UNKNOWN)
            continue
        is_on = (d10 > 0) and (tlt < 0) and (b1 > 0) and (b3 > 0)
        states.append(ON if is_on else OFF)
    return pd.Series(states, index=idx, name="raw_bei_duration_gate")


def tradable_bei_duration_gate(raw_gate: pd.Series) -> pd.Series:
    """tradable_bei_duration_gate[t] = raw_bei_duration_gate[t-1] -- the
    gate confirmed as of month t-1's close is what month t's portfolio
    may use. Month t's own raw gate (only fully known at t's close)
    never drives month t's allocation. The first month in any sample has
    no prior raw gate, so it is UNKNOWN, never defaulted to ON or OFF."""
    return raw_gate.shift(1).fillna(UNKNOWN).rename("tradable_bei_duration_gate")


def build_bei_duration_gate_signal(
    dgs10_daily: pd.Series,
    t10yie_daily: pd.Series,
    tlt_monthly_returns: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """End-to-end signal table, one row per completed month present in
    both the DGS10 and T10YIE monthly samples:
    DGS10, T10YIE, dgs10_change_1m, bei_change_1m, bei_change_3m,
    tlt_return_1m, raw_bei_duration_gate, tradable_bei_duration_gate.
    """
    dgs10 = monthly_rate_level(dgs10_daily, as_of=as_of)
    t10yie = monthly_rate_level(t10yie_daily, as_of=as_of)
    common_index = dgs10.index.intersection(t10yie.index)
    dgs10 = dgs10.reindex(common_index)
    t10yie = t10yie.reindex(common_index)
    tlt_1m = tlt_monthly_returns.reindex(common_index)

    dgs10_change_1m = dgs10.diff(1)
    bei_change_1m = t10yie.diff(1)
    bei_change_3m = t10yie - t10yie.shift(3)

    raw_gate = classify_raw_bei_duration_gate(dgs10_change_1m, bei_change_1m, bei_change_3m, tlt_1m)
    tradable_gate = tradable_bei_duration_gate(raw_gate)

    return pd.DataFrame(
        {
            "DGS10": dgs10,
            "T10YIE": t10yie,
            "dgs10_change_1m": dgs10_change_1m,
            "bei_change_1m": bei_change_1m,
            "bei_change_3m": bei_change_3m,
            "tlt_return_1m": tlt_1m,
            "raw_bei_duration_gate": raw_gate,
            "tradable_bei_duration_gate": tradable_gate,
        },
        index=common_index,
    )
