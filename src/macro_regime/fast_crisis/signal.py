"""Fast Crisis Overlay (v1.2) -- daily shock signals, the 2-of-3 raw
trigger, and the entry/exit state machine.

Not a rate-level or return forecast. A daily-frequency tail-risk brake:
three independent shock signals (VIX level+spike, SPY 5-day return,
HYG 5-day return) are combined 2-of-3 into a raw trigger, then run
through a minimum-hold / consecutive-OFF-to-exit state machine before
ever touching the portfolio -- see `fast_crisis/allocation.py` for what
Crisis Mode actually changes.

All thresholds, lookback windows, the 2-of-3 combination rule, and the
10-day minimum hold / 5-consecutive-OFF exit rule (`config/default.yaml`,
`fast_crisis:`) are ex-ante heuristics fixed before any backtest was run
and were not re-tuned after seeing results.
"""

from __future__ import annotations

import pandas as pd

ON = "ON"
OFF = "OFF"
UNKNOWN = "UNKNOWN"


def vix_shock_diagnostics(vix: pd.Series, *, ma_window_days: int) -> pd.DataFrame:
    """Raw values underlying `compute_vix_shock` (VIX level, its trailing
    `ma_window_days`-day average, and the ratio compared against
    `ma_ratio_threshold`) -- display/diagnostic only, decides nothing
    itself. `compute_vix_shock` computes this same rolling average and
    ratio internally; this function exists so a caller (e.g. a UI) can
    show the current numeric values without duplicating that formula."""
    vix_ma = vix.rolling(ma_window_days, min_periods=ma_window_days).mean()
    vix_ratio = vix / vix_ma - 1.0
    return pd.DataFrame({"vix_level": vix, "vix_ma": vix_ma, "vix_ratio": vix_ratio})


def compute_vix_shock(
    vix: pd.Series, *, threshold: float, ma_window_days: int, ma_ratio_threshold: float
) -> pd.Series:
    """vix_shock[t] = VIX[t] > threshold AND VIX[t]/MA[t] - 1 > ma_ratio_threshold,
    where MA is the trailing `ma_window_days`-day simple average of VIX.
    UNKNOWN if fewer than `ma_window_days` valid VIX observations are
    available up to and including t."""
    diag = vix_shock_diagnostics(vix, ma_window_days=ma_window_days)
    out = []
    for v, ma, r in zip(diag["vix_level"], diag["vix_ma"], diag["vix_ratio"], strict=True):
        if pd.isna(v) or pd.isna(ma) or pd.isna(r):
            out.append(UNKNOWN)
        else:
            out.append(ON if (v > threshold and r > ma_ratio_threshold) else OFF)
    return pd.Series(out, index=vix.index, name="vix_shock")


def n_day_return_diagnostic(daily_return: pd.Series, *, window_days: int) -> pd.Series:
    """The raw compounded `window_days`-trading-day return that
    `compute_equity_shock`/`compute_credit_shock` threshold against --
    display/diagnostic only, decides nothing itself."""
    return (1.0 + daily_return).rolling(window_days, min_periods=window_days).apply(
        lambda x: x.prod(), raw=True
    ) - 1.0


def _n_day_shock(daily_return: pd.Series, *, window_days: int, threshold: float, name: str) -> pd.Series:
    """shock[t] = compounded `window_days`-trading-day return of
    `daily_return` <= threshold. UNKNOWN if the full window is not yet
    available (warm-up period)."""
    n_day_return = n_day_return_diagnostic(daily_return, window_days=window_days)
    out = [UNKNOWN if pd.isna(v) else (ON if v <= threshold else OFF) for v in n_day_return]
    return pd.Series(out, index=daily_return.index, name=name)


def compute_equity_shock(spy_return: pd.Series, *, window_days: int, threshold: float) -> pd.Series:
    """equity_shock[t] = SPY's `window_days`-trading-day total return
    <= threshold."""
    return _n_day_shock(spy_return, window_days=window_days, threshold=threshold, name="equity_shock")


def compute_credit_shock(hyg_return: pd.Series, *, window_days: int, threshold: float) -> pd.Series:
    """credit_shock[t] = HYG's `window_days`-trading-day total return
    <= threshold."""
    return _n_day_shock(hyg_return, window_days=window_days, threshold=threshold, name="credit_shock")


def two_of_three(a: pd.Series, b: pd.Series, c: pd.Series) -> pd.Series:
    """3 signals observed: >=2 True -> ON, else OFF. Exactly 2 observed:
    both True -> ON; both False -> OFF (the result can never reach 2
    True regardless of the missing one, so it is not ambiguous); exactly
    1 True -> UNKNOWN (genuinely ambiguous -- the missing signal could
    tip either way). <=1 observed -> UNKNOWN."""
    out = []
    for i in range(len(a)):
        vals = [a.iloc[i], b.iloc[i], c.iloc[i]]
        observed = [v for v in vals if v != UNKNOWN]
        n_true = sum(1 for v in observed if v == ON)
        if len(observed) == 3:
            out.append(ON if n_true >= 2 else OFF)
        elif len(observed) == 2:
            if n_true == 2:
                out.append(ON)
            elif n_true == 0:
                out.append(OFF)
            else:
                out.append(UNKNOWN)
        else:
            out.append(UNKNOWN)
    return pd.Series(out, index=a.index, name="raw_fast_crisis_trigger")


def build_raw_trigger(vix_shock: pd.Series, equity_shock: pd.Series, credit_shock: pd.Series) -> pd.Series:
    return two_of_three(vix_shock, equity_shock, credit_shock)


def build_crisis_mode_state(
    raw_trigger: pd.Series, *, min_hold_days: int, min_off_days_to_exit: int
) -> pd.DataFrame:
    """Applies raw_trigger[t] to determine action starting at t+1's
    return (`tradable_trigger[t] = raw_trigger[t-1]`, the 1-day tradable
    lag -- a same-day close signal is never treated as tradable on its
    own day).

    State machine (trading-day units, evaluated once per day in
    chronological order):
    - OFF -> ON: first trading day whose `tradable_trigger == ON`.
      UNKNOWN is never used to enter.
    - Once ON: stays ON for at least `min_hold_days` trading days,
      regardless of what `tradable_trigger` does during that window.
    - After the minimum hold has elapsed, exits on the trading day
      following `min_off_days_to_exit` consecutive `tradable_trigger ==
      OFF` days. UNKNOWN does not count toward this consecutive-OFF
      streak, and does not reset it either -- it is simply skipped.
    - A fresh `tradable_trigger == ON` while already in Crisis Mode
      resets the minimum-hold counter (re-affirms the emergency) and
      resets the consecutive-OFF counter to 0.

    Returns a DataFrame indexed like `raw_trigger` with columns:
    tradable_trigger, crisis_mode (ON/OFF), days_in_mode,
    consecutive_off_count.
    """
    idx = raw_trigger.index
    tradable_trigger = raw_trigger.shift(1).fillna(UNKNOWN)

    crisis_mode = []
    days_in_mode_list = []
    consecutive_off_list = []

    in_mode = False
    days_in_mode = 0
    consecutive_off = 0

    for t in idx:
        trig = tradable_trigger.loc[t]

        if not in_mode:
            if trig == ON:
                in_mode = True
                days_in_mode = 1
                consecutive_off = 0
            else:
                days_in_mode = 0
                consecutive_off = 0
        else:
            # Exit check uses the state as it stood at the END OF
            # YESTERDAY (days_in_mode/consecutive_off from the prior
            # iteration) -- a day whose OWN trig completes the minimum
            # hold or the consecutive-OFF streak still counts as a
            # defended (ON) day; the exit only takes effect starting the
            # NEXT trading day.
            if days_in_mode >= min_hold_days and consecutive_off >= min_off_days_to_exit:
                in_mode = False
                days_in_mode = 0
                consecutive_off = 0
            else:
                days_in_mode += 1
                if trig == ON:
                    days_in_mode = 1  # re-affirm: restart minimum-hold counter
                    consecutive_off = 0
                elif trig == OFF:
                    consecutive_off += 1
                # UNKNOWN: neither increments nor resets consecutive_off

        crisis_mode.append(ON if in_mode else OFF)
        days_in_mode_list.append(days_in_mode)
        consecutive_off_list.append(consecutive_off)

    return pd.DataFrame(
        {
            "tradable_trigger": tradable_trigger,
            "crisis_mode": crisis_mode,
            "days_in_mode": days_in_mode_list,
            "consecutive_off_count": consecutive_off_list,
        },
        index=idx,
    )
