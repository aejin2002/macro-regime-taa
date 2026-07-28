"""BEI Duration Risk Gate (v1.1) -- allocation overlay.

Pure function that tilts only the nominal-Treasury duration sleeve
(intermediate_treasury + long_treasury) of Primary v1.0's regime-driven
target weights. It never changes the macro regime, Growth Basket,
SPY/KODEX200 split, HYG, LQD, TIP, DBC, GLD, or the pre-existing base
tbills weight -- those columns pass through byte-for-byte unchanged.

There is no rule here that ever *expands* TLT -- this gate only ever
reduces duration risk when ON; it is a defensive brake, not a
directional call on falling rates.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.duration_gate.signal import ON

DEFAULT_INTERMEDIATE_COL = "intermediate_treasury"
DEFAULT_LONG_COL = "long_treasury"
DEFAULT_CASH_COL = "tbills"


def apply_bei_duration_gate(
    base_weights: pd.DataFrame,
    tradable_gate: pd.Series,
    *,
    ief_weight_when_on: float,
    bil_weight_when_on: float,
    intermediate_col: str = DEFAULT_INTERMEDIATE_COL,
    long_col: str = DEFAULT_LONG_COL,
    cash_col: str = DEFAULT_CASH_COL,
) -> pd.DataFrame:
    """Apply the BEI Duration Risk Gate on top of Primary v1.0's
    regime-driven target weights (e.g. from
    `backtest.engine.build_target_weights_from_regime`).

    For each month:
    - duration_pool = base_weights[intermediate_col] + base_weights[long_col]
    - if duration_pool <= 0: the entire base allocation for that month
      is returned unchanged (there is nothing to tilt).
    - if tradable_gate[t] != ON (i.e. OFF or UNKNOWN, including a
      missing/not-yet-available value): the base IEF/TLT/BIL weights
      are kept exactly as given.
    - if tradable_gate[t] == ON: final_TLT = 0, final_IEF = duration_pool
      * ief_weight_when_on, additional_BIL = duration_pool *
      bil_weight_when_on, final_BIL = base_BIL + additional_BIL.

    Every other column in `base_weights` (Growth Basket's spy/kodex200_usd,
    high_yield, investment_grade, gold, commodities, tips, and the
    pre-existing base tbills weight) is copied through unmodified.

    Invariants that always hold on the returned frame: weights sum to 1
    per row (given `base_weights` already does), no negative weights,
    duration_pool is exactly conserved (final_IEF + final_TLT +
    additional_BIL == duration_pool), and base_BIL is preserved (never
    reduced, only ever added to).
    """
    if ief_weight_when_on < 0 or bil_weight_when_on < 0:
        raise ValueError("ief_weight_when_on and bil_weight_when_on must be non-negative")

    overlaid = base_weights.copy()
    gate_aligned = tradable_gate.reindex(base_weights.index)

    for t in base_weights.index:
        base_ief = base_weights.at[t, intermediate_col]
        base_tlt = base_weights.at[t, long_col]
        duration_pool = base_ief + base_tlt
        if duration_pool <= 0:
            continue
        if gate_aligned.at[t] != ON:
            continue

        overlaid.at[t, long_col] = 0.0
        overlaid.at[t, intermediate_col] = duration_pool * ief_weight_when_on
        overlaid.at[t, cash_col] = base_weights.at[t, cash_col] + duration_pool * bil_weight_when_on

    return overlaid
