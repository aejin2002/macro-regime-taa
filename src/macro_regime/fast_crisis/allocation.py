"""Fast Crisis Overlay (v1.2) -- allocation overlay.

Pure function that, on days Crisis Mode is ON, removes Growth Basket
(spy + kodex200_usd), HYG, and DBC entirely and moves the removed weight
to BIL. LQD, IEF, TLT, GLD, and TIP -- and whatever the BEI Duration
Risk Gate (v1.1) has already done to IEF/TLT/BIL -- pass through
unchanged. This overlay is applied on top of v1.1's already-BEI-gated
daily target, so the two gates never fight over the same weight: the BEI
gate only ever touches intermediate_treasury/long_treasury/tbills, this
overlay only ever touches spy/kodex200_usd/high_yield/commodities/tbills.

There is no rule here that ever *increases* risk exposure -- this
overlay only ever de-risks when ON; it is a defensive brake, not a
directional call.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.fast_crisis.signal import ON

CRISIS_ASSET_COLUMNS = ["spy", "kodex200_usd", "high_yield", "commodities"]
CASH_COLUMN = "tbills"


def apply_fast_crisis_overlay(
    base_weights: pd.DataFrame,
    crisis_mode: pd.Series,
    *,
    crisis_asset_columns: list[str] = CRISIS_ASSET_COLUMNS,
    cash_column: str = CASH_COLUMN,
) -> pd.DataFrame:
    """For each day:
    - `crisis_mode[t] != ON` (i.e. OFF, including a missing/not-yet-
      available value): `base_weights` for that day is returned exactly
      unchanged.
    - `crisis_mode[t] == ON`: every column in `crisis_asset_columns` is
      set to 0, and their combined removed weight is added to
      `cash_column`.

    Every other column (long_treasury, investment_grade, gold, tips, and
    whatever value `cash_column` already held) passes through unchanged.

    Invariants that always hold on the returned frame: weights sum to 1
    per row (given `base_weights` already does), no negative weights,
    and `base_weights[cash_column]` is preserved (never reduced, only
    ever added to).
    """
    overlaid = base_weights.copy()
    mode_aligned = crisis_mode.reindex(base_weights.index)
    is_on = mode_aligned == ON

    removed = base_weights.loc[is_on, crisis_asset_columns].sum(axis=1)
    overlaid.loc[is_on, crisis_asset_columns] = 0.0
    overlaid.loc[is_on, cash_column] = base_weights.loc[is_on, cash_column] + removed

    return overlaid
