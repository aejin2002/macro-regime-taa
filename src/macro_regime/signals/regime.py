"""Growth x Inflation quadrant regime classification.

Every growth-model / inflation-model pair produces its own regime series;
these are never blended into a single "house view" here. Callers name each
combination explicitly (e.g. `regime_lei_core_cpi`,
`regime_fred_minimal_core_cpi`, `regime_fred_minimal_inflation_composite`)
so it is always clear which model pair produced a given regime call.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

GROWTH_UP, GROWTH_DOWN = "Up", "Down"
INFLATION_UP, INFLATION_DOWN = "Up", "Down"


class Regime(StrEnum):
    GOLDILOCKS = "GOLDILOCKS"
    REFLATION = "REFLATION"
    STAGFLATION = "STAGFLATION"
    CONTRACTION = "CONTRACTION"
    UNKNOWN = "UNKNOWN"


_REGIME_MAP = {
    (GROWTH_UP, INFLATION_DOWN): Regime.GOLDILOCKS,
    (GROWTH_UP, INFLATION_UP): Regime.REFLATION,
    (GROWTH_DOWN, INFLATION_UP): Regime.STAGFLATION,
    (GROWTH_DOWN, INFLATION_DOWN): Regime.CONTRACTION,
}


def classify_regime(growth_label: str, inflation_label: str) -> Regime:
    return _REGIME_MAP.get((growth_label, inflation_label), Regime.UNKNOWN)


def build_regime_series(
    growth_labels: pd.Series, inflation_labels: pd.Series, *, name: str
) -> pd.Series:
    """Align a growth-label series and an inflation-label series on their
    common index and classify each date into a quadrant regime."""
    aligned = pd.concat(
        {"growth": growth_labels, "inflation": inflation_labels}, axis=1, join="outer"
    ).sort_index()
    regime = aligned.apply(
        lambda row: classify_regime(row["growth"], row["inflation"]).value, axis=1
    )
    regime.name = name
    return regime


def regime_distribution(regime_series: pd.Series) -> pd.Series:
    return regime_series.value_counts(normalize=True).reindex(
        [r.value for r in Regime], fill_value=0.0
    )


def regime_transitions(regime_series: pd.Series) -> int:
    """Count the number of month-over-month regime changes (ignoring NaN)."""
    clean = regime_series.dropna()
    return int((clean != clean.shift(1)).sum() - 1) if len(clean) > 0 else 0


def regime_durations(regime_series: pd.Series) -> pd.DataFrame:
    """Return one row per contiguous regime spell: regime, start, end, length_months."""
    clean = regime_series.dropna()
    if clean.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "length_months"])

    spell_id = (clean != clean.shift(1)).cumsum()
    rows = []
    for _, spell in clean.groupby(spell_id):
        rows.append(
            {
                "regime": spell.iloc[0],
                "start": spell.index[0],
                "end": spell.index[-1],
                "length_months": len(spell),
            }
        )
    return pd.DataFrame(rows)


def shift_to_tradable(raw_regime: pd.Series, *, lag_months: int = 1) -> pd.Series:
    """tradable_regime[t] = raw_regime[t - lag_months]: the regime known at
    month t is only actionable starting `lag_months` later. The first
    `lag_months` rows have no prior raw_regime to reference and are
    UNKNOWN. Positional shift (matches `diff_n`/`rolling_zscore` elsewhere
    in this project), so it assumes a gap-free monthly index. Returns a new
    Series -- `raw_regime` itself is never mutated.
    """
    shifted = raw_regime.shift(lag_months)
    return shifted.fillna(Regime.UNKNOWN.value).rename("tradable_regime")


def build_regime_output(
    growth_score: pd.Series,
    growth_state: pd.Series,
    inflation_score: pd.Series,
    inflation_state: pd.Series,
    *,
    tradable_lag_months: int = 1,
) -> pd.DataFrame:
    """Standard asset-allocation regime output for one growth x inflation
    model pair: growth_score, growth_state, inflation_score,
    inflation_state, raw_regime, tradable_regime. `tradable_regime` is
    `raw_regime` shifted `tradable_lag_months` forward (see
    `shift_to_tradable`) -- it is the only column derived from another;
    every other column is the model's own signal, unmodified.
    """
    raw_regime = build_regime_series(growth_state, inflation_state, name="raw_regime")
    tradable_regime = shift_to_tradable(raw_regime, lag_months=tradable_lag_months)
    return pd.concat(
        [
            growth_score.rename("growth_score"),
            growth_state.rename("growth_state"),
            inflation_score.rename("inflation_score"),
            inflation_state.rename("inflation_state"),
            raw_regime,
            tradable_regime,
        ],
        axis=1,
    )


def transition_matrix(regime_series: pd.Series) -> pd.DataFrame:
    """Row-normalized regime-to-regime transition probability matrix."""
    clean = regime_series.dropna()
    labels = [r.value for r in Regime]
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    prev = clean.shift(1)
    for a, b in zip(prev, clean, strict=True):
        if pd.isna(a):
            continue
        counts.loc[a, b] += 1
    row_sums = counts.sum(axis=1)
    return counts.div(row_sums.replace(0, pd.NA), axis=0).fillna(0.0)
