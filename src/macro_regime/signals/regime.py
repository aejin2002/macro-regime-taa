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
