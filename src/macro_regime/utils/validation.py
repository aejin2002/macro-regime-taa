"""Small validation helpers used across the pipeline."""

from __future__ import annotations

import pandas as pd


class SeriesValidationError(ValueError):
    pass


def require_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SeriesValidationError(f"{context}: missing required columns {missing}")


def require_monotonic_dates(df: pd.DataFrame, date_col: str, *, context: str) -> None:
    if not df[date_col].is_monotonic_increasing:
        raise SeriesValidationError(f"{context}: '{date_col}' is not sorted ascending")


def warn_if_empty(df: pd.DataFrame, *, context: str) -> None:
    if df.empty:
        import warnings

        warnings.warn(f"{context}: DataFrame is empty", stacklevel=2)
