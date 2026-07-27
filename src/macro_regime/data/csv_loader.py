"""Loaders for optional external CSV inputs (Conference Board LEI, ISM
New Orders / Prices Paid, Cleveland Fed nowcast).

Every loader returns `None` when its file is missing, so callers can
disable the dependent model with a clear warning instead of crashing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def _load_date_value_csv(path: Path, value_column: str, *, warning: str) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(warning, stacklevel=2)
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or value_column not in df.columns:
        raise ValueError(f"{path}: expected columns 'date' and '{value_column}'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", value_column]]


def load_conference_board_lei(path: str | Path) -> pd.DataFrame | None:
    """Load data/external/conference_board_lei.csv (columns: date, lei[, release_date, vintage_date])."""
    path = Path(path)
    if not path.exists():
        warnings.warn(
            "Conference Board LEI CSV was not found. LEI-based growth model is disabled.",
            stacklevel=2,
        )
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or "lei" not in df.columns:
        raise ValueError(f"{path}: expected columns 'date' and 'lei' (release_date/vintage_date optional)")
    df["date"] = pd.to_datetime(df["date"])
    for optional_col in ("release_date", "vintage_date"):
        if optional_col in df.columns:
            df[optional_col] = pd.to_datetime(df[optional_col])
    return df.sort_values("date").reset_index(drop=True)


def load_ism_new_orders(path: str | Path) -> pd.DataFrame | None:
    return _load_date_value_csv(
        Path(path),
        "value",
        warning="ISM New Orders CSV was not found. Growth Model C is disabled.",
    )


def load_ism_prices_paid(path: str | Path) -> pd.DataFrame | None:
    return _load_date_value_csv(
        Path(path),
        "value",
        warning=(
            "ISM Prices Paid CSV was not found. Inflation Model B will run in "
            "its 2-signal (no-ISM) variant."
        ),
    )


def load_cleveland_nowcast(path: str | Path) -> pd.DataFrame | None:
    """Load a reproducible historical Cleveland Fed Inflation Nowcast vintage
    file, if one has been supplied. Expected columns:
    forecast_date, target_month, measure, nowcast_value, vintage_date.

    Absent an official downloadable history, this stays disabled -- see
    docs/methodology.md for why scraping today's published value cannot
    stand in for history.
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(
            "Cleveland Fed Inflation Nowcast CSV was not found. Inflation Model C is disabled.",
            stacklevel=2,
        )
        return None
    df = pd.read_csv(path)
    required = {"forecast_date", "target_month", "measure", "nowcast_value", "vintage_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    for col in ("forecast_date", "target_month", "vintage_date"):
        df[col] = pd.to_datetime(df[col])
    return df.sort_values("forecast_date").reset_index(drop=True)
