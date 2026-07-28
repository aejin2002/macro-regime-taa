"""Display-only formatting helpers -- dates, percentages, ratios, and
status strings. No strategy computation lives here; every function takes
an already-computed value and returns a display string."""

from __future__ import annotations

import pandas as pd

UNKNOWN_DISPLAY = "n/a"


def fmt_date(value) -> str:
    if value is None:
        return UNKNOWN_DISPLAY
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return UNKNOWN_DISPLAY
    return pd.Timestamp(value).date().isoformat()


def fmt_pct(value: float | None, decimals: int = 2, *, signed: bool = False) -> str:
    """`value` is a fraction (0.24 == 24%), matching every backend field
    in this project -- never pre-multiplied by the caller."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_DISPLAY
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def fmt_num(value: float | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_DISPLAY
    return f"{value:.{decimals}f}"


def fmt_int(value: int | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_DISPLAY
    return str(int(value))


def fmt_status(value: str | None) -> str:
    if not value or value == "nan":
        return "UNKNOWN"
    return value


def fmt_direction(current: float | None, previous: float | None) -> str:
    """Up / Down / Flat, purely from already-computed values -- not a
    new signal, just a display label for a value that already changed
    (or didn't) between two already-computed observations."""
    if current is None or previous is None:
        return UNKNOWN_DISPLAY
    if pd.isna(current) or pd.isna(previous):
        return UNKNOWN_DISPLAY
    diff = current - previous
    if abs(diff) < 1e-12:
        return "Flat"
    return "Up" if diff > 0 else "Down"


def arrow_for(direction: str) -> str:
    return {"Up": "↑", "Down": "↓", "Flat": "→"}.get(direction, "")
