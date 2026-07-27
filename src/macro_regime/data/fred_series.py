"""Series-level orchestration on top of FredClient: fetch + validate the
full set of series used by the pipeline, and persist metadata alongside
the observations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from macro_regime.data.fred_client import FredApiError, FredClient, SeriesMetadata


def all_series_ids(config: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for group in config["series"].values():
        ids.extend(group.keys())
    return ids


def validate_all_series(client: FredClient, series_ids: list[str]) -> dict[str, bool]:
    """Check that every configured series ID still resolves on FRED.

    Invalid IDs are reported, never silently swapped for something else --
    per project policy, a broken series must fail loudly, not be papered
    over with a substitute.
    """
    return {series_id: client.validate_series_exists(series_id) for series_id in series_ids}


def fetch_all_series(
    client: FredClient,
    series_ids: list[str],
    *,
    observation_start: str | None = None,
    observation_end: str | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch and concatenate observations for all requested series (long format)."""
    frames = []
    errors: dict[str, str] = {}
    for series_id in series_ids:
        try:
            df = client.get_series(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
                refresh_cache=refresh_cache,
            )
        except FredApiError as exc:
            errors[series_id] = str(exc)
            continue
        frames.append(df)

    if errors:
        raise FredApiError(f"Failed to fetch series: {errors}")

    if not frames:
        return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end", "series_id"])
    return pd.concat(frames, ignore_index=True)


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format (date, value, series_id, ...) observations to wide
    format (one column per series, indexed by date)."""
    pivoted = long_df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    return pivoted.sort_index()


def save_metadata(metadata: dict[str, SeriesMetadata], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {series_id: asdict(meta) for series_id, meta in metadata.items()}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, default=str)


def fetch_all_metadata(client: FredClient, series_ids: list[str]) -> dict[str, SeriesMetadata]:
    return {series_id: client.get_series_metadata(series_id) for series_id in series_ids}


def get_vintage_panel(
    client: FredClient,
    series_id: str,
    as_of_dates: list[str],
) -> pd.DataFrame:
    """ALFRED-style real-time vintage panel: for each `as_of` date, the
    values of `series_id` as they were known/published at that time.

    TODO (not implemented in this build): a full vintage panel requires
    calling `get_series` once per as_of date with
    `realtime_start=realtime_end=as_of_date`, which is expensive
    (one API call per vintage x series) and needs its own caching and
    rate-limit strategy. `FredClient.get_series` already accepts
    `realtime_start`/`realtime_end` so this can be built directly on top
    of it without changing the client. Until this is implemented, any
    evaluation using current (non-vintage) FRED data must be labeled a
    "revised-data backtest" -- see docs/methodology.md -- and must not be
    presented as a real-time / ALFRED-vintage result.
    """
    raise NotImplementedError(
        "Full ALFRED vintage panels are not implemented in this build. "
        "Use FredClient.get_series(..., realtime_start=..., realtime_end=...) "
        "directly for ad-hoc vintage queries, and label any resulting "
        "backtest a 'revised-data backtest', not a real-time backtest."
    )
