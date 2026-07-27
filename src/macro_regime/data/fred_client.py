"""Minimal client for the FRED (Federal Reserve Economic Data) v1 REST API.

Design notes
------------
- `get_series` returns real-time-as-of-today data. Its `realtime_start` /
  `realtime_end` parameters are accepted and threaded through to the API
  today, but this build does NOT implement a full ALFRED vintage panel
  (i.e. "what did this series look like as of date X"). See
  `fred_series.get_vintage_panel` for the (stubbed) vintage-aware
  interface and the TODO documented there and in docs/methodology.md.
- Missing observations from FRED are encoded as the string "." — these
  are converted to NaN.
- Network calls retry on transient failures (timeouts, 5xx) with
  exponential backoff; 4xx errors (bad series ID, bad API key) raise
  immediately since retrying will not help.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from macro_regime.config import Settings, get_settings
from macro_regime.data.cache import FileCache

OBSERVATION_COLUMNS = ["date", "value", "realtime_start", "realtime_end", "series_id"]


class FredApiError(RuntimeError):
    """Raised for non-retryable FRED API errors (bad series, bad key, etc.)."""


class FredHttpError(RuntimeError):
    """Raised when all retries for a FRED HTTP request are exhausted."""


@dataclass
class SeriesMetadata:
    series_id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    source: str = "FRED (Federal Reserve Bank of St. Louis)"
    notes: str | None = None


class FredClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str = "https://api.stlouisfed.org/fred",
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        cache: FileCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.require_fred_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.use_cache = use_cache
        self.cache = cache or FileCache(self.settings.cache_dir)

    # -- low-level HTTP -----------------------------------------------------

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path}"
        query = {**params, "api_key": self.api_key, "file_type": "json"}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, params=query, timeout=self.timeout_seconds)
            except requests.exceptions.Timeout as exc:
                last_error = exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    return result
                if 400 <= response.status_code < 500:
                    raise FredApiError(
                        f"FRED API request to {path} failed with "
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                last_error = FredHttpError(
                    f"FRED API request to {path} failed with HTTP {response.status_code}"
                )

            if attempt < self.max_retries:
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        raise FredHttpError(
            f"FRED API request to {path} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    # -- public API -----------------------------------------------------

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        *,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        """Fetch observations for one series.

        Returns a DataFrame with columns:
        date, value, realtime_start, realtime_end, series_id

        `realtime_start`/`realtime_end` default to FRED's own default
        ("today's" vintage) when omitted. They are exposed here so that
        ALFRED vintage queries can be layered on later without changing
        this function's signature.
        """
        params: dict[str, Any] = {"series_id": series_id}
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        if realtime_start:
            params["realtime_start"] = realtime_start
        if realtime_end:
            params["realtime_end"] = realtime_end

        cached = None if refresh_cache else (self.cache.get(series_id, params) if self.use_cache else None)
        if cached is not None:
            payload = cached
        else:
            payload = self._request("series/observations", params)
            if self.use_cache:
                self.cache.set(series_id, params, payload)

        observations = payload.get("observations", [])
        if not observations:
            return pd.DataFrame(columns=OBSERVATION_COLUMNS)

        df = pd.DataFrame(observations)
        df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["series_id"] = series_id
        return df[OBSERVATION_COLUMNS]

    def get_series_metadata(self, series_id: str) -> SeriesMetadata:
        """Fetch and validate series metadata. Raises FredApiError if the
        series ID does not exist -- callers must not silently substitute
        another series when this fails."""
        payload = self._request("series", {"series_id": series_id})
        series_list = payload.get("seriess", [])
        if not series_list:
            raise FredApiError(f"Series '{series_id}' was not found on FRED.")
        info = series_list[0]
        return SeriesMetadata(
            series_id=series_id,
            title=info.get("title", ""),
            frequency=info.get("frequency_short", info.get("frequency", "")),
            units=info.get("units_short", info.get("units", "")),
            seasonal_adjustment=info.get("seasonal_adjustment_short", ""),
            notes=info.get("notes"),
        )

    def validate_series_exists(self, series_id: str) -> bool:
        try:
            self.get_series_metadata(series_id)
        except FredApiError:
            return False
        return True
