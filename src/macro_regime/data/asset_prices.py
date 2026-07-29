"""Minimal client for fetching asset/FX price history via Yahoo Finance
(yfinance).

Design notes
------------
- Missing observations are simply absent from the returned series (Yahoo
  Finance has no sentinel like FRED's "."). No forward/backward filling is
  ever applied here -- callers get exactly what Yahoo Finance returned, and
  any fill-across-gaps decision belongs to the caller (see
  `backtest/assets.py`).
- `auto_adjust=True` is always used: Yahoo's adjusted close already reflects
  dividends and stock splits, giving a total-return-equivalent price series
  without any additional adjustment logic in this project.
- Cached to disk like `FredClient`, keyed by (ticker, start date), so
  repeated `run-backtest` invocations do not re-hit the network every time.
- Yahoo Finance rate-limits per source IP; shared-IP hosts (e.g. Streamlit
  Community Cloud) routinely trip this when fetching many tickers back to
  back. `YFRateLimitError` is retried with exponential backoff, mirroring
  `FredClient._request`'s retry convention, and each live (non-cached)
  request is paced against the previous one to reduce the chance of
  triggering the limit at all. Any other error is not retried.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from macro_regime.config import Settings, get_settings
from macro_regime.data.cache import FileCache


class AssetPriceApiError(RuntimeError):
    """Raised when Yahoo Finance returns no data for a ticker -- callers must
    not silently substitute another ticker when this fails."""


class AssetPriceClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_retries: int = 4,
        backoff_seconds: float = 5.0,
        min_request_interval_seconds: float = 1.0,
        cache: FileCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.use_cache = use_cache
        cache_dir = self.settings.cache_dir.parent / "asset_cache"
        self.cache = cache or FileCache(cache_dir)
        self._last_request_monotonic: float | None = None
        # Guards `_last_request_monotonic` so concurrent callers (e.g. a
        # bounded ThreadPoolExecutor prefetching several tickers at once)
        # still collectively respect ONE shared >= min_request_interval_seconds
        # cadence against Yahoo Finance, rather than each thread racing
        # the same stale read and firing simultaneously.
        self._pacing_lock = threading.Lock()

    def _fetch_history(self, ticker: str, start: str) -> pd.DataFrame:
        """Fetch raw daily history for `ticker`, retrying on Yahoo
        Finance's per-IP rate limit with exponential backoff (same
        pattern as `FredClient._request`). Paces itself against the
        previous live request via `min_request_interval_seconds` --
        thread-safe, so multiple concurrent callers on the same client
        instance still serialize onto one shared pacing cadence. Any
        error other than `YFRateLimitError` is not retried and
        propagates immediately."""
        last_error: YFRateLimitError | None = None
        for attempt in range(1, self.max_retries + 1):
            with self._pacing_lock:
                if self._last_request_monotonic is not None:
                    elapsed = time.monotonic() - self._last_request_monotonic
                    if elapsed < self.min_request_interval_seconds:
                        time.sleep(self.min_request_interval_seconds - elapsed)
                self._last_request_monotonic = time.monotonic()
            try:
                return yf.Ticker(ticker).history(start=start, auto_adjust=True)
            except YFRateLimitError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise AssetPriceApiError(
            f"Yahoo Finance rate-limited requests for ticker '{ticker}' after "
            f"{self.max_retries} attempts. This is a shared-IP rate limit, not "
            f"a data or code issue -- wait a few minutes and reload."
        ) from last_error

    def get_daily_close(
        self,
        ticker: str,
        start: str,
        *,
        refresh_cache: bool = False,
    ) -> pd.Series:
        """Adjusted (split+dividend) daily close for `ticker` from `start` to
        today. Raises `AssetPriceApiError` if Yahoo Finance returns no data
        -- callers must not silently substitute a different ticker."""
        params = {"start": start}
        cached = None if refresh_cache else (self.cache.get(ticker, params) if self.use_cache else None)

        if cached is not None:
            records = cached["records"]
        else:
            hist = self._fetch_history(ticker, start)
            if hist.empty:
                raise AssetPriceApiError(
                    f"Yahoo Finance returned no data for ticker '{ticker}' from {start}."
                )
            records = [
                {"date": idx.strftime("%Y-%m-%d"), "close": float(val)}
                for idx, val in hist["Close"].items()
            ]
            if self.use_cache:
                self.cache.set(ticker, params, {"records": records})

        if not records:
            raise AssetPriceApiError(f"Yahoo Finance returned no data for ticker '{ticker}' from {start}.")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        series = df.set_index("date")["close"].sort_index()
        series.name = ticker
        return series

    def get_cache_status(self, ticker: str, start: str, *, ttl_seconds: float | None = None) -> dict | None:
        """Retrieved timestamp / cache version / last market date / TTL
        staleness for `ticker`'s cache entry at this `start`, without
        fetching anything. Returns None if nothing is cached yet for this
        exact (ticker, start) key -- callers should not assume a cache
        exists just because `get_daily_close` has been called for a
        DIFFERENT `start` value (each `start` is its own cache entry;
        see `FileCache`'s docstring)."""
        params = {"start": start}
        cached = self.cache.get(ticker, params)
        if cached is None:
            return None
        status = self.cache.cache_status(ticker, params, ttl_seconds=ttl_seconds) or {}
        records = cached.get("records") or []
        last_market_date = max((r["date"] for r in records), default=None)
        status["ticker"] = ticker
        status["start"] = start
        status["last_market_date"] = last_market_date
        return status
