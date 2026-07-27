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
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

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
        cache: FileCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.use_cache = use_cache
        cache_dir = self.settings.cache_dir.parent / "asset_cache"
        self.cache = cache or FileCache(cache_dir)

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
            hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
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
