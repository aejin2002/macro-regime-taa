"""Diagnostic-only intraday asset snapshot (5-minute bars via yfinance).

**Never used in any strategy calculation** -- Macro Regime, the BEI
Duration Risk Gate, and the Fast Crisis Overlay all run on daily-close
data through `macro_regime.fast_crisis.daily_data` /
`macro_regime.backtest.assets`, which this module never touches or
calls. This module exists purely so the Portfolio tab can show "how are
today's positions moving right now" as a monitoring aid.

Tickers are read directly from the project's own real asset mapping
(`fast_crisis.daily_data.US_TICKER_COLUMNS` + the configured KODEX/FX
tickers) -- nothing here is invented. Cached 5 minutes
(`st.cache_data(ttl=...)`) and isolated from the strategy-data cache, so
a live-price failure can never affect strategy output, and this cache is
never invalidated by "Recompute v1.2 now" (only by its own "Refresh
market data" button or the 5-minute TTL).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from macro_regime.fast_crisis.daily_data import US_TICKER_COLUMNS

INTRADAY_TTL_SECONDS = 300


def asset_ticker_map(config: dict) -> dict[str, str]:
    """{display_name: yfinance_ticker} for exactly the tickers the
    strategy actually uses -- the 9 US ETFs plus KODEX 200 and the
    USD/KRW rate. No ticker is added that the strategy doesn't hold."""
    gb_conf = config["growth_basket"]
    mapping = dict(US_TICKER_COLUMNS)
    mapping["kodex200 (KRW)"] = gb_conf["kodex200_ticker"]
    mapping["usd_krw"] = gb_conf["fx_ticker"]
    return mapping


def _extract_close(raw: pd.DataFrame) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        close_cols = [c for c in raw.columns if c[0] == "Close"]
        s = raw[close_cols[0]] if close_cols else raw.iloc[:, 0]
    elif "Close" in raw.columns:
        s = raw["Close"]
    else:
        s = raw.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("America/New_York").tz_localize(None)
    return s


@st.cache_data(ttl=INTRADAY_TTL_SECONDS, show_spinner=False)
def fetch_intraday_snapshot(ticker: str) -> dict:
    """One ticker's today-so-far 5-minute-bar snapshot. Never raises --
    any failure (unsupported ticker, closed market, network error)
    resolves to `status="unknown"` so the caller can render it as
    UNKNOWN rather than crash the page."""
    try:
        raw = yf.download(ticker, period="1d", interval="5m", auto_adjust=True, progress=False, threads=False)
    except Exception:  # noqa: BLE001 -- any fetch failure degrades to UNKNOWN, never propagates
        return {"status": "unknown", "ticker": ticker}

    if raw is None or raw.empty:
        return {"status": "unknown", "ticker": ticker}

    try:
        series = _extract_close(raw)
    except Exception:  # noqa: BLE001 -- malformed response degrades to UNKNOWN, never propagates
        return {"status": "unknown", "ticker": ticker}

    if series.empty:
        return {"status": "unknown", "ticker": ticker}

    open_price = float(series.iloc[0])
    current_price = float(series.iloc[-1])
    pct_change = (current_price / open_price - 1.0) if open_price else None
    return {
        "status": "ok",
        "ticker": ticker,
        "open_price": open_price,
        "current_price": current_price,
        "pct_change": pct_change,
        "last_updated": str(series.index[-1]),
        "series": series,
    }


def fetch_all_intraday(ticker_map: dict[str, str]) -> dict[str, dict]:
    """{display_name: snapshot} for every asset in `ticker_map`. Each
    ticker is fetched (and cached) independently, so one bad ticker
    (e.g. a KRX symbol yfinance can't serve intraday bars for) never
    blocks the others."""
    return {name: fetch_intraday_snapshot(ticker) for name, ticker in ticker_map.items()}


def clear_intraday_cache() -> None:
    fetch_intraday_snapshot.clear()
