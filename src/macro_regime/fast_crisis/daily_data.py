"""Daily price/return loading for the Fast Crisis Overlay (v1.2).

Trading-calendar convention (explicit, since NYSE and KRX differ):
- The canonical daily index is SPY's own trading-day index (a proxy for
  the NYSE calendar) -- all US-listed ETFs (SPY/HYG/DBC/BIL/LQD/IEF/TLT/
  GLD/TIP) share this calendar exactly (same exchange).
- 069500.KS (KRX) and KRW=X (near-continuous FX) are re-indexed onto the
  NYSE calendar using an as-of join: each NYSE day uses the most recent
  KRX/FX close ON OR BEFORE that day. This never uses a future KRX/FX
  price, only carries the last confirmed one forward across a KRX
  holiday that doesn't coincide with a US holiday. Whenever KRX/FX trade
  on a date the NYSE calendar has no session for (e.g. a US holiday that
  is not a KRX holiday), that extra KRX/FX movement is attributed to the
  next NYSE trading day's return -- a trading-day-alignment difference
  from the monthly engine (`backtest/assets.py`, which samples each
  asset's own last trading day of the month), not a bug. See
  docs/methodology.md, "Fast Crisis Overlay (v1.2)" for a worked example
  (2010-05-31).
- No values are ever invented for a day before a series' own first valid
  observation (no backfill).

Missing-value handling: a raw price series can have an index entry that
already exists (a real trading day) but whose value is NaN -- a data
provider gap, not a missing calendar date. Plain `reindex(calendar,
method="ffill")` only fills index positions absent from the *source*; it
passes an already-present NaN cell through unchanged. Every series is
therefore `.ffill()`-ed (carrying the last known value forward, never
fabricating, never filling before a series' own first valid value) BEFORE
the as-of join onto the canonical calendar. `validate_no_gaps_in_range`
then fails loudly, naming the offending asset(s) and date(s), if any NaN
remains in the range actually used for a backtest -- a gap this large in
raw data must be seen, never silently compounded through as an implicit
0% return.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.data.asset_prices import AssetPriceClient

US_TICKER_COLUMNS = {
    "spy": "SPY",
    "high_yield": "HYG",
    "investment_grade": "LQD",
    "intermediate_treasury": "IEF",
    "long_treasury": "TLT",
    "gold": "GLD",
    "tbills": "BIL",
    "commodities": "DBC",
    "tips": "TIP",
}
KODEX200_USD_COLUMN = "kodex200_usd"


def load_daily_prices(
    config: dict,
    vix_daily: pd.Series,
    *,
    client: AssetPriceClient | None = None,
    fetch_start: str = "2008-01-01",
    refresh_cache: bool = False,
) -> dict:
    """Fetch all daily adjusted closes needed for the Fast Crisis Overlay.
    `vix_daily` is VIXCLS as already fetched into `fred_wide.csv`
    (`config/default.yaml`, `series.diagnostics`) -- reused the same way
    DGS10/T10YIE feed the BEI Duration Risk Gate, rather than issuing a
    second, redundant live FRED call for a series `fetch` already pulled.
    Returns a dict of raw daily Series, each on its OWN native
    trading-day index (not yet aligned)."""
    client = client or AssetPriceClient()

    us_prices = {
        name: client.get_daily_close(ticker, fetch_start, refresh_cache=refresh_cache)
        for name, ticker in US_TICKER_COLUMNS.items()
    }

    gb_conf = config["growth_basket"]
    kodex_krw = client.get_daily_close(gb_conf["kodex200_ticker"], fetch_start, refresh_cache=refresh_cache)
    fx = client.get_daily_close(gb_conf["fx_ticker"], fetch_start, refresh_cache=refresh_cache)

    vix = vix_daily.dropna().sort_index()

    return {"us_prices": us_prices, "kodex_krw": kodex_krw, "fx": fx, "vix": vix}


def build_daily_return_matrix(
    raw: dict, *, as_of: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Align everything onto SPY's own trading-day calendar, as-of-join
    KODEX/FX onto it, compute daily returns for all 10 portfolio return
    columns (spy, kodex200_usd, high_yield, investment_grade,
    intermediate_treasury, long_treasury, gold, tbills, commodities,
    tips), and compute the daily-aligned VIX series (also as-of-joined,
    since VIXCLS is a FRED series on its own US-business-day calendar
    that can differ from NYSE on a small number of days).

    `as_of` (defaults to now) excludes the still-in-progress last trading
    day if data for it is only partially available -- the daily
    counterpart of the project's existing "never use an incomplete
    period" rule.
    """
    as_of = as_of or pd.Timestamp.now()
    us_prices = dict(raw["us_prices"])
    calendar = us_prices["spy"].index
    for name, s in us_prices.items():
        # `.ffill()` first repairs any internal NaN at an ALREADY-PRESENT
        # date (a provider data gap on a real trading day) -- plain
        # `reindex(..., method="ffill")` only fills index positions
        # absent from the source and silently passes an existing NaN
        # cell through unchanged. Carries forward the last known value
        # only, never fabricates, and is a no-op when there is no gap.
        us_prices[name] = s.ffill().reindex(calendar, method="ffill")

    kodex_on_calendar = raw["kodex_krw"].ffill().reindex(calendar, method="ffill")
    fx_on_calendar = raw["fx"].ffill().reindex(calendar, method="ffill")
    kodex_usd = kodex_on_calendar / fx_on_calendar

    price_df = pd.DataFrame({name: s for name, s in us_prices.items()}, index=calendar)
    price_df[KODEX200_USD_COLUMN] = kodex_usd
    price_df = price_df.loc[price_df.index <= as_of]

    returns = price_df.pct_change()

    vix_on_calendar = raw["vix"].reindex(calendar, method="ffill")
    vix_on_calendar = vix_on_calendar.loc[vix_on_calendar.index <= as_of]

    return returns, vix_on_calendar


def determine_common_start(
    returns: pd.DataFrame, vix: pd.Series, *, not_before: pd.Timestamp
) -> pd.Timestamp:
    """First trading day on/after `not_before` where every return column
    and VIX has a valid (non-NaN) value. `not_before` anchors the search
    to the first calendar month a v1.1 monthly target actually exists for
    -- starting earlier would apply the overlay to days with no real
    v1.1 base allocation to act on. Does not itself guarantee there is no
    later gap; call `validate_no_gaps_in_range` on the trimmed backtest
    range for that."""
    valid_mask = returns.notna().all(axis=1) & vix.reindex(returns.index).notna()
    candidates = returns.index[valid_mask & (returns.index >= not_before)]
    if len(candidates) == 0:
        raise ValueError("No common valid trading day found on/after not_before.")
    return candidates.min()


def validate_no_gaps_in_range(returns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Raise loudly if any return column has a NaN anywhere in
    [start, end] -- the range actually used for a backtest. Catches a raw
    data provider gap (index present, value NaN) before it can reach a
    `.prod()`/`.cumprod()` call that would otherwise treat that day as an
    implicit 0% return, silently distorting a month's compounded return.
    Never silently drops or fills at this stage -- fails with the
    offending asset(s) and date(s) named."""
    window = returns.loc[(returns.index >= start) & (returns.index <= end)]
    bad = window.isna()
    if bad.any().any():
        offenders = {
            col: [str(d.date()) for d in window.index[bad[col]]] for col in window.columns if bad[col].any()
        }
        raise ValueError(f"NaN found in required backtest range {start.date()}..{end.date()}: {offenders}")
