"""Build the monthly total-return matrix for the backtest asset universe.

Columns: spy, kodex200_usd, high_yield, investment_grade,
intermediate_treasury, long_treasury, gold, tbills, commodities, tips.

`kodex200_usd` is the only derived column: KODEX 200 (069500.KS) trades in
KRW, so its USD-equivalent total return combines the fund's own KRW return
with the USD/KRW move over the same month -- both computed from actual
month-end closes only, never predicted or filled with a future rate.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.data.asset_prices import AssetPriceClient
from macro_regime.utils.dates import resample_to_monthly

KODEX200_USD_COLUMN = "kodex200_usd"


def _to_month_end(daily_close: pd.Series) -> pd.Series:
    """Last known trading-day close within each calendar month
    (`utils/dates.py::resample_to_monthly`, how="last"). A month with zero
    observations produces NaN; that gap is never forward- or
    backward-filled here or by any caller."""
    return resample_to_monthly(daily_close, how="last")


def _drop_incomplete_trailing_month(returns: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """`resample("ME").last()` labels a still-in-progress month with its
    *future* calendar month-end date, using whatever the latest available
    trading day happens to be -- e.g. if today is 2026-07-27 and the last
    fetched close is 2026-07-24, that row is stamped "2026-07-31" even
    though July has not finished. That row's "return" is a partial-month
    return, not a real one, and must be excluded rather than mistaken for
    a completed month. Only ever affects the single most-recent row."""
    return returns.loc[returns.index <= as_of]


def build_monthly_return_matrix(
    config: dict,
    *,
    client: AssetPriceClient | None = None,
    start: str = "2000-01-01",
    refresh_cache: bool = False,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Fetch every asset + FX series, resample to month-end, compute
    month-over-month total returns, drop any trailing not-yet-complete
    month (see `_drop_incomplete_trailing_month`), and trim to the common
    start date -- the first month-end where every column has a real
    (non-NaN) return.

    `as_of` defaults to the real current time; pass it explicitly in
    tests for a deterministic "today".

    Returns (monthly_return_matrix, common_start_date).
    """
    client = client or AssetPriceClient()
    as_of = as_of or pd.Timestamp.now()
    gb_conf = config["growth_basket"]
    assets_conf = config["backtest"]["assets"]

    prices: dict[str, pd.Series] = {}
    for category, ticker in assets_conf.items():
        prices[category] = _to_month_end(client.get_daily_close(ticker, start, refresh_cache=refresh_cache))

    kodex_krw_close = _to_month_end(
        client.get_daily_close(gb_conf["kodex200_ticker"], start, refresh_cache=refresh_cache)
    )
    fx_close = _to_month_end(client.get_daily_close(gb_conf["fx_ticker"], start, refresh_cache=refresh_cache))
    prices[KODEX200_USD_COLUMN] = kodex_krw_close / fx_close

    price_df = pd.concat(prices, axis=1).sort_index()
    returns = price_df.pct_change()
    returns = _drop_incomplete_trailing_month(returns, as_of)

    valid_mask = returns.notna().all(axis=1)
    if not valid_mask.any():
        raise ValueError("No common month-end date has valid returns for every asset in the universe.")
    common_start = pd.Timestamp(returns.index[valid_mask][0])
    returns = returns.loc[returns.index >= common_start]

    return returns, common_start
