"""Computes a `BenchmarkSeries`/`BenchmarkDataStatus` from a
`BenchmarkDefinition`, reusing the existing, UNMODIFIED
`backtest.engine.{constant_target_weights, run_strategy}` for any
blended (multi-ticker, rebalanced) benchmark, and
`data.asset_prices.AssetPriceClient` for every network fetch -- exactly
the same building blocks `fast_crisis.backtest` and every `.analysis/`
experiment in this project already use. `project_6040` (UI-invisible,
research-only) is not computed here at all: it is read straight out of
an already-run `FastCrisisBacktest.results[BENCHMARK_LABEL]` via
`project_6040_series_from_backtest`, since its `growth_basket` component
needs the same KODEX200/FX construction the strategy itself uses and
there is no reason to build that a second time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macro_regime.backtest.engine import StrategyResult, constant_target_weights, run_strategy
from macro_regime.benchmarks.definitions import BenchmarkDefinition, get
from macro_regime.data.asset_prices import AssetPriceApiError, AssetPriceClient

TICKER_MAP = {"spy": "SPY", "agg": "AGG", "malox": "MALOX"}


@dataclass
class BenchmarkDataStatus:
    id: str
    available: bool
    earliest_date: pd.Timestamp | None
    latest_date: pd.Timestamp | None
    is_stale: bool
    stale_days: int
    error: str | None = None


@dataclass
class BenchmarkSeries:
    id: str
    returns: pd.Series  # daily, post-cost where a cost applies
    value: pd.Series  # NAV, starts at (1 + first day's return)
    turnover: pd.Series | None
    is_stale: pd.Series | None  # per-day stale flag (mutual-fund benchmarks only)
    status: BenchmarkDataStatus


def _unavailable(benchmark_id: str, error: str) -> BenchmarkSeries:
    return BenchmarkSeries(
        id=benchmark_id,
        returns=pd.Series(dtype=float),
        value=pd.Series(dtype=float),
        turnover=None,
        is_stale=None,
        status=BenchmarkDataStatus(
            id=benchmark_id,
            available=False,
            earliest_date=None,
            latest_date=None,
            is_stale=False,
            stale_days=0,
            error=error,
        ),
    )


def compute_benchmark_series(
    benchmark_id: str,
    *,
    strategy_calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    client: AssetPriceClient | None = None,
    refresh_cache: bool = False,
) -> BenchmarkSeries:
    definition = get(benchmark_id)
    client = client or AssetPriceClient()
    try:
        if definition.category == "mutual_fund":
            return _compute_mutual_fund_series(definition, client, strategy_calendar, as_of, refresh_cache)
        if definition.category == "single_asset":
            return _compute_single_asset_series(definition, client, strategy_calendar, as_of, refresh_cache)
        return _compute_blended_series(definition, client, strategy_calendar, as_of, refresh_cache)
    except AssetPriceApiError as exc:
        return _unavailable(benchmark_id, str(exc))


def _fetch_on_calendar(
    ticker_key: str,
    client: AssetPriceClient,
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    refresh_cache: bool,
) -> pd.Series:
    ticker = TICKER_MAP[ticker_key]
    close = client.get_daily_close(ticker, "2000-01-01", refresh_cache=refresh_cache)
    close = close.loc[close.index <= as_of]
    return close.reindex(calendar, method="ffill").pct_change()


def _compute_single_asset_series(
    definition: BenchmarkDefinition,
    client: AssetPriceClient,
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    refresh_cache: bool,
) -> BenchmarkSeries:
    ticker_key = next(iter(definition.components))
    ticker = TICKER_MAP[ticker_key]
    close = client.get_daily_close(ticker, "2000-01-01", refresh_cache=refresh_cache)
    close = close.loc[close.index <= as_of]
    common = calendar.intersection(close.index)
    returns = close.reindex(common).pct_change()
    value = (1 + returns.fillna(0.0)).cumprod()
    status = BenchmarkDataStatus(
        id=definition.id,
        available=True,
        earliest_date=close.index.min(),
        latest_date=close.index.max(),
        is_stale=False,
        stale_days=0,
    )
    return BenchmarkSeries(
        id=definition.id, returns=returns, value=value, turnover=None, is_stale=None, status=status
    )


def _compute_blended_series(
    definition: BenchmarkDefinition,
    client: AssetPriceClient,
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    refresh_cache: bool,
) -> BenchmarkSeries:
    ticker_returns = {}
    for ticker_key in definition.components:
        close = client.get_daily_close(TICKER_MAP[ticker_key], "2000-01-01", refresh_cache=refresh_cache)
        close = close.loc[close.index <= as_of]
        ticker_returns[ticker_key] = close.pct_change()

    valid = pd.Series(True, index=calendar)
    for s in ticker_returns.values():
        valid &= s.reindex(calendar).notna()
    common = calendar[valid]
    if len(common) == 0:
        return _unavailable(definition.id, "No common valid trading day across benchmark components")

    asset_returns = pd.DataFrame({k: v.reindex(common) for k, v in ticker_returns.items()})
    target = constant_target_weights(definition.components, common, list(definition.components))
    month_period = common.to_period("M")
    is_new_month = pd.Series(month_period != pd.Series(month_period).shift(1).to_numpy(), index=common)
    is_new_month.iloc[0] = True
    cost_bps = definition.transaction_cost_bps or 0.0
    result: StrategyResult = run_strategy(asset_returns, target, is_new_month, transaction_cost_bps=cost_bps)

    status = BenchmarkDataStatus(
        id=definition.id,
        available=True,
        earliest_date=common.min(),
        latest_date=common.max(),
        is_stale=False,
        stale_days=0,
    )
    return BenchmarkSeries(
        id=definition.id,
        returns=result.returns_post_cost,
        value=result.value_post_cost,
        turnover=result.turnover,
        is_stale=None,
        status=status,
    )


def _compute_mutual_fund_series(
    definition: BenchmarkDefinition,
    client: AssetPriceClient,
    calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    refresh_cache: bool,
) -> BenchmarkSeries:
    ticker_key = next(iter(definition.components))
    close = client.get_daily_close(TICKER_MAP[ticker_key], "2000-01-01", refresh_cache=refresh_cache)
    close = close.loc[close.index <= as_of]
    native_earliest, native_latest = close.index.min(), close.index.max()

    # Align onto the strategy's own calendar for charting/comparison, but
    # NEVER via backward-fill (a future NAV must never fill a past date)
    # and never treat a held-forward day as a real return: ffill the
    # NAV, and pct_change() on the ffilled series is exactly 0.0 on any
    # day the fund did not actually publish a new NAV -- no fabricated
    # return, by construction, not by a special case.
    common_calendar = calendar[
        (calendar >= native_earliest) & (calendar <= min(calendar.max(), native_latest))
    ]
    aligned_close = close.reindex(common_calendar).ffill()
    returns = aligned_close.pct_change()
    value = (1 + returns.fillna(0.0)).cumprod()

    native_days = set(close.index)
    is_stale = pd.Series([d not in native_days for d in common_calendar], index=common_calendar)

    strategy_latest = calendar.max()
    stale_days = int((strategy_latest - native_latest).days) if native_latest < strategy_latest else 0
    status = BenchmarkDataStatus(
        id=definition.id,
        available=True,
        earliest_date=native_earliest,
        latest_date=native_latest,
        is_stale=stale_days > 0,
        stale_days=stale_days,
    )
    return BenchmarkSeries(
        id=definition.id, returns=returns, value=value, turnover=None, is_stale=is_stale, status=status
    )


def project_6040_series_from_backtest(bt, benchmark_label: str) -> BenchmarkSeries:
    """Reads the project's own static 60/40 straight out of an
    already-run `FastCrisisBacktest` (built by the unmodified
    `run_fast_crisis_backtest`) instead of re-fetching/re-blending its
    Growth-Basket(SPY+KODEX200) component a second time."""
    result: StrategyResult = bt.results[benchmark_label]
    status = BenchmarkDataStatus(
        id="project_6040",
        available=True,
        earliest_date=result.returns_post_cost.index.min(),
        latest_date=result.returns_post_cost.index.max(),
        is_stale=False,
        stale_days=0,
    )
    return BenchmarkSeries(
        id="project_6040",
        returns=result.returns_post_cost,
        value=result.value_post_cost,
        turnover=result.turnover,
        is_stale=None,
        status=status,
    )
