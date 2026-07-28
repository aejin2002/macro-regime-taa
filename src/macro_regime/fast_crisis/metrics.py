"""Daily-frequency performance metrics for the Fast Crisis Overlay
(v1.2).

`backtest.metrics` hard-codes a MONTHLY annualization factor (n/12 for
CAGR, sqrt(12) for Sharpe/Sortino/vol) -- feeding it a daily return
series silently understates every annualized figure by roughly the
ratio of trading days to months (~21x). This module reimplements the
same formulas (CAGR = annualized compounded growth, Sharpe/Sortino =
annualized mean/downside-deviation of excess return, Calmar = CAGR /
|MaxDD|) with the standard 252-trading-day-per-year convention instead.
`max_drawdown`, `average_annual_turnover`, and `annual_positive_ratio`
are genuinely frequency-agnostic (no hard-coded period count) and are
reused directly from `backtest.metrics`, unmodified.

Daily and monthly metrics are never mixed on the same series -- a
function here always takes a daily-indexed input, and this module never
imports or calls anything in `backtest.metrics` that assumes monthly
periods (`cagr`, `annualized_volatility`, `sharpe_ratio`,
`sortino_ratio`, `calmar_ratio`).

v1.1's own MaxDD has two distinct, non-interchangeable readings, both
reported by `summarize_strategy_daily` and always kept in separately
labeled columns: **daily-close MaxDD** (`max_drawdown_daily_close_*`,
from the full daily NAV path -- sees the actual worst intraday-month
level) and **month-end MaxDD** (`max_drawdown_month_end_*`, from the NAV
sampled only at each calendar month's last trading day -- what the
legacy monthly v1.1 backtest itself reports). They differ because the
monthly engine only observes month-end levels; see
docs/methodology.md, "Fast Crisis Overlay (v1.2)".
"""

from __future__ import annotations

import pandas as pd

from macro_regime.backtest.engine import StrategyResult
from macro_regime.backtest.metrics import annual_positive_ratio, average_annual_turnover, max_drawdown

TRADING_DAYS_PER_YEAR = 252


def cagr_daily(value_series: pd.Series) -> float:
    n_days = len(value_series)
    if n_days == 0 or value_series.iloc[-1] <= 0:
        return float("nan")
    years = n_days / TRADING_DAYS_PER_YEAR
    return float(value_series.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_volatility_daily(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * (TRADING_DAYS_PER_YEAR**0.5))


def sharpe_ratio_daily(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    excess = (returns - risk_free_returns.reindex(returns.index)).dropna()
    std = excess.std(ddof=1)
    if excess.empty or not std:
        return float("nan")
    return float(excess.mean() / std * (TRADING_DAYS_PER_YEAR**0.5))


def sortino_ratio_daily(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    excess = (returns - risk_free_returns.reindex(returns.index)).dropna()
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if excess.empty or not downside_std:
        return float("nan")
    return float(excess.mean() / downside_std * (TRADING_DAYS_PER_YEAR**0.5))


def calmar_ratio_daily(value_series: pd.Series) -> float:
    """Calmar computed against the DAILY-close MaxDD (the more
    conservative of the two readings) -- see this module's docstring."""
    mdd = max_drawdown(value_series)
    if not mdd:
        return float("nan")
    return float(cagr_daily(value_series) / abs(mdd))


def daily_win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def month_end_value(value_series: pd.Series) -> pd.Series:
    """Re-samples a daily NAV/value series at each calendar month's last
    trading day present in the index -- the same observation points the
    legacy monthly v1.1 backtest itself uses. Never interpolates or
    invents a value; only ever selects an actual observed daily value."""
    if value_series.empty:
        return value_series
    last_day_per_month = value_series.index.to_series().groupby(value_series.index.to_period("M")).max()
    return value_series.loc[last_day_per_month.to_numpy()]


def summarize_strategy_daily(name: str, result: StrategyResult, risk_free_returns: pd.Series) -> dict:
    row: dict = {"strategy": name}
    if len(result.returns_pre_cost) > 0:
        row["start_date"] = str(result.returns_pre_cost.index.min().date())
        row["end_date"] = str(result.returns_pre_cost.index.max().date())
    else:
        row["start_date"] = None
        row["end_date"] = None

    for label, returns, value in [
        ("pre_cost", result.returns_pre_cost, result.value_pre_cost),
        ("post_cost", result.returns_post_cost, result.value_post_cost),
    ]:
        row[f"cagr_{label}"] = cagr_daily(value)
        row[f"annualized_vol_{label}"] = annualized_volatility_daily(returns)
        row[f"sharpe_{label}"] = sharpe_ratio_daily(returns, risk_free_returns)
        row[f"sortino_{label}"] = sortino_ratio_daily(returns, risk_free_returns)
        row[f"max_drawdown_daily_close_{label}"] = max_drawdown(value)
        row[f"max_drawdown_month_end_{label}"] = max_drawdown(month_end_value(value))
        row[f"calmar_{label}"] = calmar_ratio_daily(value)
        row[f"final_value_{label}"] = float(value.iloc[-1]) if len(value) else float("nan")

    row["daily_win_rate"] = daily_win_rate(result.returns_post_cost)
    row["annual_positive_year_ratio"] = annual_positive_ratio(result.returns_post_cost)
    row["avg_annual_turnover"] = average_annual_turnover(result.turnover)
    row["total_transaction_cost"] = float(
        (result.returns_pre_cost - result.returns_post_cost).clip(lower=0).sum()
    )
    return row


def build_summary_table_daily(
    results: dict[str, StrategyResult], risk_free_returns: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame([summarize_strategy_daily(name, r, risk_free_returns) for name, r in results.items()])
