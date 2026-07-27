"""Performance metrics for backtest strategy results.

All formulas are standard, unparameterized definitions -- nothing here is
fit, tuned, or chosen to flatter any particular strategy.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.backtest.engine import StrategyResult

CRISIS_YEARS = [2008, 2020, 2022]


def cagr(value_series: pd.Series) -> float:
    n_months = len(value_series)
    if n_months == 0 or value_series.iloc[-1] <= 0:
        return float("nan")
    years = n_months / 12.0
    return float(value_series.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * (12**0.5))


def sharpe_ratio(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    excess = (returns - risk_free_returns.reindex(returns.index)).dropna()
    std = excess.std(ddof=1)
    if excess.empty or not std:
        return float("nan")
    return float(excess.mean() / std * (12**0.5))


def sortino_ratio(returns: pd.Series, risk_free_returns: pd.Series) -> float:
    excess = (returns - risk_free_returns.reindex(returns.index)).dropna()
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if excess.empty or not downside_std:
        return float("nan")
    return float(excess.mean() / downside_std * (12**0.5))


def max_drawdown(value_series: pd.Series) -> float:
    if value_series.empty:
        return float("nan")
    running_max = value_series.cummax()
    drawdown = value_series / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(value_series: pd.Series) -> float:
    mdd = max_drawdown(value_series)
    if not mdd:
        return float("nan")
    return float(cagr(value_series) / abs(mdd))


def monthly_win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def annual_positive_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    annual = (1 + returns).groupby(returns.index.year).prod() - 1
    return float((annual > 0).mean()) if len(annual) else float("nan")


def average_annual_turnover(turnover: pd.Series) -> float:
    if turnover.empty:
        return float("nan")
    yearly = turnover.groupby(turnover.index.year).sum()
    return float(yearly.mean())


def summarize_strategy(name: str, result: StrategyResult, risk_free_returns: pd.Series) -> dict:
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
        row[f"cagr_{label}"] = cagr(value)
        row[f"annualized_vol_{label}"] = annualized_volatility(returns)
        row[f"sharpe_{label}"] = sharpe_ratio(returns, risk_free_returns)
        row[f"sortino_{label}"] = sortino_ratio(returns, risk_free_returns)
        row[f"max_drawdown_{label}"] = max_drawdown(value)
        row[f"calmar_{label}"] = calmar_ratio(value)
        row[f"final_value_{label}"] = float(value.iloc[-1]) if len(value) else float("nan")

    row["monthly_win_rate"] = monthly_win_rate(result.returns_post_cost)
    row["annual_positive_year_ratio"] = annual_positive_ratio(result.returns_post_cost)
    row["avg_annual_turnover"] = average_annual_turnover(result.turnover)
    return row


def build_summary_table(results: dict[str, StrategyResult], risk_free_returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([summarize_strategy(name, r, risk_free_returns) for name, r in results.items()])


def annual_returns_table(results: dict[str, StrategyResult], *, use_post_cost: bool = True) -> pd.DataFrame:
    cols = {}
    for name, r in results.items():
        returns = r.returns_post_cost if use_post_cost else r.returns_pre_cost
        cols[name] = (1 + returns).groupby(returns.index.year).prod() - 1
    return pd.DataFrame(cols)


def crisis_period_performance(annual_returns: pd.DataFrame) -> pd.DataFrame:
    """Rows for the specific calendar years the audit is required to
    cover (2008, 2020, 2022) -- whichever of them fall inside the
    common backtest sample."""
    available_years = [y for y in CRISIS_YEARS if y in annual_returns.index]
    return annual_returns.loc[available_years]


def regime_average_returns(
    asset_returns: pd.DataFrame,
    tradable_regime: pd.Series,
    portfolio_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """Average monthly return of every underlying asset and every
    strategy's portfolio, grouped by `tradable_regime`."""
    combined = asset_returns.copy()
    for name, returns in portfolio_returns.items():
        combined[f"portfolio::{name}"] = returns.reindex(asset_returns.index)
    combined["regime"] = tradable_regime.reindex(asset_returns.index)
    return combined.groupby("regime").mean(numeric_only=True)


def regime_agreement_rate(primary_regime: pd.Series, secondary_regime: pd.Series) -> float:
    """Fraction of months (on the overlapping sample) where Primary's and
    Secondary's `tradable_regime` agree exactly."""
    aligned = pd.concat(
        {"primary": primary_regime, "secondary": secondary_regime}, axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        return float("nan")
    return float((aligned["primary"] == aligned["secondary"]).mean())
