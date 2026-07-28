"""Common-period performance metrics for a `BenchmarkSeries` (or the
strategy's own daily returns), reusing the existing, UNMODIFIED daily
metric formulas in `fast_crisis.metrics` -- nothing here recomputes a
formula that already exists elsewhere in the project.
"""

from __future__ import annotations

import pandas as pd

from macro_regime.backtest.metrics import average_annual_turnover, max_drawdown
from macro_regime.fast_crisis.metrics import (
    annualized_volatility_daily,
    cagr_daily,
    calmar_ratio_daily,
    month_end_value,
    sharpe_ratio_daily,
    sortino_ratio_daily,
)


def compute_benchmark_metrics(
    returns: pd.Series, risk_free: pd.Series, *, turnover: pd.Series | None = None
) -> dict:
    if returns.empty:
        return {"n_observations": 0}
    value = (1 + returns).cumprod()
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    row = {
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "n_observations": len(returns),
        "cumulative_return": float(value.iloc[-1] - 1.0),
        "cagr": cagr_daily(value),
        "annualized_vol": annualized_volatility_daily(returns),
        "sharpe": sharpe_ratio_daily(returns, risk_free.reindex(returns.index)),
        "sortino": sortino_ratio_daily(returns, risk_free.reindex(returns.index)),
        "max_drawdown_daily_close": max_drawdown(value),
        "max_drawdown_month_end": max_drawdown(month_end_value(value)),
        "calmar": calmar_ratio_daily(value),
        "final_wealth": float(value.iloc[-1]),
        "positive_month_ratio": float((monthly > 0).mean()) if len(monthly) else float("nan"),
        "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
        "best_month": float(monthly.max()) if len(monthly) else float("nan"),
    }
    if turnover is not None:
        row["avg_annual_turnover"] = average_annual_turnover(turnover)
    return row


def beta_correlation_to_spy(returns: pd.Series, spy_returns: pd.Series) -> tuple[float, float]:
    monthly_port = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    monthly_spy = (1 + spy_returns).groupby(spy_returns.index.to_period("M")).prod() - 1
    aligned = pd.concat([monthly_port, monthly_spy], axis=1, join="inner")
    aligned.columns = ["port", "spy"]
    if len(aligned) < 3:
        return float("nan"), float("nan")
    var = aligned["spy"].var(ddof=1)
    beta = float(aligned.cov().iloc[0, 1] / var) if var else float("nan")
    corr = float(aligned["port"].corr(aligned["spy"]))
    return beta, corr


def capture_ratios(returns: pd.Series, spy_returns: pd.Series) -> tuple[float, float]:
    monthly_port = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    monthly_spy = (1 + spy_returns).groupby(spy_returns.index.to_period("M")).prod() - 1
    aligned = pd.concat([monthly_port, monthly_spy], axis=1, join="inner")
    aligned.columns = ["port", "spy"]
    up = aligned[aligned["spy"] > 0]
    down = aligned[aligned["spy"] < 0]
    up_spy = float((1 + up["spy"]).prod() - 1) if len(up) else 0.0
    up_port = float((1 + up["port"]).prod() - 1) if len(up) else 0.0
    down_spy = float((1 + down["spy"]).prod() - 1) if len(down) else 0.0
    down_port = float((1 + down["port"]).prod() - 1) if len(down) else 0.0
    upside = up_port / up_spy if (len(up) and up_spy != 0) else float("nan")
    downside = down_port / down_spy if (len(down) and down_spy != 0) else float("nan")
    return upside, downside
