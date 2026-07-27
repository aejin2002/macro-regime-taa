"""Monthly rebalancing engine shared by all 5 backtest strategies.

Rules implemented here, exactly as specified (no tuning):
- Monthly rebalance to the month's target weights, even if the target is
  identical to last month's (rule: restore to target every month-end
  regardless of whether the signal changed).
- Return for month `t` uses only the weights entering month `t` (i.e.
  known/decided before `t`'s return is realized) and month `t`'s asset
  returns -- never a later month's regime or price.
- Turnover = sum(|target - drifted_prior_weight|) / 2 ("one-way"),
  applied uniformly at every rebalance including the very first one
  (prior weights implicitly all-cash/0 before month 0).
- Transaction cost = turnover x `transaction_cost_bps`, deducted from the
  same month's return it was incurred to enter.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyResult:
    weights: pd.DataFrame
    returns_pre_cost: pd.Series
    returns_post_cost: pd.Series
    turnover: pd.Series
    value_pre_cost: pd.Series
    value_post_cost: pd.Series


def constant_target_weights(weights: dict[str, float], index: pd.Index, columns: list[str]) -> pd.DataFrame:
    """Broadcast a single {ticker: weight} dict to every date in `index`."""
    row = pd.Series(weights, index=columns).fillna(0.0)
    return pd.DataFrame([row] * len(index), index=index, columns=columns)


def build_target_weights_from_regime(
    regime_series: pd.Series,
    allocations_by_regime: dict[str, dict[str, float]],
    columns: list[str],
) -> pd.DataFrame:
    """{date: allocations_by_regime[tradable_regime[date]]}, expanded to
    the full ticker-column universe. An unrecognized regime label falls
    back to the UNKNOWN allocation (100% tbills) rather than erroring --
    the same conservative default the regime itself uses."""
    unknown_weights = allocations_by_regime["UNKNOWN"]
    rows = []
    for regime in regime_series:
        weights = allocations_by_regime.get(regime, unknown_weights)
        rows.append(pd.Series(weights, index=columns).fillna(0.0))
    return pd.DataFrame(rows, index=regime_series.index, columns=columns)


def run_strategy(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance_mask: pd.Series,
    *,
    transaction_cost_bps: float,
) -> StrategyResult:
    columns = list(asset_returns.columns)
    dates = asset_returns.index

    weights_held = pd.DataFrame(0.0, index=dates, columns=columns)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    returns_pre_cost = pd.Series(0.0, index=dates, dtype=float)

    prior_drifted = pd.Series(0.0, index=columns)  # implicit all-cash before the first month
    for t in dates:
        if bool(rebalance_mask.loc[t]):
            current = target_weights.loc[t].reindex(columns).fillna(0.0)
        else:
            current = prior_drifted
        turnover.loc[t] = float((current - prior_drifted).abs().sum() / 2.0)
        weights_held.loc[t] = current

        month_ret = asset_returns.loc[t].reindex(columns).fillna(0.0)
        returns_pre_cost.loc[t] = float((current * month_ret).sum())

        grown = current * (1.0 + month_ret)
        total = float(grown.sum())
        prior_drifted = grown / total if total != 0 else grown

    cost_fraction = turnover * (transaction_cost_bps / 10000.0)
    returns_post_cost = returns_pre_cost - cost_fraction

    value_pre_cost = (1.0 + returns_pre_cost).cumprod()
    value_post_cost = (1.0 + returns_post_cost).cumprod()

    return StrategyResult(
        weights=weights_held,
        returns_pre_cost=returns_pre_cost,
        returns_post_cost=returns_post_cost,
        turnover=turnover,
        value_pre_cost=value_pre_cost,
        value_post_cost=value_post_cost,
    )


def run_regime_strategy(
    asset_returns: pd.DataFrame,
    tradable_regime: pd.Series,
    allocations_by_regime: dict[str, dict[str, float]],
    *,
    transaction_cost_bps: float,
) -> StrategyResult:
    columns = list(asset_returns.columns)
    regime_aligned = tradable_regime.reindex(asset_returns.index)
    target = build_target_weights_from_regime(regime_aligned, allocations_by_regime, columns)
    rebalance_mask = pd.Series(True, index=asset_returns.index)
    return run_strategy(asset_returns, target, rebalance_mask, transaction_cost_bps=transaction_cost_bps)


def run_static_strategy(
    asset_returns: pd.DataFrame,
    target: dict[str, float],
    *,
    transaction_cost_bps: float,
) -> StrategyResult:
    columns = list(asset_returns.columns)
    target_df = constant_target_weights(target, asset_returns.index, columns)
    rebalance_mask = pd.Series(True, index=asset_returns.index)
    return run_strategy(asset_returns, target_df, rebalance_mask, transaction_cost_bps=transaction_cost_bps)


def run_buy_and_hold_strategy(
    asset_returns: pd.DataFrame,
    initial_weights: dict[str, float],
    *,
    transaction_cost_bps: float,
) -> StrategyResult:
    columns = list(asset_returns.columns)
    target_df = constant_target_weights(initial_weights, asset_returns.index, columns)
    rebalance_mask = pd.Series(False, index=asset_returns.index)
    if len(rebalance_mask) > 0:
        rebalance_mask.iloc[0] = True
    return run_strategy(asset_returns, target_df, rebalance_mask, transaction_cost_bps=transaction_cost_bps)
