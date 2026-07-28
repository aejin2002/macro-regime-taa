"""Fast Crisis Overlay (v1.2) -- daily backtest orchestration.

Ties together: v1.1's own monthly target (Primary v1.0 regime allocation
+ BEI Duration Risk Gate, rebuilt from the existing, UNMODIFIED
`backtest.assets` / `backtest.allocations` / `backtest.engine` /
`duration_gate.signal` / `duration_gate.allocation` functions -- never
reimplemented here), the daily asset/VIX data
(`fast_crisis.daily_data`), the daily shock signals and entry/exit state
machine (`fast_crisis.signal`), the allocation overlay
(`fast_crisis.allocation`), the existing, UNMODIFIED monthly-rebalancing
engine (`backtest.engine.run_strategy` -- index-frequency-agnostic, so it
is reused as-is for a daily walk), and daily-frequency metrics
(`fast_crisis.metrics`).

v1.1's monthly BEI-gated target is broadcast onto the daily calendar with
a period-keyed direct mapping (`_broadcast_monthly_onto_daily`) -- NOT a
temporal `ffill`, which would leave the first ~20 trading days of every
month reading a stale/absent prior-month target (there is nothing before
a month's own start to ffill from). Each daily date looks up its OWN
calendar month's target directly.

Two strategies are run over the identical daily asset-return matrix:
**v1_1_daily** (v1.1's monthly target, broadcast onto the daily
calendar, rebalanced only at each month's first trading day -- the daily
restatement of the existing monthly v1.1 backtest, not a new strategy)
and **v1_2_fast_crisis** (v1_1_daily's target, with the Fast Crisis
Overlay applied on top, rebalanced at each month's first trading day AND
on every Crisis Mode entry/exit day). Never reads or writes any of
v1.0's `backtest_*.csv` or v1.1's `bei_duration_gate_*.csv` files.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macro_regime.backtest.allocations import regime_allocations
from macro_regime.backtest.assets import build_monthly_return_matrix
from macro_regime.backtest.engine import StrategyResult, build_target_weights_from_regime, run_strategy
from macro_regime.data.asset_prices import AssetPriceClient
from macro_regime.duration_gate.allocation import apply_bei_duration_gate
from macro_regime.duration_gate.signal import UNKNOWN as BEI_UNKNOWN
from macro_regime.duration_gate.signal import build_bei_duration_gate_signal
from macro_regime.fast_crisis.allocation import CASH_COLUMN, CRISIS_ASSET_COLUMNS, apply_fast_crisis_overlay
from macro_regime.fast_crisis.daily_data import (
    build_daily_return_matrix,
    determine_common_start,
    load_daily_prices,
    validate_no_gaps_in_range,
)
from macro_regime.fast_crisis.metrics import build_summary_table_daily, month_end_value
from macro_regime.fast_crisis.signal import (
    ON,
    build_crisis_mode_state,
    build_raw_trigger,
    compute_credit_shock,
    compute_equity_shock,
    compute_vix_shock,
)

V1_1_LABEL = "v1_1_daily"
V1_2_LABEL = "v1_2_fast_crisis"


@dataclass
class FastCrisisBacktest:
    signal_table: pd.DataFrame
    allocations: pd.DataFrame
    results: dict[str, StrategyResult]
    summary: pd.DataFrame
    daily_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    turnover_breakdown: pd.DataFrame
    current_status: dict


def _build_v1_1_monthly_target(
    config: dict, primary_regime: pd.Series, wide: pd.DataFrame, monthly_asset_returns: pd.DataFrame
) -> pd.DataFrame:
    """v1.1's own monthly target: Primary v1.0 regime allocation, with
    the BEI Duration Risk Gate applied on top -- byte-for-byte the same
    construction `duration_gate.backtest.run_bei_duration_gate_backtest`
    uses, reusing the same unmodified functions (not reimplemented)."""
    gconf = config["bei_duration_gate"]
    allocations_by_regime = regime_allocations(config)
    columns = list(monthly_asset_returns.columns)

    regime_aligned = primary_regime.reindex(monthly_asset_returns.index)
    base_target = build_target_weights_from_regime(regime_aligned, allocations_by_regime, columns)

    bei_signal = build_bei_duration_gate_signal(
        wide["DGS10"], wide["T10YIE"], monthly_asset_returns["long_treasury"]
    )
    tradable_gate = bei_signal["tradable_bei_duration_gate"].reindex(monthly_asset_returns.index)
    tradable_gate = tradable_gate.fillna(BEI_UNKNOWN)

    return apply_bei_duration_gate(
        base_target,
        tradable_gate,
        ief_weight_when_on=gconf["ief_weight_when_on"],
        bil_weight_when_on=gconf["bil_weight_when_on"],
    )


def _broadcast_monthly_onto_daily(monthly: pd.Series | pd.DataFrame, daily_index: pd.DatetimeIndex):
    """Maps each daily date to ITS OWN calendar month's monthly value --
    not a temporal ffill (which has nothing to fill the first ~20
    trading days of a month from)."""
    period_to_label = dict(zip(monthly.index.to_period("M"), monthly.index, strict=True))
    mapped_labels = [period_to_label[p] for p in daily_index.to_period("M")]
    result = monthly.loc[mapped_labels].copy()
    result.index = daily_index
    return result


def run_fast_crisis_backtest(
    config: dict,
    primary_regime: pd.Series,
    wide: pd.DataFrame,
    *,
    client: AssetPriceClient | None = None,
    as_of: pd.Timestamp | None = None,
    refresh_cache: bool = False,
) -> FastCrisisBacktest:
    as_of = as_of or pd.Timestamp.now()
    bconf = config["backtest"]
    fconf = config["fast_crisis"]
    cost_bps = bconf["transaction_cost_bps"]

    monthly_asset_returns, _ = build_monthly_return_matrix(config, client=client, as_of=as_of)
    v1_1_monthly_target = _build_v1_1_monthly_target(config, primary_regime, wide, monthly_asset_returns)

    raw = load_daily_prices(config, wide["VIXCLS"], client=client, refresh_cache=refresh_cache)
    daily_returns, vix_daily = build_daily_return_matrix(raw, as_of=as_of)

    first_month = v1_1_monthly_target.index.min()
    not_before = pd.Timestamp(f"{first_month.year}-{first_month.month:02d}-01")
    common_start = determine_common_start(daily_returns, vix_daily, not_before=not_before)
    end_date = min(daily_returns.index.max(), v1_1_monthly_target.index.max())
    in_range = (daily_returns.index >= common_start) & (daily_returns.index <= end_date)
    daily_index = daily_returns.index[in_range]
    daily_returns = daily_returns.loc[daily_index]
    vix_daily = vix_daily.loc[daily_index]
    validate_no_gaps_in_range(daily_returns, daily_index.min(), daily_index.max())

    risk_free_returns = daily_returns[bconf["risk_free_asset"]]

    # -- daily signals ----------------------------------------------------
    vix_shock = compute_vix_shock(
        vix_daily,
        threshold=fconf["vix_threshold"],
        ma_window_days=fconf["vix_ma_window_days"],
        ma_ratio_threshold=fconf["vix_ma_ratio_threshold"],
    )
    equity_shock = compute_equity_shock(
        daily_returns["spy"],
        window_days=fconf["equity_shock_window_days"],
        threshold=fconf["equity_shock_threshold"],
    )
    credit_shock = compute_credit_shock(
        daily_returns["high_yield"],
        window_days=fconf["credit_shock_window_days"],
        threshold=fconf["credit_shock_threshold"],
    )
    raw_trigger = build_raw_trigger(vix_shock, equity_shock, credit_shock)
    state = build_crisis_mode_state(
        raw_trigger, min_hold_days=fconf["min_hold_days"], min_off_days_to_exit=fconf["min_off_days_to_exit"]
    )
    crisis_mode = state["crisis_mode"]

    # -- v1.1 daily target (v1.1's monthly target broadcast onto the daily calendar) --
    v1_1_daily_target = _broadcast_monthly_onto_daily(v1_1_monthly_target, daily_index)
    regime_monthly = primary_regime.reindex(v1_1_monthly_target.index)
    macro_regime_daily = _broadcast_monthly_onto_daily(regime_monthly, daily_index)

    # -- v1.2 target: Fast Crisis Overlay on top of v1.1's daily target --
    v1_2_daily_target = apply_fast_crisis_overlay(v1_1_daily_target, crisis_mode)

    month_period = daily_index.to_period("M")
    is_new_month = pd.Series(month_period != pd.Series(month_period).shift(1).to_numpy(), index=daily_index)
    is_new_month.iloc[0] = True
    mode_change = crisis_mode != crisis_mode.shift(1)
    mode_change.iloc[0] = False
    v1_2_rebalance_mask = is_new_month | mode_change

    results: dict[str, StrategyResult] = {
        V1_1_LABEL: run_strategy(
            daily_returns, v1_1_daily_target, is_new_month, transaction_cost_bps=cost_bps
        ),
        V1_2_LABEL: run_strategy(
            daily_returns, v1_2_daily_target, v1_2_rebalance_mask, transaction_cost_bps=cost_bps
        ),
    }

    summary = build_summary_table_daily(results, risk_free_returns)
    daily_returns_out = pd.DataFrame({name: r.returns_post_cost for name, r in results.items()})
    monthly_returns = pd.DataFrame(
        {name: month_end_value(r.value_post_cost).pct_change() for name, r in results.items()}
    )

    turnover_breakdown = _build_turnover_breakdown(results, is_new_month, mode_change)
    allocations = _build_allocations_table(
        daily_returns, macro_regime_daily, v1_1_daily_target, v1_2_daily_target, state
    )
    current_status = _build_current_status(
        macro_regime_daily,
        v1_1_daily_target,
        v1_2_daily_target,
        state,
        raw_trigger,
        vix_shock,
        equity_shock,
        credit_shock,
        min_hold_days=fconf["min_hold_days"],
        min_off_days_to_exit=fconf["min_off_days_to_exit"],
    )

    return FastCrisisBacktest(
        signal_table=allocations.assign(
            vix_shock=vix_shock.reindex(daily_index),
            equity_shock=equity_shock.reindex(daily_index),
            credit_shock=credit_shock.reindex(daily_index),
            raw_trigger=raw_trigger.reindex(daily_index),
            **{c: state[c].reindex(daily_index) for c in state.columns},
        ),
        allocations=allocations,
        results=results,
        summary=summary,
        daily_returns=daily_returns_out,
        monthly_returns=monthly_returns,
        turnover_breakdown=turnover_breakdown,
        current_status=current_status,
    )


def _build_turnover_breakdown(
    results: dict[str, StrategyResult], is_new_month: pd.Series, mode_change: pd.Series
) -> pd.DataFrame:
    """Splits each strategy's total turnover into 'monthly rebalance'
    (is_new_month) vs 'crisis entry/exit' (mode_change and not
    is_new_month, so a same-day overlap is never double-counted) --
    v1_1_daily has zero crisis-attributable turnover by construction."""
    crisis_only_days = mode_change & ~is_new_month
    rows = []
    for name, r in results.items():
        crisis_turnover = r.turnover.reindex(crisis_only_days.index)[crisis_only_days].sum()
        rows.append(
            {
                "strategy": name,
                "monthly_rebalance_turnover": float(r.turnover.loc[is_new_month].sum()),
                "crisis_entry_exit_turnover": float(crisis_turnover),
                "total_turnover": float(r.turnover.sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_allocations_table(
    daily_returns: pd.DataFrame,
    macro_regime_daily: pd.Series,
    v1_1_daily_target: pd.DataFrame,
    v1_2_daily_target: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    base_crisis_sum = v1_1_daily_target[CRISIS_ASSET_COLUMNS].sum(axis=1)
    final_crisis_sum = v1_2_daily_target[CRISIS_ASSET_COLUMNS].sum(axis=1)
    removed = base_crisis_sum - final_crisis_sum
    allocations = pd.DataFrame(index=daily_returns.index)
    allocations["macro_regime"] = macro_regime_daily
    allocations["crisis_mode"] = state["crisis_mode"]
    for col in v1_1_daily_target.columns:
        allocations[f"base_{col}"] = v1_1_daily_target[col]
    for col in v1_2_daily_target.columns:
        allocations[f"final_{col}"] = v1_2_daily_target[col]
    allocations["crisis_removed_weight"] = removed
    allocations["bil_increase"] = v1_2_daily_target[CASH_COLUMN] - v1_1_daily_target[CASH_COLUMN]
    return allocations


def _build_current_status(
    macro_regime_daily: pd.Series,
    v1_1_daily_target: pd.DataFrame,
    v1_2_daily_target: pd.DataFrame,
    state: pd.DataFrame,
    raw_trigger: pd.Series,
    vix_shock: pd.Series,
    equity_shock: pd.Series,
    credit_shock: pd.Series,
    *,
    min_hold_days: int,
    min_off_days_to_exit: int,
) -> dict:
    last = state.index[-1]
    days_in_mode = int(state.loc[last, "days_in_mode"])
    consecutive_off = int(state.loc[last, "consecutive_off_count"])
    in_mode = state.loc[last, "crisis_mode"] == ON
    min_hold_satisfied = bool(in_mode and days_in_mode >= min_hold_days)
    remaining_min_hold_days = max(0, min_hold_days - days_in_mode) if in_mode else None
    remaining_off_days_to_exit = (
        max(0, min_off_days_to_exit - consecutive_off) if (in_mode and min_hold_satisfied) else None
    )
    return {
        "as_of": str(last.date()),
        "macro_regime": str(macro_regime_daily.loc[last]),
        "vix_shock": str(vix_shock.loc[last]),
        "equity_shock": str(equity_shock.loc[last]),
        "credit_shock": str(credit_shock.loc[last]),
        "raw_trigger": str(raw_trigger.loc[last]),
        "tradable_trigger": str(state.loc[last, "tradable_trigger"]),
        "crisis_mode": str(state.loc[last, "crisis_mode"]),
        "days_in_mode": days_in_mode,
        "consecutive_off_days": consecutive_off,
        "min_hold_days_satisfied": min_hold_satisfied if in_mode else None,
        "remaining_min_hold_days": remaining_min_hold_days,
        "remaining_off_days_to_exit": remaining_off_days_to_exit,
        "next_exit_eligible": bool(
            in_mode and min_hold_satisfied and consecutive_off >= min_off_days_to_exit
        ),
        "base_weights": {col: float(v1_1_daily_target.loc[last, col]) for col in v1_1_daily_target.columns},
        "final_weights": {col: float(v1_2_daily_target.loc[last, col]) for col in v1_2_daily_target.columns},
    }
