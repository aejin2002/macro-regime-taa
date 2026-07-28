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

from macro_regime.backtest.allocations import regime_allocations, static_6040_allocation
from macro_regime.backtest.assets import build_monthly_return_matrix
from macro_regime.backtest.engine import (
    StrategyResult,
    build_target_weights_from_regime,
    constant_target_weights,
    run_strategy,
)
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
    n_day_return_diagnostic,
    vix_shock_diagnostics,
)

V1_1_LABEL = "v1_1_daily"
V1_2_LABEL = "v1_2_fast_crisis"
BENCHMARK_LABEL = "benchmark_60_40"


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
    bei_signal: pd.DataFrame  # raw/tradable BEI Duration Gate table (see build_bei_duration_gate_signal)


def _build_v1_1_monthly_target(
    config: dict, primary_regime: pd.Series, wide: pd.DataFrame, monthly_asset_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """v1.1's own monthly target: Primary v1.0 regime allocation, with
    the BEI Duration Risk Gate applied on top -- byte-for-byte the same
    construction `duration_gate.backtest.run_bei_duration_gate_backtest`
    uses, reusing the same unmodified functions (not reimplemented).
    Returns (regime_only_target, bei_gated_target, bei_signal) -- the
    regime-only stage is exposed purely for display (the "Macro Base"
    step of a Macro -> BEI -> Fast Crisis waterfall) and `bei_signal` is
    the full raw/tradable monthly gate table; neither changes any
    computation, only what is returned."""
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

    overlay_target = apply_bei_duration_gate(
        base_target,
        tradable_gate,
        ief_weight_when_on=gconf["ief_weight_when_on"],
        bil_weight_when_on=gconf["bil_weight_when_on"],
    )
    return base_target, overlay_target, bei_signal


def _broadcast_monthly_onto_daily(monthly: pd.Series | pd.DataFrame, daily_index: pd.DatetimeIndex):
    """Maps each daily date to ITS OWN calendar month's monthly value --
    not a temporal ffill (which has nothing to fill the first ~20
    trading days of a month from). A daily date in a calendar month that
    is STILL OPEN (later than the monthly series' own last period -- the
    monthly signal/target can only ever be computed once a month has
    closed) instead holds the LAST available monthly value forward. This
    is exactly "keep the most recently decided target until the next
    monthly rebalance" -- it uses no information from an unpublished
    period and is never a lookahead. A gap strictly BEFORE the monthly
    series' own range is a real data problem and still raises loudly."""
    monthly_periods = monthly.index.to_period("M")
    period_to_label = dict(zip(monthly_periods, monthly.index, strict=True))
    last_period = monthly_periods.max()
    last_label = period_to_label[last_period]
    daily_periods = daily_index.to_period("M")
    mapped_labels = [
        period_to_label[p] if p in period_to_label else (last_label if p > last_period else None)
        for p in daily_periods
    ]
    if any(label is None for label in mapped_labels):
        missing = sorted(
            {str(d.date()) for d, lbl in zip(daily_index, mapped_labels, strict=True) if lbl is None}
        )
        raise ValueError(
            f"No monthly value available for daily date(s) before the monthly series' own start: "
            f"{missing[:5]}"
        )
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
    v1_0_monthly_target, v1_1_monthly_target, bei_signal = _build_v1_1_monthly_target(
        config, primary_regime, wide, monthly_asset_returns
    )

    raw = load_daily_prices(config, wide["VIXCLS"], client=client, refresh_cache=refresh_cache)
    daily_returns, vix_daily = build_daily_return_matrix(raw, as_of=as_of)

    first_month = v1_1_monthly_target.index.min()
    not_before = pd.Timestamp(f"{first_month.year}-{first_month.month:02d}-01")
    common_start = determine_common_start(daily_returns, vix_daily, not_before=not_before)
    # Daily portfolio/benchmark valuation extends through the latest
    # available trading day (bounded only by `as_of` and by what the
    # asset-price data actually covers) -- NOT capped at the monthly
    # macro target's own last CLOSED month. Days in a still-open current
    # month use last month's already-decided target, held forward by
    # `_broadcast_monthly_onto_daily` (no lookahead: nothing from the
    # open month is used to pick that target).
    end_date = daily_returns.index.max()
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

    # Raw numeric values each shock thresholds against -- display/
    # diagnostic only, computed by the same formulas as the shock
    # classifications above (never re-derived differently).
    vix_diag = vix_shock_diagnostics(vix_daily, ma_window_days=fconf["vix_ma_window_days"])
    spy_5d_return = n_day_return_diagnostic(
        daily_returns["spy"], window_days=fconf["equity_shock_window_days"]
    )
    hyg_5d_return = n_day_return_diagnostic(
        daily_returns["high_yield"], window_days=fconf["credit_shock_window_days"]
    )

    # -- v1.1 daily target (v1.1's monthly target broadcast onto the daily calendar) --
    v1_1_daily_target = _broadcast_monthly_onto_daily(v1_1_monthly_target, daily_index)
    v1_0_daily_target = _broadcast_monthly_onto_daily(v1_0_monthly_target, daily_index)
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

    # Benchmark: the project's EXISTING static 60/40 definition (config's
    # `backtest.static_6040` -- Growth Basket 60% / intermediate_treasury
    # 40%, expanded via `static_6040_allocation`, same function `run-backtest`
    # uses), evaluated on the identical daily calendar/asset data as v1.1/v1.2
    # (monthly rebalance) purely so its NAV is directly comparable -- the
    # weights and rebalance rule are not redefined here.
    benchmark_target = constant_target_weights(
        static_6040_allocation(config), daily_index, list(daily_returns.columns)
    )

    results: dict[str, StrategyResult] = {
        V1_1_LABEL: run_strategy(
            daily_returns, v1_1_daily_target, is_new_month, transaction_cost_bps=cost_bps
        ),
        V1_2_LABEL: run_strategy(
            daily_returns, v1_2_daily_target, v1_2_rebalance_mask, transaction_cost_bps=cost_bps
        ),
        BENCHMARK_LABEL: run_strategy(
            daily_returns, benchmark_target, is_new_month, transaction_cost_bps=cost_bps
        ),
    }

    summary = build_summary_table_daily(results, risk_free_returns)
    daily_returns_out = pd.DataFrame({name: r.returns_post_cost for name, r in results.items()})
    monthly_returns = pd.DataFrame(
        {name: month_end_value(r.value_post_cost).pct_change() for name, r in results.items()}
    )

    turnover_breakdown = _build_turnover_breakdown(results, is_new_month, mode_change)
    last_closed_month = v1_1_monthly_target.index.to_period("M").max()
    partial_month = pd.Series(daily_index.to_period("M") > last_closed_month, index=daily_index)
    allocations = _build_allocations_table(
        daily_returns, macro_regime_daily, v1_0_daily_target, v1_1_daily_target, v1_2_daily_target, state,
        partial_month,
    )
    current_status = _build_current_status(
        macro_regime_daily,
        v1_0_daily_target,
        v1_1_daily_target,
        v1_2_daily_target,
        state,
        raw_trigger,
        vix_shock,
        equity_shock,
        credit_shock,
        vix_diag,
        spy_5d_return,
        hyg_5d_return,
        fconf,
    )

    return FastCrisisBacktest(
        signal_table=allocations.assign(
            vix_level=vix_diag["vix_level"].reindex(daily_index),
            vix_ma=vix_diag["vix_ma"].reindex(daily_index),
            vix_ratio=vix_diag["vix_ratio"].reindex(daily_index),
            spy_5d_return=spy_5d_return.reindex(daily_index),
            hyg_5d_return=hyg_5d_return.reindex(daily_index),
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
        bei_signal=bei_signal,
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
    v1_0_daily_target: pd.DataFrame,
    v1_1_daily_target: pd.DataFrame,
    v1_2_daily_target: pd.DataFrame,
    state: pd.DataFrame,
    partial_month: pd.Series,
) -> pd.DataFrame:
    """Four-stage weight waterfall, one column group per stage:
    `macro_*` (Primary v1.0 regime allocation only), `bei_*` (v1.1,
    Macro + BEI Duration Risk Gate -- the `base_*` name is kept
    alongside as an alias for backward compatibility with existing
    readers of this CSV), `final_*` (v1.2, Macro + BEI + Fast Crisis
    Overlay). Each stage is exactly what `run_fast_crisis_backtest`
    already computed -- this function only arranges it for display, it
    computes nothing new. `partial_month` (bool) marks days whose
    calendar month has no monthly macro target of its own yet -- the
    prior month's already-decided target is being held forward for
    them (see `_broadcast_monthly_onto_daily`)."""
    base_crisis_sum = v1_1_daily_target[CRISIS_ASSET_COLUMNS].sum(axis=1)
    final_crisis_sum = v1_2_daily_target[CRISIS_ASSET_COLUMNS].sum(axis=1)
    removed = base_crisis_sum - final_crisis_sum
    allocations = pd.DataFrame(index=daily_returns.index)
    allocations["macro_regime"] = macro_regime_daily
    allocations["crisis_mode"] = state["crisis_mode"]
    allocations["partial_month"] = partial_month
    for col in v1_0_daily_target.columns:
        allocations[f"macro_{col}"] = v1_0_daily_target[col]
    for col in v1_1_daily_target.columns:
        allocations[f"bei_{col}"] = v1_1_daily_target[col]
        allocations[f"base_{col}"] = v1_1_daily_target[col]  # alias, kept for existing readers
    for col in v1_2_daily_target.columns:
        allocations[f"final_{col}"] = v1_2_daily_target[col]
    allocations["crisis_removed_weight"] = removed
    allocations["bil_increase"] = v1_2_daily_target[CASH_COLUMN] - v1_1_daily_target[CASH_COLUMN]
    return allocations


def _current_crisis_entry_date(crisis_mode: pd.Series) -> str | None:
    """If currently ON, the first date of the consecutive ON run ending
    at the last observation -- display/diagnostic only, derived from
    the already-computed `crisis_mode` series, not a new calculation."""
    if crisis_mode.empty or crisis_mode.iloc[-1] != ON:
        return None
    is_on = (crisis_mode == ON).to_numpy()
    idx = len(is_on) - 1
    while idx > 0 and is_on[idx - 1]:
        idx -= 1
    return str(crisis_mode.index[idx].date())


def _build_current_status(
    macro_regime_daily: pd.Series,
    v1_0_daily_target: pd.DataFrame,
    v1_1_daily_target: pd.DataFrame,
    v1_2_daily_target: pd.DataFrame,
    state: pd.DataFrame,
    raw_trigger: pd.Series,
    vix_shock: pd.Series,
    equity_shock: pd.Series,
    credit_shock: pd.Series,
    vix_diag: pd.DataFrame,
    spy_5d_return: pd.Series,
    hyg_5d_return: pd.Series,
    fconf: dict,
) -> dict:
    min_hold_days = fconf["min_hold_days"]
    min_off_days_to_exit = fconf["min_off_days_to_exit"]
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
        "vix_level": float(vix_diag.loc[last, "vix_level"]),
        "vix_ma": float(vix_diag.loc[last, "vix_ma"]),
        "vix_ratio": float(vix_diag.loc[last, "vix_ratio"]),
        "vix_threshold": fconf["vix_threshold"],
        "vix_ma_ratio_threshold": fconf["vix_ma_ratio_threshold"],
        "equity_shock": str(equity_shock.loc[last]),
        "spy_5d_return": float(spy_5d_return.loc[last]) if pd.notna(spy_5d_return.loc[last]) else None,
        "equity_shock_threshold": fconf["equity_shock_threshold"],
        "credit_shock": str(credit_shock.loc[last]),
        "hyg_5d_return": float(hyg_5d_return.loc[last]) if pd.notna(hyg_5d_return.loc[last]) else None,
        "credit_shock_threshold": fconf["credit_shock_threshold"],
        "raw_trigger": str(raw_trigger.loc[last]),
        "tradable_trigger": str(state.loc[last, "tradable_trigger"]),
        "crisis_mode": str(state.loc[last, "crisis_mode"]),
        "crisis_entry_date": _current_crisis_entry_date(state["crisis_mode"]),
        "days_in_mode": days_in_mode,
        "consecutive_off_days": consecutive_off,
        "min_hold_days_satisfied": min_hold_satisfied if in_mode else None,
        "remaining_min_hold_days": remaining_min_hold_days,
        "remaining_off_days_to_exit": remaining_off_days_to_exit,
        "next_exit_eligible": bool(
            in_mode and min_hold_satisfied and consecutive_off >= min_off_days_to_exit
        ),
        "macro_weights": {col: float(v1_0_daily_target.loc[last, col]) for col in v1_0_daily_target.columns},
        "base_weights": {col: float(v1_1_daily_target.loc[last, col]) for col in v1_1_daily_target.columns},
        "final_weights": {col: float(v1_2_daily_target.loc[last, col]) for col in v1_2_daily_target.columns},
    }
