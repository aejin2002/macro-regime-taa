"""BEI Duration Risk Gate (v1.1) -- backtest orchestration.

Ties together the signal (`duration_gate.signal`), the allocation
overlay (`duration_gate.allocation`), and the existing, unmodified v1.0
backtest engine/metrics (`backtest.engine`, `backtest.metrics`) to
produce a two-strategy comparison: v1.0 Primary vs. v1.1 Primary + BEI
Duration Risk Gate. Never reads or writes any of v1.0's `backtest_*.csv`
files -- this module only produces the `bei_duration_gate_*` outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from macro_regime.backtest.allocations import regime_allocations
from macro_regime.backtest.engine import StrategyResult, build_target_weights_from_regime, run_strategy
from macro_regime.backtest.metrics import (
    annual_returns_table,
    build_summary_table,
    max_drawdown,
    regime_average_returns,
)
from macro_regime.duration_gate.allocation import apply_bei_duration_gate
from macro_regime.duration_gate.signal import ON, UNKNOWN, build_bei_duration_gate_signal

V1_0_LABEL = "v1_0_primary"
V1_1_LABEL = "v1_1_bei_duration_gate"

# Named windows for the period breakdown -- the same set used to audit
# the original v1.0 backtest's crisis-year coverage, plus 2021 (the
# rate-pivot year this gate specifically targets). Not tuned to flatter
# any particular result; a window is simply skipped if it falls outside
# the common backtest sample.
PERIOD_WINDOWS: dict[str, tuple[str, str]] = {
    "2013_taper_tantrum": ("2013-04-30", "2013-12-31"),
    "2018_rate_hikes": ("2018-01-31", "2018-12-31"),
    "2020_covid_rate_crash": ("2020-01-31", "2020-12-31"),
    "2021_rate_pivot": ("2021-01-31", "2021-12-31"),
    "2022_bond_selloff": ("2022-01-31", "2022-12-31"),
    "2023_2024_higher_for_longer": ("2023-01-31", "2024-12-31"),
}


@dataclass
class BeiDurationGateBacktest:
    signal_table: pd.DataFrame
    allocations: pd.DataFrame
    results: dict[str, StrategyResult]
    summary: pd.DataFrame
    annual_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    regime_analysis: pd.DataFrame


def run_bei_duration_gate_backtest(
    config: dict,
    primary_regime: pd.Series,
    wide: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> BeiDurationGateBacktest:
    """Run v1.0 Primary and v1.1 (Primary + BEI Duration Risk Gate) under
    identical conditions: same `asset_returns` (asset prices, FX, common
    start date), same `primary_regime` (tradable_regime from
    regime_output_primary.csv), same transaction cost, same monthly
    rebalance-to-target rule (`backtest.engine.run_strategy`, unmodified),
    same risk-free series for Sharpe/Sortino. The only difference between
    the two strategies is whether `apply_bei_duration_gate` ever fires.
    """
    bconf = config["backtest"]
    gconf = config["bei_duration_gate"]
    cost_bps = bconf["transaction_cost_bps"]
    risk_free_returns = asset_returns[bconf["risk_free_asset"]]
    allocations_by_regime = regime_allocations(config)
    columns = list(asset_returns.columns)

    regime_aligned = primary_regime.reindex(asset_returns.index)
    base_target = build_target_weights_from_regime(regime_aligned, allocations_by_regime, columns)
    rebalance_mask = pd.Series(True, index=asset_returns.index)

    signal_table = build_bei_duration_gate_signal(
        wide["DGS10"], wide["T10YIE"], asset_returns["long_treasury"], as_of=as_of
    )
    tradable_gate = signal_table["tradable_bei_duration_gate"].reindex(asset_returns.index).fillna(UNKNOWN)
    overlay_target = apply_bei_duration_gate(
        base_target,
        tradable_gate,
        ief_weight_when_on=gconf["ief_weight_when_on"],
        bil_weight_when_on=gconf["bil_weight_when_on"],
    )

    results: dict[str, StrategyResult] = {
        V1_0_LABEL: run_strategy(asset_returns, base_target, rebalance_mask, transaction_cost_bps=cost_bps),
        V1_1_LABEL: run_strategy(
            asset_returns, overlay_target, rebalance_mask, transaction_cost_bps=cost_bps
        ),
    }

    summary = build_summary_table(results, risk_free_returns)
    annual_returns = annual_returns_table(results)
    monthly_returns = pd.DataFrame({name: r.returns_post_cost for name, r in results.items()})

    allocations = _build_allocations_table(
        asset_returns, regime_aligned, base_target, overlay_target, tradable_gate, columns
    )
    regime_analysis = _build_regime_analysis(asset_returns, regime_aligned, results, tradable_gate)

    return BeiDurationGateBacktest(
        signal_table=signal_table,
        allocations=allocations,
        results=results,
        summary=summary,
        annual_returns=annual_returns,
        monthly_returns=monthly_returns,
        regime_analysis=regime_analysis,
    )


def _build_allocations_table(
    asset_returns: pd.DataFrame,
    regime_aligned: pd.Series,
    base_target: pd.DataFrame,
    overlay_target: pd.DataFrame,
    tradable_gate: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    duration_pool = base_target["intermediate_treasury"] + base_target["long_treasury"]
    allocations = pd.DataFrame(index=asset_returns.index)
    allocations["macro_regime"] = regime_aligned
    allocations["base_IEF"] = base_target["intermediate_treasury"]
    allocations["base_TLT"] = base_target["long_treasury"]
    allocations["base_BIL"] = base_target["tbills"]
    allocations["duration_pool"] = duration_pool
    allocations["gate_state"] = tradable_gate
    allocations["final_IEF"] = overlay_target["intermediate_treasury"]
    allocations["final_TLT"] = overlay_target["long_treasury"]
    allocations["additional_BIL"] = overlay_target["tbills"] - base_target["tbills"]
    allocations["final_BIL"] = overlay_target["tbills"]
    other_cols = [c for c in columns if c not in ("intermediate_treasury", "long_treasury", "tbills")]
    for c in other_cols:
        allocations[c] = overlay_target[c]
    return allocations


def _build_regime_analysis(
    asset_returns: pd.DataFrame,
    regime_aligned: pd.Series,
    results: dict[str, StrategyResult],
    tradable_gate: pd.Series,
) -> pd.DataFrame:
    """Long-format `analysis_type,key,series,value` table, matching the
    existing `backtest_regime_analysis.csv` convention: per-regime
    average monthly returns, plus a period-window breakdown (cumulative
    return, max drawdown, and gate-ON month count) for each entry in
    `PERIOD_WINDOWS` that falls inside the common backtest sample."""
    portfolio_returns = {name: r.returns_post_cost for name, r in results.items()}
    regime_avg = regime_average_returns(asset_returns, regime_aligned, portfolio_returns)

    rows: list[dict] = []
    for regime, row in regime_avg.iterrows():
        for series, value in row.items():
            rows.append(
                {
                    "analysis_type": "regime_avg_monthly_return",
                    "key": regime,
                    "series": series,
                    "value": value,
                }
            )

    for label, (start, end) in PERIOD_WINDOWS.items():
        window = (asset_returns.index >= start) & (asset_returns.index <= end)
        if not window.any():
            continue
        gate_on_months = int((tradable_gate.loc[window] == ON).sum())
        for name, r in results.items():
            ret = r.returns_post_cost.loc[window]
            val = (1 + ret).cumprod()
            rows.append(
                {
                    "analysis_type": "period_cum_return",
                    "key": label,
                    "series": name,
                    "value": float((1 + ret).prod() - 1),
                }
            )
            rows.append(
                {
                    "analysis_type": "period_max_drawdown",
                    "key": label,
                    "series": name,
                    "value": max_drawdown(val),
                }
            )
        rows.append(
            {
                "analysis_type": "period_gate_on_months",
                "key": label,
                "series": "tradable_bei_duration_gate",
                "value": gate_on_months,
            }
        )

    return pd.DataFrame(rows)
