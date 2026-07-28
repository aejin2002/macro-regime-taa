"""Builds the canonical v1.3 daily production artifact
(`data/processed/production_v13_daily.parquet`) -- the single source of
truth every chart/table in the Streamlit dashboard reads from. Combines
the monthly macro signal (regime/growth/inflation/BEI gate, each held
forward through the current, still-open month -- see
`fast_crisis.backtest._broadcast_monthly_onto_daily`) with genuinely
daily portfolio valuation, Fast Crisis state, and asset-level returns.
Reuses `run_fast_crisis_backtest`, `strategy_versions.build_versioned_config`,
and the daily asset-return matrix construction (`fast_crisis.daily_data`)
unmodified -- this module only assembles their outputs into one wide
daily table plus a benchmark table; it computes no new strategy logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro_regime.benchmarks import (
    compute_benchmark_series,
    list_ui_visible,
    project_6040_series_from_backtest,
)
from macro_regime.config import PROJECT_ROOT
from macro_regime.data.asset_prices import AssetPriceClient
from macro_regime.fast_crisis.backtest import BENCHMARK_LABEL, V1_2_LABEL, run_fast_crisis_backtest
from macro_regime.fast_crisis.daily_data import build_daily_return_matrix, load_daily_prices
from macro_regime.strategy_versions import build_versioned_config

ASSET_COLUMNS = [
    "spy",
    "kodex200_usd",
    "high_yield",
    "investment_grade",
    "intermediate_treasury",
    "long_treasury",
    "gold",
    "tbills",
    "commodities",
    "tips",
]

PRODUCTION_DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "production_v13_daily.parquet"
V1_2_REGRESSION_DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "v1_2_regression_daily.parquet"
BENCHMARK_DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "benchmarks_daily.parquet"


def _hold_forward(monthly: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Same hold-forward-into-the-open-month semantic as
    `fast_crisis.backtest._broadcast_monthly_onto_daily`, for series
    (growth_state/inflation_state/BEI gate) that aren't part of the
    weight-target waterfall that function already handles."""
    monthly_periods = monthly.index.to_period("M")
    period_to_label = dict(zip(monthly_periods, monthly.index, strict=True))
    last_period = monthly_periods.max()
    last_label = period_to_label[last_period]
    daily_periods = daily_index.to_period("M")
    labels = [period_to_label.get(p, last_label if p > last_period else None) for p in daily_periods]
    result = monthly.reindex(labels)
    result.index = daily_index
    return result


def build_v1_3_daily_artifact(
    config: dict,
    primary_regime: pd.Series,
    wide: pd.DataFrame,
    primary_df: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    client: AssetPriceClient | None = None,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, object]:
    """Returns (daily_artifact_df, FastCrisisBacktest) -- the backtest
    object is returned too since the benchmark-artifact builder reuses
    its already-computed `results[BENCHMARK_LABEL]` for project_6040
    rather than recomputing it."""
    as_of = as_of or pd.Timestamp.now()
    client = client or AssetPriceClient()
    v1_3_config = build_versioned_config(config, "v1_3")
    bt = run_fast_crisis_backtest(
        v1_3_config, primary_regime, wide, client=client, as_of=as_of, refresh_cache=refresh_cache
    )

    r = bt.results[V1_2_LABEL]
    idx = r.returns_post_cost.index
    alloc = bt.allocations.reindex(idx)

    raw = load_daily_prices(v1_3_config, wide["VIXCLS"], client=client, refresh_cache=refresh_cache)
    asset_returns, _ = build_daily_return_matrix(raw, as_of=as_of)
    asset_returns = asset_returns.reindex(idx)

    weights_drifted = r.weights.reindex(columns=ASSET_COLUMNS).reindex(idx)
    weights_target = alloc[[f"final_{c}" for c in ASSET_COLUMNS]].copy()
    weights_target.columns = ASSET_COLUMNS
    contribution = weights_drifted * asset_returns

    # `growth_state`/`inflation_state` in `primary_df` are the model's OWN
    # unlagged monthly reading (never shifted -- see
    # `signals.regime.build_regime_output`'s docstring). `macro_regime`
    # here, however, is `tradable_regime`, which IS `raw_regime` shifted
    # forward by `tradable_lag_months` (`signals.regime.shift_to_tradable`).
    # Showing the raw, unlagged growth/inflation reading next to a regime
    # that was actually decided a month earlier would silently explain the
    # WRONG month's regime -- so the "effective" (displayed as
    # `growth_state`/`inflation_state`, matching what actually explains
    # `macro_regime`) reading is lagged by the same amount, and the raw,
    # most-recently-observed reading is kept separately as
    # `growth_state_observed`/`inflation_state_observed`.
    lag = config["regime_output"]["tradable_lag_months"]
    growth_state_daily = _hold_forward(primary_df["growth_state"].shift(lag), idx)
    inflation_state_daily = _hold_forward(primary_df["inflation_state"].shift(lag), idx)
    growth_score_daily = _hold_forward(primary_df["growth_score"].shift(lag), idx)
    inflation_score_daily = _hold_forward(primary_df["inflation_score"].shift(lag), idx)
    growth_state_observed_daily = _hold_forward(primary_df["growth_state"], idx)
    inflation_state_observed_daily = _hold_forward(primary_df["inflation_state"], idx)
    bei_raw_daily = _hold_forward(bt.bei_signal["raw_bei_duration_gate"], idx)
    bei_tradable_daily = _hold_forward(bt.bei_signal["tradable_bei_duration_gate"], idx)

    value = r.value_post_cost
    drawdown = value / value.cummax() - 1.0

    macro_regime_change = alloc["macro_regime"] != alloc["macro_regime"].shift(1)
    crisis_mode_change = alloc["crisis_mode"] != alloc["crisis_mode"].shift(1)
    bei_change = bei_tradable_daily != bei_tradable_daily.shift(1)
    rebalance_event = r.turnover > 1e-12
    signal_event = (macro_regime_change | crisis_mode_change | bei_change).fillna(False)
    signal_event.iloc[0] = True

    out = pd.DataFrame(index=idx)
    out["date"] = idx
    out["strategy_version"] = "v1_3"
    out["daily_return_gross"] = r.returns_pre_cost
    out["daily_return_net"] = r.returns_post_cost
    out["strategy_nav"] = value
    out["strategy_drawdown"] = drawdown
    out["macro_regime"] = alloc["macro_regime"]
    out["growth_state"] = growth_state_daily
    out["inflation_state"] = inflation_state_daily
    out["growth_score"] = growth_score_daily
    out["inflation_score"] = inflation_score_daily
    out["growth_state_observed"] = growth_state_observed_daily
    out["inflation_state_observed"] = inflation_state_observed_daily
    out["bei_gate_raw"] = bei_raw_daily
    out["bei_gate_tradable"] = bei_tradable_daily
    out["fast_crisis_raw"] = bt.signal_table["raw_trigger"].reindex(idx)
    out["fast_crisis_tradable"] = alloc["crisis_mode"]
    out["fast_crisis_votes"] = (
        bt.signal_table["vix_shock"].reindex(idx).eq("ON").astype(int)
        + bt.signal_table["equity_shock"].reindex(idx).eq("ON").astype(int)
        + bt.signal_table["credit_shock"].reindex(idx).eq("ON").astype(int)
    )
    for c in ASSET_COLUMNS:
        out[f"target_weight_{c}"] = weights_target[c]
        out[f"drifted_weight_{c}"] = weights_drifted[c]
        out[f"asset_return_{c}"] = asset_returns[c]
        out[f"asset_contribution_{c}"] = contribution[c]
    out["daily_turnover"] = r.turnover
    out["transaction_cost"] = r.returns_pre_cost - r.returns_post_cost
    out["rebalance_event"] = rebalance_event
    out["signal_event"] = signal_event
    out["partial_month"] = alloc["partial_month"]
    out = out.reset_index(drop=True)
    return out, bt


def build_v1_2_regression_daily_artifact(
    config: dict,
    primary_regime: pd.Series,
    wide: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    client: AssetPriceClient | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """The v1.2 regression fixture -- same shape as the v1.3 artifact's
    core columns (not the full waterfall) so v1.2/v1.3 can be diffed
    directly. Immutable by convention: only ever regenerated from the
    UNMODIFIED v1.2 config, never from v1.3's."""
    as_of = as_of or pd.Timestamp.now()
    client = client or AssetPriceClient()
    v1_2_config = build_versioned_config(config, "v1_2")
    bt = run_fast_crisis_backtest(
        v1_2_config, primary_regime, wide, client=client, as_of=as_of, refresh_cache=refresh_cache
    )
    r = bt.results[V1_2_LABEL]
    value = r.value_post_cost
    out = pd.DataFrame(
        {
            "date": r.returns_post_cost.index,
            "strategy_version": "v1_2",
            "daily_return_gross": r.returns_pre_cost.to_numpy(),
            "daily_return_net": r.returns_post_cost.to_numpy(),
            "strategy_nav": value.to_numpy(),
            "strategy_drawdown": (value / value.cummax() - 1.0).to_numpy(),
            "macro_regime": bt.allocations["macro_regime"].reindex(r.returns_post_cost.index).to_numpy(),
            "daily_turnover": r.turnover.to_numpy(),
        }
    ).reset_index(drop=True)
    return out


def build_benchmark_daily_artifact(
    bt,
    strategy_calendar: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    client: AssetPriceClient | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    client = client or AssetPriceClient()
    frames = []
    for definition in list_ui_visible():
        series = compute_benchmark_series(
            definition.id,
            strategy_calendar=strategy_calendar,
            as_of=as_of,
            client=client,
            refresh_cache=refresh_cache,
        )
        if not series.status.available:
            frames.append(
                pd.DataFrame(
                    {
                        "benchmark_id": [definition.id],
                        "date": [pd.NaT],
                        "available": [False],
                        "error": [series.status.error],
                    }
                )
            )
            continue
        df = pd.DataFrame(
            {
                "date": series.returns.index,
                "benchmark_id": definition.id,
                "daily_return": series.returns.to_numpy(),
                "nav": series.value.to_numpy(),
                "available": True,
            }
        )
        if series.is_stale is not None:
            df["is_stale"] = series.is_stale.reindex(series.returns.index).to_numpy()
        frames.append(df)
    # project_6040 -- internal/research only, kept out of the UI-visible loop above
    proj = project_6040_series_from_backtest(bt, BENCHMARK_LABEL)
    frames.append(
        pd.DataFrame(
            {
                "date": proj.returns.index,
                "benchmark_id": "project_6040",
                "daily_return": proj.returns.to_numpy(),
                "nav": proj.value.to_numpy(),
                "available": True,
            }
        )
    )
    return pd.concat(frames, ignore_index=True, sort=False)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
