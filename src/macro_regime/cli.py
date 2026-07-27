"""Typer CLI: fetch -> build-signals -> evaluate -> run-all."""

from __future__ import annotations

import json

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from macro_regime.backtest.allocations import (
    equal_weight_allocation,
    growth_basket_allocation,
    regime_allocations,
    static_6040_allocation,
)
from macro_regime.backtest.assets import build_monthly_return_matrix
from macro_regime.backtest.engine import (
    StrategyResult,
    run_buy_and_hold_strategy,
    run_regime_strategy,
    run_static_strategy,
)
from macro_regime.backtest.metrics import (
    annual_returns_table,
    build_summary_table,
    crisis_period_performance,
    regime_agreement_rate,
    regime_average_returns,
)
from macro_regime.config import PROJECT_ROOT, MissingApiKeyError, load_config
from macro_regime.data.csv_loader import load_ism_new_orders, load_ism_prices_paid
from macro_regime.data.fred_client import FredClient
from macro_regime.data.fred_series import (
    all_series_ids,
    fetch_all_metadata,
    fetch_all_series,
    save_metadata,
    to_wide,
    validate_all_series,
)
from macro_regime.evaluation.lead_lag import growth_forward_target, inflation_forward_target
from macro_regime.evaluation.report import EvaluationReport, evaluate_model
from macro_regime.signals.growth import (
    classify_growth_model_a,
    cli_diagnostics,
    growth_model_b_fred_minimal,
    growth_model_c_simple_two_signal,
    growth_model_d_amtmno_claims,
)
from macro_regime.signals.inflation import (
    cleveland_median_cpi_momentum,
    commodity_core_composite,
    core_inflation_momentum,
    leading_inflation_composite,
)
from macro_regime.signals.regime import build_regime_output, build_regime_series
from macro_regime.utils.dates import normalize_month_end_index

app = typer.Typer(add_completion=False, help="Growth/Inflation macro regime research CLI")
console = Console()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WIDE_PATH = PROCESSED_DIR / "fred_wide.csv"
METADATA_PATH = PROCESSED_DIR / "series_metadata.json"
SIGNALS_PATH = PROCESSED_DIR / "signals.csv"
REGIME_METADATA_PATH = PROCESSED_DIR / "regime_metadata.json"
EVAL_PATH = PROCESSED_DIR / "evaluation_report.json"
REGIME_OUTPUT_PRIMARY_PATH = PROCESSED_DIR / "regime_output_primary.csv"
REGIME_OUTPUT_SECONDARY_PATH = PROCESSED_DIR / "regime_output_secondary.csv"
BACKTEST_SUMMARY_PATH = PROCESSED_DIR / "backtest_summary.csv"
BACKTEST_ANNUAL_RETURNS_PATH = PROCESSED_DIR / "backtest_annual_returns.csv"
BACKTEST_MONTHLY_RETURNS_PATH = PROCESSED_DIR / "backtest_monthly_returns.csv"
BACKTEST_ALLOCATIONS_PRIMARY_PATH = PROCESSED_DIR / "backtest_allocations_primary.csv"
BACKTEST_ALLOCATIONS_SECONDARY_PATH = PROCESSED_DIR / "backtest_allocations_secondary.csv"
BACKTEST_REGIME_ANALYSIS_PATH = PROCESSED_DIR / "backtest_regime_analysis.csv"


@app.command()
def fetch(
    start_date: str = typer.Option(None, "--start-date"),
    end_date: str = typer.Option(None, "--end-date"),
    refresh_cache: bool = typer.Option(False, "--refresh-cache"),
) -> None:
    """Validate and download all configured FRED series to data/processed/."""
    config = load_config()
    start_date = start_date or config["default_start_date"]

    try:
        client = FredClient()
    except MissingApiKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    series_ids = all_series_ids(config)
    console.print(f"Validating {len(series_ids)} configured FRED series...")
    validity = validate_all_series(client, series_ids)
    invalid = [sid for sid, ok in validity.items() if not ok]
    if invalid:
        console.print(
            f"[red]Invalid/unavailable FRED series (not silently substituted): {invalid}[/red]"
        )
        raise typer.Exit(code=1)

    metadata = fetch_all_metadata(client, series_ids)
    save_metadata(metadata, METADATA_PATH)

    long_df = fetch_all_series(
        client,
        series_ids,
        observation_start=start_date,
        observation_end=end_date,
        refresh_cache=refresh_cache,
    )
    wide = to_wide(long_df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(WIDE_PATH)
    console.print(f"[green]Saved {wide.shape[0]} rows x {wide.shape[1]} series to {WIDE_PATH}[/green]")


def _growth_signals(wide: pd.DataFrame, config: dict, growth_model: str) -> dict[str, pd.Series]:
    zconf = config["zscore"]
    gmodels = config["growth_models"]
    out: dict[str, pd.Series] = {}

    if growth_model in ("all", "model_a_cli"):
        a_conf = gmodels["model_a_cli"]
        us_cli = wide.get(a_conf["us_series"])
        if us_cli is not None and us_cli.dropna().shape[0] > 0:
            out["growth_model_us_cli"] = classify_growth_model_a(
                us_cli.dropna(), primary_window=a_conf["primary_window_months"]
            )

    if growth_model in ("all", "model_b_fred_minimal"):
        b_conf = gmodels["model_b_fred_minimal"]
        claims = wide.get(b_conf["claims_series"])
        permits = wide.get(b_conf["permits_series"])
        cfnai = wide.get(b_conf["activity_series"])
        if claims is not None and permits is not None and cfnai is not None:
            df_b = growth_model_b_fred_minimal(
                claims.dropna(),
                permits.dropna(),
                cfnai.dropna(),
                claims_change_months=b_conf["claims_change_months"],
                permits_change_months=b_conf["permits_change_months"],
                zscore_window=zconf["rolling_window_months"],
                zscore_min_periods=zconf["min_periods_months"],
            )
            df_b.to_csv(PROCESSED_DIR / "growth_model_b_detail.csv")
            out["growth_model_fred_minimal"] = df_b["growth_label"]

    if growth_model in ("all", "model_c_simple_two_signal"):
        c_conf = gmodels["model_c_simple_two_signal"]
        ism_new_orders = load_ism_new_orders(config["ism_new_orders_csv"])
        claims = wide.get(c_conf["claims_series"])
        if ism_new_orders is not None and claims is not None:
            ism_series = ism_new_orders.set_index("date")["value"]
            df_c = growth_model_c_simple_two_signal(
                ism_series,
                claims.dropna(),
                change_months=c_conf["ism_new_orders_change_months"],
                zscore_window=zconf["rolling_window_months"],
                zscore_min_periods=zconf["min_periods_months"],
            )
            if df_c is not None:
                df_c.to_csv(PROCESSED_DIR / "growth_model_c_detail.csv")
                out["growth_model_two_signal"] = df_c["growth_label"]

    if growth_model in ("all", "model_d_amtmno_claims"):
        d_conf = gmodels["model_d_amtmno_claims"]
        amtmno = wide.get(d_conf["amtmno_series"])
        claims = wide.get(d_conf["claims_series"])
        if amtmno is not None and claims is not None:
            df_d = growth_model_d_amtmno_claims(
                amtmno.dropna(),
                claims.dropna(),
                amtmno_change_months=d_conf["amtmno_change_months"],
                claims_change_months=d_conf["claims_change_months"],
                zscore_window=zconf["rolling_window_months"],
                zscore_min_periods=zconf["min_periods_months"],
            )
            df_d.to_csv(PROCESSED_DIR / "growth_model_d_detail.csv")
            out["growth_model_amtmno_claims"] = df_d["growth_label"]

    # Growth models are sourced from a mix of FRED conventions -- some
    # (e.g. PERMIT, CFNAIMA3, and single-series models like the CLI) are
    # stamped first-of-month, others (anything resampled from weekly/daily
    # data, e.g. ICSA-derived claims) land on month-end. Normalizing every
    # model's *output* index here, once, guarantees all growth columns
    # share one convention before they are combined in `build_signals` and
    # reindexed against evaluation targets in `evaluate` -- the same class
    # of bug that `normalize_month_end_index` fixes inside
    # `growth_model_b_fred_minimal` would otherwise resurface one layer up.
    return {name: normalize_month_end_index(series) for name, series in out.items()}


def _inflation_signals(wide: pd.DataFrame, config: dict, inflation_model: str) -> dict[str, pd.Series]:
    zconf = config["zscore"]
    imodels = config["inflation_models"]
    out: dict[str, pd.Series] = {}

    if inflation_model in ("all", "model_a_realized_core"):
        a_conf = imodels["model_a_realized_core"]
        core = wide.get(a_conf["core_series"])
        if core is not None:
            df_ia = core_inflation_momentum(
                core.dropna(),
                short_window_months=a_conf["short_window_months"],
                long_window_months=a_conf["long_window_months"],
            )
            df_ia.to_csv(PROCESSED_DIR / "inflation_model_a_detail.csv")
            out["inflation_model_core_momentum"] = df_ia["inflation_label"]

    if inflation_model in ("all", "model_b_leading_composite"):
        b_conf = imodels["model_b_leading_composite"]
        breakeven = wide.get(b_conf["breakeven_series"])
        core_b = wide.get(b_conf["core_series"])
        ism_pp = load_ism_prices_paid(config["ism_prices_paid_csv"])
        if breakeven is not None and core_b is not None:
            ism_pp_series = ism_pp.set_index("date")["value"] if ism_pp is not None else None
            df_ib = leading_inflation_composite(
                core_b.dropna(),
                breakeven.dropna(),
                ism_pp_series,
                breakeven_change_months=b_conf["breakeven_change_months"],
                zscore_window=zconf["rolling_window_months"],
                zscore_min_periods=zconf["min_periods_months"],
            )
            df_ib.to_csv(PROCESSED_DIR / "inflation_model_b_detail.csv")
            out["inflation_model_leading_composite"] = df_ib["inflation_label"]

    if inflation_model in ("all", "model_c_cleveland_median_cpi"):
        c_conf = imodels["model_c_cleveland_median_cpi"]
        median_cpi = wide.get(c_conf["median_cpi_series"])
        if median_cpi is not None and median_cpi.dropna().shape[0] > 0:
            df_ic = cleveland_median_cpi_momentum(
                median_cpi.dropna(),
                ma_window_months=c_conf["ma_window_months"],
                lag_months=c_conf["lag_months"],
            )
            df_ic.to_csv(PROCESSED_DIR / "inflation_model_c_detail.csv")
            out["inflation_model_cleveland_median_cpi"] = df_ic["inflation_label"]

    if inflation_model in ("all", "model_d_commodity_core_aux"):
        d_conf = imodels["model_d_commodity_core_aux"]
        commodity = wide.get(d_conf["commodity_series"])
        core_d = wide.get(d_conf["core_series"])
        if commodity is not None and core_d is not None:
            df_id = commodity_core_composite(
                core_d.dropna(),
                commodity.dropna(),
                commodity_change_months=d_conf["commodity_change_months"],
                zscore_window=zconf["rolling_window_months"],
                zscore_min_periods=zconf["min_periods_months"],
            )
            df_id.to_csv(PROCESSED_DIR / "inflation_model_d_detail.csv")
            out["inflation_model_commodity_core_aux"] = df_id["inflation_label"]

    # See the matching comment in `_growth_signals`: normalize every
    # inflation model's output index to month-end here so it shares a
    # convention with the (also-normalized) growth columns and with the
    # evaluation targets.
    return {name: normalize_month_end_index(series) for name, series in out.items()}


@app.command("build-signals")
def build_signals(
    growth_model: str = typer.Option(
        "all",
        "--growth-model",
        help=(
            "all | model_a_cli | model_b_fred_minimal | "
            "model_c_simple_two_signal | model_d_amtmno_claims"
        ),
    ),
    inflation_model: str = typer.Option(
        "all",
        "--inflation-model",
        help=(
            "all | model_a_realized_core | model_b_leading_composite | "
            "model_c_cleveland_median_cpi | model_d_commodity_core_aux"
        ),
    ),
) -> None:
    """Compute growth/inflation direction labels and combine them into
    every growth-model x inflation-model regime series."""
    if not WIDE_PATH.exists():
        console.print("[red]No fetched data found. Run `fetch` first.[/red]")
        raise typer.Exit(code=1)

    config = load_config()
    wide = pd.read_csv(WIDE_PATH, index_col=0, parse_dates=True)

    growth = _growth_signals(wide, config, growth_model)
    inflation = _inflation_signals(wide, config, inflation_model)

    if not growth or not inflation:
        console.print("[yellow]Warning: no growth or inflation models produced output.[/yellow]")

    signals_df = pd.concat({**growth, **inflation}, axis=1).sort_index()

    regime_metadata: dict[str, dict[str, str]] = {}
    for gname in growth:
        for iname in inflation:
            g_suffix = gname.replace("growth_model_", "")
            i_suffix = iname.replace("inflation_model_", "")
            regime_col = f"regime_{g_suffix}_{i_suffix}"
            signals_df[regime_col] = build_regime_series(
                signals_df[gname], signals_df[iname], name=regime_col
            )
            regime_metadata[regime_col] = {"growth_model": gname, "inflation_model": iname}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    signals_df.to_csv(SIGNALS_PATH)
    with REGIME_METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(regime_metadata, fh, indent=2)

    console.print(
        f"[green]Saved {signals_df.shape[0]} rows x {signals_df.shape[1]} columns to {SIGNALS_PATH}[/green]"
    )
    console.print(f"Regime combinations: {list(regime_metadata.keys())}")


@app.command()
def evaluate() -> None:
    """Score every growth/inflation model against forward realized targets,
    alongside always-Up / always-Down naive baselines."""
    if not SIGNALS_PATH.exists() or not WIDE_PATH.exists():
        console.print("[red]Missing data. Run `fetch` and `build-signals` first.[/red]")
        raise typer.Exit(code=1)

    config = load_config()
    econf = config["evaluation"]
    wide = pd.read_csv(WIDE_PATH, index_col=0, parse_dates=True)
    signals = pd.read_csv(SIGNALS_PATH, index_col=0, parse_dates=True)

    report = EvaluationReport()

    # Normalize to the same month-end convention as the (already-normalized)
    # signal columns in `signals.csv` -- otherwise `.reindex(target.index)`
    # below silently returns all-NaN predictions for any date convention
    # mismatch, exactly as it did before `_growth_signals`/`_inflation_signals`
    # started normalizing their output.
    indpro = normalize_month_end_index(wide[econf["growth_target_series"]].dropna())
    growth_cols = [c for c in signals.columns if c.startswith("growth_model")]
    for horizon in econf["growth_horizons_months"]:
        target = growth_forward_target(indpro, horizon)
        for gcol in growth_cols:
            pred = signals[gcol].reindex(target.index)
            ev = evaluate_model(
                gcol,
                f"{econf['growth_target_series']}_forward",
                horizon,
                pred,
                target["actual_label"],
                signal_score=None,
                target_level=indpro.reindex(target.index),
            )
            report.add(ev)

    core = normalize_month_end_index(wide[econf["inflation_target_series"]].dropna())
    trailing_12m = core / core.shift(12) - 1.0
    inflation_cols = [c for c in signals.columns if c.startswith("inflation_model")]
    for horizon in econf["inflation_horizons_months"]:
        target = inflation_forward_target(core, horizon, compare_to=trailing_12m)
        for icol in inflation_cols:
            pred = signals[icol].reindex(target.index)
            ev = evaluate_model(
                icol,
                f"{econf['inflation_target_series']}_forward",
                horizon,
                pred,
                target["actual_label"],
            )
            report.add(ev)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with EVAL_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)

    frame = report.to_frame()
    columns = [
        "model",
        "target",
        "horizon_months",
        "n_obs",
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "baseline_always_up_accuracy",
        "baseline_always_down_accuracy",
    ]
    table = Table(title="Evaluation Report (vs. naive baselines)")
    for col in columns:
        table.add_column(col)
    for _, row in frame.iterrows():
        table.add_row(
            *[
                f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])
                for c in columns
            ]
        )
    console.print(table)
    console.print(f"[green]Saved full report to {EVAL_PATH}[/green]")


def _build_one_regime_output(
    wide: pd.DataFrame,
    config: dict,
    *,
    growth_model_key: str,
    inflation_model_key: str,
) -> pd.DataFrame:
    """Compute growth_score/growth_state and inflation_score/inflation_state
    directly from the frozen model functions (not from `_growth_signals` /
    `_inflation_signals`, which only expose labels) and assemble the
    standard regime-output schema. Only `model_a_cli`, `model_d_amtmno_claims`,
    and `model_a_realized_core` are supported -- the two frozen core models."""
    zconf = config["zscore"]
    gmodels = config["growth_models"]
    imodels = config["inflation_models"]

    if growth_model_key == "model_a_cli":
        g_conf = gmodels["model_a_cli"]
        cli_series = wide[g_conf["us_series"]].dropna()
        growth_score = normalize_month_end_index(
            cli_diagnostics(cli_series)[f"chg_{g_conf['primary_window_months']}m"]
        )
        growth_state = normalize_month_end_index(
            classify_growth_model_a(cli_series, primary_window=g_conf["primary_window_months"])
        )
    elif growth_model_key == "model_d_amtmno_claims":
        g_conf = gmodels["model_d_amtmno_claims"]
        amtmno = wide[g_conf["amtmno_series"]].dropna()
        claims = wide[g_conf["claims_series"]].dropna()
        df_g = growth_model_d_amtmno_claims(
            amtmno,
            claims,
            amtmno_change_months=g_conf["amtmno_change_months"],
            claims_change_months=g_conf["claims_change_months"],
            zscore_window=zconf["rolling_window_months"],
            zscore_min_periods=zconf["min_periods_months"],
        )
        growth_score = normalize_month_end_index(df_g["growth_score"])
        growth_state = normalize_month_end_index(df_g["growth_label"])
    else:
        raise ValueError(f"Unsupported growth_model_key for regime output: {growth_model_key}")

    if inflation_model_key == "model_a_realized_core":
        i_conf = imodels["model_a_realized_core"]
        core = wide[i_conf["core_series"]].dropna()
        df_i = core_inflation_momentum(
            core,
            short_window_months=i_conf["short_window_months"],
            long_window_months=i_conf["long_window_months"],
        )
        inflation_score = normalize_month_end_index(df_i["signal_raw"])
        inflation_state = normalize_month_end_index(df_i["inflation_label"])
    else:
        raise ValueError(f"Unsupported inflation_model_key for regime output: {inflation_model_key}")

    return build_regime_output(
        growth_score,
        growth_state,
        inflation_score,
        inflation_state,
        tradable_lag_months=config["regime_output"]["tradable_lag_months"],
    )


@app.command("build-regime-output")
def build_regime_output_cmd() -> None:
    """Build the standard asset-allocation regime output for the two frozen
    core models (Primary: US OECD CLI x Core Inflation Momentum; Secondary:
    AMTMNO+Claims x Core Inflation Momentum) -- see `regime_output` in
    config/default.yaml. Computed directly from `fred_wide.csv`; does not
    read or modify `signals.csv` / `build-signals` / `evaluate` output."""
    if not WIDE_PATH.exists():
        console.print("[red]No fetched data found. Run `fetch` first.[/red]")
        raise typer.Exit(code=1)

    config = load_config()
    wide = pd.read_csv(WIDE_PATH, index_col=0, parse_dates=True)
    roconf = config["regime_output"]

    primary = _build_one_regime_output(
        wide,
        config,
        growth_model_key=roconf["primary"]["growth_model"],
        inflation_model_key=roconf["primary"]["inflation_model"],
    )
    secondary = _build_one_regime_output(
        wide,
        config,
        growth_model_key=roconf["secondary"]["growth_model"],
        inflation_model_key=roconf["secondary"]["inflation_model"],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    primary.to_csv(REGIME_OUTPUT_PRIMARY_PATH)
    secondary.to_csv(REGIME_OUTPUT_SECONDARY_PATH)
    console.print(
        f"[green]Saved regime_output_primary.csv ({len(primary)} rows) and "
        f"regime_output_secondary.csv ({len(secondary)} rows)[/green]"
    )


@app.command("run-backtest")
def run_backtest_cmd(
    refresh_cache: bool = typer.Option(False, "--refresh-cache"),
) -> None:
    """Run the v1.0 fixed-regime asset allocation backtest. Fetches asset
    and FX prices from Yahoo Finance (yfinance) -- a separate, slower,
    independently-cached network call from the FRED pipeline, so this is
    not chained into `run-all`. Requires `build-regime-output` to have
    already produced `regime_output_primary.csv` / `regime_output_secondary.csv`.
    Weights are used exactly as configured in `backtest.regime_allocations`
    -- nothing here is optimized or fit to data."""
    if not REGIME_OUTPUT_PRIMARY_PATH.exists() or not REGIME_OUTPUT_SECONDARY_PATH.exists():
        console.print("[red]Missing regime output. Run `build-regime-output` first.[/red]")
        raise typer.Exit(code=1)

    config = load_config()
    primary_df = pd.read_csv(REGIME_OUTPUT_PRIMARY_PATH, index_col=0, parse_dates=True)
    secondary_df = pd.read_csv(REGIME_OUTPUT_SECONDARY_PATH, index_col=0, parse_dates=True)
    primary_regime = primary_df["tradable_regime"]
    secondary_regime = secondary_df["tradable_regime"]

    console.print("[cyan]Fetching asset + FX prices from Yahoo Finance...[/cyan]")
    asset_returns, common_start = build_monthly_return_matrix(config, refresh_cache=refresh_cache)
    console.print(f"[green]Common backtest start date: {common_start.date()}[/green]")

    bconf = config["backtest"]
    cost_bps = bconf["transaction_cost_bps"]
    risk_free_returns = asset_returns[bconf["risk_free_asset"]]

    allocations = regime_allocations(config)
    results: dict[str, StrategyResult] = {
        "primary": run_regime_strategy(
            asset_returns, primary_regime, allocations, transaction_cost_bps=cost_bps
        ),
        "secondary": run_regime_strategy(
            asset_returns, secondary_regime, allocations, transaction_cost_bps=cost_bps
        ),
        "static_60_40": run_static_strategy(
            asset_returns, static_6040_allocation(config), transaction_cost_bps=cost_bps
        ),
        "equal_weight": run_static_strategy(
            asset_returns, equal_weight_allocation(config), transaction_cost_bps=cost_bps
        ),
        "growth_basket_buy_and_hold": run_buy_and_hold_strategy(
            asset_returns, growth_basket_allocation(config), transaction_cost_bps=cost_bps
        ),
    }

    summary = build_summary_table(results, risk_free_returns)
    annual_returns = annual_returns_table(results)
    monthly_returns = pd.DataFrame({name: r.returns_post_cost for name, r in results.items()})
    crisis = crisis_period_performance(annual_returns)

    primary_aligned = primary_regime.reindex(asset_returns.index)
    secondary_aligned = secondary_regime.reindex(asset_returns.index)
    regime_avg = regime_average_returns(
        asset_returns, primary_aligned, {name: r.returns_post_cost for name, r in results.items()}
    )
    agreement = regime_agreement_rate(primary_aligned, secondary_aligned)

    regime_analysis_rows: list[dict] = []
    for regime, row in regime_avg.iterrows():
        for series, value in row.items():
            regime_analysis_rows.append(
                {
                    "analysis_type": "regime_avg_monthly_return",
                    "key": regime,
                    "series": series,
                    "value": value,
                }
            )
    for year, row in crisis.iterrows():
        for series, value in row.items():
            regime_analysis_rows.append(
                {"analysis_type": "crisis_year_return", "key": str(year), "series": series, "value": value}
            )
    regime_analysis_rows.append(
        {
            "analysis_type": "primary_secondary_agreement_rate",
            "key": "overall",
            "series": "tradable_regime",
            "value": agreement,
        }
    )
    regime_analysis = pd.DataFrame(regime_analysis_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BACKTEST_SUMMARY_PATH, index=False)
    annual_returns.to_csv(BACKTEST_ANNUAL_RETURNS_PATH)
    monthly_returns.to_csv(BACKTEST_MONTHLY_RETURNS_PATH)
    results["primary"].weights.to_csv(BACKTEST_ALLOCATIONS_PRIMARY_PATH)
    results["secondary"].weights.to_csv(BACKTEST_ALLOCATIONS_SECONDARY_PATH)
    regime_analysis.to_csv(BACKTEST_REGIME_ANALYSIS_PATH, index=False)

    table = Table(title="Backtest Summary (post-cost)")
    summary_cols = [
        "strategy",
        "start_date",
        "end_date",
        "cagr_post_cost",
        "annualized_vol_post_cost",
        "sharpe_post_cost",
        "max_drawdown_post_cost",
        "final_value_post_cost",
    ]
    for col in summary_cols:
        table.add_column(col)
    for _, row in summary.iterrows():
        table.add_row(*[f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in summary_cols])
    console.print(table)
    console.print(f"[green]Saved 6 backtest output files to {PROCESSED_DIR}[/green]")


@app.command("run-all")
def run_all(
    start_date: str = typer.Option(None, "--start-date"),
    end_date: str = typer.Option(None, "--end-date"),
    refresh_cache: bool = typer.Option(False, "--refresh-cache"),
    growth_model: str = typer.Option("all", "--growth-model"),
    inflation_model: str = typer.Option("all", "--inflation-model"),
) -> None:
    """Run fetch -> build-signals -> evaluate -> build-regime-output end to end."""
    fetch(start_date=start_date, end_date=end_date, refresh_cache=refresh_cache)
    build_signals(growth_model=growth_model, inflation_model=inflation_model)
    evaluate()
    build_regime_output_cmd()


if __name__ == "__main__":
    app()
