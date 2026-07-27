"""Typer CLI: fetch -> build-signals -> evaluate -> run-all."""

from __future__ import annotations

import json

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

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
    growth_model_b_fred_minimal,
    growth_model_c_simple_two_signal,
)
from macro_regime.signals.inflation import core_inflation_momentum, leading_inflation_composite
from macro_regime.signals.regime import build_regime_series

app = typer.Typer(add_completion=False, help="Growth/Inflation macro regime research CLI")
console = Console()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WIDE_PATH = PROCESSED_DIR / "fred_wide.csv"
METADATA_PATH = PROCESSED_DIR / "series_metadata.json"
SIGNALS_PATH = PROCESSED_DIR / "signals.csv"
REGIME_METADATA_PATH = PROCESSED_DIR / "regime_metadata.json"
EVAL_PATH = PROCESSED_DIR / "evaluation_report.json"


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
        kr_cli = wide.get(a_conf["kr_series"])
        if kr_cli is not None and kr_cli.dropna().shape[0] > 0:
            out["growth_model_kr_cli"] = classify_growth_model_a(
                kr_cli.dropna(), primary_window=a_conf["primary_window_months"]
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

    return out


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

    return out


@app.command("build-signals")
def build_signals(
    growth_model: str = typer.Option(
        "all", "--growth-model", help="all | model_a_cli | model_b_fred_minimal | model_c_simple_two_signal"
    ),
    inflation_model: str = typer.Option(
        "all", "--inflation-model", help="all | model_a_realized_core | model_b_leading_composite"
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

    indpro = wide[econf["growth_target_series"]].dropna()
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

    core = wide[econf["inflation_target_series"]].dropna()
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


@app.command("run-all")
def run_all(
    start_date: str = typer.Option(None, "--start-date"),
    end_date: str = typer.Option(None, "--end-date"),
    refresh_cache: bool = typer.Option(False, "--refresh-cache"),
    growth_model: str = typer.Option("all", "--growth-model"),
    inflation_model: str = typer.Option("all", "--inflation-model"),
) -> None:
    """Run fetch -> build-signals -> evaluate end to end."""
    fetch(start_date=start_date, end_date=end_date, refresh_cache=refresh_cache)
    build_signals(growth_model=growth_model, inflation_model=inflation_model)
    evaluate()


if __name__ == "__main__":
    app()
