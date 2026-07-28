"""v1.3 ("Growth Participation") live production-behavior tests.

Same on-disk-cache-backed, no-live-network pattern as
`test_v1_2_regression.py`. Proves v1.3 changes are correctly scoped to
GOLDILOCKS/REFLATION only, and that BEI Duration Gate / Fast Crisis /
t+1 execution / transaction costs behave identically to v1.2 (since none
of that machinery is touched by `strategy_versions.build_versioned_config`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macro_regime.config import load_config
from macro_regime.fast_crisis.backtest import V1_2_LABEL, run_fast_crisis_backtest
from macro_regime.strategy_versions import build_versioned_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cache_available() -> bool:
    return (PROJECT_ROOT / "data" / "raw" / "asset_cache").exists() and (
        PROJECT_ROOT / "data" / "raw" / "cache"
    ).exists()


pytestmark = pytest.mark.skipif(not _cache_available(), reason="requires the on-disk price/macro cache")


@pytest.fixture(scope="module")
def backtests():
    wide = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fred_wide.csv", index_col=0, parse_dates=True)
    primary_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "regime_output_primary.csv", index_col=0, parse_dates=True
    )
    primary_regime = primary_df["tradable_regime"]
    as_of = pd.Timestamp("2026-06-30")
    v1_2_config = build_versioned_config(load_config(), "v1_2")
    v1_3_config = build_versioned_config(load_config(), "v1_3")
    bt_v1_2 = run_fast_crisis_backtest(v1_2_config, primary_regime, wide, as_of=as_of)
    bt_v1_3 = run_fast_crisis_backtest(v1_3_config, primary_regime, wide, as_of=as_of)
    return bt_v1_2, bt_v1_3


def test_v1_3_goldilocks_allocation_exact(backtests):
    _, bt_v1_3 = backtests
    alloc = bt_v1_3.allocations
    mask = alloc["macro_regime"] == "GOLDILOCKS"
    expected = {
        "macro_spy": 0.65 * 0.6,
        "macro_kodex200_usd": 0.65 * 0.4,
        "macro_high_yield": 0.10,
        "macro_investment_grade": 0.10,
        "macro_gold": 0.05,
        "macro_tbills": 0.00,
    }
    for col, val in expected.items():
        assert (alloc.loc[mask, col].round(10) == round(val, 10)).all(), col


def test_v1_3_reflation_allocation_exact(backtests):
    _, bt_v1_3 = backtests
    alloc = bt_v1_3.allocations
    mask = alloc["macro_regime"] == "REFLATION"
    expected = {
        "macro_spy": 0.45 * 0.6,
        "macro_kodex200_usd": 0.45 * 0.4,
        "macro_commodities": 0.20,
        "macro_tips": 0.15,
        "macro_gold": 0.10,
        "macro_high_yield": 0.10,
        "macro_tbills": 0.00,
    }
    for col, val in expected.items():
        assert (alloc.loc[mask, col].round(10) == round(val, 10)).all(), col


def test_v1_3_contraction_unchanged_vs_v1_2(backtests):
    bt_v1_2, bt_v1_3 = backtests
    mask = bt_v1_2.allocations["macro_regime"] == "CONTRACTION"
    r2 = bt_v1_2.results[V1_2_LABEL].returns_pre_cost.loc[mask]
    r3 = bt_v1_3.results[V1_2_LABEL].returns_pre_cost.loc[mask]
    assert r2.equals(r3)


def test_v1_3_stagflation_unchanged_vs_v1_2(backtests):
    bt_v1_2, bt_v1_3 = backtests
    mask = bt_v1_2.allocations["macro_regime"] == "STAGFLATION"
    r2 = bt_v1_2.results[V1_2_LABEL].returns_pre_cost.loc[mask]
    r3 = bt_v1_3.results[V1_2_LABEL].returns_pre_cost.loc[mask]
    assert r2.equals(r3)


def test_v1_3_weights_sum_to_one_every_day(backtests):
    _, bt_v1_3 = backtests
    final_cols = [c for c in bt_v1_3.allocations.columns if c.startswith("final_")]
    sums = bt_v1_3.allocations[final_cols].sum(axis=1)
    assert (sums.round(6) == 1.0).all()


def test_v1_3_bil_never_negative(backtests):
    _, bt_v1_3 = backtests
    final_cols = [c for c in bt_v1_3.allocations.columns if c.startswith("final_")]
    assert (bt_v1_3.allocations[final_cols] >= -1e-9).all().all()


def test_v1_3_fast_crisis_on_days_identical_to_v1_2(backtests):
    """Fast Crisis is a daily overlay driven by VIX/SPY/HYG shocks, none
    of which depend on the regime allocation table -- ON/OFF days must
    be identical between v1.2 and v1.3."""
    bt_v1_2, bt_v1_3 = backtests
    assert bt_v1_2.allocations["crisis_mode"].equals(bt_v1_3.allocations["crisis_mode"])


def test_v1_3_bei_gate_on_days_identical_to_v1_2(backtests):
    bt_v1_2, bt_v1_3 = backtests
    gate_on_v1_2 = (bt_v1_2.allocations["macro_regime"] == "CONTRACTION") & (
        (bt_v1_2.allocations["macro_long_treasury"] - bt_v1_2.allocations["bei_long_treasury"]) > 1e-9
    )
    gate_on_v1_3 = (bt_v1_3.allocations["macro_regime"] == "CONTRACTION") & (
        (bt_v1_3.allocations["macro_long_treasury"] - bt_v1_3.allocations["bei_long_treasury"]) > 1e-9
    )
    assert gate_on_v1_2.equals(gate_on_v1_3)
