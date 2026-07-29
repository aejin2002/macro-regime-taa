"""v1.2 exact-reproduction regression tests.

Unlike the rest of the test suite, these exercise the real
`run_fast_crisis_backtest` pipeline end-to-end (reading real FRED/asset
price data from the on-disk `FileCache`/`asset_cache` already populated
by prior pipeline runs -- no live network call is made as long as that
cache is warm). They are slower (~10s) than the rest of the suite.

The primary check compares v1.2 built via
`strategy_versions.build_versioned_config(config, "v1_2")` against v1.2
built the OLD way (`load_config()` used directly, with no versioning
layer at all) -- run in the SAME session against the SAME data vintage.
This isolates the real regression concern (did adding
`strategy_versions.py` change v1.2's behavior at all?) from upstream
FRED data revisions, which legitimately shift the committed
`fast_crisis_daily_returns.csv` snapshot by float-noise-scale amounts
every time `update-all --refresh-cache` re-fetches (FRED routinely
revises recent historical values by tiny amounts; this is expected and
is not a strategy-logic change -- see docs/methodology.md).
A secondary, loosely-toleranced sanity check against the committed
snapshot catches anything larger than that expected drift.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macro_regime.config import load_config
from macro_regime.fast_crisis.backtest import V1_2_LABEL, run_fast_crisis_backtest
from macro_regime.fast_crisis.metrics import cagr_daily
from macro_regime.strategy_versions import build_versioned_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROD_DAILY_RETURNS = PROJECT_ROOT / "data" / "processed" / "fast_crisis_daily_returns.csv"
PROD_SUMMARY = PROJECT_ROOT / "data" / "processed" / "fast_crisis_summary.csv"

# Upstream FRED revisions (recent-vintage historical values changing by a
# few ten-thousandths) can move final CAGR/returns by up to ~1e-4 without
# any regime/allocation/rule actually changing -- see module docstring.
DATA_REVISION_TOLERANCE = 2e-4


def _cache_available() -> bool:
    return (PROJECT_ROOT / "data" / "raw" / "asset_cache").exists() and (
        PROJECT_ROOT / "data" / "raw" / "cache"
    ).exists()


pytestmark = pytest.mark.skipif(
    not (_cache_available() and PROD_DAILY_RETURNS.exists() and PROD_SUMMARY.exists()),
    reason="requires the on-disk price/macro cache and existing production artifacts",
)


@pytest.fixture(scope="module")
def live_pair():
    """(bt_versioned, bt_unversioned) -- both built in this same session
    against identical, current data, differing ONLY in whether they went
    through `strategy_versions.build_versioned_config`. Must be
    byte-identical if that module truly changes nothing for v1_2.

    `as_of` is derived from the COMMITTED SNAPSHOT's own latest date
    (`PROD_DAILY_RETURNS`), not hardcoded: a fixed literal here silently
    drifts out of sync every time `update-all` regenerates the snapshot
    at a later "today" than whatever date was hardcoded when this file
    was last touched, making `test_v1_2_daily_returns_close_to_committed_snapshot`/
    `test_v1_2_summary_metrics_close_to_committed_snapshot` compare two
    different date ranges (a real month-plus gap, not float-noise) --
    that was reproduced as a pre-existing failure identically on
    `origin/main`, unrelated to any live-pipeline work, before this fix."""
    wide = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fred_wide.csv", index_col=0, parse_dates=True)
    primary_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "regime_output_primary.csv", index_col=0, parse_dates=True
    )
    prod_daily = pd.read_csv(PROD_DAILY_RETURNS, index_col=0, parse_dates=True)
    as_of = prod_daily.index.max() + pd.Timedelta(days=1)
    raw_config = load_config()
    versioned_config = build_versioned_config(raw_config, "v1_2")
    bt_unversioned = run_fast_crisis_backtest(raw_config, primary_df["tradable_regime"], wide, as_of=as_of)
    bt_versioned = run_fast_crisis_backtest(
        versioned_config, primary_df["tradable_regime"], wide, as_of=as_of
    )
    return bt_versioned, bt_unversioned


def test_versioned_v1_2_byte_identical_to_unversioned(live_pair):
    """The true regression check: strategy_versions.py must not have
    changed v1.2's behavior by even a single bit."""
    bt_versioned, bt_unversioned = live_pair
    r_v = bt_versioned.results[V1_2_LABEL].returns_post_cost
    r_u = bt_unversioned.results[V1_2_LABEL].returns_post_cost
    assert r_v.equals(r_u)
    assert bt_versioned.allocations["macro_regime"].equals(bt_unversioned.allocations["macro_regime"])
    assert bt_versioned.allocations["crisis_mode"].equals(bt_unversioned.allocations["crisis_mode"])


def test_v1_2_daily_returns_close_to_committed_snapshot(live_pair):
    bt_versioned, _ = live_pair
    prod = pd.read_csv(PROD_DAILY_RETURNS, index_col=0, parse_dates=True)["v1_2_fast_crisis"]
    recomputed = bt_versioned.results[V1_2_LABEL].returns_post_cost
    common = recomputed.index.intersection(prod.index)
    assert len(common) == len(prod)  # no dates lost
    diff = (recomputed.reindex(common) - prod.reindex(common)).abs()
    assert diff.max() < DATA_REVISION_TOLERANCE


def test_v1_2_summary_metrics_close_to_committed_snapshot(live_pair):
    bt_versioned, _ = live_pair
    prod_row = pd.read_csv(PROD_SUMMARY)
    prod_row = prod_row[prod_row["strategy"] == V1_2_LABEL].iloc[0]
    value = bt_versioned.results[V1_2_LABEL].value_post_cost
    assert abs(cagr_daily(value) - prod_row["cagr_post_cost"]) < DATA_REVISION_TOLERANCE


def test_v1_2_macro_regime_dates_match_committed_regime_output(live_pair):
    bt_versioned, _ = live_pair
    prod_regime = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "regime_output_primary.csv", index_col=0, parse_dates=True
    )["tradable_regime"]
    daily_regime = bt_versioned.allocations["macro_regime"]
    for month_end, expected in prod_regime.items():
        days_in_month = daily_regime.loc[(daily_regime.index.to_period("M") == month_end.to_period("M"))]
        if len(days_in_month):
            assert (days_in_month == expected).all(), f"{month_end}: expected {expected}"


def test_v1_2_bei_gate_only_active_in_contraction(live_pair):
    bt_versioned, _ = live_pair
    alloc = bt_versioned.allocations
    gate_on = (alloc["macro_regime"] == "CONTRACTION") & (
        (alloc["macro_long_treasury"] - alloc["bei_long_treasury"]) > 1e-9
    )
    assert (alloc.loc[gate_on, "macro_regime"] == "CONTRACTION").all()


def test_end_date_extends_past_last_closed_macro_month_with_default_as_of():
    """Sanity check on the daily/monthly separation fix, using live
    'now': the daily backtest must not be silently truncated at the
    monthly signal's last closed month when as_of is left at its live
    default."""
    wide = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fred_wide.csv", index_col=0, parse_dates=True)
    primary_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "regime_output_primary.csv", index_col=0, parse_dates=True
    )
    config = build_versioned_config(load_config(), "v1_2")
    bt_live = run_fast_crisis_backtest(config, primary_df["tradable_regime"], wide)
    last_closed_month = primary_df.index.max()
    assert bt_live.allocations.index.max() > last_closed_month
    assert bt_live.allocations["partial_month"].any()
