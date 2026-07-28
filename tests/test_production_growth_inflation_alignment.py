"""Regression test for a QA-discovered bug: `growth_state`/`inflation_state`
in `regime_output_primary.csv` are the model's OWN unlagged monthly
reading (never shifted), but `macro_regime` in the daily production
artifact is `tradable_regime` -- `raw_regime` shifted forward by
`tradable_lag_months`. Showing the raw growth/inflation reading next to
`macro_regime` without applying the same lag silently explains the WRONG
month's regime. `production.build_v1_3_daily_artifact` now lags
`growth_state`/`inflation_state`/`growth_score`/`inflation_score` by the
same amount (exposed as `growth_state`/`inflation_state`, matching what
actually explains `macro_regime`), keeping the raw, most-recent reading
separately as `growth_state_observed`/`inflation_state_observed`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macro_regime.config import load_config
from macro_regime.production import build_v1_3_daily_artifact
from macro_regime.signals.regime import classify_regime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cache_available() -> bool:
    return (PROJECT_ROOT / "data" / "raw" / "asset_cache").exists() and (
        PROJECT_ROOT / "data" / "raw" / "cache"
    ).exists()


pytestmark = pytest.mark.skipif(not _cache_available(), reason="requires the on-disk price/macro cache")


@pytest.fixture(scope="module")
def daily_artifact():
    wide = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fred_wide.csv", index_col=0, parse_dates=True)
    primary_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "regime_output_primary.csv", index_col=0, parse_dates=True
    )
    config = load_config()
    df, _bt = build_v1_3_daily_artifact(config, primary_df["tradable_regime"], wide, primary_df)
    return df


def test_effective_growth_inflation_state_reconciles_with_macro_regime(daily_artifact):
    """The lagged (effective) growth_state + inflation_state must combine,
    via the SAME classify_regime function the engine itself uses, to
    exactly the macro_regime shown for that day -- for every day, not
    just the latest."""
    df = daily_artifact.dropna(subset=["growth_state", "inflation_state"])
    mismatches = []
    for _, row in df.iterrows():
        expected = classify_regime(row["growth_state"], row["inflation_state"]).value
        if expected != row["macro_regime"] and row["macro_regime"] != "UNKNOWN":
            mismatches.append(
                (row["date"], row["growth_state"], row["inflation_state"], row["macro_regime"], expected)
            )
    assert not mismatches, (
        f"{len(mismatches)} day(s) where effective growth/inflation state doesn't explain "
        f"macro_regime: {mismatches[:5]}"
    )


def test_observed_state_is_not_lagged_and_can_differ_from_effective(daily_artifact):
    """Sanity check that the two are genuinely different columns (not an
    accidental duplicate) -- there must be at least some days where the
    raw, most-recently-observed reading differs from the lagged one."""
    df = daily_artifact.dropna(subset=["growth_state", "growth_state_observed"])
    assert (df["growth_state"] != df["growth_state_observed"]).any()
