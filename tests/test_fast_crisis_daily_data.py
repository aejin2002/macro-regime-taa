import numpy as np
import pandas as pd
import pytest

from macro_regime.fast_crisis.daily_data import (
    KODEX200_USD_COLUMN,
    build_daily_return_matrix,
    determine_common_start,
    validate_no_gaps_in_range,
)


class _FakeAssetPriceClient:
    """No-network stand-in for AssetPriceClient, mirroring the pattern in
    tests/test_backtest.py's _FakeAssetPriceClient."""

    def __init__(self, series_by_ticker: dict[str, pd.Series]):
        self._series_by_ticker = series_by_ticker

    def get_daily_close(self, ticker: str, start: str, *, refresh_cache: bool = False) -> pd.Series:
        return self._series_by_ticker[ticker]


def _spy_calendar(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


US_ASSET_NAMES = [
    "spy",
    "high_yield",
    "investment_grade",
    "intermediate_treasury",
    "long_treasury",
    "gold",
    "tbills",
    "commodities",
    "tips",
]


def _flat_raw(n: int, *, spy_value: float = 100.0) -> dict:
    idx = _spy_calendar(n)
    us_prices = {name: pd.Series(spy_value + i, index=idx) for i, name in enumerate(US_ASSET_NAMES)}
    kodex_krw = pd.Series(50000.0, index=idx)
    fx = pd.Series(1300.0, index=idx)
    vix = pd.Series(20.0, index=idx)
    return {"us_prices": us_prices, "kodex_krw": kodex_krw, "fx": fx, "vix": vix}


# -- A. missing-value handling -----------------------------------------------


def test_internal_nan_at_existing_date_is_repaired_by_ffill_not_left_unfixed():
    """Reproduces the 2024-10-30-style bug: an already-present trading day
    whose value is NaN (a provider data gap, not a missing calendar
    date). Plain reindex(method='ffill') alone would NOT fix this --
    the source series itself must be .ffill()-ed first. After the fix,
    the gap day carries forward the last known value, and no NaN
    remains in the return matrix from that point on."""
    raw = _flat_raw(10)
    idx = raw["kodex_krw"].index
    raw["kodex_krw"] = raw["kodex_krw"].copy()
    raw["kodex_krw"].iloc[5] = np.nan  # index entry exists, value is NaN

    returns, _ = build_daily_return_matrix(raw)
    assert not returns[KODEX200_USD_COLUMN].iloc[6:].isna().any()
    # the gap day itself carries the last known price forward -> its
    # OWN return is 0%, not NaN, and not silently skipped later.
    assert returns[KODEX200_USD_COLUMN].loc[idx[5]] == pytest.approx(0.0)


def test_monthly_compounding_is_not_distorted_by_a_repaired_internal_gap():
    """The specific failure mode this bug caused: prod(skipna=True)
    silently treating a NaN day as an implicit 0% return, breaking the
    telescoping identity so a month's compounded return no longer
    equals its start/end price ratio. After the fix, compounding across
    a repaired gap must exactly telescope."""
    raw = _flat_raw(15)
    idx = raw["kodex_krw"].index
    raw["kodex_krw"] = raw["kodex_krw"].copy()
    raw["kodex_krw"].iloc[7] = 50500.0
    raw["kodex_krw"].iloc[8] = np.nan  # gap in the middle of the window
    raw["kodex_krw"].iloc[9] = 51000.0

    returns, _ = build_daily_return_matrix(raw)
    window = returns[KODEX200_USD_COLUMN].loc[idx[1:10]]
    compounded = (1 + window).prod() - 1
    direct = raw["kodex_krw"].iloc[9] / raw["kodex_krw"].iloc[0] - 1.0
    assert compounded == pytest.approx(direct, abs=1e-9)


def test_validate_no_gaps_in_range_raises_and_names_offender():
    idx = _spy_calendar(10)
    returns = pd.DataFrame({"kodex200_usd": [1.0] * 10, "spy": [1.0] * 10}, index=idx)
    returns.loc[idx[5], "kodex200_usd"] = np.nan
    with pytest.raises(ValueError, match="kodex200_usd"):
        validate_no_gaps_in_range(returns, idx.min(), idx.max())


def test_validate_no_gaps_in_range_passes_silently_when_clean():
    idx = _spy_calendar(10)
    returns = pd.DataFrame({"kodex200_usd": [1.0] * 10, "spy": [1.0] * 10}, index=idx)
    validate_no_gaps_in_range(returns, idx.min(), idx.max())  # must not raise


def test_no_backfill_before_series_own_first_valid_value():
    """.ffill() only carries a value FORWARD -- a NaN before a series'
    own first valid observation must remain NaN, never invented."""
    raw = _flat_raw(10)
    raw["kodex_krw"] = raw["kodex_krw"].copy()
    raw["kodex_krw"].iloc[:3] = np.nan  # series "starts late"
    returns, _ = build_daily_return_matrix(raw)
    idx = raw["kodex_krw"].index
    # day 0's return is always NaN (pct_change has no prior day); days
    # 1-3 must also stay NaN since there is no valid KODEX price yet.
    assert returns[KODEX200_USD_COLUMN].loc[idx[:4]].isna().all()


# -- B. trading-day alignment -------------------------------------------------


def test_kodex_fx_trading_on_a_us_holiday_attributes_to_next_us_trading_day():
    """2010-05-31-style case: KRX/FX trade on a date the SPY (NYSE)
    calendar has no session for. That extra KRX/FX movement must show
    up on the NEXT SPY trading day's return -- not be lost, and not
    look ahead into it before that day arrives."""
    spy_idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])  # gap: no US session on 01-04
    kodex_idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    fx_idx = kodex_idx

    us_prices = {name: pd.Series(100.0, index=spy_idx) for name in US_ASSET_NAMES}
    kodex_krw = pd.Series([50000.0, 50000.0, 51000.0, 51000.0], index=kodex_idx)  # KRX moves on 01-04
    fx = pd.Series(1300.0, index=fx_idx)
    vix = pd.Series(20.0, index=spy_idx)
    raw = {"us_prices": us_prices, "kodex_krw": kodex_krw, "fx": fx, "vix": vix}

    returns, _ = build_daily_return_matrix(raw)
    # the SPY calendar has no 01-04 row at all; the KRX move that day is
    # only visible in the return computed on 01-05 (the next SPY day).
    assert pd.Timestamp("2024-01-04") not in returns.index
    ret_0105 = returns.loc[pd.Timestamp("2024-01-05"), KODEX200_USD_COLUMN]
    assert ret_0105 == pytest.approx(51000.0 / 50000.0 - 1.0)


def test_as_of_join_never_uses_a_future_krx_price():
    """Every SPY day's KODEX/FX value must be the most recent one ON OR
    BEFORE that day -- confirmed by mutating a future KRX price and
    checking past returns are bit-identical."""
    raw_a = _flat_raw(10)
    returns_a, _ = build_daily_return_matrix(raw_a)

    raw_b = _flat_raw(10)
    raw_b["kodex_krw"] = raw_b["kodex_krw"].copy()
    raw_b["kodex_krw"].iloc[-1] = 999999.0  # mutate only the last (future-most) day
    returns_b, _ = build_daily_return_matrix(raw_b)

    pd.testing.assert_series_equal(
        returns_a[KODEX200_USD_COLUMN].iloc[:-1], returns_b[KODEX200_USD_COLUMN].iloc[:-1]
    )


# -- common start / as_of trimming -------------------------------------------


def test_determine_common_start_respects_not_before():
    raw = _flat_raw(30)
    returns, vix = build_daily_return_matrix(raw)
    not_before = returns.index[10]
    common_start = determine_common_start(returns, vix, not_before=not_before)
    assert common_start >= not_before


def test_as_of_excludes_days_after_as_of():
    raw = _flat_raw(10)
    as_of = list(raw["us_prices"]["spy"].index)[5]
    returns, vix = build_daily_return_matrix(raw, as_of=as_of)
    assert returns.index.max() <= as_of
    assert vix.index.max() <= as_of
