import numpy as np
import pandas as pd

from macro_regime.backtest.assets import KODEX200_USD_COLUMN, build_monthly_return_matrix
from macro_regime.backtest.engine import (
    build_target_weights_from_regime,
    run_regime_strategy,
)

UP, DOWN, UNKNOWN = "Up", "Down", "Unknown"


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-31", periods=n, freq="ME")


def _synthetic_returns(n: int, columns: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    data = rng.normal(loc=0.005, scale=0.03, size=(n, len(columns)))
    return pd.DataFrame(data, index=idx, columns=columns)


ALLOCATIONS = {
    "GOLDILOCKS": {"spy": 0.6, "high_yield": 0.3, "tbills": 0.1},
    "REFLATION": {"spy": 0.3, "gold": 0.5, "tbills": 0.2},
    "STAGFLATION": {"gold": 0.7, "tbills": 0.3},
    "CONTRACTION": {"tbills": 0.6, "gold": 0.4},
    "UNKNOWN": {"tbills": 1.0},
}
COLUMNS = ["spy", "high_yield", "gold", "tbills"]


def test_weights_sum_to_one_every_month():
    n = 12
    returns = _synthetic_returns(n, COLUMNS)
    regime = pd.Series(
        (["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION", "UNKNOWN"] * 3)[:n],
        index=returns.index[:n],
    )
    result = run_regime_strategy(returns, regime, ALLOCATIONS, transaction_cost_bps=10)
    sums = result.weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)


def test_no_negative_weights():
    n = 10
    returns = _synthetic_returns(n, COLUMNS)
    regime = pd.Series(["GOLDILOCKS", "STAGFLATION"] * 5, index=returns.index[:n])[:n]
    result = run_regime_strategy(returns, regime, ALLOCATIONS, transaction_cost_bps=10)
    assert (result.weights >= -1e-12).all().all()


def test_unknown_regime_maps_to_full_tbills():
    n = 6
    returns = _synthetic_returns(n, COLUMNS)
    labels = ["GOLDILOCKS", "UNKNOWN", "GOLDILOCKS", "UNKNOWN", "GOLDILOCKS", "UNKNOWN"]
    regime = pd.Series(labels, index=returns.index[:n])
    result = run_regime_strategy(returns, regime, ALLOCATIONS, transaction_cost_bps=10)
    unknown_rows = result.weights[regime.reindex(result.weights.index) == "UNKNOWN"]
    assert (unknown_rows["tbills"] == 1.0).all()
    other_cols = [c for c in COLUMNS if c != "tbills"]
    assert (unknown_rows[other_cols] == 0.0).all().all()


def test_no_lookahead_future_regime_does_not_affect_past_returns():
    n = 12
    returns = _synthetic_returns(n, COLUMNS)
    regime_a = pd.Series(["GOLDILOCKS"] * n, index=returns.index)
    regime_b = regime_a.copy()
    regime_b.iloc[-1] = "STAGFLATION"  # change only the last (future-most) month

    result_a = run_regime_strategy(returns, regime_a, ALLOCATIONS, transaction_cost_bps=10)
    result_b = run_regime_strategy(returns, regime_b, ALLOCATIONS, transaction_cost_bps=10)

    pd.testing.assert_series_equal(
        result_a.returns_pre_cost.iloc[:-1], result_b.returns_pre_cost.iloc[:-1]
    )
    pd.testing.assert_frame_equal(result_a.weights.iloc[:-1], result_b.weights.iloc[:-1])


def test_post_cost_value_never_exceeds_pre_cost_value():
    n = 12
    returns = _synthetic_returns(n, COLUMNS)
    # A regime that changes every month forces non-zero turnover throughout.
    regime = pd.Series(
        (["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION"] * 3)[:n], index=returns.index
    )
    result = run_regime_strategy(returns, regime, ALLOCATIONS, transaction_cost_bps=10)
    assert (result.value_post_cost <= result.value_pre_cost + 1e-12).all()
    assert (result.turnover > 0).any()


def test_reproducibility_identical_inputs_identical_outputs():
    n = 12
    returns = _synthetic_returns(n, COLUMNS)
    regime = pd.Series(
        (["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION"] * 3)[:n], index=returns.index
    )
    result_1 = run_regime_strategy(returns, regime, ALLOCATIONS, transaction_cost_bps=10)
    result_2 = run_regime_strategy(returns, regime, ALLOCATIONS, transaction_cost_bps=10)

    pd.testing.assert_series_equal(result_1.returns_post_cost, result_2.returns_post_cost)
    pd.testing.assert_frame_equal(result_1.weights, result_2.weights)
    pd.testing.assert_series_equal(result_1.turnover, result_2.turnover)


def test_build_target_weights_unrecognized_regime_falls_back_to_unknown():
    idx = _idx(3)
    regime = pd.Series(["GOLDILOCKS", "SOME_UNRECOGNIZED_LABEL", "GOLDILOCKS"], index=idx)
    target = build_target_weights_from_regime(regime, ALLOCATIONS, COLUMNS)
    assert target.loc[idx[1], "tbills"] == 1.0


class _FakeAssetPriceClient:
    """No-network stand-in for AssetPriceClient, for testing
    build_monthly_return_matrix's gap-handling and common-start-date
    logic without hitting Yahoo Finance."""

    def __init__(self, series_by_ticker: dict[str, pd.Series]):
        self._series_by_ticker = series_by_ticker

    def get_daily_close(self, ticker: str, start: str, *, refresh_cache: bool = False) -> pd.Series:
        return self._series_by_ticker[ticker]


def test_no_forward_fill_across_a_missing_month_and_correct_common_start():
    # ICSA-style daily series with a real gap: no trading days at all in
    # month 3 (e.g. as if that asset had a data outage), and "gold" starts
    # two months later than everything else (forces a later common start).
    daily_idx_full = pd.bdate_range("2020-01-01", periods=260)
    # Remove all business days falling in March 2020 to simulate a gap.
    daily_idx_gapped = daily_idx_full[~((daily_idx_full.month == 3) & (daily_idx_full.year == 2020))]

    rng = np.random.default_rng(1)
    n_full = len(daily_idx_full)
    spy = pd.Series(100 + np.cumsum(rng.normal(size=len(daily_idx_gapped))), index=daily_idx_gapped)
    tbills = pd.Series(100 + np.cumsum(rng.normal(scale=0.1, size=n_full)), index=daily_idx_full)
    kodex_krw = pd.Series(50000 + np.cumsum(rng.normal(scale=50, size=n_full)), index=daily_idx_full)
    fx = pd.Series(1200 + np.cumsum(rng.normal(scale=1, size=n_full)), index=daily_idx_full)
    # "gold" only starts in April 2020 -- the latest-inception asset, so it
    # should drive the common start date.
    gold_idx = daily_idx_full[daily_idx_full >= "2020-04-01"]
    gold = pd.Series(1500 + np.cumsum(rng.normal(scale=5, size=len(gold_idx))), index=gold_idx)

    fake_series = {"SPY": spy, "BIL": tbills, "GLD": gold, "069500.KS": kodex_krw, "KRW=X": fx}
    client = _FakeAssetPriceClient(fake_series)

    config = {
        "growth_basket": {
            "spy_weight": 0.6,
            "kodex200_weight": 0.4,
            "kodex200_ticker": "069500.KS",
            "fx_ticker": "KRW=X",
        },
        "backtest": {"assets": {"spy": "SPY", "gold": "GLD", "tbills": "BIL"}},
    }

    returns, common_start = build_monthly_return_matrix(config, client=client, start="2020-01-01")

    # Common start must be driven by gold's later inception, not an earlier date.
    assert common_start >= pd.Timestamp("2020-04-30")

    # March 2020's SPY gap: confirm it produced NaN in the pre-trim matrix
    # rather than being silently filled from a later (future) month. We
    # rebuild the untrimmed matrix the same way to inspect it directly.
    from macro_regime.backtest.assets import _to_month_end

    spy_monthly = _to_month_end(spy)
    assert pd.isna(spy_monthly.get(pd.Timestamp("2020-03-31")))

    assert KODEX200_USD_COLUMN in returns.columns
    assert returns.notna().all().all()


def _fake_client_with_daily_index(idx: pd.DatetimeIndex, seed: int) -> _FakeAssetPriceClient:
    rng = np.random.default_rng(seed)
    n = len(idx)
    return _FakeAssetPriceClient(
        {
            "SPY": pd.Series(100 + np.cumsum(rng.normal(size=n)), index=idx),
            "BIL": pd.Series(100 + np.cumsum(rng.normal(size=n)), index=idx),
            "GLD": pd.Series(100 + np.cumsum(rng.normal(size=n)), index=idx),
            "069500.KS": pd.Series(100 + np.cumsum(rng.normal(size=n)), index=idx),
            "KRW=X": pd.Series(1200 + np.cumsum(rng.normal(scale=1, size=n)), index=idx),
        }
    )


_MINI_CONFIG = {
    "growth_basket": {
        "spy_weight": 0.6,
        "kodex200_weight": 0.4,
        "kodex200_ticker": "069500.KS",
        "fx_ticker": "KRW=X",
    },
    "backtest": {"assets": {"spy": "SPY", "gold": "GLD", "tbills": "BIL"}},
}


def test_in_progress_month_is_excluded_until_it_actually_completes():
    # Daily data runs through 2020-07-10 -- July has not finished yet.
    # resample("ME").last() would otherwise stamp that partial data
    # "2020-07-31", a future date relative to `as_of`.
    idx_partial_july = pd.bdate_range("2020-01-01", "2020-07-10")
    client_partial = _fake_client_with_daily_index(idx_partial_july, seed=2)

    as_of_mid_july = pd.Timestamp("2020-07-10")
    returns_partial, _ = build_monthly_return_matrix(
        _MINI_CONFIG, client=client_partial, start="2020-01-01", as_of=as_of_mid_july
    )
    assert pd.Timestamp("2020-07-31") not in returns_partial.index
    assert returns_partial.index.max() <= pd.Timestamp("2020-06-30")

    # Once July has actually closed (daily data covers the full month and
    # `as_of` is past month-end), the July row must be included.
    idx_full_july = pd.bdate_range("2020-01-01", "2020-07-31")
    client_full = _fake_client_with_daily_index(idx_full_july, seed=2)
    returns_full, _ = build_monthly_return_matrix(
        _MINI_CONFIG, client=client_full, start="2020-01-01", as_of=pd.Timestamp("2020-08-05")
    )
    assert pd.Timestamp("2020-07-31") in returns_full.index
