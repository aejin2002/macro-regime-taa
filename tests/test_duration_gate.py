import numpy as np
import pandas as pd
import pytest

from macro_regime.duration_gate.allocation import apply_bei_duration_gate
from macro_regime.duration_gate.signal import (
    OFF,
    ON,
    UNKNOWN,
    build_bei_duration_gate_signal,
    classify_raw_bei_duration_gate,
    monthly_rate_level,
    tradable_bei_duration_gate,
)
from macro_regime.utils.dates import drop_incomplete_trailing_month, resample_to_monthly


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2018-01-31", periods=n, freq="ME")


# -- data: monthly averaging / incomplete-month handling --------------------


def test_monthly_rate_level_is_a_true_average_not_last_value():
    idx = pd.date_range("2020-01-01", "2020-01-31", freq="D")
    values = np.linspace(1.0, 2.0, len(idx))
    series = pd.Series(values, index=idx)
    as_of = pd.Timestamp("2020-02-15")
    levels = monthly_rate_level(series, as_of=as_of)
    assert levels.loc[pd.Timestamp("2020-01-31")] == pytest.approx(values.mean())
    assert levels.loc[pd.Timestamp("2020-01-31")] != values[-1]


def test_monthly_rate_level_excludes_in_progress_month():
    idx = pd.date_range("2020-01-01", "2020-06-15", freq="D")
    series = pd.Series(1.0, index=idx)
    as_of = pd.Timestamp("2020-06-15")  # June not finished
    levels = monthly_rate_level(series, as_of=as_of)
    assert pd.Timestamp("2020-06-30") not in levels.index
    assert levels.index.max() == pd.Timestamp("2020-05-31")


def test_monthly_rate_level_never_forward_fills_a_gap_month():
    idx = pd.to_datetime(["2020-01-05", "2020-01-20", "2020-03-05", "2020-03-20"])
    series = pd.Series([1.0, 1.1, 2.0, 2.1], index=idx)
    as_of = pd.Timestamp("2020-04-01")
    levels = monthly_rate_level(series, as_of=as_of)
    assert pd.isna(levels.loc[pd.Timestamp("2020-02-29")])


def test_monthly_rate_level_does_not_backfill_before_series_start():
    idx = pd.date_range("2020-03-01", "2020-03-31", freq="D")
    series = pd.Series(1.0, index=idx)
    as_of = pd.Timestamp("2020-04-01")
    levels = monthly_rate_level(series, as_of=as_of)
    assert pd.Timestamp("2020-01-31") not in levels.index
    assert pd.Timestamp("2020-02-29") not in levels.index


def test_drop_incomplete_trailing_month_only_trims_the_future_row():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    series = pd.Series(range(5), index=idx)
    as_of = pd.Timestamp("2020-03-15")
    trimmed = drop_incomplete_trailing_month(series, as_of)
    assert trimmed.index.max() == pd.Timestamp("2020-02-29")
    assert len(trimmed) == 2


def test_resample_to_monthly_mean_matches_manual_average():
    idx = pd.date_range("2021-05-01", "2021-05-10", freq="D")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    series = pd.Series(values, index=idx)
    monthly = resample_to_monthly(series, how="mean")
    assert monthly.loc[pd.Timestamp("2021-05-31")] == pytest.approx(sum(values) / len(values))


# -- signal classification ---------------------------------------------------


def test_all_four_conditions_true_gives_on():
    idx = _dates(1)
    dgs10_1m = pd.Series([0.05], index=idx)
    bei_1m = pd.Series([0.03], index=idx)
    bei_3m = pd.Series([0.08], index=idx)
    tlt_1m = pd.Series([-0.02], index=idx)
    state = classify_raw_bei_duration_gate(dgs10_1m, bei_1m, bei_3m, tlt_1m)
    assert state.iloc[0] == ON


def test_dgs10_change_not_positive_gives_off():
    idx = _dates(2)
    dgs10_1m = pd.Series([0.0, -0.01], index=idx)  # zero and negative
    bei_1m = pd.Series([0.03, 0.03], index=idx)
    bei_3m = pd.Series([0.08, 0.08], index=idx)
    tlt_1m = pd.Series([-0.02, -0.02], index=idx)
    state = classify_raw_bei_duration_gate(dgs10_1m, bei_1m, bei_3m, tlt_1m)
    assert list(state) == [OFF, OFF]


def test_tlt_return_not_negative_gives_off():
    idx = _dates(2)
    dgs10_1m = pd.Series([0.05, 0.05], index=idx)
    bei_1m = pd.Series([0.03, 0.03], index=idx)
    bei_3m = pd.Series([0.08, 0.08], index=idx)
    tlt_1m = pd.Series([0.0, 0.01], index=idx)  # zero and positive
    state = classify_raw_bei_duration_gate(dgs10_1m, bei_1m, bei_3m, tlt_1m)
    assert list(state) == [OFF, OFF]


def test_bei_change_1m_not_positive_gives_off():
    idx = _dates(2)
    dgs10_1m = pd.Series([0.05, 0.05], index=idx)
    bei_1m = pd.Series([0.0, -0.01], index=idx)  # zero and negative
    bei_3m = pd.Series([0.08, 0.08], index=idx)
    tlt_1m = pd.Series([-0.02, -0.02], index=idx)
    state = classify_raw_bei_duration_gate(dgs10_1m, bei_1m, bei_3m, tlt_1m)
    assert list(state) == [OFF, OFF]


def test_bei_change_3m_not_positive_gives_off():
    idx = _dates(2)
    dgs10_1m = pd.Series([0.05, 0.05], index=idx)
    bei_1m = pd.Series([0.03, 0.03], index=idx)
    bei_3m = pd.Series([0.0, -0.01], index=idx)  # zero and negative
    tlt_1m = pd.Series([-0.02, -0.02], index=idx)
    state = classify_raw_bei_duration_gate(dgs10_1m, bei_1m, bei_3m, tlt_1m)
    assert list(state) == [OFF, OFF]


def test_missing_any_required_value_gives_unknown():
    idx = _dates(4)
    dgs10_1m = pd.Series([np.nan, 0.05, 0.05, 0.05], index=idx)
    bei_1m = pd.Series([0.03, np.nan, 0.03, 0.03], index=idx)
    bei_3m = pd.Series([0.08, 0.08, np.nan, 0.08], index=idx)
    tlt_1m = pd.Series([-0.02, -0.02, -0.02, np.nan], index=idx)
    state = classify_raw_bei_duration_gate(dgs10_1m, bei_1m, bei_3m, tlt_1m)
    assert list(state) == [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN]


def test_t10yie_percent_scale_not_confused_with_decimal():
    """T10YIE from FRED is already in percentage-point units (e.g. 2.35
    means 2.35%). A 0.05 percentage-point rise (dgs10_change_1m=0.05)
    must be classified the same way regardless of whether the caller's
    raw levels happen to be small (~0.5) or large (~5.0) -- the gate
    must never internally divide or multiply by 100."""
    idx = _dates(1)
    small_scale = classify_raw_bei_duration_gate(
        pd.Series([0.05], index=idx),
        pd.Series([0.03], index=idx),
        pd.Series([0.08], index=idx),
        pd.Series([-0.02], index=idx),
    )
    # Same *changes*, computed from levels an order of magnitude larger
    # (e.g. T10YIE ~= 2.3% vs ~= 0.3%) -- the classification only looks
    # at the change values themselves, so it must be identical.
    large_scale = classify_raw_bei_duration_gate(
        pd.Series([0.05], index=idx),
        pd.Series([0.03], index=idx),
        pd.Series([0.08], index=idx),
        pd.Series([-0.02], index=idx),
    )
    assert list(small_scale) == list(large_scale) == [ON]


# -- timing: t+1 application, no lookahead -----------------------------


def test_tradable_gate_is_raw_gate_shifted_by_one_month():
    idx = _dates(4)
    raw = pd.Series([ON, OFF, UNKNOWN, ON], index=idx)
    tradable = tradable_bei_duration_gate(raw)
    assert tradable.iloc[0] == UNKNOWN  # no prior month
    assert list(tradable.iloc[1:]) == list(raw.iloc[:-1])


def test_no_lookahead_future_rate_data_does_not_affect_past_signal():
    idx = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(0)
    dgs10 = pd.Series(2.0 + np.cumsum(rng.normal(scale=0.02, size=len(idx))), index=idx)
    t10yie = pd.Series(2.0 + np.cumsum(rng.normal(scale=0.01, size=len(idx))), index=idx)
    monthly_idx = pd.date_range("2015-01-31", "2020-12-31", freq="ME")
    tlt_returns = pd.Series(rng.normal(loc=0.003, scale=0.02, size=len(monthly_idx)), index=monthly_idx)
    as_of = pd.Timestamp("2021-01-15")

    table_a = build_bei_duration_gate_signal(dgs10, t10yie, tlt_returns, as_of=as_of)

    dgs10_mutated = dgs10.copy()
    dgs10_mutated.iloc[-1] = -999.0  # mutate only the very last daily observation
    table_b = build_bei_duration_gate_signal(dgs10_mutated, t10yie, tlt_returns, as_of=as_of)

    pd.testing.assert_series_equal(
        table_a["raw_bei_duration_gate"].iloc[:-1], table_b["raw_bei_duration_gate"].iloc[:-1]
    )
    pd.testing.assert_series_equal(
        table_a["tradable_bei_duration_gate"].iloc[:-1], table_b["tradable_bei_duration_gate"].iloc[:-1]
    )


def test_no_lookahead_future_gate_does_not_affect_past_allocation():
    idx = _dates(6)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.2] * 6, "long_treasury": [0.35] * 6, "tbills": [0.2] * 6},
        index=idx,
    )
    state_a = pd.Series([ON] * 6, index=idx)
    state_b = state_a.copy()
    state_b.iloc[-1] = OFF  # mutate only the future-most month

    overlaid_a = apply_bei_duration_gate(weights, state_a, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    overlaid_b = apply_bei_duration_gate(weights, state_b, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    pd.testing.assert_frame_equal(overlaid_a.iloc[:-1], overlaid_b.iloc[:-1])


def test_portfolio_at_month_t_never_uses_raw_gate_at_month_t():
    """The month-t raw gate (only fully known at t's own close) must not
    be the value driving month-t's allocation -- only tradable_gate
    (raw shifted by 1) may be passed to apply_bei_duration_gate."""
    idx = _dates(3)
    raw = pd.Series([OFF, ON, OFF], index=idx)  # raw ON only in month 2
    tradable = tradable_bei_duration_gate(raw)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.2] * 3, "long_treasury": [0.35] * 3, "tbills": [0.2] * 3}, index=idx
    )
    overlaid = apply_bei_duration_gate(weights, tradable, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    # raw ON at month 2 (idx[1]) should only affect month 3 (idx[2]), not month 2 itself
    assert overlaid["long_treasury"].iloc[1] == 0.35  # month 2 unaffected by its own raw ON
    assert overlaid["long_treasury"].iloc[2] == 0.0  # month 3 affected by month 2's raw ON


# -- allocation mechanics -----------------------------------------------


def test_gate_on_sets_tlt_zero_ief_30pct_bil_70pct_of_pool():
    idx = _dates(1)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.20], "long_treasury": [0.35], "tbills": [0.20]}, index=idx
    )
    state = pd.Series([ON], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    duration_pool = 0.55
    assert overlaid["long_treasury"].iloc[0] == 0.0
    assert overlaid["intermediate_treasury"].iloc[0] == pytest.approx(duration_pool * 0.30)
    assert overlaid["tbills"].iloc[0] == pytest.approx(0.20 + duration_pool * 0.70)


def test_contraction_example_from_spec():
    """Reproduces the worked example in the request: Contraction base
    TLT 35% / IEF 20% / BIL 20% -> Gate ON -> TLT 0% / IEF 16.5% /
    final BIL 58.5%."""
    idx = _dates(1)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.20], "long_treasury": [0.35], "tbills": [0.20]}, index=idx
    )
    state = pd.Series([ON], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    assert overlaid["long_treasury"].iloc[0] == pytest.approx(0.0)
    assert overlaid["intermediate_treasury"].iloc[0] == pytest.approx(0.165)
    assert overlaid["tbills"].iloc[0] == pytest.approx(0.585)


def test_goldilocks_example_from_spec():
    """Goldilocks base IEF 10% / TLT 0% / BIL 5% -> Gate ON -> IEF 3% /
    TLT 0% / final BIL 12%."""
    idx = _dates(1)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.10], "long_treasury": [0.0], "tbills": [0.05]}, index=idx
    )
    state = pd.Series([ON], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    assert overlaid["intermediate_treasury"].iloc[0] == pytest.approx(0.03)
    assert overlaid["long_treasury"].iloc[0] == pytest.approx(0.0)
    assert overlaid["tbills"].iloc[0] == pytest.approx(0.12)


def test_off_or_unknown_leaves_allocation_exactly_equal_to_base():
    idx = _dates(2)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.20, 0.10], "long_treasury": [0.35, 0.0], "tbills": [0.20, 0.05]},
        index=idx,
    )
    state = pd.Series([OFF, UNKNOWN], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    pd.testing.assert_frame_equal(overlaid, weights)


def test_zero_duration_pool_leaves_month_fully_unchanged():
    idx = _dates(1)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.0], "long_treasury": [0.0], "tbills": [0.20], "growth_basket": [0.80]},
        index=idx,
    )
    state = pd.Series([ON], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    pd.testing.assert_frame_equal(overlaid, weights)


def test_weights_sum_to_one_and_no_negatives():
    idx = _dates(4)
    weights = pd.DataFrame(
        {
            "spy": [0.21, 0.0, 0.06, 0.0],
            "kodex200_usd": [0.14, 0.0, 0.04, 0.0],
            "high_yield": [0.10, 0.0, 0.0, 0.0],
            "investment_grade": [0.10, 0.15, 0.0, 0.0],
            "intermediate_treasury": [0.10, 0.20, 0.20, 0.0],
            "long_treasury": [0.0, 0.35, 0.0, 0.0],
            "gold": [0.05, 0.10, 0.25, 0.0],
            "tbills": [0.05, 0.20, 0.20, 1.0],
            "commodities": [0.0, 0.0, 0.25, 0.0],
            "tips": [0.25, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    state = pd.Series([ON, ON, OFF, UNKNOWN], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    assert np.allclose(overlaid.sum(axis=1), 1.0, atol=1e-9)
    assert (overlaid >= -1e-12).all().all()


def test_base_bil_is_preserved_never_reduced():
    idx = _dates(2)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.20, 0.10], "long_treasury": [0.35, 0.0], "tbills": [0.20, 0.05]},
        index=idx,
    )
    state = pd.Series([ON, ON], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    assert (overlaid["tbills"] >= weights["tbills"] - 1e-12).all()


def test_other_assets_never_touched():
    idx = _dates(2)
    weights = pd.DataFrame(
        {
            "spy": [0.30, 0.30],
            "kodex200_usd": [0.10, 0.10],
            "high_yield": [0.10, 0.10],
            "investment_grade": [0.05, 0.05],
            "gold": [0.05, 0.05],
            "commodities": [0.0, 0.0],
            "tips": [0.0, 0.0],
            "intermediate_treasury": [0.20, 0.20],
            "long_treasury": [0.15, 0.15],
            "tbills": [0.05, 0.05],
        },
        index=idx,
    )
    state = pd.Series([ON, OFF], index=idx)
    overlaid = apply_bei_duration_gate(weights, state, ief_weight_when_on=0.30, bil_weight_when_on=0.70)
    for col in ["spy", "kodex200_usd", "high_yield", "investment_grade", "gold", "commodities", "tips"]:
        pd.testing.assert_series_equal(overlaid[col], weights[col])


def test_negative_weight_parameters_raise():
    idx = _dates(1)
    weights = pd.DataFrame(
        {"intermediate_treasury": [0.2], "long_treasury": [0.35], "tbills": [0.2]}, index=idx
    )
    state = pd.Series([ON], index=idx)
    with pytest.raises(ValueError):
        apply_bei_duration_gate(weights, state, ief_weight_when_on=-0.1, bil_weight_when_on=0.7)


# -- end-to-end signal table -----------------------------------------------


def test_build_signal_table_end_to_end():
    idx = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(1)
    dgs10 = pd.Series(2.0 + np.cumsum(rng.normal(scale=0.02, size=len(idx))), index=idx)
    t10yie = pd.Series(2.0 + np.cumsum(rng.normal(scale=0.01, size=len(idx))), index=idx)
    monthly_idx = pd.date_range("2015-01-31", "2020-12-31", freq="ME")
    tlt_returns = pd.Series(rng.normal(loc=0.003, scale=0.02, size=len(monthly_idx)), index=monthly_idx)

    table = build_bei_duration_gate_signal(dgs10, t10yie, tlt_returns, as_of=pd.Timestamp("2021-01-15"))
    required_cols = {
        "DGS10",
        "T10YIE",
        "dgs10_change_1m",
        "bei_change_1m",
        "bei_change_3m",
        "tlt_return_1m",
        "raw_bei_duration_gate",
        "tradable_bei_duration_gate",
    }
    assert required_cols.issubset(table.columns)
    assert table["raw_bei_duration_gate"].isin([ON, OFF, UNKNOWN]).all()
    assert table["tradable_bei_duration_gate"].isin([ON, OFF, UNKNOWN]).all()
    assert list(table["tradable_bei_duration_gate"].iloc[1:]) == list(
        table["raw_bei_duration_gate"].iloc[:-1]
    )
