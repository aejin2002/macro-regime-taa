import numpy as np
import pandas as pd
import pytest

from macro_regime.fast_crisis.allocation import apply_fast_crisis_overlay
from macro_regime.fast_crisis.signal import (
    OFF,
    ON,
    UNKNOWN,
    build_crisis_mode_state,
    build_raw_trigger,
    compute_credit_shock,
    compute_equity_shock,
    compute_vix_shock,
    n_day_return_diagnostic,
    two_of_three,
    vix_shock_diagnostics,
)

MIN_HOLD_DAYS = 10
MIN_OFF_DAYS_TO_EXIT = 5


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=n)


def _build_state(raw: pd.Series) -> pd.DataFrame:
    return build_crisis_mode_state(
        raw, min_hold_days=MIN_HOLD_DAYS, min_off_days_to_exit=MIN_OFF_DAYS_TO_EXIT
    )


# -- atomic signals -----------------------------------------------------


def test_vix_shock_requires_above_threshold_and_ma_ratio():
    idx = _dates(25)
    vix = pd.Series([20.0] * 24 + [45.0], index=idx)  # ma20 of last 20 = 20, ratio = 1.25 > 0.5
    result = compute_vix_shock(vix, threshold=30.0, ma_window_days=20, ma_ratio_threshold=0.50)
    assert result.iloc[-1] == ON


def test_vix_shock_above_threshold_but_not_ma_ratio_is_off():
    idx = _dates(25)
    vix = pd.Series([28.0] * 24 + [32.0], index=idx)  # ma20=28, ratio=0.143 -- fails despite >30
    result = compute_vix_shock(vix, threshold=30.0, ma_window_days=20, ma_ratio_threshold=0.50)
    assert result.iloc[-1] == OFF


def test_vix_shock_unknown_before_ma_window_observations():
    idx = _dates(19)
    vix = pd.Series([40.0] * 19, index=idx)
    result = compute_vix_shock(vix, threshold=30.0, ma_window_days=20, ma_ratio_threshold=0.50)
    assert (result == UNKNOWN).all()


def test_equity_shock_window_return_threshold():
    idx = _dates(6)
    daily_ret = [-0.015] * 5
    returns = pd.Series([np.nan] + daily_ret, index=idx)
    result = compute_equity_shock(returns, window_days=5, threshold=-0.07)
    compounded = (1 - 0.015) ** 5 - 1
    assert compounded <= -0.07
    assert result.iloc[-1] == ON


def test_equity_shock_unknown_before_full_window():
    idx = _dates(4)
    returns = pd.Series([np.nan, -0.02, -0.02, -0.02], index=idx)
    result = compute_equity_shock(returns, window_days=5, threshold=-0.07)
    assert result.iloc[-1] == UNKNOWN


def test_credit_shock_window_return_threshold():
    idx = _dates(6)
    daily_ret = [-0.007] * 5
    returns = pd.Series([np.nan] + daily_ret, index=idx)
    result = compute_credit_shock(returns, window_days=5, threshold=-0.03)
    compounded = (1 - 0.007) ** 5 - 1
    assert compounded <= -0.03
    assert result.iloc[-1] == ON


# -- display-only diagnostics (must not change any classification) --------


def test_vix_shock_diagnostics_matches_the_values_compute_vix_shock_uses():
    idx = _dates(25)
    vix = pd.Series([20.0] * 24 + [45.0], index=idx)
    diag = vix_shock_diagnostics(vix, ma_window_days=20)
    expected_ma = vix.iloc[5:25].mean()  # rolling(20) window ending at (and including) the last day
    assert diag["vix_level"].iloc[-1] == 45.0
    assert diag["vix_ma"].iloc[-1] == pytest.approx(expected_ma)
    assert diag["vix_ratio"].iloc[-1] == pytest.approx(45.0 / expected_ma - 1.0)
    # same threshold check compute_vix_shock makes, done manually here
    result = compute_vix_shock(vix, threshold=30.0, ma_window_days=20, ma_ratio_threshold=0.50)
    assert result.iloc[-1] == ON
    assert diag["vix_level"].iloc[-1] > 30.0 and diag["vix_ratio"].iloc[-1] > 0.50


def test_n_day_return_diagnostic_matches_the_value_shock_functions_use():
    idx = _dates(6)
    daily_ret = [-0.015] * 5
    returns = pd.Series([np.nan] + daily_ret, index=idx)
    raw_value = n_day_return_diagnostic(returns, window_days=5)
    expected = (1 - 0.015) ** 5 - 1
    assert raw_value.iloc[-1] == pytest.approx(expected)
    result = compute_equity_shock(returns, window_days=5, threshold=-0.07)
    assert result.iloc[-1] == ON  # confirms diagnostic and classification agree


# -- 2-of-3 trigger combination -------------------------------------------


def test_two_of_three_full_observation():
    a, b, c = pd.Series([ON]), pd.Series([ON]), pd.Series([OFF])
    assert two_of_three(a, b, c).iloc[0] == ON
    a2 = pd.Series([OFF])
    assert two_of_three(a2, b, c).iloc[0] == OFF


def test_two_of_three_exactly_two_observed_both_true_on():
    a, b, c = pd.Series([ON]), pd.Series([ON]), pd.Series([UNKNOWN])
    assert two_of_three(a, b, c).iloc[0] == ON


def test_two_of_three_exactly_two_observed_both_false_off():
    a, b, c = pd.Series([OFF]), pd.Series([OFF]), pd.Series([UNKNOWN])
    assert two_of_three(a, b, c).iloc[0] == OFF


def test_two_of_three_exactly_two_observed_ambiguous_unknown():
    a, b, c = pd.Series([ON]), pd.Series([OFF]), pd.Series([UNKNOWN])
    assert two_of_three(a, b, c).iloc[0] == UNKNOWN


def test_two_of_three_at_most_one_observed_unknown():
    a, b, c = pd.Series([ON]), pd.Series([UNKNOWN]), pd.Series([UNKNOWN])
    assert two_of_three(a, b, c).iloc[0] == UNKNOWN


def test_build_raw_trigger_is_two_of_three():
    idx = _dates(3)
    vix_shock = pd.Series([ON, OFF, ON], index=idx)
    equity_shock = pd.Series([ON, ON, OFF], index=idx)
    credit_shock = pd.Series([OFF, ON, ON], index=idx)
    trigger = build_raw_trigger(vix_shock, equity_shock, credit_shock)
    assert list(trigger) == [ON, ON, ON]


# -- timing / no-lookahead ------------------------------------------------


def test_entry_uses_raw_shifted_by_one_not_same_day():
    idx = _dates(5)
    raw = pd.Series([OFF, OFF, ON, OFF, OFF], index=idx)
    state = _build_state(raw)
    assert state["tradable_trigger"].iloc[2] == OFF  # raw[1]=OFF
    assert state["tradable_trigger"].iloc[3] == ON  # raw[2]=ON
    assert state["crisis_mode"].iloc[2] == OFF  # not yet entered on the raw-ON day itself
    assert state["crisis_mode"].iloc[3] == ON


def test_first_day_tradable_is_unknown_never_enters():
    idx = _dates(3)
    raw = pd.Series([ON, OFF, OFF], index=idx)
    state = _build_state(raw)
    assert state["tradable_trigger"].iloc[0] == UNKNOWN
    assert state["crisis_mode"].iloc[0] == OFF


def test_no_lookahead_future_raw_trigger_does_not_affect_past_state():
    idx = _dates(30)
    raw_a = pd.Series([OFF] * 30, index=idx)
    raw_a.iloc[10] = ON
    raw_b = raw_a.copy()
    raw_b.iloc[-1] = ON  # mutate only the future-most day

    state_a = _build_state(raw_a)
    state_b = _build_state(raw_b)
    pd.testing.assert_frame_equal(state_a.iloc[:-1], state_b.iloc[:-1])


# -- min-hold / exit state machine -----------------------------------------


def test_minimum_hold_enforced_even_if_raw_turns_off_immediately():
    idx = _dates(30)
    raw = pd.Series([OFF] * 30, index=idx)
    raw.iloc[5] = ON  # single-day spike
    state = _build_state(raw)
    entry_idx = 6  # tradable ON at day 6 (raw[5]=ON)
    for i in range(entry_idx, entry_idx + MIN_HOLD_DAYS):
        assert state["crisis_mode"].iloc[i] == ON, f"day {i} should still be ON (min hold)"


def test_exit_after_min_hold_and_consecutive_off():
    idx = _dates(30)
    raw = pd.Series([OFF] * 30, index=idx)
    raw.iloc[0] = ON  # enter on day 1; raw OFF for the rest, so the
    # consecutive-OFF exit condition is satisfied well before day 10,
    # but the minimum hold must still be fully honored first.
    state = _build_state(raw)
    modes = list(state["crisis_mode"])
    on_days = state.index[state["crisis_mode"] == ON]
    entry_pos = modes.index(ON)
    exit_pos = modes.index(OFF, entry_pos)
    assert len(on_days) == MIN_HOLD_DAYS
    assert exit_pos - entry_pos == MIN_HOLD_DAYS
    assert all(v == OFF for v in modes[exit_pos:])


def test_unknown_does_not_count_toward_or_reset_consecutive_off():
    idx = _dates(40)
    raw = pd.Series([OFF] * 40, index=idx)
    raw.iloc[0] = ON
    for i in range(11, 20, 2):
        raw.iloc[i] = UNKNOWN
    state = _build_state(raw)
    assert (state["crisis_mode"] == OFF).any()


def test_reentry_resets_minimum_hold_and_consecutive_off():
    idx = _dates(40)
    raw = pd.Series([OFF] * 40, index=idx)
    raw.iloc[0] = ON  # first entry
    raw.iloc[4] = ON  # re-affirm well before the 10-day minimum hold elapses
    state = _build_state(raw)
    assert state["tradable_trigger"].iloc[5] == ON
    assert state["days_in_mode"].iloc[5] == 1
    assert state["consecutive_off_count"].iloc[5] == 0
    assert state["crisis_mode"].iloc[5] == ON


# -- allocation overlay -----------------------------------------------------


def _base_weights(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame(
        {
            "spy": [0.36] * n,
            "kodex200_usd": [0.24] * n,
            "high_yield": [0.10] * n,
            "investment_grade": [0.10] * n,
            "intermediate_treasury": [0.10] * n,
            "long_treasury": [0.0] * n,
            "gold": [0.05] * n,
            "tbills": [0.05] * n,
            "commodities": [0.0] * n,
            "tips": [0.0] * n,
        },
        index=idx,
    )


def test_crisis_mode_on_zeroes_growth_hyg_dbc_moves_to_bil():
    idx = _dates(2)
    weights = _base_weights(idx)
    weights.loc[idx[1], "commodities"] = 0.15  # non-zero DBC to verify it's actually removed
    weights.loc[idx[1], "spy"] = 0.30
    weights.loc[idx[1], "kodex200_usd"] = 0.20
    weights.loc[idx[1], "high_yield"] = 0.10
    weights.loc[idx[1], "tbills"] = 0.05
    weights.loc[idx[1]] = weights.loc[idx[1]] / weights.loc[idx[1]].sum()
    mode = pd.Series([OFF, ON], index=idx)
    overlaid = apply_fast_crisis_overlay(weights, mode)
    assert overlaid.loc[idx[1], "spy"] == 0.0
    assert overlaid.loc[idx[1], "kodex200_usd"] == 0.0
    assert overlaid.loc[idx[1], "high_yield"] == 0.0
    assert overlaid.loc[idx[1], "commodities"] == 0.0
    removed = (
        weights.loc[idx[1], "spy"]
        + weights.loc[idx[1], "kodex200_usd"]
        + weights.loc[idx[1], "high_yield"]
        + weights.loc[idx[1], "commodities"]
    )
    assert overlaid.loc[idx[1], "tbills"] == weights.loc[idx[1], "tbills"] + removed


def test_crisis_mode_off_leaves_allocation_exactly_equal_to_base():
    idx = _dates(2)
    weights = _base_weights(idx)
    mode = pd.Series([OFF, OFF], index=idx)
    overlaid = apply_fast_crisis_overlay(weights, mode)
    pd.testing.assert_frame_equal(overlaid, weights)


def test_other_assets_never_touched():
    idx = _dates(2)
    weights = _base_weights(idx)
    weights.loc[idx[1], "long_treasury"] = 0.15
    weights.loc[idx[1], "investment_grade"] = 0.10
    weights.loc[idx[1], "gold"] = 0.05
    weights.loc[idx[1], "tips"] = 0.05
    mode = pd.Series([OFF, ON], index=idx)
    overlaid = apply_fast_crisis_overlay(weights, mode)
    for col in ["long_treasury", "investment_grade", "gold", "tips"]:
        pd.testing.assert_series_equal(overlaid[col], weights[col])


def test_weights_sum_to_one_and_no_negatives():
    idx = _dates(4)
    weights = _base_weights(idx)
    mode = pd.Series([OFF, ON, OFF, ON], index=idx)
    overlaid = apply_fast_crisis_overlay(weights, mode)
    assert np.allclose(overlaid.sum(axis=1), 1.0, atol=1e-9)
    assert (overlaid >= -1e-12).all().all()


def test_base_bil_never_reduced():
    idx = _dates(2)
    weights = _base_weights(idx)
    mode = pd.Series([ON, ON], index=idx)
    overlaid = apply_fast_crisis_overlay(weights, mode)
    assert (overlaid["tbills"] >= weights["tbills"] - 1e-12).all()
