import pandas as pd

from macro_regime.signals.regime import Regime, build_regime_output, shift_to_tradable


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def test_tradable_regime_equals_raw_regime_shifted_one_month():
    idx = _idx(6)
    raw = pd.Series(
        ["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION", "GOLDILOCKS", "REFLATION"],
        index=idx,
    )
    tradable = shift_to_tradable(raw, lag_months=1)
    for t in range(1, len(idx)):
        assert tradable.iloc[t] == raw.iloc[t - 1]


def test_first_row_is_unknown():
    idx = _idx(4)
    raw = pd.Series(["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION"], index=idx)
    tradable = shift_to_tradable(raw, lag_months=1)
    assert tradable.iloc[0] == Regime.UNKNOWN.value


def test_future_raw_regime_changes_do_not_affect_past_tradable_regime():
    idx = _idx(5)
    raw_a = pd.Series(["GOLDILOCKS", "REFLATION", "STAGFLATION", "CONTRACTION", "GOLDILOCKS"], index=idx)
    raw_b = raw_a.copy()
    raw_b.iloc[-1] = "REFLATION"  # change only the last (future-most) value

    tradable_a = shift_to_tradable(raw_a, lag_months=1)
    tradable_b = shift_to_tradable(raw_b, lag_months=1)

    # Every row except the one that reads the changed future value must be identical.
    pd.testing.assert_series_equal(tradable_a.iloc[:-1], tradable_b.iloc[:-1])


def test_shift_to_tradable_does_not_mutate_input():
    idx = _idx(3)
    raw = pd.Series(["GOLDILOCKS", "REFLATION", "STAGFLATION"], index=idx)
    raw_copy = raw.copy()
    shift_to_tradable(raw, lag_months=1)
    pd.testing.assert_series_equal(raw, raw_copy)


def test_unknown_propagates_forward_one_month():
    idx = _idx(4)
    raw = pd.Series(["GOLDILOCKS", "UNKNOWN", "STAGFLATION", "CONTRACTION"], index=idx)
    tradable = shift_to_tradable(raw, lag_months=1)
    # tradable at t=2 reads raw at t=1, which is UNKNOWN.
    assert tradable.iloc[2] == Regime.UNKNOWN.value


def test_build_regime_output_schema():
    idx = _idx(6)
    growth_score = pd.Series(range(6), index=idx, dtype=float)
    growth_state = pd.Series(["Up", "Up", "Down", "Down", "Up", "Unknown"], index=idx)
    inflation_score = pd.Series(range(6), index=idx, dtype=float) * -1
    inflation_state = pd.Series(["Down", "Up", "Up", "Down", "Down", "Up"], index=idx)

    out = build_regime_output(growth_score, growth_state, inflation_score, inflation_state)

    assert list(out.columns) == [
        "growth_score",
        "growth_state",
        "inflation_score",
        "inflation_state",
        "raw_regime",
        "tradable_regime",
    ]
    # raw_regime[0] = Up/Down -> GOLDILOCKS; tradable_regime[0] must be UNKNOWN
    # (no prior raw_regime to reference) regardless of raw_regime[0]'s value.
    assert out["tradable_regime"].iloc[0] == Regime.UNKNOWN.value
    assert out["tradable_regime"].iloc[1] == out["raw_regime"].iloc[0]


def test_primary_and_secondary_share_identical_schema():
    idx = _idx(5)
    inflation_score = pd.Series(range(5), index=idx, dtype=float)
    inflation_state = pd.Series(["Up", "Down", "Up", "Down", "Up"], index=idx)

    primary = build_regime_output(
        pd.Series(range(5), index=idx, dtype=float),
        pd.Series(["Up", "Down", "Up", "Down", "Up"], index=idx),
        inflation_score,
        inflation_state,
    )
    secondary = build_regime_output(
        pd.Series(range(5, 10), index=idx, dtype=float),
        pd.Series(["Down", "Down", "Up", "Up", "Down"], index=idx),
        inflation_score,
        inflation_state,
    )

    assert list(primary.columns) == list(secondary.columns)
    assert primary.index.equals(secondary.index)
    for col in primary.columns:
        assert primary[col].dtype == secondary[col].dtype
