import pandas as pd

from macro_regime.signals.regime import Regime, build_regime_series, classify_regime


def test_quadrant_regime_mapping():
    assert classify_regime("Up", "Down") == Regime.GOLDILOCKS
    assert classify_regime("Up", "Up") == Regime.REFLATION
    assert classify_regime("Down", "Up") == Regime.STAGFLATION
    assert classify_regime("Down", "Down") == Regime.CONTRACTION


def test_unknown_regime_for_unrecognized_labels():
    assert classify_regime("Unknown", "Up") == Regime.UNKNOWN
    assert classify_regime("Up", "Neutral") == Regime.UNKNOWN


def test_build_regime_series_aligns_and_classifies():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    growth = pd.Series(["Up", "Down", "Up"], index=idx)
    inflation = pd.Series(["Down", "Up", "Unknown"], index=idx)
    regime = build_regime_series(growth, inflation, name="test_regime")
    assert list(regime) == ["GOLDILOCKS", "STAGFLATION", "UNKNOWN"]
