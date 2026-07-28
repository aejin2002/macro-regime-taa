from __future__ import annotations

import copy

import pytest

from macro_regime.config import load_config
from macro_regime.strategy_versions import (
    STRATEGY_VERSIONS,
    UnknownStrategyVersionError,
    build_versioned_config,
)


@pytest.fixture
def config():
    return load_config()


def test_v1_2_config_is_unmodified_copy(config):
    v1_2 = build_versioned_config(config, "v1_2")
    assert v1_2["backtest"]["regime_allocations"] == config["backtest"]["regime_allocations"]
    assert v1_2 is not config  # deep copy, not the same object


def test_v1_2_mutation_does_not_leak_back_into_source_config(config):
    original = copy.deepcopy(config)
    v1_2 = build_versioned_config(config, "v1_2")
    v1_2["backtest"]["regime_allocations"]["GOLDILOCKS"]["growth_basket"] = 0.99
    assert config["backtest"]["regime_allocations"] == original["backtest"]["regime_allocations"]


def test_v1_3_goldilocks_reflation_overridden(config):
    v1_3 = build_versioned_config(config, "v1_3")
    goldilocks = v1_3["backtest"]["regime_allocations"]["GOLDILOCKS"]
    reflation = v1_3["backtest"]["regime_allocations"]["REFLATION"]
    assert goldilocks == {
        "growth_basket": 0.65,
        "high_yield": 0.10,
        "investment_grade": 0.10,
        "intermediate_treasury": 0.10,
        "gold": 0.05,
        "tbills": 0.00,
    }
    assert reflation == {
        "growth_basket": 0.45,
        "commodities": 0.20,
        "tips": 0.15,
        "gold": 0.10,
        "high_yield": 0.10,
        "tbills": 0.00,
    }


def test_v1_3_contraction_stagflation_unknown_unchanged(config):
    v1_3 = build_versioned_config(config, "v1_3")
    for regime in ["CONTRACTION", "STAGFLATION", "UNKNOWN"]:
        assert (
            v1_3["backtest"]["regime_allocations"][regime] == config["backtest"]["regime_allocations"][regime]
        )


def test_v1_3_weights_sum_to_one_and_nonnegative(config):
    v1_3 = build_versioned_config(config, "v1_3")
    for regime, weights in v1_3["backtest"]["regime_allocations"].items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, regime
        assert all(w >= 0 for w in weights.values()), regime


def test_v1_2_v1_3_are_independent_configs(config):
    v1_2 = build_versioned_config(config, "v1_2")
    v1_3 = build_versioned_config(config, "v1_3")
    v1_3["backtest"]["regime_allocations"]["GOLDILOCKS"]["growth_basket"] = 0.01
    assert v1_2["backtest"]["regime_allocations"]["GOLDILOCKS"]["growth_basket"] == 0.60


def test_unknown_version_raises(config):
    with pytest.raises(UnknownStrategyVersionError):
        build_versioned_config(config, "v1_4")


def test_default_production_version_is_v1_3(config):
    assert config["strategy_versions"]["default_production_version"] == "v1_3"
    assert "v1_3" in STRATEGY_VERSIONS
