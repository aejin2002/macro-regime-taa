"""Product release versions ("Macro Regime TAA v1.2" / "v1.3").

This is a DIFFERENT axis from `fast_crisis.backtest`'s `V1_0`/`V1_1`/
`V1_2_LABEL` internal overlay-stack labels (Macro-only -> +BEI Duration
Gate -> +Fast Crisis Overlay) -- those describe which OVERLAYS are
applied and exist for every release. This module only ever changes which
REGIME ALLOCATION TABLE a release starts from; `run_fast_crisis_backtest`
itself is never modified or branched on version, so a v1.3 config change
can never propagate into a v1.2 result -- they are simply two different,
independently-constructed config dicts passed into the same unmodified
pipeline.

v1.2 is `config/default.yaml`'s `backtest.regime_allocations` exactly as
written, frozen. v1.3 ("Growth Participation") overrides only
GOLDILOCKS/REFLATION per `strategy_versions.v1_3.regime_allocation_overrides`
in that same file -- STAGFLATION/CONTRACTION/UNKNOWN, BEI Duration Gate,
Fast Crisis, transaction costs, and execution lag are identical.
"""

from __future__ import annotations

import copy

STRATEGY_VERSIONS = ("v1_2", "v1_3")
DISPLAY_NAMES = {"v1_2": "Macro Regime TAA v1.2", "v1_3": "Macro Regime TAA v1.3 — Growth Participation"}


class UnknownStrategyVersionError(ValueError):
    pass


def build_versioned_config(config: dict, version: str) -> dict:
    """Returns a deep-copied config with the given release's regime
    allocation table applied. `config` itself is never mutated, so
    callers building multiple versions from the same loaded config never
    see cross-contamination."""
    if version not in STRATEGY_VERSIONS:
        raise UnknownStrategyVersionError(
            f"Unknown strategy version {version!r}; must be one of {STRATEGY_VERSIONS}"
        )
    out = copy.deepcopy(config)
    if version == "v1_2":
        return out

    overrides = config["strategy_versions"]["v1_3"]["regime_allocation_overrides"]
    for regime, weights in overrides.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"v1.3 {regime} allocation does not sum to 1.0: {total}")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"v1.3 {regime} allocation has a negative weight: {weights}")
        out["backtest"]["regime_allocations"][regime] = dict(weights)
    return out
