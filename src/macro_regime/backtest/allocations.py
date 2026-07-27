"""Regime -> asset-category -> underlying-ticker weight expansion.

Every weight here comes straight from `config/default.yaml`'s
`backtest.regime_allocations` / `growth_basket` -- nothing in this module
is computed, tuned, or optimized.
"""

from __future__ import annotations


def _growth_basket_weights(config: dict) -> dict[str, float]:
    gb_conf = config["growth_basket"]
    return {
        "spy": gb_conf["spy_weight"],
        "kodex200_usd": gb_conf["kodex200_weight"],
    }


def expand_category_weights(category_weights: dict[str, float], config: dict) -> dict[str, float]:
    """Expand a {category: weight} dict (as written in config, e.g.
    {"growth_basket": 0.6, "high_yield": 0.1, ...}) into
    {underlying_ticker_column: weight}. `growth_basket` splits into its
    two underlying tickers at its configured sub-weights; every other
    category name already matches its return-matrix column 1:1."""
    gb_weights = _growth_basket_weights(config)
    expanded: dict[str, float] = {}
    for category, weight in category_weights.items():
        if category == "growth_basket":
            for ticker, sub_weight in gb_weights.items():
                expanded[ticker] = expanded.get(ticker, 0.0) + weight * sub_weight
        else:
            expanded[category] = expanded.get(category, 0.0) + weight
    return expanded


def regime_allocations(config: dict) -> dict[str, dict[str, float]]:
    """{regime_name: {ticker_column: weight}} for all 5 fixed regime
    tables (GOLDILOCKS/REFLATION/STAGFLATION/CONTRACTION/UNKNOWN)."""
    raw = config["backtest"]["regime_allocations"]
    return {regime: expand_category_weights(weights, config) for regime, weights in raw.items()}


def static_6040_allocation(config: dict) -> dict[str, float]:
    return expand_category_weights(config["backtest"]["static_6040"], config)


def equal_weight_categories(config: dict) -> list[str]:
    """The 9 distinct asset categories named anywhere in the regime
    tables: `growth_basket` plus every other `backtest.assets` entry
    except `spy` (SPY is only reachable as part of growth_basket, not a
    standalone category in any regime table)."""
    assets_conf = config["backtest"]["assets"]
    return ["growth_basket"] + [k for k in assets_conf if k != "spy"]


def equal_weight_allocation(config: dict) -> dict[str, float]:
    categories = equal_weight_categories(config)
    weight = 1.0 / len(categories)
    return expand_category_weights(dict.fromkeys(categories, weight), config)


def growth_basket_allocation(config: dict) -> dict[str, float]:
    """100% Growth Basket, expanded to its underlying SPY/KODEX200
    weights -- used for the standalone buy-and-hold comparison strategy."""
    return _growth_basket_weights(config)
