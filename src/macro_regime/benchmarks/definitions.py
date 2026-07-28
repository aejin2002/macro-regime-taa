"""Benchmark Registry -- a reusable definition + lookup layer so adding a
new benchmark never requires touching multiple Streamlit files directly.

Each `BenchmarkDefinition` is pure metadata; `benchmarks.engine` is what
actually computes a series/metrics from one. Kept intentionally simple
(one dataclass, one dict-backed registry) -- no factory pattern, no
plugin discovery, since there are only a handful of benchmarks and the
project's own convention favors explicit, auditable config over
abstraction for its own sake.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkDefinition:
    id: str
    display_name: str
    category: str  # "blended" | "single_asset" | "mutual_fund"
    data_source: str
    components: dict[
        str, float
    ]  # {ticker_or_category: weight}; single-asset benchmarks have one entry at 1.0
    calculation_method: str
    rebalance_rule: str  # "monthly" | "none"
    transaction_cost_bps: float | None
    calendar_rule: str  # "spy_nyse_calendar" | "native_nav_calendar"
    total_return_treatment: str
    known_inception_date: str | None
    ui_visible: bool
    default_selected: bool
    supports_relative_performance: bool
    notes: str = field(default="")


REGISTRY: dict[str, BenchmarkDefinition] = {
    "us_60_40": BenchmarkDefinition(
        id="us_60_40",
        display_name="US 60/40",
        category="blended",
        data_source="yfinance",
        components={"spy": 0.6, "agg": 0.4},
        calculation_method="60% SPY / 40% AGG, monthly rebalance, "
        "weights drift with daily returns between rebalances",
        rebalance_rule="monthly",
        transaction_cost_bps=10.0,
        calendar_rule="spy_nyse_calendar",
        total_return_treatment="auto_adjust=True (dividend/split adjusted close)",
        known_inception_date="2003-09-29",  # AGG inception; SPY is much earlier
        ui_visible=True,
        default_selected=True,
        supports_relative_performance=True,
        notes="Default production benchmark. Standard, externally-recognizable 60/40.",
    ),
    "malox": BenchmarkDefinition(
        id="malox",
        display_name="MALOX",
        category="mutual_fund",
        data_source="yfinance",
        components={"malox": 1.0},
        calculation_method="BlackRock Global Allocation Fund, Institutional Shares -- reported fund "
        "NAV/adjusted total return used as-is, no synthetic rebalancing cost added",
        rebalance_rule="none",
        transaction_cost_bps=None,
        calendar_rule="native_nav_calendar",
        total_return_treatment="auto_adjust=True (distribution-adjusted, confirmed against actual "
        "distribution events -- see docs/methodology.md for the verification)",
        known_inception_date="2000-01-03",
        ui_visible=True,
        default_selected=False,
        supports_relative_performance=True,
        notes="Mutual fund: one NAV per day, no intraday price, may lag the strategy's market date by "
        "up to a day. Never backward-filled; a day with no new MALOX NAV is flagged stale, not "
        "treated as a 0% or estimated return.",
    ),
    "spy": BenchmarkDefinition(
        id="spy",
        display_name="SPY",
        category="single_asset",
        data_source="yfinance",
        components={"spy": 1.0},
        calculation_method="Buy-and-hold SPY, no rebalancing",
        rebalance_rule="none",
        transaction_cost_bps=None,
        calendar_rule="spy_nyse_calendar",
        total_return_treatment="auto_adjust=True",
        known_inception_date="1993-01-29",
        ui_visible=True,
        default_selected=False,
        supports_relative_performance=True,
    ),
    "agg": BenchmarkDefinition(
        id="agg",
        display_name="AGG",
        category="single_asset",
        data_source="yfinance",
        components={"agg": 1.0},
        calculation_method="Buy-and-hold AGG, no rebalancing",
        rebalance_rule="none",
        transaction_cost_bps=None,
        calendar_rule="spy_nyse_calendar",
        total_return_treatment="auto_adjust=True",
        known_inception_date="2003-09-29",
        ui_visible=True,
        default_selected=False,
        supports_relative_performance=True,
    ),
    "project_6040": BenchmarkDefinition(
        id="project_6040",
        display_name="Project 60/40 (internal/research only)",
        category="blended",
        data_source="yfinance",
        components={"growth_basket": 0.6, "intermediate_treasury": 0.4},
        calculation_method="Growth Basket(SPY60%+KODEX200 40%) 60% + IEF 40%, monthly rebalance",
        rebalance_rule="monthly",
        transaction_cost_bps=10.0,
        calendar_rule="spy_nyse_calendar",
        total_return_treatment="auto_adjust=True",
        known_inception_date="2007-06-01",
        ui_visible=False,
        default_selected=False,
        supports_relative_performance=True,
        notes="v1.0-v1.2's original internal benchmark. Retained for historical-research reproducibility "
        "(regression fixtures, .analysis/ reports) -- deliberately excluded from every user-facing "
        "comparison surface (see docs/methodology.md, 'Benchmark history').",
    ),
}

MAX_CONCURRENT_BENCHMARKS = 3  # + the strategy line itself, keeps charts readable


def list_ui_visible() -> list[BenchmarkDefinition]:
    return [b for b in REGISTRY.values() if b.ui_visible]


def get(benchmark_id: str) -> BenchmarkDefinition:
    if benchmark_id not in REGISTRY:
        raise KeyError(f"Unknown benchmark id {benchmark_id!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[benchmark_id]
