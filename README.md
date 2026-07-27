# macro-regime-taa

Macro regime research engine using growth and inflation signals for
tactical asset allocation (TAA).

This project's core deliverable is Layer 1 of a TAA process: classifying
and evaluating the US Growth x Inflation macro regime. As of v1.0 it also
includes a fixed-regime **asset allocation backtest** (see "Backtest"
below) -- a fixed, ex-ante allocation applied to `tradable_regime`, not
an optimized or actively-managed trading system. See
`docs/methodology.md` for the full research write-up and
`docs/data_dictionary.md` for every data source and schema used.

## Purpose

Two axes, classified monthly, **using US growth and US inflation only**
(no separate per-country regime):

- **Growth:** Up / Down
- **Inflation:** Up / Down

| Growth | Inflation | Regime |
|---|---|---|
| Up | Down | GOLDILOCKS |
| Up | Up | REFLATION |
| Down | Up | STAGFLATION |
| Down | Down | CONTRACTION |

Each growth model is evaluated against each inflation model
independently -- there is no single blended "house regime" in this
version. The priority is computing the signals correctly and measuring
their forecast skill, not trading them.

**Core model data source policy:** the two core operational models
(Primary, Secondary below) use only FRED-API-auto-fetched data. No
external XLSX, manually-supplied CSV, or scraped data in a core model.
Older models that require an optional external CSV remain in the
codebase for baseline comparison, clearly labeled legacy/auxiliary --
they are never part of Primary/Secondary.

**As of commit `40e43d7`, the macro signal models are frozen:**

| Core model | Growth | Inflation |
|---|---|---|
| **Primary** | US OECD CLI | Core Inflation Momentum |
| **Secondary** (robustness check) | AMTMNO + Initial Claims | Core Inflation Momentum |

Cleveland Median CPI Momentum and Commodity + Core are auxiliary/research
models only -- a challenger experiment (see `docs/methodology.md`) found
Core Inflation Momentum the stronger, more consistent inflation axis for
both. No further indicator search or formula tuning happens on Primary/
Secondary without an explicit decision to unfreeze them.

## Growth Asset Basket

When the growth axis is "Up", it is expressed as a single fixed-weight
basket rather than a country-specific asset pick: **SPY 60% + KODEX 200
ETF (`069500.KS`) 40%, USD-converted returns.** There is no separate
KOSPI-oriented regime -- KOSPI 200 exposure (via the KODEX 200 ETF, a
real KOSPI-200-tracking fund) is a constituent of this one basket,
allocated against the same US Growth x US Inflation regime as SPY.
Configured under `growth_basket` in `config/default.yaml`:

```yaml
growth_basket:
  spy_weight: 0.6
  kodex200_weight: 0.4
  kodex200_ticker: "069500.KS"
  fx_ticker: "KRW=X"
```

KODEX 200 is KRW-denominated; its USD-equivalent return is computed
using the **actual month-end** USD/KRW rate (`KRW=X`) only -- never
predicted, hedged, or filled forward. This fixed 60/40 split is used
exactly as configured, not optimized -- see `docs/methodology.md`,
"Backtest", for the full rebalancing rules and FX methodology.

## Installation

Requires Python >= 3.11.

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
make setup
# equivalent to: uv venv && uv pip install -e ".[dev]"
```

Without uv (standard venv + pip):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## FRED API key setup

1. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and set `FRED_API_KEY=<your key>`. `.env` is git-ignored
   and the key is never hardcoded anywhere in the codebase. If the key is
   missing, every command that needs it fails immediately with:

   > FRED_API_KEY is missing. Create a .env file from .env.example and add
   > your FRED API key.

## Data sources

- **FRED** (auto-fetched): Industrial Production, CFNAI, Initial Claims,
  Building Permits, OECD CLI (US), AMTMNO, Core CPI, Core PCE, headline
  CPI, Cleveland Fed Median CPI, IMF commodity price index, yield-curve
  spreads, 5Y breakeven inflation. Full list and roles in
  `docs/data_dictionary.md`. Series IDs are validated against the live
  FRED API on every `fetch` -- an invalid or retired ID fails loudly
  rather than being silently swapped for something else. **Both core
  models (Model A, Model B) are built entirely from this list** -- no
  external file is required to compute or evaluate them.
- **External CSVs** (optional, supplied locally, never committed; feed
  only legacy/auxiliary models, never the core Model A/B pairing):
  - `data/external/conference_board_lei.csv` -- Conference Board LEI.
    **Not the same as FRED's `USSLIND`** (see below).
  - `data/external/ism_new_orders.csv`, `data/external/ism_prices_paid.csv`
    -- ISM sub-indices, not reliably available directly from FRED.

  Each file's absence disables only the model(s) that depend on it; the
  rest of the pipeline keeps running with a clear warning. Exact schemas
  are in `docs/data_dictionary.md`.

### Conference Board LEI vs. FRED USSLIND

FRED's `USSLIND` is the **Philadelphia Fed's State Leading Indexes**
series, not the Conference Board's Leading Economic Index, and it was
discontinued after 2020. This project never uses `USSLIND` as a stand-in
for the Conference Board LEI -- the LEI must be supplied via
`data/external/conference_board_lei.csv` or that model is skipped.

## Growth models

| Model | Inputs | Rule | Status |
|---|---|---|---|
| A -- OECD CLI | `USALOLITOAASTSAM` (US) | Sign of `CLI_t - CLI_t-3`, with an optional 3m/6m-agreement stabilization rule | **Primary** growth axis |
| B -- FRED Minimal | Initial Claims, Building Permits, CFNAI-MA3 | `mean(-z(Δ3m claims), z(Δ6m permits), z(CFNAI-MA3))` | Baseline/proxy |
| C -- Simple Two-Signal | ISM New Orders (CSV), Initial Claims | `(z(Δ3m ISM) - z(Δ3m claims)) / 2`; disabled without the ISM CSV | Legacy (external CSV) |
| D -- AMTMNO + Claims | `AMTMNO`, Initial Claims | `mean(z(Δ3m AMTMNO), -z(Δ3m claims))`; fully FRED-native | **Secondary** growth axis |

## Inflation models

| Model | Inputs | Rule | Status |
|---|---|---|---|
| A -- Realized Core Inflation Momentum | Core CPI | `core_3m_annualized - core_12m` | **Primary and Secondary** inflation axis (current-environment classifier -- see Methodology) |
| B -- Leading Inflation Composite | ISM Prices Paid (CSV, optional), 5Y breakeven, Core CPI momentum | `mean(z(Δ3m ISM prices paid), z(Δ3m breakeven), z(core momentum))`; falls back to a 2-signal variant without ISM | Baseline/proxy |
| C -- Cleveland Median CPI Momentum | `MEDCPIM158SFRBCLE` | 3m MA of Median CPI vs. itself 3 months prior; **not** the Cleveland Fed Inflation Nowcast (see Methodology) | Auxiliary/research (rejected as core inflation axis) |
| D -- Commodity + Core | `PALLFNFINDEXM`, Core CPI momentum | `mean(z(Δ3m commodity index), z(core momentum))`; **2-signal only, no ISM input** | Auxiliary/secondary |

Full formulas, windows, and rationale: `docs/methodology.md`.

FRED Minimal, ISM Simple Two-Signal, Leading Inflation Composite,
Cleveland Median CPI, and Commodity+Core remain active as legacy/
auxiliary baselines only -- never part of Primary/Secondary.

## Regime mapping

See the table under Purpose above. Every `growth_model x inflation_model`
pair produces its own `regime_<growth>_<inflation>` column; the mapping
from column name to model pair is written to
`data/processed/regime_metadata.json` by `build-signals`.

## Standard regime output (asset allocation)

`build-regime-output` produces the asset-allocation-ready output for
Primary and Secondary: `data/processed/regime_output_primary.csv` /
`regime_output_secondary.csv`, each with columns `growth_score`,
`growth_state`, `inflation_score`, `inflation_state`, `raw_regime`,
`tradable_regime`. `tradable_regime[t] = raw_regime[t-1]`
(`regime_output.tradable_lag_months` in `config/default.yaml`) -- a
1-month portfolio-application lag, not the same mechanism as FRED
publication lag. See `docs/methodology.md`, "Standard regime output",
for the full semantics and caveats (revised-data backtest, Core
Inflation Momentum as a current-environment classifier).

## Backtest (v1.0, fixed-regime asset allocation)

`run-backtest` applies fixed, ex-ante regime allocations to Primary's and
Secondary's `tradable_regime` (only `tradable_regime`, never
`raw_regime`), and compares against Static 60/40, Static equal-weight,
and Growth Basket buy-and-hold. **Weights are used exactly as specified
-- nothing is optimized, fit to data, or threshold-tuned. No
Rate/Credit/Crisis overlay in this version.**

Asset prices come from Yahoo Finance (`yfinance`) -- the only asset-price
source in this repo; there is no existing internal price infrastructure
to reuse. **The common backtest sample runs 2009-05-31 to 2026-06-30**,
not the BIL-inception-driven ~2007-06 start originally expected:
`069500.KS` (KODEX 200) has a real Yahoo Finance data gap from 2007-03
through 2009-03 (data resumes 2009-04), and the no-forward-fill rule
means the common sample can't begin until that gap clears. **2008 (GFC)
is therefore not covered by this backtest's sample** -- only 2020 and
2022 of the three requested crisis years are. The end date is the last
*fully elapsed* calendar month as of when the backtest is run (a
still-in-progress current month is always excluded, never included as a
partial-month return). See `docs/methodology.md`, "Backtest", for the
full findings. Monthly rebalance to target every month (restoring full
target weights even when unchanged), one-way turnover x
`backtest.transaction_cost_bps` (10bp) transaction cost, every metric
reported pre-cost and post-cost.

**The per-regime allocation percentages (60/10/10/10/5/5 for
GOLDILOCKS, etc.) are ex-ante heuristic weights based on each asset
class's expected economic *direction* under that regime -- they are not
copied from any academic paper or third-party model, and they are not
fit, backtested, or optimized against this data.**

### Results (v1.0, post-cost, 2009-05-31 to 2026-06-30)

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | Turnover/yr | Final Value |
|---|---|---|---|---|---|---|---|---|
| Primary | 9.92% | 9.92% | 0.879 | 1.334 | -15.71% | **0.632** | 2.03 | 5.07x |
| Secondary | 9.18% | 10.39% | 0.779 | 1.171 | -19.75% | 0.465 | 3.18 | 4.51x |
| Static 60/40 | **10.34%** | 11.06% | 0.835 | 1.233 | -24.74% | 0.418 | 0.17 | 5.42x |
| Equal-weight | 5.63% | 6.26% | 0.707 | 1.098 | -13.95% | 0.404 | 0.15 | 2.56x |
| Growth Basket B&H | **14.64%** | 16.48% | 0.839 | 1.256 | -27.48% | 0.533 | 0.03 | 10.44x |

**Read this as a risk-management result, not a return-maximization
one.** Primary's CAGR is *not* higher than Static 60/40's or Growth
Basket buy-and-hold's -- it is lower than both. What Primary does show
is a materially smaller max drawdown (-15.7% vs. -24.7% for Static 60/40
and -27.5% for buy-and-hold), the best Calmar ratio of all five
strategies, and a modestly higher Sharpe/Sortino than the passive
alternatives -- i.e., comparable-to-better risk-adjusted return at
substantially lower realized drawdown, not better absolute performance.
Secondary does not show the same risk-adjusted edge as Primary in this
sample. See `docs/methodology.md`, "Backtest", for the full metric
definitions and caveats.

```bash
python -m macro_regime.cli run-backtest
```

Writes `data/processed/backtest_summary.csv`,
`backtest_annual_returns.csv`, `backtest_monthly_returns.csv`,
`backtest_allocations_primary.csv`, `backtest_allocations_secondary.csv`,
`backtest_regime_analysis.csv`. Not chained into `run-all` (separate,
slower, independently-cached network source) -- run it after
`build-regime-output`.

## Running the pipeline

```bash
make fetch          # validate + download FRED series -> data/processed/fred_wide.csv
make build-signals  # compute growth/inflation labels + all regime combinations
make evaluate        # score every model vs. naive Up/Down baselines
```

Or via the CLI directly:

```bash
python -m macro_regime.cli fetch --start-date 1990-01-01
python -m macro_regime.cli build-signals --growth-model all --inflation-model all
python -m macro_regime.cli evaluate
python -m macro_regime.cli build-regime-output
python -m macro_regime.cli run-backtest
python -m macro_regime.cli run-all   # fetch -> build-signals -> evaluate -> build-regime-output
```

Options: `--start-date`, `--end-date`, `--refresh-cache`,
`--growth-model`, `--inflation-model`.

## Streamlit app

```bash
make app
# equivalent to: streamlit run app/streamlit_app.py
```

Pages: Overview, Data Explorer, Growth Models, Inflation Models, Regime
Comparison, Regime Output, Backtest, Evaluation, Methodology. Run `make
run-all` and `run-backtest` (or the CLI commands above) at least once
before launching the app so `data/processed/*` exists.

## Tests and linting

```bash
make test   # pytest
make lint   # ruff check + mypy
```

## Look-ahead bias prevention

Every signal carries `observation_date`, `release_date`,
`availability_date`, `signal_date`, and `effective_date` columns
(`src/macro_regime/data/availability.py`). Series without a modeled
release calendar fall back to a conservative lag
(`lookahead.monthly_macro_lag_months` / `daily_market_lag_days` in
`config/default.yaml`, defaulting to 1 month / 1 day). Signals are
generated as of month-end and only become effective from the first
calendar day of the following month. All z-scores use a strictly backward
rolling window (`utils/stats.py::rolling_zscore`), which cannot see future
data by construction -- verified in `tests/test_lookahead.py`.

## Revised data vs. real-time vintage

Every evaluation in this repository is a **revised-data backtest**: it
uses today's fully-revised FRED values, not what was actually published
and known at each historical date. `FredClient.get_series()` accepts
`realtime_start`/`realtime_end` for ad-hoc ALFRED vintage queries, but a
full real-time vintage panel is **not implemented** in this build (see the
documented stub and TODO in `src/macro_regime/data/fred_series.py::get_vintage_panel`).
Do not interpret any result here as real-time/live performance until that
gap is closed.

## Current limitations

- No full ALFRED real-time vintage panel yet.
- `effective_date` is a calendar-day approximation, not a trading-day
  calendar.
- Average lead-time diagnostic is a simple cross-correlation heuristic,
  not a formal causality test.
- The backtest's common sample runs 2009-05-31 to 2026-06-30 (driven by
  a real Yahoo Finance data gap in `069500.KS`, not the full 1990+
  macro-signal history) -- **2008 (GFC) is not covered**; no ETF
  survivorship-bias modeling.
- No Rate/Credit/Crisis overlay in the backtest yet -- see Future plans.
- Past backtest performance does not represent live/real-time
  performance.

## Future plans

Layered on top of this regime engine and backtest, not yet implemented:

- **Rate Overlay** -- yield-curve / policy-rate risk adjustment
- **Credit Overlay** -- credit-spread confirmation of the growth call
- **Price Confirmation** -- price/momentum filter before acting on a regime
- **Crisis Gate** -- tail-risk override to suspend normal regime logic
  during acute stress
- **Allocation weight optimization** -- the fixed regime allocation
  tables and the 60/40 SPY/KODEX200 Growth Basket split
  (`config/default.yaml`) are used exactly as specified, untuned;
  backtesting and optimizing those weights is future work
