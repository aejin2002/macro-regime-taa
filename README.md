# macro-regime-taa

Macro regime research engine using growth and inflation signals for
tactical asset allocation (TAA).

This project is **not** a full asset-allocation or backtesting system. It
is Layer 1 of a TAA process: classifying and evaluating the US
Growth x Inflation macro regime. See `docs/methodology.md` for the full
research write-up and `docs/data_dictionary.md` for every data source and
schema used.

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
(Model A, Model B below) use only FRED-API-auto-fetched data. No
external XLSX, manually-supplied CSV, or scraped data in a core model.
Older models that require an optional external CSV remain in the
codebase for baseline comparison, clearly labeled legacy/auxiliary --
they are never part of the core Model A/B pairing.

## Growth Asset Basket

When the growth axis is "Up", it is expressed as a single fixed-weight
basket rather than a country-specific asset pick:

- S&P 500
- KOSPI 200

There is no separate KOSPI-oriented regime -- KOSPI 200 is a constituent
of this one basket, allocated against the same US Growth x US Inflation
regime as the S&P 500. Weights are configured under `growth_basket` in
`config/default.yaml`:

```yaml
growth_basket:
  sp500_weight: 0.5
  kospi200_weight: 0.5
```

The default is an untuned 50:50 split. This build does not compute
basket returns, backtest the weights, or optimize them -- that is future
work layered on top of this regime engine (see Future plans below).

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
| A -- OECD CLI | `USALOLITOAASTSAM` (US) | Sign of `CLI_t - CLI_t-3`, with an optional 3m/6m-agreement stabilization rule | Core (Model A) |
| B -- FRED Minimal | Initial Claims, Building Permits, CFNAI-MA3 | `mean(-z(Δ3m claims), z(Δ6m permits), z(CFNAI-MA3))` | Baseline/proxy |
| C -- Simple Two-Signal | ISM New Orders (CSV), Initial Claims | `(z(Δ3m ISM) - z(Δ3m claims)) / 2`; disabled without the ISM CSV | Legacy (external CSV) |
| D -- AMTMNO + Claims | `AMTMNO`, Initial Claims | `mean(z(Δ3m AMTMNO), -z(Δ3m claims))`; fully FRED-native | Core (Model B) |

## Inflation models

| Model | Inputs | Rule | Status |
|---|---|---|---|
| A -- Realized Core Inflation Momentum | Core CPI | `core_3m_annualized - core_12m` | Core (Model B inflation axis) |
| B -- Leading Inflation Composite | ISM Prices Paid (CSV, optional), 5Y breakeven, Core CPI momentum | `mean(z(Δ3m ISM prices paid), z(Δ3m breakeven), z(core momentum))`; falls back to a 2-signal variant without ISM | Baseline/proxy |
| C -- Cleveland Median CPI Momentum | `MEDCPIM158SFRBCLE` | 3m MA of Median CPI vs. itself 3 months prior; **not** the Cleveland Fed Inflation Nowcast (see Methodology) | Core (Model A inflation axis) |
| D -- Commodity + Core | `PALLFNFINDEXM`, Core CPI momentum | `mean(z(Δ3m commodity index), z(core momentum))`; **2-signal only, no ISM input** | Auxiliary/secondary |

Full formulas, windows, and rationale: `docs/methodology.md`.

**Core research models** (fully FRED-native, always computable from a
live fetch alone): **Model A** = Growth CLI x Inflation Cleveland Median
CPI (`regime_us_cli_cleveland_median_cpi`); **Model B** = Growth
AMTMNO+Claims x Inflation Core Momentum
(`regime_amtmno_claims_core_momentum`). FRED Minimal, ISM Simple
Two-Signal, Leading Inflation Composite, and Commodity+Core remain
active as legacy/auxiliary baselines only.

## Regime mapping

See the table under Purpose above. Every `growth_model x inflation_model`
pair produces its own `regime_<growth>_<inflation>` column; the mapping
from column name to model pair is written to
`data/processed/regime_metadata.json` by `build-signals`.

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
python -m macro_regime.cli run-all
```

Options: `--start-date`, `--end-date`, `--refresh-cache`,
`--growth-model`, `--inflation-model`.

## Streamlit app

```bash
make app
# equivalent to: streamlit run app/streamlit_app.py
```

Pages: Overview, Data Explorer, Growth Models, Inflation Models, Regime
Comparison, Evaluation, Methodology. Run `make run-all` (or the CLI
commands above) at least once before launching the app so
`data/processed/*` exists.

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
- No asset returns, position sizing, or trading backtest in this version
  -- by design; see Future plans.

## Future plans

Layered on top of this regime engine, not yet implemented:

- **Rate Overlay** -- yield-curve / policy-rate risk adjustment
- **Credit Overlay** -- credit-spread confirmation of the growth call
- **Price Confirmation** -- price/momentum filter before acting on a regime
- **Crisis Gate** -- tail-risk override to suspend normal regime logic
  during acute stress
- **Growth Asset Basket weight optimization** -- the S&P 500 / KOSPI 200
  split in `growth_basket` (`config/default.yaml`) is a fixed, untuned
  50:50 default; backtesting and optimizing that split is future work
