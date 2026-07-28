# Methodology

## Scope

This project studies the **Growth x Inflation regime layer** ("Layer 1")
of a tactical asset allocation (TAA) process. It does not implement asset
selection, position sizing, or ETF backtesting. The deliverable is a
research engine for classifying and evaluating macro regimes, not a
trading system.

## Core model data source policy

**The core operational models (Model A, Model B -- see "Research models"
below) use only data auto-fetched from the FRED API.** No external XLSX,
manually-supplied CSV, or web-scraped data is implemented in a core
model. A handful of older models predate this policy and remain in the
codebase as explicitly-labeled **legacy/auxiliary** signals for baseline
comparison -- they still require an optional external CSV and stay
inactive without one; they are never promoted into the core Model A/B
pairing. See "Research models" for the exact classification.

## Regime definition

Two binary axes, classified monthly from **US growth and US inflation
only** -- there is no separate per-country regime:

- **Growth**: Up / Down
- **Inflation**: Up / Down

| Growth | Inflation | Regime |
|---|---|---|
| Up | Down | GOLDILOCKS |
| Up | Up | REFLATION |
| Down | Up | STAGFLATION |
| Down | Down | CONTRACTION |
| (missing/disagreement) | | UNKNOWN |

Every growth model is paired with every inflation model independently;
regime columns are named `regime_<growth_suffix>_<inflation_suffix>` and
the mapping from column name to model pair is recorded in
`data/processed/regime_metadata.json`.

## Growth Asset Basket

The growth axis is spent on a single fixed-weight basket -- S&P 500 +
KOSPI 200 -- not a country-specific pick, and KOSPI 200 does not get its
own regime. Both assets sit under the one US Growth x US Inflation
classification above. Weights live in `growth_basket` in
`config/default.yaml` (`sp500_weight` / `kospi200_weight`, default
50:50, untuned in this build); see "Planned future layers" below for
where weight optimization and basket-return backtesting fit in.

## Growth models

**Model A -- OECD Composite Leading Indicator (CLI).**
Uses `USALOLITOAASTSAM` (US), a monthly, amplitude-adjusted OECD CLI
series pulled directly from FRED. There is no Korea CLI input and no
separate KR growth regime -- KOSPI 200 is evaluated against this same US
CLI signal as part of the Growth Asset Basket (see above), not against a
Korea-specific growth call.

- Primary rule: `cli_change_3m = CLI_t - CLI_t-3`. Up if positive, Down if
  negative, Unknown if insufficient data or exactly zero.
- Diagnostics also computed: level vs. 100, 1/3/6-month changes, and
  whether the 3-month and 6-month changes agree in sign.
- Optional stabilization rule: Up when both the 3-month and 6-month
  changes are positive, Down when both are negative; when they disagree,
  the previous non-Unknown state is carried forward (reduces whipsaw at
  the cost of a lag).

**Model B -- FRED Minimal.** Uses only series with no external-CSV
dependency:

```
claims_component  = -z(change_3m(monthly-average(ICSA)))
permits_component =  z(change_6m(PERMIT))
activity_component = z(CFNAIMA3)
growth_score = mean(claims_component, permits_component, activity_component)
```

The claims component's sign is flipped because *falling* initial claims
signal an *improving* labor market (i.e. growth up).

**Model C -- Simple Two-Signal (legacy/auxiliary, not core).** Only
active when `data/external/ism_new_orders.csv` is present -- this is
exactly the kind of external-CSV dependency the core model data source
policy above excludes from Model A/B:

```
growth_score = (z(change_3m(ISM New Orders)) - z(change_3m(Initial Claims))) / 2
```

If the CSV is absent, this model returns `None` and is skipped everywhere
downstream (signals, regimes, evaluation) -- the other growth models
are unaffected.

**Model D -- AMTMNO + Initial Claims (fully FRED-native, core Model B
growth axis).** Replaces Model C's ISM New Orders input with `AMTMNO`
(Manufacturers' New Orders: Total Manufacturing), which is FRED-native --
this model has no external-CSV dependency at all:

```
amtmno_component  =  z(change_3m(AMTMNO))
claims_component  = -z(change_3m(monthly-average(ICSA)))
growth_score = mean(amtmno_component, claims_component)
label = Up if growth_score > 0, Down if growth_score < 0, Unknown if tied/insufficient history
```

## Inflation models

**Model A -- Realized Core Inflation Momentum.**

```
core_3m_annualized = (core_t / core_t-3) ** 4 - 1
core_12m           = core_t / core_t-12 - 1
signal_raw         = core_3m_annualized - core_12m
```

Default core series: `CPILFESL` (Core CPI, SA). Up if `signal_raw > 0`.

**Model B -- Leading Inflation Composite (legacy/auxiliary baseline, not
core).**

```
inflation_score = mean(z(change_3m(ISM Prices Paid)), z(change_3m(T5YIE)), z(core momentum))
```

If `data/external/ism_prices_paid.csv` is absent, a 2-signal variant
(`inflation_score_no_ism`, dropping the ISM term) is computed and used
instead; both columns are always present in the detail output so it is
clear which one was actually active. This is an established pre-existing
baseline and is kept exactly as-is; the core model data source policy
above applies to Model A/B (the two named research models), not to this
proxy.

**Model C -- Cleveland Median CPI Momentum.**

> This is an institutional underlying-inflation trend signal, not the
> Cleveland Fed Inflation Nowcast.

The Cleveland Fed Inflation Nowcast itself was evaluated and rejected as
a model input: it publishes only a *current* value, with no official,
reproducible, downloadable historical vintage file, so backtesting it
would require scraping today's displayed value and treating it as if it
were the historical nowcast at each past date -- a severe form of
look-ahead bias. In its place, Model C uses the Cleveland Fed's **Median
CPI** series (`MEDCPIM158SFRBCLE`), which is FRED-native, auto-fetchable,
and has monthly history back to 1983:

```
ma3_t  = mean(MEDCPIM158SFRBCLE over months [t-2, t])
change = ma3_t - ma3_(t-3)
label  = Up if change > 0, Down if change < 0, Unknown if change == 0 or NaN
```

This is an exact rule with no threshold tuning: the 3-month moving
average is compared to itself 3 months prior, with ties/insufficient
history mapping to `Unknown` (`utils/stats.py::sign_label`).

**Model D -- Commodity + Core (2-signal, auxiliary/secondary research
model, not core).**

```
inflation_score = mean(z(change_3m(PALLFNFINDEXM)), z(core momentum))
```

`PALLFNFINDEXM` (IMF Global Price Index of All Commodities, FRED-native,
monthly since 1992) plus core inflation momentum -- a global commodity
basket rather than a US-only producer-price measure. **This is
structurally a 2-signal model with no ISM Prices Paid input at all**
(`signals/inflation.py::commodity_core_composite` does not accept an ISM
argument). An earlier iteration of this model attempted to be a 3-signal
composite (ISM Prices Paid + commodity + core) that silently fell back to
2 signals when the ISM CSV was absent, while still being labeled/reported
as if the composite design were active -- that was corrected: the model
now has no code path that can produce or claim a 3-signal result. It is
kept only as a secondary research signal for comparison, not part of the
core Model A/B pairing.

## Core models: Primary / Secondary (frozen as of commit `40e43d7`)

**As of commit `40e43d7`, the macro signal models are frozen.** No further
indicator search or formula tuning happens on these two models absent an
explicit decision to unfreeze them:

| Core model | Growth | Inflation | Regime column (`signals.csv`) |
|---|---|---|---|
| **Primary** | US OECD CLI (Growth Model A) | Core Inflation Momentum (Inflation Model A) | `regime_us_cli_core_momentum` |
| **Secondary** (robustness check) | AMTMNO + Initial Claims (Growth Model D) | Core Inflation Momentum (Inflation Model A) | `regime_amtmno_claims_core_momentum` |

Both share the same inflation axis (Core Inflation Momentum) and differ
only in the growth axis -- Secondary exists to check whether Primary's
regime call is corroborated by an independent, fully FRED-native growth
measure. See "Standard regime output" below for the asset-allocation-ready
form of these two models.

**Cleveland Median CPI Momentum (Model C) and Commodity + Core (Model D)
are auxiliary/research models only** -- an earlier iteration used
Cleveland Median CPI as Primary's inflation axis, but a dedicated
challenger experiment (Sticky Price CPI vs. Core Inflation Momentum, and
a full audit of all four candidate axes) found Core Inflation Momentum
the stronger, more consistent choice at both 3m and 6m horizons and under
a 1-month signal-lag stress test. Cleveland Median CPI and Commodity +
Core remain in the codebase and are still evaluated in `signals.csv` /
`evaluation_report.json` for research visibility, but are never part of
Primary or Secondary.

Everything else the pipeline computes is legacy or auxiliary, kept for
baseline comparison only, and explicitly **not** part of Primary/Secondary:

| Column | Status | Why |
|---|---|---|
| `growth_model_fred_minimal` | Baseline/proxy | Pre-existing baseline, FRED-native, kept as-is |
| `growth_model_two_signal` (Growth Model C) | Legacy | Requires `ism_new_orders.csv`; core policy excludes external-CSV models |
| `inflation_model_leading_composite` (Inflation Model B) | Baseline/proxy | Pre-existing baseline, requires `ism_prices_paid.csv` for its full form; kept as-is |
| `inflation_model_cleveland_median_cpi` (Inflation Model C) | Auxiliary/research | Rejected as Primary's inflation axis after the challenger experiment; kept for research only |
| `inflation_model_commodity_core_aux` (Inflation Model D) | Auxiliary/secondary | 2-signal only, FRED-native, but not part of Primary/Secondary |

`build-signals` still cross-joins every active growth model with every
active inflation model into its own `regime_*` column (per "Regime
definition" above) -- these auxiliary combinations remain visible for
research purposes, they are just not Primary or Secondary.

## Standard regime output (asset allocation)

**This output is a revised-data backtest**, same as every other result in
this project (see "Revised data vs. real-time (ALFRED) vintages" below):
`growth_score`/`inflation_score`/`raw_regime` are computed from FRED's
current, fully-revised values, not what was actually known on each
historical date. Do not present `regime_output_*.csv` as a real-time/live
track record.

`build-regime-output` (also run as the last step of `run-all`) produces
the asset-allocation-ready output for Primary and Secondary, written to
`data/processed/regime_output_primary.csv` / `regime_output_secondary.csv`.
Both files share an identical schema, computed directly from the frozen
model functions (not re-derived from `signals.csv`):

| Column | Meaning |
|---|---|
| `growth_score` | The growth model's own continuous score (US CLI: `chg_3m`; AMTMNO+Claims: `growth_score` from `growth_model_d_amtmno_claims`) |
| `growth_state` | Up / Down / Unknown |
| `inflation_score` | Core Inflation Momentum's `signal_raw` |
| `inflation_state` | Up / Down / Unknown |
| `raw_regime` | GOLDILOCKS / REFLATION / STAGFLATION / CONTRACTION / UNKNOWN, from `growth_state` x `inflation_state` (either axis Unknown -> UNKNOWN) |
| `tradable_regime` | `raw_regime` shifted 1 calendar month forward (`regime_output.tradable_lag_months` in `config/default.yaml`, default 1) |

**`tradable_regime` semantics:** `tradable_regime[t] = raw_regime[t-1]`.
The regime computed from month `t`'s data is only applied to month
`t+1`'s portfolio -- this is a **portfolio-application lag**, a
conservative assumption that a regime call cannot be acted on until the
following month. It is a *different* mechanism from FRED
publication/availability lag (see "Look-ahead bias and effective dates"
below): even if the underlying FRED data had zero publication delay,
`tradable_regime` would still lag `raw_regime` by construction. The first
month of any series has no prior `raw_regime` to reference and is always
UNKNOWN. `raw_regime` itself is never modified by this process
(`signals/regime.py::shift_to_tradable` returns a new Series).

In practice, this 1-month portfolio-application lag also functions as a
**rough, approximate stand-in for real FRED publication/availability
lag**, since `evaluate()` / `build-signals` do not apply the
`effective_date` machinery from `data/availability.py` (see "Look-ahead
bias" below -- it exists and is unit-tested but is not wired into the
live pipeline). `regime_output.tradable_lag_months: 1` roughly matches
`lookahead.monthly_macro_lag_months: 1`, but the two are not the same
mechanism and should not be assumed equivalent; treat `tradable_regime`
as a conservative approximation, not a validated point-in-time backtest.

**Do not treat Core Inflation Momentum as a strong short-term inflation
predictor.** The audit at commit `40e43d7` found it to be, at best, a
**weak predictive signal** (small, inconsistent edge over persistence and
majority-class baselines, strongest at 6m) -- functionally it is closer
to a **current-environment classifier**: it describes the recent
inflation trend more reliably than it forecasts the next 3-6 months.
Both Primary and Secondary inherit this limitation on their inflation
axis; only the growth axis differs between them.

## Backtest (v1.0, fixed-regime asset allocation)

`run-backtest` turns Primary's and Secondary's `tradable_regime` (only
`tradable_regime` -- `raw_regime` is never used for return computation)
into an actual fixed-weight portfolio, and compares it against three
static benchmarks. **This is a fixed, ex-ante allocation: the weights
below were specified in advance and are used exactly as given -- nothing
in this backtest is optimized, fit to data, or threshold-tuned.** No
Rate/Credit/Crisis overlay exists in this version (explicitly deferred).

### Growth Asset Basket

**SPY 60% + KODEX 200 ETF (`069500.KS`) 40%, USD-converted returns.**
KODEX 200 is a real KOSPI-200-tracking fund (Yahoo Finance ticker
`069500.KS`), KRW-denominated -- not a USD proxy like an MSCI-Korea ETF.
Its USD-equivalent return combines the fund's own KRW return with the
USD/KRW move over the same month:

```
KODEX_USD_price_t = KODEX_KRW_price_t / USDKRW_t   (USDKRW from Yahoo Finance "KRW=X")
```

Only the **actual month-end** `KRW=X` rate is ever used -- never
predicted, averaged, hedged, or filled forward with a later rate. This
project implements no other FX model; if `069500.KS` or `KRW=X` ever
becomes unfetchable, the backtest must stop rather than silently
substitute a USD proxy (this was a hard constraint during development,
not just a preference).

The 60/40 split is **rebalanced monthly** wherever "Growth Basket"
appears inside a regime/static/equal-weight strategy (its internal
turnover and transaction cost count toward that strategy's total). The
standalone **"Growth Basket buy-and-hold" comparison strategy is true
buy-and-hold**: bought once at 60/40, never rebalanced again, zero
turnover after month 0.

### Fixed regime allocations

**These are ex-ante heuristic weights, not copied from any academic
paper or third-party model.** Each percentage reflects the expected
economic *direction* of that asset class under that regime (e.g.
Treasuries and Gold overweighted in CONTRACTION, Commodities and TIPS
overweighted when inflation is rising) -- a judgment call made in
advance, not a value fit, backtested, or optimized against this
project's data. Exactly as specified
(`backtest.regime_allocations` in `config/default.yaml`), all rows sum
to 100%:

| | Growth Basket | High Yield | Inv. Grade | Interm. Treas. | Long Treas. | Gold | T-bills | Commodities | TIPS |
|---|---|---|---|---|---|---|---|---|---|
| GOLDILOCKS | 60% | 10% | 10% | 10% | -- | 5% | 5% | -- | -- |
| REFLATION | 40% | 10% | -- | -- | -- | 10% | 5% | 20% | 15% |
| STAGFLATION | 10% | -- | -- | -- | -- | 25% | 20% | 25% | 20% |
| CONTRACTION | -- | -- | 15% | 20% | 35% | 10% | 20% | -- | -- |
| UNKNOWN | -- | -- | -- | -- | -- | -- | 100% | -- | -- |

Underlying tickers: High Yield=`HYG`, Investment Grade=`LQD`,
Intermediate Treasury=`IEF`, Long Treasury=`TLT`, Gold=`GLD`,
T-bills=`BIL`, Commodities=`DBC`, TIPS=`TIP` (`backtest.assets` in
config).

### Comparison strategies

- **Primary** / **Secondary** -- regime allocation driven by each
  model's `tradable_regime`.
- **Static 60/40** -- Growth Basket 60% + Intermediate Treasury 40%,
  rebalanced monthly (constant target, not "no rebalancing").
- **Static equal-weight** -- 1/9 each across the 9 distinct categories
  named anywhere in the regime tables above (Growth Basket counted once,
  not as two slots), rebalanced monthly. This 1/9 definition was not
  specified further upstream and is documented here as an explicit
  choice, not hidden.
- **Growth Basket buy-and-hold** -- 100% Growth Basket, bought once,
  never rebalanced (see above).

### Mechanics

- **Monthly rebalance to target, every month**, even when the target is
  unchanged from the prior month (regime strategies restore full target
  weights on schedule, not only on a regime change).
- Month `t`'s portfolio return uses only the weights entering month `t`
  (decided from `tradable_regime[t]` or the constant static target) and
  month `t`'s realized asset returns -- never a later month's regime or
  price.
- **Turnover** = sum(|target weight - drifted prior weight|) / 2 ("one-way"),
  applied uniformly including the very first rebalance (implicit
  all-cash prior state).
- **Transaction cost** = turnover x `backtest.transaction_cost_bps`
  (10bp), deducted from the same month's return it was incurred to
  enter. Every metric is reported **pre-cost and post-cost** side by
  side; post-cost cumulative value is checked to never exceed pre-cost
  (`tests/test_backtest.py`).
- **Risk-free rate** (Sharpe/Sortino): the realized monthly return of
  `BIL` (already in the universe), not an arbitrary constant -- also an
  explicit, undirected choice, documented rather than hidden.

### Common sample and data limitations

Monthly returns are built from each ticker's Yahoo-Finance adjusted
close (`auto_adjust=True`, split+dividend-adjusted = total-return
equivalent), resampled to month-end using the **last known trading-day
close within the month only** -- a month with no data produces `NaN`
and is never forward- or backward-filled across the gap
(`utils/dates.py::resample_to_monthly`, reused as-is).

**The most recent, still-in-progress calendar month is always excluded.**
`resample("ME").last()` labels whatever the latest available trading day
is with that *future* calendar month-end date -- e.g. if today is
2026-07-27 and the latest fetched close is 2026-07-24, that row gets
stamped "2026-07-31" even though July has four more trading days left.
Using it as a "monthly return" would silently understate/misstate the
month. `backtest/assets.py::build_monthly_return_matrix` drops any row
whose date is after `as_of` (real current time by default, injectable
for tests) before computing the common start date or any downstream
metric -- confirmed live: this moved the reported end date from
2026-07-31 (partial) to 2026-06-30 (the last fully-elapsed month) the
first time it was checked.

**The backtest starts at the first month-end where every asset in the
universe has a real return -- not the 1990+ history used by the macro
signals.** All 9 US-listed tickers have data by 2007-05 (`BIL`'s
inception, the latest of the nine). **The actual binding constraint,
discovered when this backtest was first run, is a real gap in Yahoo
Finance's `069500.KS` (KODEX 200) history: data exists for 2007-01 and
2007-02, then is completely missing from 2007-03 through 2009-03, then
resumes 2009-04 onward.** Per the no-forward-fill rule, that gap is
never bridged, so the common sample begins **2009-05** (the first month
with a valid month-over-month return for every asset) -- **not** ~2007
as a BIL-only inception check would suggest. This means **2008 (GFC) is
not covered by this backtest's common sample** -- only 2020 and 2022 of
the three requested crisis years are; `backtest_annual_returns.csv` /
`backtest_regime_analysis.csv` correctly omit 2008 rather than
fabricating them. This build does **not** model ETF survivorship bias
(each fund's full available history as currently listed is used; no
adjustment for funds that may have closed or merged elsewhere in the
industry).

**Past performance in this backtest does not represent live or
real-time performance** -- it is a revised-regime, historical-price
backtest with a 1-month portfolio-application lag approximation, not a
tracked, executed strategy.

## BEI Duration Risk Gate (v1.1)

**v1.1 is a single, narrow overlay on top of the frozen v1.0 Primary
backtest, not a new macro regime model.** It does not forecast interest
rate *levels*, and it does not change the Primary growth/inflation
regime classification or any asset outside the nominal-Treasury
duration sleeve. It only tilts how much of that sleeve's allocation
(`intermediate_treasury` + `long_treasury`, i.e. IEF + TLT) is held when
a long-end nominal rate sell-off, confirmed by actually-falling TLT
prices, is being driven by rising market inflation compensation. It
exists to reduce long-duration drawdown risk in that specific
situation, not to maximize return.

`run-bei-duration-gate` compares two strategies under otherwise
identical conditions to the v1.0 Primary backtest above (same asset
prices, same Growth Basket, same transaction cost, same monthly
rebalance-to-target rule, same common start date, same metric
formulas): **v1.0 Primary** (unmodified) and **v1.1 Primary + BEI
Duration Risk Gate**. Because it targets only the duration sleeve, this
comparison is scoped to those two strategies -- it does not re-run
Secondary, Static 60/40, Equal-weight, or Growth Basket buy-and-hold.

### Data

Two additional FRED series are fetched, alongside the existing pipeline
series, via the same `fetch`/`FredClient` cache (`config/default.yaml`,
`series.diagnostics`):

- **DGS10** -- 10-Year Treasury Constant Maturity Rate (the long-end
  nominal rate the gate confirms is actually rising).
- **T10YIE** -- 10-Year Breakeven Inflation Rate. **This is the market's
  inflation *compensation* embedded in the spread between nominal and
  TIPS Treasury yields -- it is not a pure survey- or model-based
  expected-inflation measure.** It is also influenced by an inflation
  risk premium and by TIPS-market liquidity conditions, which can move
  independently of actual inflation expectations. This module and the
  product UI always refer to it as "market inflation compensation," not
  "expected inflation."

Both are daily series, averaged to month-end
(`duration_gate/signal.py::monthly_rate_level`, reusing
`utils/dates.py::resample_to_monthly(how="mean")` and the same
`drop_incomplete_trailing_month` still-in-progress-month exclusion used
throughout this project -- no forward fill, no backfilling before a
series' own start). TLT's monthly return is the same adjusted
(split+dividend) monthly return already computed for the v1.0 backtest
(`backtest/assets.py::build_monthly_return_matrix`'s `long_treasury`
column) -- not recomputed separately.

DGS10 and T10YIE are both available on FRED well before the v1.0
backtest's own binding constraint (the 2009-05 `069500.KS` gap
described above), so adding them **does not move the common backtest
start date** -- verified live: the rate-signal series' own common start
is 2003-01, while the asset-price-driven common start remains 2009-05.

### The gate: `raw_bei_duration_gate`

For a completed month `t` (never a still-in-progress month):

```
dgs10_change_1m = DGS10[t] - DGS10[t-1]
bei_change_1m   = T10YIE[t] - T10YIE[t-1]
bei_change_3m   = T10YIE[t] - T10YIE[t-3]
tlt_return_1m   = TLT's adjusted monthly return for month t
```

`raw_bei_duration_gate[t]` = **ON** iff all four hold:

1. `dgs10_change_1m > 0` -- the long-end nominal rate actually rose.
2. `tlt_return_1m < 0` -- TLT's own price actually fell (a real,
   realized loss, not just a rate move).
3. `bei_change_1m > 0` -- market inflation compensation rising, 1-month.
4. `bei_change_3m > 0` -- market inflation compensation rising,
   3-month (the move is not a single-month blip).

**OFF** if every one of the four values needed is present but the AND
fails. **UNKNOWN** if any of the four is missing (e.g. the warm-up
period before a 3-month lookback is available). No real-yield
condition, no DGS2-based signal, and no combination with any other
signal is used -- this gate is deliberately narrower than the
DFII10/real-yield and DGS2/DGS10 `rate_score`-based variants explored
during development (see "Prior experiment models," below).

**Only the ex-ante thresholds above are used.** The 1-month/3-month
windows, the strict `> 0` / `< 0` comparisons, and the 30%/70% IEF/BIL
split below were fixed before any backtest was run and are not
optimized, tuned, or adjusted to past performance -- including after
seeing that performance.

### Application timing

```
tradable_bei_duration_gate[t] = raw_bei_duration_gate[t-1]
```

The gate confirmed as of month `t-1`'s close is what month `t`'s
portfolio may use; month `t`'s own raw gate (only fully known at `t`'s
own close) never drives month `t`'s allocation. The very first month in
any sample has no prior raw gate and is UNKNOWN, never defaulted to ON
or OFF.

### Allocation rule

For each month, `duration_pool = base_IEF + base_TLT` (from Primary
v1.0's regime allocation table, unmodified). `base_BIL` is tracked
separately and always preserved.

- **Gate ON**: `final_TLT = 0`, `final_IEF = duration_pool * 0.30`,
  `additional_BIL = duration_pool * 0.70`, `final_BIL = base_BIL +
  additional_BIL`.
- **Gate OFF or UNKNOWN**: `final_IEF = base_IEF`, `final_TLT =
  base_TLT`, `final_BIL = base_BIL` -- Primary v1.0's allocation
  exactly, unchanged.
- **`duration_pool == 0`** (e.g. REFLATION, STAGFLATION, UNKNOWN
  regimes, which hold no IEF/TLT to begin with): the entire month's
  allocation is left unchanged regardless of gate state -- there is
  nothing to tilt.

There is **no rule that ever expands TLT** -- this is a one-directional
duration-reduction brake, not a directional call on falling rates.
Growth Basket, the SPY/KODEX200 60/40 split, HYG, LQD, TIP, DBC, and
GLD are never touched. `duration_pool` and `base_BIL` are always exactly
conserved; total portfolio weight always sums to 1 with no negative
weights (`tests/test_duration_gate.py`).

### Output files

`run-bei-duration-gate` writes six files to `data/processed/`, none of
which overwrite or are read by the v1.0 `backtest_*.csv` files:
`bei_duration_gate_signals.csv` (the raw signal table), `_allocations.csv`
(base vs. final IEF/TLT/BIL per month plus every other asset's final
weight), `_monthly_returns.csv`, `_annual_returns.csv`, `_summary.csv`
(CAGR/Vol/Sharpe/Sortino/MaxDD/Calmar/turnover, pre- and post-cost, for
both strategies), and `_regime_analysis.csv` (per-regime average
returns, plus a period breakdown -- 2013 Taper Tantrum, 2018, 2020,
2021, 2022, 2023-2024 -- of cumulative return, max drawdown, and
gate-ON-month count for each strategy).

### Known limitations

- **T10YIE is market inflation compensation, not expected inflation** --
  repeated here because it is easy to conflate the two; risk premia and
  TIPS-market liquidity can move this series independent of actual
  inflation expectations.
- **This is a revised-FRED-data backtest**, like v1.0 -- not a
  real-time/ALFRED-vintage track record.
- **2008 (GFC) is not covered**, for the same reason as v1.0: the
  `069500.KS` price gap sets the common backtest start at 2009-05.
- **The 1-month/3-month windows and 30%/70% split are ex-ante
  heuristics**, not values taken from a published paper and not
  re-optimized after seeing backtest results.
- **The most recent 24 months of the backtest sample showed a small
  performance drag relative to v1.0 Primary** in development testing --
  this is disclosed, not hidden, and is consistent with the gate being
  a risk-reduction brake rather than a return-maximizing signal: it
  will not help in every period, including recent ones.
- **No guarantee of forward repeatability.** Historical improvement,
  where present, does not imply the gate will behave the same way in a
  future rate cycle.

### Prior experiment models (not in this product)

During development, three additional decomposition/hybrid gate variants
and a FALLING-state TLT-*expansion* rule were backtested in an isolated
temporary environment to understand which components of a broader
"rate overlay" idea actually added value beyond this BEI-only gate:
**Rising-only** (a DGS2/DGS10-based `rate_score` gate, no T10YIE/DFII10
involved), **Real-yield-only** (DFII10-based, the mirror-image of this
BEI gate), **Decomposition OR** (BEI OR real-yield), **Hybrid**
(Rising-only OR Decomposition OR), and the FALLING-state rule that
*expands* TLT when rates are falling. None of these are implemented in
this codebase -- they were experiment-only, run in a disposable
temporary directory, and are not reachable from any CLI command,
Streamlit page, or config in this repository. This BEI-only gate was
selected because it was the only variant that continued to outperform
v1.0 Primary after excluding 2022 (the single year that otherwise drove
most of the other variants' apparent edge) and after excluding
2021-2023 entirely.

## Fast Crisis Overlay (v1.2)

**v1.2 is a daily-frequency tail-risk brake on top of v1.1 (Primary v1.0
+ BEI Duration Risk Gate), not a new macro regime model.** v1.1's
monthly regime/BEI logic is completely unchanged and untouched by v1.2
-- `run-fast-crisis-overlay` never reads or writes any `backtest_*.csv`
or `bei_duration_gate_*.csv` file, and v1.1 remains independently runnable
via `run-bei-duration-gate` exactly as before. v1.2 exists because v1.1's
own worst drawdown (the 2020 COVID shock) was diagnosed as occurring
*within* a single calendar month, invisible to a monthly-rebalance
engine until the month had already closed -- v1.2 adds a same-week-scale
daily watch layer that can act mid-month, which a monthly engine
structurally cannot.

Layer order: **1. Macro Regime -> 2. BEI Duration Risk Gate -> 3. Fast
Crisis Overlay.** The BEI gate only ever touches
`intermediate_treasury`/`long_treasury`/`tbills`; the Fast Crisis Overlay
only ever touches `spy`/`kodex200_usd`/`high_yield`/`commodities`/`tbills`
-- the two gates never act on the same weight, so they cannot fight each
other or double-count a de-risking move.

### Official daily evaluation calendar

v1.2's engine walks one canonical daily calendar: **SPY's own US trading
days** (all US-listed ETFs in the universe share this exactly). KODEX
200 (069500.KS, KRX) and KRW=X are as-of joined onto it -- each SPY day
uses the most recent KRX/FX close on or before that day, never a future
one. Whenever KRX/FX trade on a date SPY has no session for (a US
holiday that isn't a KRX holiday), that day's KRX/FX movement is
attributed to the *next* SPY trading day's return. This is a genuine,
structural difference from the legacy monthly engine (which samples each
asset's own last trading day of the calendar month) -- **not a bug**.
Worked example: 2010-05-31 was a US holiday; SPY's last May session was
2010-05-28, while KRX/FX traded through 05-31. The legacy monthly engine
uses the 05-31 KODEX/FX price for May's return; v1.2's daily engine
attributes that same 3 days of movement to June instead. Verified live:
this class of difference tops out at 0.36 percentage points in any
single month across the full 2009-05 to 2026-06 sample, and daily-close
returns for every OTHER asset (spy, high_yield, investment_grade,
intermediate_treasury, long_treasury, gold, tbills, commodities, tips)
match the legacy monthly engine's own asset returns to floating-point
precision (<= 2e-6).

### Missing-value handling

A raw daily price series can have an index entry that already exists (a
real trading day) with a `NaN` value -- a data-provider gap, not a
missing calendar date. This actually occurred once in the verification
sample: **069500.KS on 2024-10-30**. A plain `reindex(calendar,
method="ffill")` only fills index positions *absent* from the source; it
silently passes an already-present `NaN` cell through unchanged, and a
downstream `.prod()` compounding call treats that day as an implicit 0%
return -- distorting that month's real return (verified: true October
2024 KODEX return ~-6.44%, distorted value ~-4.51%, a 0.46-point swing in
that single month's portfolio return). `fast_crisis/daily_data.py` fixes
this by `.ffill()`-ing every raw series (carrying the last known value
forward only, never fabricating, never filling before a series' own
first valid observation) *before* the as-of join, and
`validate_no_gaps_in_range` fails loudly -- naming the exact asset(s) and
date(s) -- if any `NaN` remains in the range a backtest actually uses.
**This fix lives only in `fast_crisis/`, not in `backtest/assets.py`**:
the legacy monthly engine was independently verified unaffected by this
specific gap (its own October 2024 return was already correct, since
month-end resampling happened to land on a valid day), so v1.1's
official numbers are untouched by this fix.

### The three shock signals

```
vix_shock[t]    = VIX[t] > 30 AND VIX[t] / MA20(VIX)[t] - 1 > 0.50
equity_shock[t] = SPY's 5-trading-day total return <= -7%
credit_shock[t] = HYG's 5-trading-day total return <= -3%
```

Each is **UNKNOWN**, never guessed, until its own warm-up window (20
valid VIX observations; 5 trading days of return history) is available.
`raw_fast_crisis_trigger[t]` = **ON** iff at least 2 of the 3 signals are
ON; with exactly 2 observed it is ON only if both are True, OFF only if
both are False (the third, missing signal can never change either of
those two outcomes), and **UNKNOWN** only in the genuinely ambiguous
case of exactly 1 of 2 observed signals being True. Fewer than 2
observed signals is always UNKNOWN.

All three thresholds, the 20-day VIX window, and the 2-of-3 combination
rule (`config/default.yaml`, `fast_crisis:`) are **ex-ante heuristics**
fixed before any backtest was run and were not re-tuned after seeing
results -- see "Research basis and limitations," below.

### Application timing and the entry/exit state machine

```
tradable_trigger[t] = raw_fast_crisis_trigger[t-1]
```

A same-day close signal is never treated as tradable on its own day --
the 1-trading-day lag mirrors the BEI gate's own `raw[t-1]` convention.

- **OFF -> ON**: the first trading day whose `tradable_trigger == ON`.
  UNKNOWN never triggers an entry.
- **Once ON**: stays ON for at least **10 trading days**, regardless of
  what the trigger does during that window.
- **Exit**: after the minimum hold has elapsed, exits on the trading day
  following **5 consecutive** `tradable_trigger == OFF` days. UNKNOWN
  neither counts toward this streak nor resets it -- it is skipped.
- **Re-affirmation**: a fresh `tradable_trigger == ON` while already in
  Crisis Mode restarts the minimum-hold counter and resets the
  consecutive-OFF counter to 0.

### Allocation rule

**Crisis Mode ON**: Growth Basket (`spy` + `kodex200_usd`), `high_yield`,
and `commodities` are all set to 0; their combined weight moves entirely
to `tbills`. `investment_grade`, `intermediate_treasury`, `long_treasury`,
`gold`, and `tips` -- and whatever the BEI gate has already done to
`intermediate_treasury`/`long_treasury`/`tbills` -- pass through
unchanged. **Crisis Mode OFF or UNKNOWN**: v1.1's (already BEI-gated)
allocation is returned exactly unchanged. There is no rule that ever
*increases* risk exposure -- this is a one-directional de-risking brake.
A day that is both a monthly rebalance day and a Crisis Mode entry/exit
day trades exactly once, to the final combined target, with exactly one
transaction cost applied -- never double-counted.

### Daily portfolio mechanics

The existing, **unmodified** `backtest.engine.run_strategy` -- already
index-frequency-agnostic -- is reused as-is for the daily walk; no new
portfolio engine was written. Weights are only re-set to target on an
actual rebalance day (each month's first trading day, or a Crisis Mode
entry/exit day); every other day, the strategy's actual held weights
drift with realized asset returns and no trade or cost occurs. Verified
live: `v1_1_daily`'s own crisis-attributable turnover is exactly 0 (it
has no overlay), and no strategy ever has nonzero turnover on a
non-rebalance day.

### Two MaxDD readings -- never mixed

**Daily-close MaxDD** (`max_drawdown_daily_close_*`) is computed from
the full daily NAV path and sees the actual worst intra-month level.
**Month-end MaxDD** (`max_drawdown_month_end_*`) is computed from the
NAV sampled only at each month's last trading day -- what the legacy
monthly v1.1 backtest itself reports, and matches it closely (verified
live: v1.1's daily-restated month-end MaxDD is -15.387% vs. the legacy
monthly engine's own -15.387%). These two numbers describe different
things and are always kept in separately labeled columns, never averaged
or substituted for one another; all daily-frequency metrics use 252-day
annualization, never the monthly engine's 12-month convention
(`fast_crisis/metrics.py`, deliberately separate from `backtest/metrics.py`).

### Results (v1.1 vs. v1.2, post-cost, 2009-05-01 to 2026-06-30, live run)

| Strategy | CAGR | Vol | Sharpe | Sortino | Daily-close MaxDD | Month-end MaxDD | Calmar | Turnover/yr |
|---|---|---|---|---|---|---|---|---|
| v1.1 (daily-restated) | 10.22% | 8.60% | 1.026 | 1.305 | -22.61% | -15.39% | 0.452 | 2.19 |
| v1.2 Fast Crisis Overlay | 10.80% | 7.98% | 1.165 | 1.585 | -16.64% | -13.37% | 0.649 | 2.54 |

v1.2 improves every one of these metrics over v1.1, driven overwhelmingly
by the 2020 COVID shock (the only period where v1.1's own worst drawdown
occurred). Excluding 2020 entirely, v1.2's Sharpe improvement over v1.1
narrows to roughly neutral-to-slightly-positive rather than large; it
remains non-negative excluding 2020+2022 combined and even excluding the
most recent 24 months -- i.e. the improvement is not manufactured by a
single crisis alone, but the bulk of its economic value is concentrated
in exactly the kind of acute, fast-moving shock it was designed for.

### Output files

`run-fast-crisis-overlay` writes seven files to `data/processed/`, none
of which overwrite or are read by v1.0's `backtest_*.csv` or v1.1's
`bei_duration_gate_*.csv` files: `fast_crisis_signals_daily.csv` (every
daily signal, trigger, and state-machine column), `_allocations_daily.csv`
(base v1.1 vs. final v1.2 weights per day), `_daily_returns.csv`,
`_monthly_returns.csv` (both strategies' NAV re-sampled to month-end),
`_summary.csv` (the dual-MDD performance table above, pre- and post-cost),
`_turnover_breakdown.csv` (monthly-rebalance vs. crisis-entry/exit
turnover per strategy), and `_current_status.json` (today's regime, gate
states, Crisis Mode status, and both strategies' current weights -- the
same snapshot a live dashboard would show).

### Known limitations

- **The three thresholds, the 20-day VIX window, the 2-of-3 rule, and
  the 10-day/5-day hold/exit rule are ex-ante heuristics**, fixed before
  any backtest was run and not re-optimized after seeing results.
- **This is a revised-price backtest**, like v1.0/v1.1 -- not a
  real-time/live track record.
- **Most of the measured benefit comes from a single event (2020)** --
  disclosed, not hidden; see "Results," above.
- **VIX-only and Equity+VIX combination candidates were also tested and
  rejected** during development in favor of the 2-of-3 rule shipped here
  (VIX-only alone had a ~70% false-positive rate on its own entries) --
  those candidates are not implemented in this codebase.
- **No guarantee of forward repeatability.** A future crisis may not
  resemble 2020's shape closely enough for these specific daily
  thresholds to fire in time, or at all.

### Research basis and limitations

- **Moreira & Muir** (volatility-managed portfolios): reducing risk
  exposure in high-volatility regimes is a well-studied direction in the
  academic literature -- cited here as directional motivation for a
  vol-based brake, not as a source of this overlay's specific numeric
  thresholds.
- **VIX** is the market's own ~30-day-forward implied volatility, priced
  into S&P 500 options -- a market-implied risk gauge, not a realized
  historical volatility measure.
- **Credit-market stress research** motivates treating credit spreads as
  informationally distinct from equity price action -- credit markets
  can reflect funding/liquidity stress that hasn't yet fully shown up in
  equity prices. **HYG is an imperfect, ETF-price-based proxy for credit
  spreads (OAS)**, not the spread itself -- it is affected by ETF
  premium/discount dynamics and liquidity conditions of the fund, not
  purely the underlying bonds' credit risk.
- **The 2-of-3 combination across three largely independent signal
  families** is a standard confirmation-style design intended to reduce
  the false-positive rate any single signal would have alone -- verified
  live to matter: VIX-only's entries had a substantially higher
  false-positive rate than the 2-of-3 rule's entries over the same
  sample.

## Growth Participation (v1.3)

Current production version, adopted after an independent robustness
study of Candidate "Both +5" (`.analysis/growth_participation_robustness/`,
itself following on from a recent-underperformance-attribution analysis
that found v1.2's structural equity underweight -- not BEI or Fast
Crisis -- as the dominant driver of recent lag vs. a 60/40 benchmark).

**Exact change vs. v1.2** (`config/default.yaml`,
`strategy_versions.v1_3.regime_allocation_overrides`):

| Regime | Change |
|---|---|
| GOLDILOCKS | Growth Basket 60%→65%, tbills 5%→0% |
| REFLATION | Growth Basket 40%→45%, tbills 5%→0% |
| STAGFLATION | unchanged |
| CONTRACTION | unchanged |
| UNKNOWN | unchanged |

Growth Basket's internal SPY 60% / KODEX200(USD) 40% split, BEI Duration
Risk Gate, Fast Crisis Overlay (including its ON-day behavior: Growth
Basket/HYG/DBC → 0, moved to BIL, GLD/TIP/LQD/IEF/TLT unchanged),
transaction costs, and `t+1` execution are all identical to v1.2 -- the
only thing v1.3 changes is which two rows of the regime allocation table
are used. This is enforced structurally, not just by convention:
`macro_regime.strategy_versions.build_versioned_config` deep-copies the
loaded config and only ever overwrites `backtest.regime_allocations[
"GOLDILOCKS"]`/`["REFLATION"]` for `version="v1_3"`; `version="v1_2"`
returns an unmodified copy. `run_fast_crisis_backtest` itself is never
edited or branched on version, so a v1.3 config change can never
propagate into a v1.2 result -- v1.2 and v1.3 are always two
independently-constructed config dicts passed through the same
unmodified pipeline. `tests/test_v1_3_production.py` verifies live that
CONTRACTION/STAGFLATION daily returns, BEI-gate-ON days, and Fast-
Crisis-ON days are byte-identical between the two versions.

**Adoption record** (Candidate "Both +5" robustness study; see
`.analysis/growth_participation_robustness/report_ko.md` for the full
20-section analysis -- pre-registered PASS criteria, parameter
sensitivity at ±2.5pp/±5pp/±7.5pp/±10pp shifts, transaction-cost
sensitivity at 0/10/25/50bp, execution-delay stress test, rolling
36/60-month robustness, regime-attribution isolation checks):

- Full CAGR: 10.80% → ~11.41%
- Sharpe: 1.165 → ~1.184
- Daily MDD: -16.64% → -16.64% (unchanged -- the full-period worst
  drawdown episode occurs in a regime v1.3 never touches)
- Last 3Y CAGR: 17.75% → ~18.93%
- Last 5Y CAGR: 11.03% → ~11.70%
- COVID MDD: -7.81% → ~-8.34%
- 2022 performance: unchanged (2022 was Contraction/Stagflation-dominated)
- Rolling 36-month CAGR win rate vs. v1.2: ~99.7%
- Rolling 60-month CAGR win rate vs. v1.2: 100%
- Calm-market (SPY monthly return 0-3%) upside capture vs. US 60/40: 79.7%

> The candidate narrowly missed the pre-specified 80% calm-market
> upside-capture threshold, but was adopted based on its higher CAGR,
> improved Sharpe, unchanged full-period drawdown, strong rolling
> consistency and limited degradation during crisis periods.

These figures are recorded here as the study's own findings, for
context. Every number the Streamlit dashboard or
`data/processed/production_v13_daily.parquet` displays is computed live
from the current engine and current data -- this table is never read by
any product code path.

**Daily/monthly separation fix**: the daily backtest previously computed
`end_date = min(daily_returns.index.max(), v1_1_monthly_target.index.max())`
-- since the monthly macro target can only ever be computed for a
CLOSED calendar month, this silently truncated the ENTIRE daily
NAV/benchmark/drawdown series at the last complete month, even on days
the underlying daily asset-price data already covered (e.g. dashboard
artifacts appeared frozen at the end of the last closed month while live
market data for the following, still-open month was already available).
Root cause confirmed by direct inspection, not guessed: it was not a
stale on-disk cache (a *separate*, real issue was found and fixed too --
see below), but this specific `end_date` computation. Fixed by (1)
setting `end_date = daily_returns.index.max()` and (2) changing
`_broadcast_monthly_onto_daily` so a daily date in a calendar month with
no monthly value of its own yet holds the LAST available monthly value
forward (previously it did an exact-month dict lookup that would
`KeyError` on any such date, which is why `end_date` had to be capped in
the first place). Holding a month's already-decided target forward
introduces no lookahead -- it uses nothing from the still-open month.
Verified: v1.2 daily returns for the closed-month period are unchanged
to floating-point precision after the fix
(`tests/test_v1_2_regression.py`), and the new `partial_month` column
(`FastCrisisBacktest.allocations`, and in the canonical daily parquet
artifact) flags exactly the open-month days now being served this way.

**Secondary finding -- asset-price cache fragmentation**: independently
of the above, `AssetPriceClient`'s on-disk cache is keyed on
`(ticker, start_date)` with no expiry, so two different call sites
requesting the same ticker with different `start` parameters can hold
independently stale copies indefinitely (observed live: a `SPY` cache
entry fetched with `start="2000-01-01"` was three days staler than one
fetched with `start="2009-01-01"`, purely because of when each was last
refreshed). This does not affect the product's own daily backtest (which
always uses one consistent `start`), but it did initially leak into the
new Benchmark Registry's US 60/40 series until `update-all` was fixed to
always pass `refresh_cache=True`. Never rely on an un-refreshed cache
call for a production artifact. `AssetPriceClient.get_cache_status` and
`FileCache.cache_status` now expose retrieved-timestamp/cache-version/
TTL-based staleness for any (ticker, start) entry, and the Streamlit
sidebar warns if the cache entry the daily engine actually uses is
older than 24h.

**Observed vs. effective (lagged) macro signal**: `growth_state`/
`inflation_state` in `regime_output_primary.csv` are each model's OWN
unlagged monthly reading (never shifted). `macro_regime`, however, is
`tradable_regime = raw_regime.shift(tradable_lag_months)` -- the regime
actually in effect today was decided from LAST month's growth/inflation
reading, not this month's. The canonical daily artifact
(`production.build_v1_3_daily_artifact`) exposes both: `growth_state`/
`inflation_state`/`growth_score`/`inflation_score` are the LAGGED,
EFFECTIVE reading that actually explains `macro_regime` (verified for
the entire history via `classify_regime(growth_state, inflation_state)
== macro_regime`, `tests/test_production_growth_inflation_alignment.py`),
while `growth_state_observed`/`inflation_state_observed` are the raw,
most-recently-published reading, not yet acted on. The Streamlit Signals
tab shows both, labeled "raw signal (unlagged)" vs. "effective month
(lagged)", alongside the real underlying series value/comparison value/
calculated change for each -- never just an internal model name.

## Benchmark Registry

`src/macro_regime/benchmarks/` (`BenchmarkDefinition`, `BenchmarkRegistry`,
`BenchmarkSeries`, `BenchmarkMetrics`, `BenchmarkDataStatus`) replaces
the single hardcoded 60/40 benchmark with a small, explicit registry --
each definition records id, display name, category, data source,
components/weights, calculation method, rebalance rule, transaction-cost
rule, calendar rule, total-return treatment, known inception date,
UI-visibility flag, default-selection flag, and relative-performance
support. Adding a benchmark means adding one `BenchmarkDefinition`, not
editing multiple Streamlit files.

**User-facing (`ui_visible=True`)**:
- **US 60/40** (`us_60_40`) -- SPY 60% / AGG 40%, monthly rebalance,
  10bp transaction cost, daily valuation with drift between rebalances.
  Default-selected everywhere; every default strategy comparison (NAV,
  drawdown, CAGR, Sharpe, Sortino, MDD, relative NAV, annual/rolling
  returns, upside/downside capture, crisis performance) uses this.
- **MALOX** (`malox`) -- see below.
- **SPY** (`spy`), **AGG** (`agg`) -- single-asset, buy-and-hold,
  opt-in reference lines.

**Internal-only (`ui_visible=False`)**: **Project 60/40**
(`project_6040`, Growth Basket 60% + IEF 40%) -- the project's own
original v1.0-v1.2 benchmark. Kept registered and fully functional (its
series still comes straight from `FastCrisisBacktest.results[
BENCHMARK_LABEL]`, computed by the unmodified `run_fast_crisis_backtest`)
so historical `.analysis/` reports and the v1.0-v1.2 regression fixtures
remain reproducible, but it is excluded from every user-facing surface:
the Overview/Performance benchmark selectors, checkboxes, chart
legends, the downloadable comparison export, and the pitch copy. Not
deleted, not hidden behind a secret flag -- just never registered as
`ui_visible`.

### MALOX data treatment

Ticker identity verified live against provider metadata before use:
`quoteType == "MUTUALFUND"`, `longName == "BlackRock Global Allocation
Fund"` -- the Institutional Shares class. Distribution-adjustment
verified against an actual event (a $0.90/share distribution on
2025-12-16): the auto-adjusted price for 2025-12-15 is reduced from the
raw close by exactly that distribution, confirming the adjusted series
is a genuine total-return series, not a naive close price. This is a
single-event spot-check, not a full independent cross-provider
reconciliation -- recorded as a limitation.

Because MALOX is a mutual fund, not an ETF: one NAV is published per
trading day (no intraday price), and that NAV can post up to a day
later than the strategy's own market date (both a NYSE-vs-fund-NAV
calendar difference and an ordinary end-of-day-pricing lag). Alignment
rule: MALOX's own NAV series is forward-filled onto the strategy's
calendar for VALUATION purposes only -- a day with no new published NAV
is never treated as a real return; forward-filling a flat price
naturally produces exactly 0.0% return on that day (not a special case
in the code, a direct consequence of `ffill().pct_change()`), and that
day is separately flagged `is_stale=True` so the UI can show a badge
without ever implying a fabricated gain/loss. MALOX's own NAV is never
backward-filled into a date before its first real observation. Every
MALOX comparison (NAV chart, drawdown, relative performance, metric
table) uses the common period between MALOX's own available history and
the strategy's -- the strategy's/US 60/40's own full-period numbers are
never silently truncated to MALOX's shorter or later-starting window.

No synthetic rebalancing transaction cost is added to MALOX -- its
reported NAV/adjusted total return already reflects the fund's real
internal expense ratio; this project does not model an external
investor's own sales load or platform fee on top of that, and the two
should not be confused.

## Conference Board LEI vs. FRED USSLIND

`USSLIND` on FRED is the **Philadelphia Fed's State Leading Indexes**
aggregate, not the Conference Board's Leading Economic Index. `USSLIND`
was also discontinued after 2020. This project never uses `USSLIND` as a
stand-in for the Conference Board LEI. If a Conference Board LEI-based
model is wanted, it must be supplied via
`data/external/conference_board_lei.csv` (columns: `date, lei`, optional
`release_date`, `vintage_date`). Absent that file, a clear warning is
raised and the model is skipped -- it is not silently disabled without
explanation.

## Date-index normalization

FRED stamps native monthly series (e.g. `PERMIT`, `CFNAIMA3`,
`USALOLITOAASTSAM`, `CPILFESL`) first-of-month, while series resampled
from weekly/daily data (`ICSA`, `T5YIE`) land on month-end after
resampling. `utils/dates.py::normalize_month_end_index` re-stamps a
series to month-end regardless of which convention it started from, and
is applied twice: inside each growth/inflation model that joins
multiple raw series (`growth_model_b_fred_minimal`,
`leading_inflation_composite`), and again to every model's *output*
column plus the evaluation targets (`INDPRO`, `CPILFESL`) in
`cli.py::_growth_signals` / `_inflation_signals` / `evaluate`. Skipping
either layer produces the same failure mode: two economically identical
months land on different index labels, so joins/reindexes silently
return `NaN` instead of erroring, and the affected model evaluates with
`n_obs = 0` even though its underlying signal is computed correctly.

## Standardization (z-scores)

All z-scores use `rolling_zscore` (`utils/stats.py`): a rolling window
(default 120 months) with a minimum history requirement (default 60
months), `ddof=1`, and `NaN` wherever the rolling standard deviation is
exactly 0. Because `pandas.Series.rolling()` only ever looks backward from
each timestamp, this construction cannot use future information --
verified in `tests/test_lookahead.py::test_rolling_zscore_has_no_lookahead`,
which checks that truncating a series to any date `t` does not change the
z-score computed at `t`. Winsorization is available in config but disabled
by default. Full-sample mean/std are never used.

## Look-ahead bias and effective dates

Every signal carries five date columns (`data/availability.py`):

- `observation_date` -- the period the data describes
- `release_date` -- the agency's actual publication date, if known
- `availability_date` -- `release_date` if known, else `observation_date`
  plus a conservative configured lag (`monthly_macro_lag_months`, default
  1 month for monthly macro data; `daily_market_lag_days` for market data)
- `signal_date` -- equals `observation_date`
- `effective_date` -- the first calendar day of the month *after*
  `availability_date`'s month; this is the earliest date a backtest may
  actually use the signal

Most FRED series in this build do not have a modeled release calendar, so
they fall back to the conservative lag rather than assuming same-day
availability.

## Revised data vs. real-time (ALFRED) vintages

**This build evaluates against currently revised FRED data.** Any
evaluation result in this repository should be read as a
**"revised-data backtest"**, not a real-time backtest: it uses each
series' latest, fully-revised values, not what was actually published and
known at each historical date. `FredClient.get_series()` accepts
`realtime_start`/`realtime_end` so real ALFRED vintage queries are
possible ad hoc, but a full vintage panel (`fred_series.get_vintage_panel`)
is **not implemented** in this build -- it is a documented `NotImplementedError`
stub with a TODO explaining the (expensive, one-call-per-vintage-per-series)
implementation path. Do not present revised-data results as real-time
performance.

## Evaluation

- **Growth target:** `INDPRO` forward log change over 3 and 6 months
  (`ip_forward_3m`, `ip_forward_6m`); Up if positive, Down if negative.
- **Inflation target:** forward annualized core inflation (3/6 months)
  compared against trailing 12-month core inflation as of `t`; Up if the
  forward rate exceeds the trailing rate.
- **Metrics:** accuracy, balanced accuracy, Up recall, Down recall, Down
  precision, false alarm rate, confusion matrix, plus a simple
  cross-correlation-based average lead time diagnostic
  (`evaluation/lead_lag.py::average_lead_time_months`).
- **Baselines:** every model is reported alongside always-Up and
  always-Down naive baselines. Results are never hidden or re-thresholded
  to look better than the baseline -- if a model underperforms naive, that
  is reported as-is.

## Current limitations

- No full ALFRED real-time vintage panel (see above).
- `effective_date` uses calendar-day approximations, not an exchange
  trading calendar.
- Lead-time estimation is a simple cross-correlation heuristic, not a
  formal causality test.
- No asset returns, position sizing, or backtest P&L in this version --
  by design.

## Planned future layers

This regime engine is intended as the foundation for later additions,
not yet implemented:

- **Rate Overlay** -- yield curve / policy-rate-based risk adjustment.
- **Credit Overlay** -- credit spread confirmation of the growth regime.
- **Price Confirmation** -- price/momentum filter before acting on a
  regime signal.
- **Crisis Gate** -- a tail-risk override that can suspend normal regime
  logic during acute stress.
- **Growth Asset Basket weight optimization** -- `growth_basket` in
  `config/default.yaml` currently holds a fixed, untuned 50:50 S&P 500 /
  KOSPI 200 split; no basket-return computation or backtest exists yet.
