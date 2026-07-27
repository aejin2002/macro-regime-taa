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

Exactly as specified (`backtest.regime_allocations` in
`config/default.yaml`), all rows sum to 100%:

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
