# Methodology

## Scope

This project studies the **Growth x Inflation regime layer** ("Layer 1")
of a tactical asset allocation (TAA) process. It does not implement asset
selection, position sizing, or ETF backtesting. The deliverable is a
research engine for classifying and evaluating macro regimes, not a
trading system.

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

**Model C -- Simple Two-Signal.** Only active when
`data/external/ism_new_orders.csv` is present:

```
growth_score = (z(change_3m(ISM New Orders)) - z(change_3m(Initial Claims))) / 2
```

If the CSV is absent, this model returns `None` and is skipped everywhere
downstream (signals, regimes, evaluation) -- the other two growth models
are unaffected.

## Inflation models

**Model A -- Realized Core Inflation Momentum.**

```
core_3m_annualized = (core_t / core_t-3) ** 4 - 1
core_12m           = core_t / core_t-12 - 1
signal_raw         = core_3m_annualized - core_12m
```

Default core series: `CPILFESL` (Core CPI, SA). Up if `signal_raw > 0`.

**Model B -- Leading Inflation Composite.**

```
inflation_score = mean(z(change_3m(ISM Prices Paid)), z(change_3m(T5YIE)), z(core momentum))
```

If `data/external/ism_prices_paid.csv` is absent, a 2-signal variant
(`inflation_score_no_ism`, dropping the ISM term) is computed and used
instead; both columns are always present in the detail output so it is
clear which one was actually active.

**Model C -- Cleveland Fed Inflation Nowcast: disabled by default.**
The Cleveland Fed publishes a *current* nowcast on its website, but no
official, reproducible, downloadable **historical vintage** file was
located at build time. Scraping today's displayed value and treating it
as if it were the historical nowcast at each past date would be a severe
form of look-ahead bias (the nowcast is revised continuously and today's
page only shows the latest vintage). Until an official historical vintage
source is identified, this model:

- ships as a stub (`signals/inflation.py::cleveland_nowcast_signal`),
- documents the exact input schema
  (`forecast_date, target_month, measure, nowcast_value, vintage_date`)
  expected at `data/external/cleveland_fed_inflation_nowcast.csv`,
- produces **no backtest output** even if a file is supplied, until the
  classification rule is implemented and validated against the nowcast's
  disclosed methodology.

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
- Cleveland Fed Inflation Nowcast backtest is disabled pending an official
  historical vintage source.
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
