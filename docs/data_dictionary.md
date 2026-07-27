# Data Dictionary

## FRED series (auto-fetched, `config/default.yaml`)

| Series ID | Title | Frequency | Units | Role |
|---|---|---|---|---|
| `INDPRO` | Industrial Production Index | Monthly | Index | Growth target (evaluation) |
| `CFNAI` | Chicago Fed National Activity Index | Monthly | Index | Growth confirmation |
| `CFNAIMA3` | CFNAI, 3-Month Moving Average | Monthly | Index | Growth Model B input |
| `ICSA` | Initial Claims | Weekly | Number | Growth Models B/C/D input |
| `PERMIT` | New Private Housing Units Authorized by Building Permits | Monthly | Thousands | Growth Model B input |
| `USPHCI` | Coincident Economic Activity Index, US | Monthly | Index | Growth confirmation |
| `USALOLITOAASTSAM` | OECD CLI, United States, Amplitude Adjusted | Monthly | Index | Growth Model A input (US only -- no Korea CLI, no separate KR regime); **Primary growth axis** |
| `AMTMNO` | Manufacturers' New Orders: Total Manufacturing | Monthly | Millions of Dollars | Growth Model D input; **Secondary growth axis** (fully FRED-native replacement for ISM New Orders) |
| `CPILFESL` | Core CPI, Seasonally Adjusted | Monthly | Index | Inflation Models A/B/D input, inflation target; Model A is the **Primary and Secondary inflation axis** (frozen as of `40e43d7`; see Methodology re: current-environment classifier, not a strong short-term predictor) |
| `PCEPILFE` | Core PCE Price Index | Monthly | Index | Alternate inflation core series |
| `CPIAUCSL` | Headline CPI, Seasonally Adjusted | Monthly | Index | Diagnostic |
| `T10Y3M` | 10Y minus 3M Treasury | Daily | Percent | Diagnostic (yield curve) |
| `T10Y2Y` | 10Y minus 2Y Treasury | Daily | Percent | Diagnostic (yield curve) |
| `T5YIE` | 5-Year Breakeven Inflation Rate | Daily | Percent | Inflation Model B input |
| `MEDCPIM158SFRBCLE` | Median CPI (Cleveland Fed) | Monthly | Percent | Inflation Model C input -- underlying-inflation trend signal, **not** the Cleveland Fed Inflation Nowcast; **auxiliary/research only**, rejected as Primary's inflation axis after a challenger experiment (see Methodology) |
| `PALLFNFINDEXM` | Global Price Index of All Commodities (IMF) | Monthly | Index | Inflation Model D input (auxiliary/secondary, 2-signal only -- see Methodology) |

All series are validated against the live FRED API at fetch time
(`FredClient.get_series_metadata`); an invalid/retired series ID raises an
error rather than being silently dropped or substituted. Metadata
(source, title, frequency, units) for every fetched series is written to
`data/processed/series_metadata.json`.

`USSLIND` is intentionally **not** in this list -- see
`docs/methodology.md` for why it must not be treated as the Conference
Board LEI.

## Growth Asset Basket (`config/default.yaml`, not a FRED series)

`growth_basket.sp500_weight` / `growth_basket.kospi200_weight` (default
0.5 / 0.5) are structural allocation weights for when the growth axis is
"Up" -- S&P 500 and KOSPI 200 are both constituents of one basket, not
separate per-country regime picks. No price series for either asset is
fetched or used in this build; the weights are declared for the future
asset-allocation layer only (see `docs/methodology.md`).

## External CSV inputs (optional, `data/external/`)

None of these are committed to the repository; supply your own locally.
Their absence disables only the model(s) that depend on them. **None of
these feed the core Model A/B pairing** -- they only feed
legacy/auxiliary models kept for baseline comparison (see
`docs/methodology.md`, "Core model data source policy").

### `conference_board_lei.csv`

| Column | Required | Type | Notes |
|---|---|---|---|
| `date` | yes | date | Observation month |
| `lei` | yes | float | Conference Board LEI level |
| `release_date` | no | date | Actual publication date, used for availability |
| `vintage_date` | no | date | Data vintage, for real-time-aware use |

### `ism_new_orders.csv` / `ism_prices_paid.csv`

| Column | Required | Type | Notes |
|---|---|---|---|
| `date` | yes | date | Observation month |
| `value` | yes | float | ISM sub-index level |

## Derived / processed artifacts (`data/processed/`, git-ignored inputs
excluded but outputs are local-only build products, not committed)

| File | Produced by | Contents |
|---|---|---|
| `fred_wide.csv` | `fetch` | Wide date x series matrix of raw FRED values |
| `series_metadata.json` | `fetch` | Per-series source/title/frequency/units |
| `signals.csv` | `build-signals` | Growth/inflation labels + all regime columns |
| `regime_metadata.json` | `build-signals` | Maps each `regime_*` column to its growth/inflation model pair |
| `growth_model_b_detail.csv`, `growth_model_c_detail.csv`, `growth_model_d_detail.csv` | `build-signals` | Component-level growth model diagnostics |
| `inflation_model_a_detail.csv`, `inflation_model_b_detail.csv`, `inflation_model_c_detail.csv`, `inflation_model_d_detail.csv` | `build-signals` | Component-level inflation model diagnostics |
| `evaluation_report.json` | `evaluate` | Full per-model, per-horizon evaluation results vs. naive baselines |
| `regime_output_primary.csv`, `regime_output_secondary.csv` | `build-regime-output` | Standard asset-allocation output for the two frozen core models: `growth_score`, `growth_state`, `inflation_score`, `inflation_state`, `raw_regime`, `tradable_regime` (see Methodology, "Standard regime output") |
