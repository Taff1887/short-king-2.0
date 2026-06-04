# Data audit — short-king-2.0

_Generated 2026-06-04 06:43 UTC_

## 1. Universe coverage
- **asic_long**: 261,597 rows | 500 unique tickers | 830 dates | window 2010-07-05 → 2026-05-25
- **prices_long**: 476,557 rows | 410 symbols | 1420 unique daily dates
- **master_clean**: 261,597 rows | 86,492 investable (33.1%)
- **features**: 261,597 rows × 1155 cols

### Coverage by calendar year
| Year | Rows | Unique tickers | Weeks |
|---|---:|---:|---:|
| 2010 | 5,939 | 262 | 26 |
| 2011 | 13,065 | 284 | 52 |
| 2012 | 13,549 | 293 | 53 |
| 2013 | 14,174 | 312 | 52 |
| 2014 | 15,450 | 333 | 52 |
| 2015 | 16,181 | 370 | 52 |
| 2016 | 16,123 | 384 | 52 |
| 2017 | 16,429 | 385 | 52 |
| 2018 | 17,201 | 395 | 53 |
| 2019 | 17,791 | 405 | 52 |
| 2020 | 18,851 | 427 | 52 |
| 2021 | 19,756 | 421 | 52 |
| 2022 | 19,232 | 402 | 52 |
| 2023 | 18,461 | 380 | 52 |
| 2024 | 17,433 | 366 | 53 |
| 2025 | 15,913 | 338 | 52 |
| 2026 | 6,049 | 307 | 21 |

### Top-20 tickers by ASIC report frequency
| Rank | Ticker | Weeks present |
|---:|---|---:|
| 1 | TLS | 830 |
| 2 | ALL | 830 |
| 3 | TAH | 830 |
| 4 | SXL | 830 |
| 5 | SUL | 830 |
| 6 | STO | 830 |
| 7 | SHL | 830 |
| 8 | SGM | 830 |
| 9 | SFR | 830 |
| 10 | SEK | 830 |
| 11 | RSG | 830 |
| 12 | RRL | 830 |
| 13 | RIO | 830 |
| 14 | RHC | 830 |
| 15 | QBE | 830 |
| 16 | QAN | 830 |
| 17 | PRU | 830 |
| 18 | ANZ | 830 |
| 19 | ANN | 830 |
| 20 | AMP | 830 |

## 2. Data quality checks

**Look-ahead audit**: 0 violations across 261,597 rows (max violation = 0 days).

**Extreme weekly returns** (|ret_1w| > 150%): 0 cells (0 symbols).

**FMP fundamental column non-null %** (selected key fields):

| Endpoint | Field | Non-null % |
|---|---|---:|
| income_statement | netIncome | 100.0 |
| income_statement | revenue | 100.0 |
| income_statement | operatingIncome | 100.0 |
| income_statement | ebitda | 100.0 |
| balance_sheet | totalAssets | 100.0 |
| balance_sheet | totalDebt | 100.0 |
| balance_sheet | totalStockholdersEquity | 100.0 |
| balance_sheet | commonStockSharesOutstanding | _absent_ |
| cash_flow | operatingCashFlow | 100.0 |
| cash_flow | freeCashFlow | 100.0 |
| ratios | priceEarningsRatio | _absent_ |
| ratios | returnOnEquity | _absent_ |
| ratios | returnOnInvestedCapital | _absent_ |
| key_metrics | roic | _absent_ |
| key_metrics | fcfYield | _absent_ |
| key_metrics | earningsYield | 100.0 |
| financial_growth | revenueGrowth | 100.0 |
| financial_growth | epsgrowth | 100.0 |

**Sector / industry coverage** on the cleaned panel: sector 0.0%, industry 0.0%. (Profile pull is not yet wired into the pipeline — known limitation.)

**Market cap distribution** (A$m) on the cleaned panel:

- p10: 111  p25: 318  p50: 989  p75: 3,535  p90: 11,851
- min: 0  max: 3,182,459

- **fwd_ret_1w**: n=86,475, mean=+0.0018, std=0.0914, |ret|>200% outliers=2, max=14.65, min=-0.99
- **fwd_ret_4w**: n=85,794, mean=+0.0064, std=0.1789, |ret|>200% outliers=10, max=16.34, min=-0.94
- **fwd_ret_12w**: n=83,048, mean=+0.0167, std=0.2926, |ret|>200% outliers=102, max=14.37, min=-0.95

## 3. FMP vs Yahoo Finance cross-check (monthly returns)

Sample 50 ASX tickers (random, seed 42). For each: fetch Yahoo prices, compute month-end returns, compare to FMP's `adjClose` resampled to month-end. Reports Spearman correlation + median absolute difference of the month-end price levels.

- **Coverage**: 50 symbols sampled, 40 ok, 2 warn, 1 mismatch, 7 insufficient.
- **Median Spearman correlation** (monthly returns): 0.9996
- **5th-percentile correlation** (worst-fit): 0.9760
- **Median absolute price-level diff** (month-end): 0.054%

Five worst-fit symbols:

| Symbol | n_months | corr | median_abs_diff_% | flag |
|---|---:|---:|---:|---|
| VRL.AX | 61 | 0.010 | 100.00 | mismatch |
| SUN.AX | 61 | 0.890 | 14.62 | warn |
| HLS.AX | 61 | 0.974 | 19.44 | warn |
| AFG.AX | 61 | 0.996 | 1.81 | ok |
| TOE.AX | 51 | 0.997 | 0.00 | ok |
