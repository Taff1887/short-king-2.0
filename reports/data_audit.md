# Data audit — short-king-2.0

_Generated 2026-06-04 23:10 UTC_

## 1. Universe coverage
- **asic_long**: 262,251 rows | 500 unique tickers | 833 dates | window 2010-06-14 → 2026-05-25
- **prices_long**: 1,327,654 rows | 356 symbols | 4376 unique daily dates
- **master_clean**: 262,251 rows | 192,165 investable (73.3%)
- **features**: 262,251 rows × 1178 cols

### Coverage by calendar year
| Year | Rows | Unique tickers | Weeks |
|---|---:|---:|---:|
| 2010 | 6,593 | 262 | 29 |
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
| 1 | TLS | 833 |
| 2 | ALL | 833 |
| 3 | TAH | 833 |
| 4 | SXL | 833 |
| 5 | SUL | 833 |
| 6 | STO | 833 |
| 7 | SHL | 833 |
| 8 | SGM | 833 |
| 9 | SFR | 833 |
| 10 | SEK | 833 |
| 11 | RSG | 833 |
| 12 | RRL | 833 |
| 13 | RIO | 833 |
| 14 | RHC | 833 |
| 15 | QBE | 833 |
| 16 | QAN | 833 |
| 17 | PRU | 833 |
| 18 | ANZ | 833 |
| 19 | ANN | 833 |
| 20 | AMP | 833 |

## 2. Data quality checks

**Look-ahead audit**: 0 violations across 262,251 rows (max violation = 0 days).

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

- p10: 100  p25: 297  p50: 918  p75: 3,254  p90: 10,506
- min: 0  max: 268,473

- **fwd_ret_1w**: n=192,221, mean=+0.0022, std=0.0716, |ret|>200% outliers=2, max=2.37, min=-0.97
- **fwd_ret_4w**: n=191,310, mean=+0.0094, std=0.1467, |ret|>200% outliers=27, max=3.59, min=-0.97
- **fwd_ret_12w**: n=188,908, mean=+0.0277, std=0.2643, |ret|>200% outliers=244, max=9.38, min=-0.97

## 3. FMP vs Yahoo Finance cross-check (monthly returns)

Sample 50 ASX tickers (random, seed 42). For each: fetch Yahoo prices, compute month-end returns, compare to FMP's `adjClose` resampled to month-end. Reports Spearman correlation + median absolute difference of the month-end price levels.

- **Coverage**: 50 symbols sampled, 50 ok, 0 warn, 0 mismatch, 0 insufficient.
- **Median Spearman correlation** (monthly returns): 1.0000
- **5th-percentile correlation** (worst-fit): 0.9989
- **Median absolute price-level diff** (month-end): 0.000%

Five worst-fit symbols:

| Symbol | n_months | corr | median_abs_diff_% | flag |
|---|---:|---:|---:|---|
| EWC.AX | 206 | 0.990 | 0.00 | ok |
| TUA.AX | 73 | 0.998 | 0.00 | ok |
| AFG.AX | 134 | 0.999 | 0.00 | ok |
| LOV.AX | 139 | 0.999 | 0.00 | ok |
| BAP.AX | 147 | 0.999 | 0.00 | ok |
