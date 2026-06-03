# Data Dictionary

Authoritative column-by-column reference for every parquet emitted by the
short-king-2.0 pipeline. Each section lists the on-disk path, the row grain
(natural key), the upstream source, and the typed schema. Units are stated
explicitly — in particular **`ShortPct` is on a 0–100 scale (not 0–1)**, market
caps are in **AUD**, and ratio columns such as `ev_ebitda` are unitless
multiples.

All parquet files are written via `short_king.utils.io.write_parquet` (snappy
compression, pyarrow engine). Dates are `datetime64[ns]` (tz-naive, normalised
to midnight) unless otherwise noted.

---

## 1. `data/processed/asic_long.parquet`

- **Row grain:** `(Date, Ticker)` — one row per ticker per ASIC weekly snapshot.
- **Source:** ASIC daily aggregate short-position PDF
  (`download.asic.gov.au/short-selling/RR{YYYYMMDD}-001-SSDailyAggShortPos.pdf`),
  parsed by `short_king.data.asic`. Restricted to ORDINARY shares via
  `filter_ordinary_only`.

| name            | dtype           | units         | description                                                             | source / formula                              |
|-----------------|-----------------|---------------|-------------------------------------------------------------------------|-----------------------------------------------|
| Date            | datetime64[ns]  | calendar date | As-of date of the short position (release date minus 4 business days). | `asof_date(ReleaseDate)`                      |
| ReleaseDate     | datetime64[ns]  | calendar date | Date ASIC published the PDF.                                            | ASIC filename `RR{YYYYMMDD}`                  |
| Ticker          | string          | —             | ASX product code (1-6 alphanumerics, optional dot).                     | ASIC "Product Code" column                    |
| Company         | string          | —             | Issuer description as printed on the ASIC PDF.                          | ASIC "Product" column                         |
| ShortPositions  | Int64 (nullable)| shares        | Reported short positions (shares).                                      | ASIC "Reported Short Position"                |
| TotalVolume     | Int64 (nullable)| shares        | Total product in issue (shares on register).                            | ASIC "Total Product in Issue"                 |
| ShortPct        | float64         | percent (0–100)| Short interest as % of shares outstanding. **0–100 scale, NOT 0–1.**   | ASIC "% Reported as Short" or `100 * ShortPositions / TotalVolume` when missing |

---

## 2. `data/processed/prices_long.parquet`

- **Row grain:** `(symbol, date)` — daily bars per ASX symbol.
- **Source:** FMP `historical-price-eod/dividend-adjusted` via
  `short_king.data.prices.fetch_many_adjusted`. Total-return adjusted (splits
  and dividends).

| name      | dtype           | units                  | description                                                  | source / formula |
|-----------|-----------------|------------------------|--------------------------------------------------------------|------------------|
| symbol    | string          | —                      | FMP symbol, ASX names suffixed `.AX` (e.g. `BHP.AX`).        | FMP              |
| date      | datetime64[ns]  | calendar date          | Trading date (midnight-normalised, tz-naive).                | FMP              |
| adjClose  | float64         | total-return index (AUD basis) | Split- and dividend-adjusted close. Use for returns, momentum, vol — **not** for level-based ratios. | FMP `adjClose`   |
| volume    | Int64 (nullable)| shares                 | Daily traded share volume.                                   | FMP `volume`     |

---

## 3. `data/raw/fmp_raw/income_statement.parquet`

- **Row grain:** `(symbol, fiscalYear, period)` — one row per quarterly filing.
- **Source:** FMP `stable/income-statement` via
  `short_king.data.fundamentals.fetch_many` (`period="quarter"`).
- **Notes:** Carries the public-availability timestamps (`filingDate`,
  `acceptedDate`) used as the point-in-time anchor downstream. All money
  columns are in the issuer's reported currency (AUD for ASX names).

| name                                 | dtype           | units                  | description                                                          | source / formula |
|--------------------------------------|-----------------|------------------------|----------------------------------------------------------------------|------------------|
| symbol                               | string          | —                      | FMP symbol (e.g. `BHP.AX`).                                          | FMP              |
| date                                 | datetime64[ns]  | calendar date          | Fiscal period-end date.                                              | FMP `date`       |
| filingDate                           | datetime64[ns]  | calendar date          | Date the filing was lodged.                                          | FMP `filingDate` |
| acceptedDate                         | datetime64[ns]  | timestamp              | Public-availability timestamp — anchor for PIT lagging.              | FMP `acceptedDate` |
| fiscalYear                           | string          | year (e.g. "2023")     | Issuer's fiscal year label.                                          | FMP              |
| period                               | string          | `Q1`/`Q2`/`Q3`/`Q4`/`FY`| Fiscal period label.                                                 | FMP              |
| reportedCurrency                     | string          | ISO 4217               | Currency of monetary columns (typically `AUD`).                      | FMP              |
| revenue                              | float64         | currency               | Total revenue.                                                       | FMP              |
| costOfRevenue                        | float64         | currency               | Cost of revenue.                                                     | FMP              |
| grossProfit                          | float64         | currency               | Revenue − cost of revenue.                                           | FMP              |
| researchAndDevelopmentExpenses       | float64         | currency               | R&D expense.                                                         | FMP              |
| sellingGeneralAndAdministrativeExpenses | float64      | currency               | SG&A expense.                                                        | FMP              |
| operatingExpenses                    | float64         | currency               | Total operating expense.                                             | FMP              |
| ebitda                               | float64         | currency               | Earnings before interest, tax, depreciation & amortisation.          | FMP              |
| operatingIncome                      | float64         | currency               | Operating income (EBIT proxy).                                       | FMP              |
| interestExpense                      | float64         | currency               | Interest expense (positive number).                                  | FMP              |
| incomeBeforeTax                      | float64         | currency               | Pre-tax income.                                                      | FMP              |
| incomeTaxExpense                     | float64         | currency               | Tax expense.                                                         | FMP              |
| netIncome                            | float64         | currency               | Net income attributable to the parent.                               | FMP              |
| eps                                  | float64         | currency / share       | Basic earnings per share.                                            | FMP              |
| epsDiluted                           | float64         | currency / share       | Diluted EPS.                                                         | FMP              |
| weightedAverageShsOut                | float64         | shares                 | Weighted-average basic shares.                                       | FMP              |
| weightedAverageShsOutDil             | float64         | shares                 | Weighted-average diluted shares.                                     | FMP              |

(FMP may include a few additional fields — e.g. `depreciationAndAmortization`,
`ebitdaratio`, `interestIncome` — that flow through unchanged. Vendor
`link`/`finalLink`/`cik`/`fillingDate` are dropped by the assembler.)

---

## 4. `data/raw/fmp_raw/balance_sheet.parquet`

- **Row grain:** `(symbol, fiscalYear, period)` — one row per quarterly filing.
- **Source:** FMP `stable/balance-sheet-statement`. Carries `filingDate`/
  `acceptedDate`. Money columns in `reportedCurrency`.

| name                                | dtype           | units    | description                                                      | source / formula |
|-------------------------------------|-----------------|----------|------------------------------------------------------------------|------------------|
| symbol                              | string          | —        | FMP symbol.                                                      | FMP              |
| date                                | datetime64[ns]  | date     | Period-end date.                                                 | FMP              |
| filingDate                          | datetime64[ns]  | date     | Filing lodgement date.                                           | FMP              |
| acceptedDate                        | datetime64[ns]  | timestamp| Public-availability timestamp.                                   | FMP              |
| fiscalYear                          | string          | —        | Fiscal year label.                                               | FMP              |
| period                              | string          | —        | Fiscal period label.                                             | FMP              |
| reportedCurrency                    | string          | ISO 4217 | Reporting currency.                                              | FMP              |
| cashAndCashEquivalents              | float64         | currency | Cash + equivalents.                                              | FMP              |
| cashAndShortTermInvestments         | float64         | currency | Cash + short-term investments.                                   | FMP              |
| netReceivables                      | float64         | currency | Receivables net of allowance.                                    | FMP              |
| inventory                           | float64         | currency | Inventory.                                                       | FMP              |
| totalCurrentAssets                  | float64         | currency | Current assets.                                                  | FMP              |
| propertyPlantEquipmentNet           | float64         | currency | PP&E net of depreciation.                                        | FMP              |
| goodwill                            | float64         | currency | Goodwill.                                                        | FMP              |
| intangibleAssets                    | float64         | currency | Intangible assets.                                               | FMP              |
| totalAssets                         | float64         | currency | Total assets.                                                    | FMP              |
| accountPayables                     | float64         | currency | Accounts payable.                                                | FMP              |
| shortTermDebt                       | float64         | currency | Short-term debt.                                                 | FMP              |
| totalCurrentLiabilities             | float64         | currency | Current liabilities.                                             | FMP              |
| longTermDebt                        | float64         | currency | Long-term debt.                                                  | FMP              |
| totalDebt                           | float64         | currency | Total interest-bearing debt.                                     | FMP              |
| netDebt                             | float64         | currency | Total debt − cash (FMP-published).                               | FMP              |
| totalLiabilities                    | float64         | currency | Total liabilities.                                               | FMP              |
| totalStockholdersEquity             | float64         | currency | Common equity attributable to parent.                            | FMP              |
| totalEquity                         | float64         | currency | Total equity (incl. minority interest).                          | FMP              |
| commonStockSharesOutstanding        | float64         | shares   | Shares outstanding as reported on the balance sheet.             | FMP              |

---

## 5. `data/raw/fmp_raw/cash_flow.parquet`

- **Row grain:** `(symbol, fiscalYear, period)`.
- **Source:** FMP `stable/cash-flow-statement`. Carries `filingDate`/
  `acceptedDate`.

| name                                   | dtype           | units    | description                                              | source / formula |
|----------------------------------------|-----------------|----------|----------------------------------------------------------|------------------|
| symbol                                 | string          | —        | FMP symbol.                                              | FMP              |
| date                                   | datetime64[ns]  | date     | Period-end date.                                         | FMP              |
| filingDate                             | datetime64[ns]  | date     | Filing lodgement date.                                   | FMP              |
| acceptedDate                           | datetime64[ns]  | timestamp| Public-availability timestamp.                           | FMP              |
| fiscalYear                             | string          | —        | Fiscal year.                                             | FMP              |
| period                                 | string          | —        | Fiscal period.                                           | FMP              |
| reportedCurrency                       | string          | ISO 4217 | Reporting currency.                                      | FMP              |
| netIncome                              | float64         | currency | Net income (repeated for convenience).                   | FMP              |
| depreciationAndAmortization            | float64         | currency | D&A non-cash charge.                                     | FMP              |
| stockBasedCompensation                 | float64         | currency | Stock-based comp non-cash charge.                        | FMP              |
| changeInWorkingCapital                 | float64         | currency | ∆ working capital.                                       | FMP              |
| operatingCashFlow                      | float64         | currency | Cash from operations.                                    | FMP              |
| netCashProvidedByOperatingActivities   | float64         | currency | Cash from operations (alt label).                        | FMP              |
| capitalExpenditure                     | float64         | currency | Capex (negative number).                                 | FMP              |
| acquisitionsNet                        | float64         | currency | Net acquisition spend.                                   | FMP              |
| netCashUsedForInvestingActivities      | float64         | currency | Net cash from investing.                                 | FMP              |
| debtRepayment                          | float64         | currency | Repayment of debt.                                       | FMP              |
| commonStockIssued                      | float64         | currency | Equity issuance proceeds.                                | FMP              |
| commonStockRepurchased                 | float64         | currency | Buybacks (negative number).                              | FMP              |
| dividendsPaid                          | float64         | currency | Dividends paid (negative number).                        | FMP              |
| netCashUsedProvidedByFinancingActivities| float64        | currency | Net cash from financing.                                 | FMP              |
| freeCashFlow                           | float64         | currency | FMP-published FCF (OCF − capex).                         | FMP              |

---

## 6. `data/raw/fmp_raw/ratios.parquet`

- **Row grain:** `(symbol, fiscalYear, period)`.
- **Source:** FMP `stable/ratios`. Period-end-anchored — no public-availability
  timestamp; the assembler joins on `(symbol, fiscalYear, period)` so this
  table inherits the parent income filing's `acceptedDate`.

| name                          | dtype          | units                     | description                                       | source / formula |
|-------------------------------|----------------|---------------------------|---------------------------------------------------|------------------|
| symbol                        | string         | —                         | FMP symbol.                                       | FMP              |
| date                          | datetime64[ns] | date                      | Period-end date.                                  | FMP              |
| fiscalYear / period           | string         | —                         | Fiscal period label.                              | FMP              |
| currentRatio                  | float64        | multiple                  | Current assets / current liabilities.             | FMP              |
| quickRatio                    | float64        | multiple                  | (CA − inventory) / CL.                            | FMP              |
| debtToEquityRatio             | float64        | multiple                  | Total debt / equity.                              | FMP              |
| debtToAssetsRatio             | float64        | multiple                  | Total debt / total assets.                        | FMP              |
| interestCoverageRatio         | float64        | multiple                  | EBIT / interest expense.                          | FMP              |
| returnOnAssets                | float64        | fraction (0–1)            | Net income / total assets (decimal — e.g. 0.08 = 8%). | FMP          |
| returnOnEquity                | float64        | fraction (0–1)            | Net income / equity (decimal).                    | FMP              |
| returnOnInvestedCapital       | float64        | fraction (0–1)            | NOPAT / invested capital (decimal).               | FMP              |
| grossProfitMargin             | float64        | fraction (0–1)            | Gross profit / revenue.                           | FMP              |
| operatingProfitMargin         | float64        | fraction (0–1)            | Operating income / revenue.                       | FMP              |
| netProfitMargin               | float64        | fraction (0–1)            | Net income / revenue.                             | FMP              |
| priceEarningsRatio            | float64        | multiple                  | Market cap / net income.                          | FMP              |
| priceToBookRatio              | float64        | multiple                  | Market cap / book value.                          | FMP              |
| priceToSalesRatio             | float64        | multiple                  | Market cap / revenue.                             | FMP              |
| evToEBITDA                    | float64        | multiple                  | EV / EBITDA.                                      | FMP              |
| earningsYield                 | float64        | fraction (0–1)            | Net income / market cap (decimal).                | FMP              |
| freeCashFlowYield             | float64        | fraction (0–1)            | FCF / market cap (decimal).                       | FMP              |

(FMP exposes several dozen additional ratios; the assembler prefixes every
column with `ratios_` when joining into the master panel so the namespace stays
disjoint.)

---

## 7. `data/raw/fmp_raw/key_metrics.parquet`

- **Row grain:** `(symbol, fiscalYear, period)`.
- **Source:** FMP `stable/key-metrics`. Period-end-anchored.

| name                    | dtype          | units                  | description                                                  | source / formula |
|-------------------------|----------------|------------------------|--------------------------------------------------------------|------------------|
| symbol                  | string         | —                      | FMP symbol.                                                  | FMP              |
| date                    | datetime64[ns] | date                   | Period-end date.                                             | FMP              |
| fiscalYear / period     | string         | —                      | Fiscal period label.                                         | FMP              |
| marketCap               | float64        | currency               | Market cap at period-end (vendor-computed).                  | FMP              |
| enterpriseValue         | float64        | currency               | EV at period-end (vendor-computed).                          | FMP              |
| peRatio                 | float64        | multiple               | P/E at period-end.                                           | FMP              |
| priceToSalesRatio       | float64        | multiple               | P/S at period-end.                                           | FMP              |
| pocfratio               | float64        | multiple               | Price / operating cash flow.                                 | FMP              |
| pfcfRatio               | float64        | multiple               | Price / free cash flow.                                      | FMP              |
| pbRatio                 | float64        | multiple               | P/B at period-end.                                           | FMP              |
| enterpriseValueOverEBITDA | float64      | multiple               | EV / EBITDA.                                                 | FMP              |
| evToSales               | float64        | multiple               | EV / revenue.                                                | FMP              |
| evToOperatingCashFlow   | float64        | multiple               | EV / OCF.                                                    | FMP              |
| evToFreeCashFlow        | float64        | multiple               | EV / FCF.                                                    | FMP              |
| earningsYield           | float64        | fraction (0–1)         | E/P (decimal).                                               | FMP              |
| freeCashFlowYield       | float64        | fraction (0–1)         | FCF/P (decimal).                                             | FMP              |
| debtToEquity            | float64        | multiple               | Total debt / equity.                                         | FMP              |
| debtToAssets            | float64        | multiple               | Total debt / total assets.                                   | FMP              |
| netDebtToEBITDA         | float64        | multiple               | Net debt / EBITDA.                                           | FMP              |
| currentRatio            | float64        | multiple               | CA / CL.                                                     | FMP              |
| interestCoverage        | float64        | multiple               | EBIT / interest.                                             | FMP              |
| incomeQuality           | float64        | multiple               | OCF / net income.                                            | FMP              |
| dividendYield           | float64        | fraction (0–1)         | TTM dividend / price.                                        | FMP              |
| payoutRatio             | float64        | fraction (0–1)         | Dividends paid / net income.                                 | FMP              |
| revenuePerShare         | float64        | currency / share       | Revenue per diluted share.                                   | FMP              |
| netIncomePerShare       | float64        | currency / share       | EPS.                                                         | FMP              |
| bookValuePerShare       | float64        | currency / share       | Book value per share.                                        | FMP              |
| tangibleBookValuePerShare | float64      | currency / share       | Tangible book per share.                                     | FMP              |

---

## 8. `data/raw/fmp_raw/enterprise_values.parquet`

- **Row grain:** `(symbol, fiscalYear, period)`.
- **Source:** FMP `stable/enterprise-values`. Period-end-anchored.

| name                    | dtype          | units    | description                                              | source / formula |
|-------------------------|----------------|----------|----------------------------------------------------------|------------------|
| symbol                  | string         | —        | FMP symbol.                                              | FMP              |
| date                    | datetime64[ns] | date     | Period-end date.                                         | FMP              |
| fiscalYear / period     | string         | —        | Fiscal period label.                                     | FMP              |
| stockPrice              | float64        | currency | Period-end close used by FMP for cap calc.               | FMP              |
| numberOfShares          | float64        | shares   | Shares outstanding at period-end.                        | FMP              |
| marketCapitalization    | float64        | currency | Period-end market cap (= stockPrice × shares).           | FMP              |
| minusCashAndCashEquivalents | float64    | currency | Cash netted off (positive number).                       | FMP              |
| addTotalDebt            | float64        | currency | Total debt added on.                                     | FMP              |
| enterpriseValue         | float64        | currency | EV = market cap + debt − cash.                           | FMP              |

---

## 9. `data/raw/fmp_raw/financial_growth.parquet`

- **Row grain:** `(symbol, fiscalYear, period)`.
- **Source:** FMP `stable/financial-growth`. Period-end-anchored; all values
  are YoY (or TTM) growth rates expressed as decimals (e.g. 0.15 = 15%).

| name                            | dtype          | units            | description                                       | source / formula |
|---------------------------------|----------------|------------------|---------------------------------------------------|------------------|
| symbol                          | string         | —                | FMP symbol.                                       | FMP              |
| date                            | datetime64[ns] | date             | Period-end date.                                  | FMP              |
| fiscalYear / period             | string         | —                | Fiscal period label.                              | FMP              |
| revenueGrowth                   | float64        | fraction         | YoY revenue growth (decimal).                     | FMP              |
| grossProfitGrowth               | float64        | fraction         | YoY gross-profit growth.                          | FMP              |
| ebitgrowth                      | float64        | fraction         | YoY EBIT growth.                                  | FMP              |
| operatingIncomeGrowth           | float64        | fraction         | YoY operating-income growth.                      | FMP              |
| netIncomeGrowth                 | float64        | fraction         | YoY net-income growth.                            | FMP              |
| epsgrowth                       | float64        | fraction         | YoY EPS growth.                                   | FMP              |
| epsdilutedGrowth                | float64        | fraction         | YoY diluted EPS growth.                           | FMP              |
| operatingCashFlowGrowth         | float64        | fraction         | YoY OCF growth.                                   | FMP              |
| freeCashFlowGrowth              | float64        | fraction         | YoY FCF growth.                                   | FMP              |
| assetGrowth                     | float64        | fraction         | YoY total-asset growth (when published).          | FMP              |
| bookValueperShareGrowth         | float64        | fraction         | YoY book-value-per-share growth.                  | FMP              |
| debtGrowth                      | float64        | fraction         | YoY total-debt growth.                            | FMP              |

---

## 10. `data/processed/master_pit.parquet`

- **Row grain:** `(Date, Ticker)` — one row per ticker per ASIC Friday rebalance.
- **Source:** Derived. Output of
  `short_king.data.assemble.assemble_pit_panel`. Joins `asic_long`,
  `prices_long` (as-of), the seven FMP fundamentals tables (as-of by
  `acceptedDate` on the income filing, then `(symbol, fiscalYear, period)` for
  the derived endpoints), and computes forward-return labels.
- **PIT guarantee:** every column is knowable strictly as-of `Date`, **except**
  the explicit `fwd_ret_*` label columns.

### 10a. Declared core columns

| name              | dtype          | units                  | description                                                                                       | source / formula |
|-------------------|----------------|------------------------|---------------------------------------------------------------------------------------------------|------------------|
| Date              | datetime64[ns] | date                   | ASIC as-of date (Friday-anchored).                                                                | ASIC             |
| ReleaseDate       | datetime64[ns] | date                   | ASIC PDF release date.                                                                            | ASIC             |
| Ticker            | string         | —                      | ASX product code (no suffix).                                                                     | ASIC             |
| Symbol            | string         | —                      | FMP symbol = `Ticker + ".AX"`.                                                                    | derived          |
| Company           | string         | —                      | Issuer description from ASIC.                                                                     | ASIC             |
| sector            | string         | —                      | FMP profile sector (static snapshot).                                                             | FMP `profile`    |
| industry          | string         | —                      | FMP profile industry.                                                                             | FMP `profile`    |
| ShortPositions    | Int64          | shares                 | Reported short positions.                                                                         | ASIC             |
| TotalVolume       | Int64          | shares                 | Total product in issue.                                                                           | ASIC             |
| ShortPct          | float64        | percent (0–100)        | Short interest %, **0–100 scale**.                                                                | ASIC             |
| adjClose          | float64        | total-return index (AUD)| Last total-return-adjusted close on or before `Date` (≤ 7-day tolerance).                         | FMP prices       |
| volume            | Int64          | shares                 | Daily share volume on that bar.                                                                   | FMP prices       |
| mktCap            | float64        | AUD                    | PIT market cap = `sharesOutstanding × adjClose`.                                                  | derived          |
| sharesOutstanding | float64        | shares                 | Common shares outstanding from the most recent PIT balance sheet.                                  | FMP `balance_sheet_commonStockSharesOutstanding` |
| fwd_ret_1w        | float64        | fraction (decimal)     | **Label.** Forward 1-week total return: `adjClose[Date+1w] / adjClose[Date] − 1`.                | derived (look-ahead by design — label only) |
| fwd_ret_4w        | float64        | fraction (decimal)     | **Label.** Forward 4-week total return.                                                           | derived          |
| fwd_ret_12w       | float64        | fraction (decimal)     | **Label.** Forward 12-week total return.                                                          | derived          |

### 10b. Fundamentals payload columns

Every non-key column from the seven FMP endpoints is carried through prefixed
with the endpoint name (so namespaces stay disjoint). Concretely:

| prefix                    | source endpoint                       | example columns                                                  |
|---------------------------|---------------------------------------|------------------------------------------------------------------|
| `income_statement_*`      | `income_statement.parquet`            | `income_statement_revenue`, `income_statement_ebitda`, `income_statement_netIncome` |
| `balance_sheet_*`         | `balance_sheet.parquet`               | `balance_sheet_totalAssets`, `balance_sheet_totalDebt`, `balance_sheet_commonStockSharesOutstanding` |
| `cash_flow_*`             | `cash_flow.parquet`                   | `cash_flow_operatingCashFlow`, `cash_flow_freeCashFlow`          |
| `ratios_*`                | `ratios.parquet`                      | `ratios_debtToEquityRatio`, `ratios_currentRatio`, `ratios_priceEarningsRatio` |
| `key_metrics_*`           | `key_metrics.parquet`                 | `key_metrics_marketCap`, `key_metrics_enterpriseValue`, `key_metrics_evToEBITDA` |
| `enterprise_values_*`     | `enterprise_values.parquet`           | `enterprise_values_enterpriseValue`, `enterprise_values_marketCapitalization` |
| `financial_growth_*`      | `financial_growth.parquet`            | `financial_growth_revenueGrowth`, `financial_growth_epsgrowth`   |

Units and meaning match the source-table entries above. The
`link`/`finalLink`/`cik`/`calendarYear`/`fillingDate`/`reportedCurrency`
columns are dropped during the merge.

### 10c. Trailing PIT / quality columns

| name                    | dtype          | units    | description                                                                                  | source / formula |
|-------------------------|----------------|----------|----------------------------------------------------------------------------------------------|------------------|
| fiscalYear              | string         | —        | Fiscal year of the as-of filing.                                                             | FMP income filing |
| period                  | string         | —        | Fiscal period (Q1/Q2/Q3/Q4/FY) of the as-of filing.                                          | FMP income filing |
| acceptedDate            | datetime64[ns] | timestamp| Public-availability timestamp of the as-of filing.                                           | FMP income filing |
| period_end              | datetime64[ns] | date     | Fiscal period-end of the as-of filing.                                                       | FMP income filing |
| filing_lag_days         | Int64          | days     | `Date − acceptedDate` in calendar days (data freshness).                                     | derived          |
| filing_stale_quarters   | float64        | quarters | `filing_lag_days / 92` — staleness in approximate quarters.                                  | derived          |
| investable              | bool           | —        | Eligibility flag: has price, filing ≤ 4 quarters stale, `mktCap ≥ A$200m`.                   | derived          |

---

## 11. `data/processed/master_clean.parquet`

- **Row grain:** `(Date, Ticker)` — same as `master_pit.parquet` but restricted
  to `investable == True` rows and with any per-pipeline data-cleaning
  (de-duplication, dtype coercion, outlier flagging) applied.
- **Source:** Derived from `master_pit.parquet`.
- **Schema:** Identical column contract to `master_pit.parquet` (sections 10a /
  10b / 10c). The `investable` column is preserved (all `True`) for downstream
  consumers that expect the field.

---

## 12. `data/processed/features.parquet`

- **Row grain:** `(Date, Ticker)` — one row per ticker per weekly rebalance,
  derived from `master_clean.parquet`.
- **Source:** Derived. Output of
  `short_king.features.build.build_feature_panel`, which stitches per-family
  modules (short, price, liquidity, valuation, quality, leverage_growth) and
  appends a cross-sectional percentile rank (`<col>_rk`) for every numeric raw
  feature plus one-hot sector dummies (`sec_<sector>`).

### 12a. Identifier and label passthroughs

| name              | dtype          | units              | description                                                              | source |
|-------------------|----------------|--------------------|--------------------------------------------------------------------------|--------|
| Date / Ticker / Symbol / Company / sector / industry | various | — | Carried through from `master_clean.parquet`.                                | passthrough |
| adjClose / volume / mktCap / sharesOutstanding | various | — | Carried through (level units as before).                                     | passthrough |
| ShortPositions / TotalVolume / ShortPct        | various | — | ASIC inputs carried through.                                                 | passthrough |
| fwd_ret_1w / fwd_ret_4w / fwd_ret_12w          | float64 | fraction (decimal) | Forward-return **labels** — never ranked, never dummied.                     | passthrough |
| investable                                     | bool    | —                  | Eligibility flag carried through.                                            | passthrough |
| filing_lag_days / filing_stale_quarters / acceptedDate / period_end / fiscalYear / period | various | — | PIT metadata. | passthrough |

### 12b. Raw feature columns (added by family modules)

All raw features are oriented so **higher = better fundamental quality / more
expensive / more momentum** (the short-side scoring layer flips signs as
needed). Rolling windows are per-ticker and look-back only.

| name                | dtype   | units                  | description                                                                                                     | source / formula                  |
|---------------------|---------|------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------|
| mom_1w              | float64 | fraction               | 1-week momentum: `adjClose.pct_change(1)`.                                                                      | `features.price.momentum`         |
| mom_4w              | float64 | fraction               | 4-week momentum.                                                                                                | `features.price.momentum`         |
| mom_12w             | float64 | fraction               | 12-week momentum.                                                                                               | `features.price.momentum`         |
| mom_26w             | float64 | fraction               | 26-week momentum.                                                                                               | `features.price.momentum`         |
| mom_52w             | float64 | fraction               | 52-week momentum.                                                                                               | `features.price.momentum`         |
| mom_12w_skip1       | float64 | fraction               | Skip-1-week 12-week momentum: `adjClose[t-1]/adjClose[t-13] − 1` (removes microstructure reversal).             | `features.price.momentum_skip_1w` |
| vol_4w              | float64 | fraction (std)         | Std of weekly returns over a trailing 4-week window per ticker.                                                 | `features.price.realised_vol`     |
| vol_12w             | float64 | fraction (std)         | Std of weekly returns over a trailing 12-week window per ticker.                                                | `features.price.realised_vol`     |
| drawdown_52w        | float64 | fraction (≤ 0)         | Current drawdown vs trailing 52-week high: `adjClose / running_max − 1`.                                        | `features.price.max_drawdown_52w` |
| beta_52w            | float64 | unitless               | Rolling 52-week beta of weekly returns to `mkt_ret_1w` (if joined).                                             | `features.price.beta_to_market`   |
| adv_aud             | float64 | AUD / day              | Trailing 20-week mean of `close × adv_shares` — average daily dollar volume.                                    | `features.liquidity.adv_aud`      |
| turnover_pct        | float64 | fraction               | Trailing 20-week mean of `adv_shares / sharesOutstanding`.                                                      | `features.liquidity.turnover_pct` |
| amihud              | float64 | 1 / AUD (×1e-?)        | Amihud illiquidity: trailing 20-week mean of `|ret_1w| / dollar_volume`. Higher = less liquid.                  | `features.liquidity.amihud_illiquidity` |
| log_mktcap          | float64 | log(AUD)               | `log(marketCap)` — size feature.                                                                                | `features.liquidity.log_mktcap`   |
| ev_ebitda           | float64 | multiple               | EV / EBITDA (prefers FMP column, else `enterpriseValue / ebitda`).                                              | `features.valuation.ev_ebitda`    |
| pe                  | float64 | multiple               | P/E (prefers FMP column, else `marketCap / netIncome`).                                                         | `features.valuation.pe`           |
| pb                  | float64 | multiple               | Price / book.                                                                                                   | `features.valuation.pb`           |
| ps                  | float64 | multiple               | Price / sales.                                                                                                  | `features.valuation.ps`           |
| fcf_yield           | float64 | fraction (decimal)     | FCF / market cap.                                                                                               | `features.valuation.fcf_yield`    |
| earnings_yield      | float64 | fraction (decimal)     | Net income / market cap (signed — loss-makers stay negative).                                                   | `features.valuation.earnings_yield`|
| sales_yield         | float64 | fraction (decimal)     | Revenue / market cap.                                                                                           | `features.valuation.sales_yield`  |
| roe                 | float64 | fraction (decimal)     | Return on equity.                                                                                               | `features.quality.roe`            |
| roic                | float64 | fraction (decimal)     | Return on invested capital.                                                                                     | `features.quality.roic`           |
| roa                 | float64 | fraction (decimal)     | Return on assets.                                                                                               | `features.quality.roa`            |
| gross_margin        | float64 | fraction (decimal)     | Gross profit / revenue.                                                                                         | `features.quality.gross_margin`   |
| operating_margin    | float64 | fraction (decimal)     | Operating income / revenue.                                                                                     | `features.quality.operating_margin`|
| net_margin          | float64 | fraction (decimal)     | Net income / revenue.                                                                                           | `features.quality.net_margin`     |
| accruals            | float64 | fraction (decimal)     | Sign-flipped Sloan accruals: `−(netIncome − OCF) / totalAssets` — higher = better quality (less accrual distortion). | `features.quality.accruals`  |
| cfo_to_ni           | float64 | multiple               | Operating cash flow / net income (NaN where NI ≤ 0).                                                            | `features.quality.cfo_to_ni`      |
| debt_equity         | float64 | multiple               | Total debt / equity.                                                                                            | `features.leverage_growth.debt_equity` |
| net_debt_to_ebitda  | float64 | years (multiple)       | Net debt / EBITDA.                                                                                              | `features.leverage_growth.net_debt_to_ebitda` |
| interest_coverage   | float64 | multiple               | EBIT / interest expense.                                                                                        | `features.leverage_growth.interest_coverage` |
| current_ratio       | float64 | multiple               | Current assets / current liabilities.                                                                           | `features.leverage_growth.current_ratio` |
| revenue_growth_yoy  | float64 | fraction (decimal)     | YoY revenue growth.                                                                                             | `features.leverage_growth.revenue_growth_yoy` |
| eps_growth_yoy      | float64 | fraction (decimal)     | YoY diluted-EPS growth.                                                                                         | `features.leverage_growth.eps_growth_yoy` |
| asset_growth_yoy    | float64 | fraction (decimal)     | YoY total-asset growth (Cooper-Gulen-Schill asset-growth anomaly).                                              | `features.leverage_growth.asset_growth_yoy` |

(The `short` family adds short-interest features — typically `short_pct`,
`short_pct_chg_4w`, `short_pct_z_26w`, `days_to_cover` — orientations match
the convention "higher = more crowded short / more bearish".)

### 12c. Cross-sectional rank columns

For every raw numeric feature column listed in 12b (and any extras emitted by
the `short` family) a within-date percentile rank is appended:

| name                | dtype   | units             | description                                                                                    | source / formula                  |
|---------------------|---------|-------------------|------------------------------------------------------------------------------------------------|-----------------------------------|
| `<feature>_rk`      | float64 | fraction (0–1)    | Percentile rank within the row's `Date`, computed via `df.groupby(Date).rank(pct=True)`. NaNs stay NaN. The `_rk` columns are the actual model inputs. | `features.build.cross_sectional_rank` |

### 12d. Sector dummy columns

| name                | dtype | units | description                                                       | source / formula                  |
|---------------------|-------|-------|-------------------------------------------------------------------|-----------------------------------|
| `sec_<sector>`      | int8  | 0/1   | One-hot indicator from `pd.get_dummies(sector, prefix='sec')`.    | `features.build._sector_dummies`  |

---

## 13. `reports/oof_predictions.parquet`

- **Row grain:** one row per `(Date, original-panel-row)` test-fold prediction.
- **Source:** Derived. Output of
  `short_king.models.walk_forward.fit_predict_walkforward` — out-of-fold
  predictions from the expanding-window walk-forward CV with purge + embargo.

| name    | dtype          | units                  | description                                                                                                            | source / formula                  |
|---------|----------------|------------------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| date    | datetime64[ns] | date                   | Rebalance date of the test row.                                                                                        | passthrough                       |
| idx     | int64          | row index              | Positional index back into the feature matrix `X` that was passed to `fit_predict_walkforward` — rejoin metadata via this. | passthrough                   |
| y_true  | float64        | fraction (decimal) or 0/1 | Realised target (e.g. `fwd_ret_4w` for regression, or 1{ret < threshold} for the crash-classification head).            | passthrough                       |
| y_pred  | float64        | score                  | Model output: `predict_proba(...)[:, -1]` for classifiers, `decision_function` if no proba, else `predict`.            | `walk_forward._emit_predictions`  |
| fold    | int64          | —                      | Fold ordinal (0-indexed) of the walk-forward split that produced this row.                                             | `walk_forward.fit_predict_walkforward` |

---

## 14. `reports/backtest_*.parquet`

One parquet per portfolio construction recipe (e.g. `backtest_top_k_short.parquet`,
`backtest_decile_short.parquet`, `backtest_long_short_decile.parquet`). Each
contains two row-grains depending on the variant — typically a long-format
target-book panel and an aggregated equity curve. The shared contract is:

### 14a. Target-book rows (one per `(Date, Ticker)` position)

- **Row grain:** `(Date, Ticker)`.
- **Source:** Derived. Output of `short_king.portfolio.construct.top_k_short`,
  `decile_short` or `long_short_decile`.

| name    | dtype          | units               | description                                                                                                | source / formula                  |
|---------|----------------|---------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------|
| Date    | datetime64[ns] | date                | Rebalance date.                                                                                            | passthrough                       |
| Ticker  | string         | —                   | ASX product code held on `Date`.                                                                           | passthrough                       |
| weight  | float64        | signed weight       | Target position weight. Per-leg gross sums to 1.0 (long) / −1.0 (short).                                   | `portfolio.construct`             |
| side    | int8           | −1 / +1             | +1 long, −1 short. Redundant with sign(weight) but explicit for filtering.                                 | `portfolio.construct`             |

### 14b. Per-date P&L / equity-curve rows (if emitted)

- **Row grain:** `Date` (one row per rebalance).
- **Source:** Derived from the target-book joined to `fwd_ret_*` labels.

| name           | dtype          | units               | description                                                                                                | source / formula                  |
|----------------|----------------|---------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------|
| Date           | datetime64[ns] | date                | Rebalance date.                                                                                            | passthrough                       |
| n_positions    | int64          | —                   | Number of names held on `Date`.                                                                            | derived                           |
| gross_exposure | float64        | fraction            | `sum(|weight|)` — gross leverage (1.0 for single-leg, 2.0 for long-short).                                 | derived                           |
| net_exposure   | float64        | fraction            | `sum(weight)` — net market exposure (−1.0 short-only, 0.0 dollar-neutral).                                 | derived                           |
| period_return  | float64        | fraction (decimal)  | Realised per-period return: `sum(weight × fwd_ret_<horizon>w)`. Horizon matches the construction config.   | derived                           |
| cum_return     | float64        | fraction (decimal)  | Compounded equity curve: `cumprod(1 + period_return) − 1`.                                                  | derived                           |

Variants may add columns for transaction-cost-adjusted return
(`period_return_net`, `turnover`), per-leg attribution
(`long_return` / `short_return`), or benchmark-relative metrics
(`active_return`) — units follow the same fraction-decimal convention.
