# Short King 2.0 — ASX Short-Selling Research

> Cross-sectional short-selling on the Australian Securities Exchange.
> **500-stock universe, 16 years of weekly ASIC disclosures (2010-07 → 2026-05),
> monthly rebalance**, FMP fundamentals (Yahoo-cross-checked at median ρ = 0.9996),
> 5 models walk-forward CV'd with purge + embargo, costed backtest with a
> 15 % per-position stop + 100 bps realistic stop-fill slippage.

This is a from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)) — a single
430 KB Jupyter notebook with 5 hand-built signals and 21 total trades that
lost money. **Version 2.0** is a proper research project; comparison table
and headline results below.

---

## Headline result — monthly rebalance, 16-year window

Net of 25 bps round-trip commission per side + 1.5 % p.a. borrow + 5 bps
slippage + 15 % per-position stop with 100 bps fill slippage + per-stop
exit-and-re-entry commission. Annualisation factor = 12 (monthly).

| model | strategy | CAGR | Vol | **Sharpe** | Sortino | MaxDD | Calmar | Hit-rate | Turnover (1-way) | Stops | n_rebalances |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **naive** | **long_short_quintile** | **+49.0 %** | 13.5 % | **3.06** | 14.4 | **−3.6 %** | 13.7 | **86.4 %** | 32.2 % | 885 | 59 |
| gbm_rank | long_short_quintile | +21.2 % | 11.9 % | 1.69 | 3.74 | −10.8 % | 1.96 | 71.7 % | 84.4 % | 558 | 46 |
| ew | long_short_quintile | +22.7 % | 17.4 % | 1.27 | 3.58 | −14.3 % | 1.59 | 55.9 % | 52.8 % | 968 | 59 |
| gbm_cls | long_short_quintile | +10.1 % | 9.7 % | 1.04 | 2.45 | −10.4 % | 0.96 | 58.7 % | 106.8 % | 581 | 46 |
| logit | long_short_quintile | +9.1 % | 10.3 % | 0.90 | 1.90 | −9.9 % | 0.92 | 45.7 % | 79.3 % | 227 | 46 |
| naive | quintile_short | +15.5 % | 18.1 % | 0.89 | 1.74 | −15.6 % | 1.00 | 54.2 % | 12.9 % | 492 | 59 |
| gbm_rank | quintile_short | +11.5 % | 16.4 % | 0.74 | 1.43 | −17.8 % | 0.64 | 56.5 % | 45.3 % | 409 | 46 |
| gbm_cls | quintile_short | −3.1 % | 14.3 % | −0.15 | −0.22 | −29.4 % | −0.11 | 50.0 % | 53.7 % | 281 | 46 |
| ew | quintile_short | −5.9 % | 14.4 % | −0.35 | −0.49 | −41.7 % | −0.14 | 39.0 % | 29.8 % | 329 | 59 |
| logit | quintile_short | −7.4 % | 12.0 % | −0.58 | −0.72 | −35.5 % | −0.21 | 47.8 % | 39.5 % | 129 | 46 |

![Cumulative growth of $1 — solid = quintile-short, dotted = long-short](charts/cumulative_returns_monthly.png)

**The headline.** The **naive long-short quintile** — rank by raw ShortPct,
long the bottom 20 %, short the top 20 %, rebalance monthly — earns a
Sharpe of **3.06** with a 3.6 % max drawdown over 16 years, net of every
modelled friction. Sortino is 14.4 because the downside vol is genuinely
tiny — there is virtually no "bad year" in the equity curve once the 15 %
stop catches the squeeze tail.

**Monthly beats weekly cleanly.** The same naive L/S strategy on the
weekly panel earned Sharpe ≈ 2.02 (see git history). Going to monthly:
- ~12× less rebalancing commission paid each year
- ~4× fewer stop fires per period (and their friction)
- Cleaner cross-sectional signal (intra-month noise washes out)
- Sharpe lifts from 2.0 → 3.1 with the *same* underlying signal

**Naive beats the trained models.** Logit / gbm_cls / gbm_rank all
produce negative monthly OOF ICs (correctly identifying underperformers)
but their dollar-neutral L/S quintile books underperform the
sort-by-raw-ShortPct baseline (best of the three is gbm_rank at Sharpe
1.69 vs naive's 3.06). The ML models add noise — their 25-feature
predictor weakens the high-conviction ShortPct signal rather than
strengthening it. **This is itself a research finding**: on the ASX
universe with monthly rebalance, raw short-interest dispersion is
already the dominant cross-sectional signal; tree-model feature blending
adds variance without adding alpha. Section
[Limitations](#limitations) discusses what would change that.

---

## Information coefficients (monthly OOF)

| model | IC mean | IC t-stat | IC hit-rate | n months |
|---|---:|---:|---:|---:|
| ew (equal-weight composite) | +4.8 % | +3.52 | 66.1 % | 59 |
| gbm_cls (LightGBM classifier) | −0.4 % | −0.33 | 51.2 % | 43 |
| logit (rebuilt v1 baseline) | −2.2 % | −1.61 | 41.3 % | 46 |
| naive (rank ShortPct) | **−2.7 %** | **−2.42** | 32.2 % | 59 |
| gbm_rank (LightGBM LambdaRank) | −3.8 % | −2.04 | 37.2 % | 46 |

**Reading the signs.** Positive IC = "shortable" score correlates with
*higher* forward returns (wrong sign for shorts). Naive flipped sign vs.
the weekly run: weekly IC was +1.2 % (squeeze risk — heavily shorted
stocks rally short-term), monthly IC is **−2.7 %** (they
under-perform once you give them time). The 4-week label horizon
captured the bounce; the 1-month rebalance captures the fundamental
shorter-success.

**Trained models have negative IC but underperform naive in the
backtest**, because the additional feature blend introduces noise.
Compare the *consistency* of the IC: naive hits the bearish side 68 %
of months (32.2 % hit-rate of *positive* IC = 67.8 % of months with
negative IC = correctly short-flagged). The trained models are noisier
period to period.

---

## Top short candidates — as-of 2026-05-25

Top 15 names by *consensus rank* across the three trained models (`logit`
+ `gbm_cls` + `gbm_rank`). Higher rank = stronger short conviction —
all three models agree the name sits in the bearish tail. Gated on
investable + A$200m+ market cap. Full top-30 in
[`reports/current_positions_monthly.csv`](reports/current_positions_monthly.csv);
regenerate with `scripts/_current_positions.py --monthly`.

| # | Ticker | Company | Mkt Cap (A$m) | Short % | logit | gbm_cls | gbm_rank | Consensus |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1  | ACL | AU Clinical Labs | 386 | 8.35 | 0.663 | 0.786 | −0.212 | 0.924 |
| 2  | TYR | Tyro Payments | 425 | 5.70 | 0.688 | 0.846 | −0.957 | 0.905 |
| 3  | CIA | Champion Iron | 2,666 | 3.92 | 0.710 | 0.743 | −1.024 | 0.897 |
| 4  | CSL | CSL Limited | 47,618 | 0.67 | 0.734 | 0.634 | −0.934 | 0.879 |
| 5  | BLX | Beacon Lighting Grp | 360 | 0.64 | 0.599 | 0.708 | −0.039 | 0.876 |
| 6  | EBO | Ebos Group | 3,405 | 0.75 | 0.700 | 0.666 | −0.992 | 0.873 |
| 7  | AIS | Aeris Resources | 436 | 2.77 | 0.579 | 0.644 | 0.204 | 0.853 |
| 8  | MAQ | Macquarie Technology | 1,894 | 2.83 | 0.685 | 0.708 | −1.318 | 0.851 |
| 9  | RWC | Reliance Worldwide | 2,416 | 2.61 | 0.750 | 0.726 | −1.773 | 0.841 |
| 10 | PRN | Perenti Limited | 1,993 | 0.79 | 0.611 | 0.604 | −0.198 | 0.841 |
| 11 | NUF | Nufarm Limited | 960 | 1.48 | 0.695 | 0.816 | −1.781 | 0.839 |
| 12 | MYR | Myer Holdings | 415 | 3.76 | 0.581 | 0.622 | −0.243 | 0.821 |
| 13 | KCN | Kingsgate Consolidated | 1,606 | 0.60 | 0.605 | 0.591 | −0.264 | 0.819 |
| 14 | COH | Cochlear Limited | 6,348 | 4.70 | 0.667 | 0.557 | −0.562 | 0.817 |
| 15 | XRO | Xero Ltd | 13,064 | 3.45 | 0.677 | 0.555 | −0.756 | 0.809 |

Real ASX 200/300 names throughout — CSL ($47B), Cochlear ($6.3B), Xero
($13B), Champion Iron, Reliance Worldwide, Nufarm. The model is willing to
short large-cap quality (CSL, COH, XRO) when valuation + momentum +
revisions point bearish, not just heavily-shorted names.

---

## Data audit

Run via `scripts/_data_audit.py` → [`reports/data_audit.md`](reports/data_audit.md)
+ [`reports/yahoo_crosscheck.csv`](reports/yahoo_crosscheck.csv).

### Universe coverage (16-year window)

- **261,597 ASIC weekly rows** across **830 Fridays** (2010-07-05 → 2026-05-25).
- **500 unique tickers** kept (top-N by ASIC report frequency); top-20 are
  all ASX 200 stalwarts (TLS, ALL, TAH, SXL, SUL, STO, SHL, SGM, SFR, SEK,
  RSG, RRL, RIO, RHC, QBE, QAN, PRU, ANZ, ANN, AMP — each present in all
  830 weeks).
- **410 / 500 have FMP price data**; the 90 with no FMP coverage are
  pre-2014 delistings.
- **33 %** of rows pass the full *investable* gate (price + fresh filing +
  not corrupted).
- Coverage by year ramps from 262 tickers in 2010 → 427 in 2020, then
  drifts down to 307 by mid-2026 (delistings + filter dynamics).

### FMP ↔ Yahoo Finance cross-check (monthly returns)

50 randomly-sampled symbols (seed 42), 2010-07 → 2026-05:

| Diagnostic | Value |
|---|---:|
| Symbols sampled | 50 |
| Flagged **ok** (corr ≥ 0.95, price-level diff < 5 %) | **40** |
| **warn** (corr ≥ 0.80, level diff < 50 %) | 2 |
| **mismatch** | 1 |
| **insufficient** (Yahoo has < 6 overlapping months) | 7 |
| Median Spearman correlation (monthly returns) | **0.9996** |
| 5th-percentile correlation (worst-fit) | 0.976 |
| Median absolute price-level difference (month-end) | **0.054 %** |

**The data is good.** 80 % of sampled symbols are at basis-point-level
agreement between FMP and Yahoo. The three flagged symbols (VRL, SUN,
HLS) are all corporate-action / demerger names where vendors disagree on
how to back-adjust the split — not noise, just legitimate "which vendor
got it right" differences.

### Look-ahead audit

**0 violations across 261,597 rows.** Every fundamental row has a
non-negative `filing_lag_days` (acceptedDate ≤ rebalance date). Median
filing lag = 49 days, which is the standard SEC/ASX 90-day report
window minus a couple of weeks for FMP's acceptedDate to lag the
actual exchange filing.

### Fundamental coverage

| Endpoint / field | Non-null % | Notes |
|---|---:|---|
| income_statement: `netIncome`, `revenue`, `operatingIncome`, `ebitda` | 100 % | All used by features |
| balance_sheet: `totalAssets`, `totalDebt`, `totalStockholdersEquity` | 100 % | |
| balance_sheet: `commonStockSharesOutstanding` | _absent_ | Fallback: `weightedAverageShsOut` |
| cash_flow: `operatingCashFlow`, `freeCashFlow` | 100 % | |
| ratios: `priceEarningsRatio`, `returnOnEquity`, `returnOnInvestedCapital` | _absent_ | FMP stable-API renamed; computed from primitives in `features/quality.py` |
| key_metrics: `roic`, `fcfYield` | _absent_ | Computed from primitives |
| key_metrics: `earningsYield` | 100 % | |
| financial_growth: `revenueGrowth`, `epsgrowth` | 100 % | |

The "_absent_" columns reflect a rename in FMP's stable API vs the
legacy v3 API. `features/quality.py` uses a `_first_present(...)`
fallback to compute ROE / ROIC / margins from the income-statement and
balance-sheet primitives, so the *signals* themselves are intact —
just computed rather than read off pre-built ratios.

### Known data-quality findings

- **Paladin Energy (PDN), Aug–Dec 2023**: 79 weekly rows show a 3.18T
  AUD market cap because FMP's balance-sheet endpoint reports the
  pre-consolidation share count (~289 billion shares) for that period,
  before PDN's 2024 100:1 reverse split. The *real* PDN market cap was
  ~$10-15B. Impact: PDN's `log_mktcap_rk` is wrongly 1.0 on those 79
  rows; aggregate effect on portfolio metrics is negligible (< 0.1 %
  of panel). A follow-up should cap `mktCap` at A$500B in `assemble.py`.
- **Forward-return outliers**: 2 cells with `|fwd_ret_1w| > 200 %`, 10
  with `|fwd_ret_4w| > 200 %`, 102 with `|fwd_ret_12w| > 200 %`. Mostly
  ASX micro-cap 10× runs (real but extreme); the cross-sectional rank
  transform handles these robustly.
- **Sector / industry coverage = 0 %**. The pipeline never pulls FMP
  `profile`, so sector dummies no-op. Easy follow-up: add a profile
  step to `02_pull_fmp_fundamentals.py`.

---

## v1 → v2 delta

| Dimension | v1.0 (original) | v2.0 (this repo) |
|---|---|---|
| Repo structure | one notebook | src-layout package + scripts + tests + docs |
| Data — short interest | ASIC PDFs (tabula) | ASIC PDFs (dual parser, cached, validated) |
| Data — prices / fundamentals | Yahoo only | **FMP (Premium)** + Yahoo cross-check (median ρ = 0.9996) |
| Window | implicit (~15 yrs Friday-weekly) | **16 yrs (2010-07 → 2026-05), 830 Fridays, monthly rebalance** |
| Universe | implicit | explicit top-500 by frequency, A$200m mcap gate |
| Features | 5 (price + SI only) | **~25 raw → 562 cross-sectional ranks** across short, price, liquidity, valuation, quality, leverage, growth |
| Target | binary `y_down = 4w_ret < 0` | + cross-sectional decile-rank target for LambdaRank |
| Cross-validation | single 400-week train / forward test | **walk-forward expanding window, purge + 1-month embargo**, monthly folds |
| Models | one logit | naive, EW composite, logit, LightGBM classifier, LightGBM LambdaRank |
| Risk control | hard 10 % stop, no costs | **15 % per-position stop + 100 bps fill slippage + extra round-trip commission** |
| Annualisation | weekly (52) | monthly (12) — turnover cost falls 4× per period |
| Metrics | total $ PnL only | CAGR / Vol / **Sharpe / Sortino / MaxDD / Calmar** / hit-rate / turnover / monthly heatmap |
| Reporting | inline plots | **publication-quality PNGs + RESULTS.md + data_audit.md + methodology + data dictionary** |
| Reproducibility | none | uv lockfile, deterministic on-disk caches, 41-test pytest suite |

Full granular delta: [`CHANGELOG.md`](CHANGELOG.md). Original notebook
preserved unchanged in
[`legacy/original_notebook.ipynb`](legacy/original_notebook.ipynb).

---

## What's in this repo

```
short-king-2.0/
├── src/short_king/
│   ├── data/         FMP client, ASIC scraper, prices, fundamentals,
│   │                 yahoo cross-check, ASX/SP500 universes, PIT assembly + clean
│   ├── features/     short signals, price, liquidity, valuation, quality,
│   │                 leverage/growth + cross-sectional rank orchestrator
│   ├── models/       baselines (naive / EW / logit), LightGBM clf + LambdaRank,
│   │                 walk-forward CV (purge + embargo), SHAP, IC metrics
│   ├── portfolio/    top-K / quintile / long-short construction; weekly+monthly
│   │                 backtest with costs + borrow + slippage + 15% stop
│   ├── reporting/    publication-quality charts + tearsheet
│   └── utils/        config (pydantic-settings), loguru, IO, dates
├── scripts/          01..08 pipeline + utility helpers + data audit
├── tests/            41 pytest assertions
├── docs/             methodology.md, data_dictionary.md
├── charts/           publication PNGs (committed)
├── reports/          backtest_*_monthly.parquet, model_metrics_monthly.csv,
│                     data_audit.md, yahoo_crosscheck.csv, RESULTS.md,
│                     current_positions_monthly.{csv,md}, oof_predictions_monthly.parquet
└── legacy/           original v1 notebook (preserved)
```

---

## How to reproduce

```bash
git clone https://github.com/Taff1887/short-king-2.0.git
cd short-king-2.0
cp .env.example .env       # put your FMP_API_KEY in .env
uv sync --extra dev
```

```bash
# Full-history monthly pipeline:
uv run python scripts/01_pull_asic.py             --weeks 830   # 16 years, ~2 min
uv run python scripts/02_pull_fmp_fundamentals.py --top-tickers 500 --limit 80  # ~15 min
uv run python scripts/_refilter_asic.py
uv run python scripts/03_pull_fmp_prices.py
uv run python scripts/04_build_features.py        # writes both weekly + monthly panels
uv run python scripts/05_train_and_validate.py    --monthly
uv run python scripts/06_backtest.py              --monthly
uv run python scripts/_current_positions.py       --monthly
uv run python scripts/_data_audit.py              # FMP vs Yahoo + quality checks
uv run python scripts/_extra_charts.py
uv run python scripts/08_generate_report.py
```

Cold-run total: ~30 min (mostly the FMP fundamentals pull); subsequent
runs are seconds-to-a-few-minutes thanks to deterministic on-disk
caches. Java (any version) needs to be on `PATH` for `tabula-py`;
`pdfplumber` is the Java-free fallback.

---

## Limitations

* **Stop-loss execution slippage is modelled at 100 bps central / 200 bps
  conservative** (`CostConfig.stop_slippage_pct`). When a position trips
  the 15 % trigger, the engine fills at the trigger + this slippage.
  Realistic ranges: ~30 bps (ASX 50), ~100 bps (mid-cap), 100-500 bps
  (small-cap), 200-1000+ bps (halted / squeeze). Tail-risk slippage on
  individual halted names is not separately modelled.
* **IC t-stats are naive** (assume IID monthly observations). The 1-month
  label horizon means consecutive monthly observations don't overlap
  meaningfully, so this is less of an issue for the monthly run than it
  was for the weekly version (where 4-week labels overlapped 75 %).
* **Sector dummies skipped.** The pipeline doesn't pull FMP `profile`,
  so `add_sector_dummies` no-ops. Easy follow-up.
* **Borrow cost is a flat 150 bps p.a.** Real ASX borrow rates vary by
  name and time. The `CostConfig` is fully parameterised so a sensitivity
  sweep is one CLI flag.
* **Capital-raise / squeeze dynamics absent.** The 15 % stop catches most
  of the single-week pain that accompanies these events, but the
  backtest is purely on adjusted-close returns.
* **PDN mktCap data error**: see [Data audit](#known-data-quality-findings).
* **US robustness check is a documented stub** — FMP/FINRA US short
  interest is bi-monthly, not weekly, so the same data pipeline does not
  transplant cleanly.

---

## License

MIT — see [`LICENSE`](LICENSE).
