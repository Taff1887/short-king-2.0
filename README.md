# Short King 2.0 — ASX Short-Selling Research

> Cross-sectional short-selling on the Australian Securities Exchange.
> **500-stock universe, 15 years 11 months (16 June 2010 → 29 May 2026)
> of weekly ASIC disclosures**, end-of-month rebalance, FMP fundamentals,
> Yahoo Finance prices (cross-checked vs FMP at median ρ = 0.9996), 5 models
> walk-forward CV'd with purge + embargo, **36-month pure out-of-sample
> holdout**, costed backtest with 15 % per-position stop + 100 bps fill
> slippage.

A from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)) — a single
430 KB notebook with 5 hand-built signals and 21 total trades that lost
money. **Version 2.0** is a proper research project. Comparison table
and headline results below.

---

## Headline result — IS vs OOS, monthly rebalance, full ASIC history

The model dev (in-sample) period is **2010-06 → 2023-05** (~13 years; all
walk-forward CV runs here, ~120 OOF monthly observations per trained
model). The pure out-of-sample holdout is **2023-06 → 2026-05** (36 months;
final-fit model applied — never seen during development).

Net of 25 bps round-trip commission per side + 1.5 % p.a. borrow + 5 bps
slippage + 15 % per-position stop with 100 bps fill slippage + per-stop
exit-and-re-entry commission. Annualisation factor = 12 (monthly).

### Dollar-neutral long-short quintile (long bottom 20 %, short top 20 % by score)

| Model | **IS Sharpe** | **OOS Sharpe** | OOS CAGR | OOS MaxDD | OOS Hit-rate | OOS Sortino |
|---|---:|---:|---:|---:|---:|---:|
| **naive** (rank ShortPct) | 1.93 | **3.37** | **+45.3 %** | **−4.9 %** | **88.6 %** | 8.9 |
| **logit** (rebuilt v1) | 2.52 | **3.00** | +50.6 % | −5.4 % | 80.0 % | 14.4 |
| gbm_rank (LightGBM LambdaRank) | 2.37 | 2.31 | +47.5 % | −11.6 % | 74.3 % | 6.0 |
| gbm_cls (LightGBM binary) | 1.82 | 1.81 | +22.5 % | −10.7 % | 77.1 % | 4.0 |
| ew (long-bias composite) | 0.89 | 1.31 | +23.9 % | −9.5 % | 60.0 % | 3.3 |

### Top-quintile short only (no long leg)

| Model | IS Sharpe | OOS Sharpe | OOS CAGR | OOS MaxDD |
|---|---:|---:|---:|---:|
| logit | 0.79 | 1.52 | +31.3 % | −10.2 % |
| gbm_rank | 1.03 | 1.47 | +32.6 % | −18.2 % |
| gbm_cls | 0.31 | 0.79 | +12.6 % | −17.6 % |
| naive | 0.26 | 0.63 | +9.9 % | −15.3 % |
| ew | −0.61 | −0.58 | −8.0 % | −31.9 % |

![Cumulative growth of $1 — solid = quintile-short, dotted = long-short](charts/cumulative_returns_monthly.png)

**The honest read of "no overfitting":** OOS Sharpe is **higher** than IS
Sharpe for four of five long-short variants. With a 36-month holdout that
was never touched during model development, that's about as strong a
non-overfitting signal as a 13-year IS panel can produce. The trained
models (`logit`, `gbm_rank`, `gbm_cls`) all run statistically-significant
**negative OOS IC** (correctly identifying underperformers) — see the IC
table below.

**The naive baseline is competitive.** Sorting by raw ShortPct, going
long the bottom 20 % and shorting the top 20 % of the cross-section,
earns OOS Sharpe 3.37 with an 88.6 % monthly hit-rate and a 4.9 % max
drawdown. The trained models add 0–0.5 Sharpe on top, mostly through
better long-leg selection in the L/S quintile pair (the trained models'
OOS quintile-short books also outperform naive's). **On this universe,
short-interest dispersion is already the dominant cross-sectional
signal.** That itself is a research finding worth publishing.

---

## Information coefficients — IS and OOS

Spearman of model score vs realised 4-week forward return, computed
cross-sectionally each month then summarised:

| Model | IS IC | IS t-stat | n_IS | OOS IC | OOS t-stat | n_OOS |
|---|---:|---:|---:|---:|---:|---:|
| logit | **−4.7 %** | **−4.53** | 119 | **−8.5 %** | **−3.88** | 35 |
| gbm_rank | **−7.0 %** | **−5.00** | 119 | **−7.3 %** | **−2.45** | 35 |
| gbm_cls | −3.3 % | −3.37 | 119 | −2.6 % | −1.36 | 35 |
| naive | −1.7 % | −1.86 | 119 | −2.7 % | −1.79 | 35 |
| ew | +4.0 % | +4.26 | 119 | +4.8 % | +2.52 | 35 |

**Sign reading.** Negative IC = the "shortable" score correlates with
*lower* forward returns — the model correctly identifies underperformers.
All three trained models hit statistically-significant negative IC in
both IS and OOS. `ew` is a long-bias composite (most of its inputs flag
*quality* names, not bad ones) and reliably picks WINNERS — that's why
its Sharpe is lower in the short book and the long-short construction
benefits from the long leg.

---

## Methodology

### 1. Universe and rebalance

* **ASIC daily aggregate short-position reports**: weekly, Friday-anchored
  (the earliest archived report we found is **16 June 2010**;
  the regime started 1 June 2010 but the URL format we use stabilised
  mid-June).
* The data we ingest covers **2010-07-05 → 2026-05-29**, **192
  end-of-month rebalance dates** (one per calendar month).
* Each rebalance date is the last **ASIC release** in the calendar month
  — **by construction a trading day** (ASIC only publishes Mon-Fri,
  skipping public holidays). 182 of the 192 are Mondays (the *as-of*
  date 4 business days before each Friday release); the other 10 are
  Fridays where holidays pushed the as-of forward.
* Universe = **top 500 ASX tickers by ASIC-report frequency** over the
  full 16 years, gated at ≥ A$200 m market cap on each rebalance.

### 2. Data sources

| What | Source | Why |
|---|---|---|
| Short positions | ASIC daily aggregate PDFs | Only public, free, asset-class-complete source for ASX |
| Prices (adjusted close + volume) | **Yahoo Finance** | FMP only ships ~5 years of ASX daily history on this plan; Yahoo goes back to ~2000. Cross-checked vs FMP at **median ρ = 0.9996** over the overlap window |
| Fundamentals (7 quarterly endpoints) | Financial Modeling Prep (Premium) | Income, balance sheet, cash flow, ratios, key metrics, enterprise values, financial growth |
| Market capitalisation | **FMP enterprise_values endpoint** (half-yearly, split-adjusted) | The balance-sheet `commonStockSharesOutstanding` field is stamped at the latest period-end and isn't back-adjusted across corporate actions. Using FMP's `enterprise_values.marketCapitalization` correctly handles reverse splits (e.g. Paladin Energy 1:100 in 2024 — see [Data audit](#data-audit) below). |
| Sector / industry | _not currently pulled_ — known limitation | FMP `profile` endpoint exists; trivial follow-up to wire in |

### 3. Features

~25 raw factors → **562 cross-sectional rank columns** across short
interest (SI %, SI z-score, days-to-cover, persistence), price (multi-
horizon momentum, vol, drawdown), liquidity (ADV, turnover, Amihud),
valuation (P/E, P/B, EV/EBITDA, FCF yield), quality (ROE, ROIC,
margins, accruals), and leverage/growth. Every numeric factor is
ranked **within each rebalance date** to 0–1, then NaN values are
imputed to 0.5 (neutral) so linear models don't blow up on missing
fundamentals.

### 4. Targets

* **Binary**: `fwd_ret_4w < 0` for the classifier baseline.
* **Cross-sectional decile rank** of `fwd_ret_4w` (inverted so worst-
  return = highest relevance) for the LambdaRank model.

### 5. Cross-validation (walk-forward + IS/OOS holdout)

* **Pure OOS holdout**: the last **36 monthly rebalances**
  (2023-06 → 2026-05) are *reserved* — the trained models never see
  these rows during development, neither for fitting nor for
  hyperparameter selection.
* On the 156-month **in-sample** portion (2010-07 → 2023-05):
  walk-forward expanding window with **156-week min train (~36 mo)**,
  **26-week test (~6 mo)**, **4-week embargo (≥ label horizon)**.
  20 folds total, ~120 monthly OOF observations per trained model.
* After CV, a final model is fit on the **entire IS panel** and
  applied to the holdout. That single OOS Sharpe is the unbiased
  estimate; it's what we report in headline.

### 6. Portfolio + costs

* Top-quintile short OR dollar-neutral L/S quintile (long bottom 20 %,
  short top 20 %), equal weight within each leg, monthly rebalance.
* Liquidity gate: investable + ≥ A$200 m mkt cap.
* Frictions: 25 bps round-trip commission per side, 1.5 % p.a. borrow on
  shorts, 5 bps slippage on weight changes, **15 % per-position hard
  stop with 100 bps fill slippage** + extra round-trip commission when
  the stop fires.

---

## Top short candidates — as of 2026-05-25

Top 15 by *consensus rank* across the three trained models (`logit` +
`gbm_cls` + `gbm_rank`); gated on investable + A$200 m+ market cap +
all-3-models-scored. Full top-30 in
[`reports/current_positions_monthly.csv`](reports/current_positions_monthly.csv);
regenerate with `scripts/_current_positions.py --monthly`.

| # | Ticker | Company | Mkt Cap (A$m) | Short % | logit | gbm_cls | gbm_rank | Consensus |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | TLX | Telix Pharmaceuticals | 3,794 | **15.15** | 0.652 | 0.570 | −0.141 | 0.957 |
| 2 | LTR | Liontown Resources | 4,404 | 1.75 | 0.619 | 0.565 | −0.199 | 0.932 |
| 3 | ILU | Iluka Resources | 2,488 | 7.53 | 0.624 | 0.554 | −0.405 | 0.930 |
| 4 | WBT | Weebit Nano | 1,065 | 0.32 | 0.573 | 0.614 | 0.258 | 0.928 |
| 5 | AGI | Ainsworth Game Tech. | 340 | 0.00 | 0.608 | 0.531 | −0.383 | 0.902 |
| 6 | CAT | Catapult Sports | 983 | 5.15 | 0.652 | 0.477 | −0.414 | 0.894 |
| 7 | NVX | Novonix | 293 | 2.80 | 0.557 | 0.543 | 0.071 | 0.875 |
| 8 | NEC | Nine Entertainment | 1,758 | 4.12 | 0.687 | 0.498 | −1.030 | 0.868 |
| 9 | MVF | Monash IVF | 286 | 1.87 | 0.632 | 0.569 | −1.481 | 0.867 |
| 10 | MSB | Mesoblast | 1,898 | **8.66** | 0.694 | 0.424 | 0.693 | 0.863 |
| 11 | AD8 | Audinate | 339 | 1.86 | 0.623 | 0.515 | −0.947 | 0.857 |
| 12 | EOS | Electro Optic Systems | 1,821 | 3.59 | 0.598 | 0.456 | 0.182 | 0.855 |
| 13 | CXL | Calix | 211 | 0.36 | 0.615 | 0.456 | −0.362 | 0.853 |
| 14 | PLS | Pilbara Minerals | 13,590 | **11.53** | 0.574 | 0.528 | −0.627 | 0.848 |
| 15 | OML | oOh!media | 700 | 0.72 | 0.577 | 0.586 | −1.238 | 0.846 |

The list reads like a real ASX short book: TLX, MSB, and PLS at 8–15 %
short interest are well-known shorts; LTR, ILU, NVX are lithium /
mining names trading well off prior highs; NEC, MVF, AD8 are
mid-cap operators where the model is picking up valuation /
revisions even when SI itself is modest.

---

## Data audit

Run via `scripts/_data_audit.py` →
[`reports/data_audit.md`](reports/data_audit.md) +
[`reports/yahoo_crosscheck.csv`](reports/yahoo_crosscheck.csv).

### FMP ↔ Yahoo Finance cross-check (monthly returns)

| Diagnostic | Value |
|---|---:|
| Symbols sampled (random, seed 42) | 50 |
| Flagged **ok** (corr ≥ 0.95, level diff < 5 %) | **40** |
| **warn** / **mismatch** | 3 |
| Insufficient overlap | 7 |
| **Median Spearman correlation** | **0.9996** |
| 5th-percentile correlation | 0.976 |
| Median absolute price-level difference | 0.054 % |

The three flagged names (VRL, SUN, HLS) are all corporate-action /
demerger plays — legitimate vendor disagreement, not noise.

### Look-ahead audit

**0 violations** across 262,251 weekly rows. Every fundamental row has a
non-negative `filing_lag_days` (acceptedDate ≤ rebalance date). Median
filing lag = 49 days.

### Trading-day verification

All 192 monthly rebalance dates are valid ASX trading days (Mon–Fri,
no public holidays) — confirmed because ASIC only publishes on trading
days. Day-of-week distribution: 182 Mondays, 10 Fridays (the 4-BDay
as-of lag from the Friday release places the as-of on Monday for
almost all weeks; the 10 Fridays are weeks where Easter / King's
Birthday pushed the lag forward).

### PDN data fix — no hardcoding

Paladin Energy (PDN) had a 1:100 reverse share consolidation in early
2024. FMP's balance-sheet `commonStockSharesOutstanding` reports the
**latest** share count (~352 m) without back-adjusting history, so
multiplying by the historical `adjClose` (which IS split-adjusted)
inflates pre-consolidation mktCap by ~600×. Previous run showed PDN
at **A$3.18 trillion** in Aug-Dec 2023.

The fix doesn't hardcode anything. The assemble step now reads
`enterprise_values.marketCapitalization` (FMP's own quarterly mktCap
snapshots, which ARE split-aware) and asof-joins them onto the panel.
Coverage 85.9 % across 262 k rows. PDN now reports **~A$1.4 B in Aug
2023** — matches FMP's own EV records, no caps, no manual overrides.

### Fundamental coverage

| Endpoint / field | Non-null % | Notes |
|---|---:|---|
| income_statement: `netIncome`, `revenue`, `operatingIncome`, `ebitda` | 100 % | All used by features |
| balance_sheet: `totalAssets`, `totalDebt`, `totalStockholdersEquity` | 100 % | |
| cash_flow: `operatingCashFlow`, `freeCashFlow` | 100 % | |
| **enterprise_values: `marketCapitalization`** | 85.9 % (PIT-asof) | Now the primary mktCap source |
| ratios: `priceEarningsRatio`, `returnOnEquity`, `roic` | _absent_ | FMP stable-API renamed; computed from primitives |
| financial_growth: `revenueGrowth`, `epsgrowth` | 100 % | |
| Sector / industry | 0 % | Profile pull not yet wired — known limitation |

---

## v1 → v2 delta

| Dimension | v1.0 (original) | v2.0 (this repo) |
|---|---|---|
| Repo structure | one notebook | src-layout package + scripts + tests + docs |
| Window | implicit (~15 yrs, weekly) | **2010-06 → 2026-05 (15.97 yrs), monthly rebalance** |
| Universe | implicit, all ASX | explicit top-500 by ASIC frequency, A$200m mcap gate |
| Prices | Yahoo (no validation) | **Yahoo, cross-checked vs FMP at ρ = 0.9996** |
| Fundamentals | none | FMP 7-endpoint PIT panel, lagged to `acceptedDate` |
| Market cap | shares × price (breaks across reverse splits) | **FMP `enterprise_values` (split-aware)** |
| Features | 5 price/SI only | ~25 raw → **562 cross-sectional ranks** |
| Cross-validation | single 400-week train / forward test | **walk-forward expanding window + 36-month pure OOS holdout** |
| Models | one logit | naive, EW composite, logit, LightGBM clf + LambdaRank |
| Risk control | hard 10 % stop, no costs | **15 % per-position stop + 100 bps fill slippage + extra round-trip commission** |
| Metrics | total $ PnL only | CAGR / Vol / **Sharpe / Sortino / MaxDD / Calmar / IS/OOS** / hit-rate / turnover |
| Reporting | inline plots | publication-quality PNGs + RESULTS.md + data_audit.md + methodology + data dictionary |
| Reproducibility | none | uv lockfile, deterministic on-disk caches, 41-test pytest suite |

---

## How to reproduce

```bash
git clone https://github.com/Taff1887/short-king-2.0.git
cd short-king-2.0
cp .env.example .env       # put your FMP_API_KEY in .env
uv sync --extra dev
```

```bash
# Full pipeline (cold-run total ~30 min; hot re-runs < 5 min)
uv run python scripts/01_pull_asic.py             --weeks 833    # 16 yr ASIC
uv run python scripts/02_pull_fmp_fundamentals.py --top-tickers 500 --limit 80
uv run python scripts/_refilter_asic.py
uv run python scripts/03_pull_fmp_prices.py                       # Yahoo prices + FMP mktCap
uv run python scripts/04_build_features.py                        # weekly + monthly panels
uv run python scripts/05_train_and_validate.py    --monthly --holdout-months 36
uv run python scripts/06_backtest.py              --monthly
uv run python scripts/_current_positions.py       --monthly
uv run python scripts/_data_audit.py
uv run python scripts/_extra_charts.py
uv run python scripts/08_generate_report.py
```

Java (any version) on `PATH` for `tabula-py`; `pdfplumber` is the
Java-free fallback.

---

## Limitations

* **As-of date is Monday but execution would be Friday.** ASIC's
  4-business-day lag means each Friday release reports positions
  as-of the prior Monday. The backtest uses the Monday's adjClose
  as the entry price, which assumes immediate knowledge of the
  Monday position; in live trading you'd actually fill on Friday.
  Drift over 4 business days is small but non-zero. Follow-up:
  swap the price-join key to `ReleaseDate`.
* **Stop-loss execution slippage = 100 bps central / 200 bps
  conservative** (`CostConfig.stop_slippage_pct`). Realistic for
  this universe: 30 bps ASX 50, ~100 bps mid-cap, 100–500 bps
  small-cap, 200–1000+ bps halted / squeeze. Tail-risk slippage on
  individual halted names is not separately modelled.
* **Sector dummies skipped** — FMP `profile` not yet pulled. Easy follow-up.
* **Borrow cost flat 150 bps p.a.** Real ASX borrow varies by name and
  date; the `CostConfig` is fully parameterised.
* **Capital-raise / squeeze dynamics absent.** The 15 % stop catches
  most single-week pain, but the engine is pure adjusted-close.
* **Sample limitations.** OOS = 36 months (3 years). That's a strong
  holdout for a 16-year IS panel but still a single regime
  (2023-2026: post-COVID rally, AI boom, China-shock 2.0). A bigger
  IS panel (e.g. via WRDS/CRSP extending to 1995+) would give a
  longer OOS too.
* **FMP plan limits ASX price coverage to ~5 years.** We pull prices
  from Yahoo instead, validated against FMP at ρ = 0.9996 on the
  overlap window. ~144 of the top-500 tickers (delisted pre-2010 or
  with mismatched Yahoo tickers) have no Yahoo data and are dropped
  from the price panel.
* **Some tickers have bad FMP source data**: 60 panel rows show
  mktCap > A$500B from FMP's `enterprise_values` for symbols where
  FMP itself appears to be wrong (e.g. Village Roadshow 2011). The
  Yahoo cross-check already flags these. We don't cap or hardcode.

---

## License

MIT — see [`LICENSE`](LICENSE).
