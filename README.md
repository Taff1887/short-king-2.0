# Short King 2.0 — ASX Short-Selling Research

> Cross-sectional short-selling on the Australian Securities Exchange.
> **500-stock universe, 15 years 11 months (16 June 2010 → 29 May 2026)
> of weekly ASIC disclosures**, **Friday-release end-of-month rebalance**
> (4-business-day lag respected — we trade on the day positions become
> known, not the as-of date), FMP fundamentals, Yahoo Finance prices
> (cross-checked vs FMP at median ρ = 0.9996), 3 models walk-forward CV'd
> with purge + embargo, **36-month pure out-of-sample holdout**, costed
> backtest (25 bps commission + 1.5 % p.a. borrow + 5 bps slippage)
> **with a realistic 20 % per-position stop loss + 10 % execution
> slippage + daily-OHLC gap handling applied to every short position**.
> Benchmarked against ASX 200 buy & hold.

A from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)) — a single
430 KB notebook with 5 hand-built signals and 21 total trades that lost
money. **Version 2.0** is a proper research project. Headline results
below.

---

## ⚠️ Headline finding — the honest version

Applying a **realistic stop loss** (20 % trigger + 10 % slippage + gap rule)
to every short position **destroys the strategy** on this universe.
Every (model × strategy) combination prints negative Sharpe over the full
16-year panel and over the 35-month OOS holdout. The only thing that
makes money is **the ASX 200 buy & hold benchmark**.

This is the realistic answer to the question that broke earlier iterations
of this project: "*does a stop loss save the strategy from squeeze risk?*"
The answer is no — a realistic stop loss with gap handling fires on
**17-26 % of all monthly short positions** and the gap penalty on the
real squeezes eats more alpha than the stop saves on the false alarms.

What this project really shows:

* **The cross-sectional short-interest signal is real** — every model has
  statistically-significant negative information coefficient (their score
  correctly anti-correlates with forward returns). See [Information
  coefficients](#information-coefficients--is-and-oos).
* **The signal is strong enough to make money in a frictionless world**
  — *without* a stop loss, the EW polarity-aware composite and the naive
  short-interest baseline both clear OOS Sharpe ≈ 0.9. See
  [Table 1b](#table-1b--no-stop-comparison-what-the-strategy-could-do-without-the-gap-penalty).
* **But the squeeze tail is too fat to trade in practice.** A realistic
  stop loss + gap rule converts the latent alpha into realised
  drawdowns. On this universe, at this rebalance frequency, with these
  execution assumptions, **the strategy is not investable.**
* **The honest takeaway** for an interviewer / reviewer: this is a clean
  demonstration of methodology — universe construction, point-in-time
  fundamentals, walk-forward CV with purge / embargo, IS vs OOS
  discipline, realistic execution costs, and the discipline of *not
  hiding the gap risk*. The strategy doesn't work; that's a real finding.

---

## Headline results

* **3 models**: `naive` (rank by ShortPct), `ew` (polarity-aware
  12-factor composite), `logit` (L2 logistic regression).
* **2 strategies** per model: **quintile-short only** (top 20 % most
  shortable) and **dollar-neutral long-short quintile** (long bottom
  20 %, short top 20 %).
* **+ ASX 200 buy & hold** as the equity-only benchmark.
* All numbers are **stop-loss applied** (20 % trigger / 10 % slippage /
  gap rule on daily OHLC). The "no stop" comparison is in Table 1b.

### Table 1 — OOS holdout (n=35 monthly rebalances, 2023-06 → 2026-05)

Ranked by OOS Sharpe.

| Strategy | n_months | Sharpe | CAGR | Ann. vol | MaxDD | Hit-rate |
|---|---:|---:|---:|---:|---:|---:|
| **ASX 200 (buy & hold)** | 35 | **+0.71** | **+7.30 %** | 10.77 % | **−7.79 %** | **62.9 %** |
| ew / long_short_quintile | 35 | −0.94 | −20.37 % | 21.57 % | −56.16 % | 34.3 % |
| naive / long_short_quintile | 35 | −0.99 | −13.91 % | 14.02 % | −37.38 % | 42.9 % |
| naive / quintile_short | 35 | −1.50 | −28.69 % | 20.74 % | −66.17 % | 40.0 % |
| logit / long_short_quintile | 35 | −1.54 | −27.64 % | 19.48 % | −62.48 % | 25.7 % |
| ew / quintile_short | 35 | −1.64 | −35.29 % | 24.25 % | −74.57 % | 31.4 % |
| logit / quintile_short | 35 | −1.68 | −34.44 % | 23.11 % | −72.78 % | 37.1 % |

### Table 2 — Full-period (n=191 monthly rebalances, 2010-06 → 2026-05)

| Strategy | n_months | Sharpe | CAGR | Ann. vol | MaxDD | Hit-rate |
|---|---:|---:|---:|---:|---:|---:|
| **ASX 200 (buy & hold)** | 190 | **+0.39** | **+4.45 %** | 13.79 % | −31.0 % | 61.6 % |
| naive / long_short_quintile | 191 | −0.97 | −15.87 % | 16.24 % | −95.35 % | 41.4 % |
| naive / quintile_short | 191 | −1.30 | −29.24 % | 23.91 % | −99.60 % | 34.6 % |
| ew / long_short_quintile | 191 | −1.47 | −28.15 % | 20.73 % | −99.50 % | 34.0 % |
| ew / quintile_short | 191 | −1.72 | −41.36 % | 27.92 % | −99.98 % | 28.8 % |
| logit / quintile_short | 154 | −1.80 | −38.58 % | 24.76 % | −99.82 % | 27.3 % |
| logit / long_short_quintile | 154 | −1.93 | −28.64 % | 16.55 % | −98.73 % | 27.9 % |

> Trained-model rows have n=154 instead of 191 because the walk-forward
> CV needs a ~3-year warm-up window — `logit` only has OOF scores from
> 2013-07 onwards. The naive and EW composites are parameter-free, so
> they cover the full panel.

![Cumulative growth of $1 — solid = quintile-short, dotted = long-short, dashed = ASX 200 buy & hold](charts/cumulative_returns_monthly.png)

### Table 1b — No-stop comparison: what the strategy could do without the gap penalty

The same backtest configuration but **without the stop loss** (every
position realises its full monthly P&L). This is the realistic-execution-
free version — useful to isolate "where the signal lives" from "where the
costs eat it":

| OOS Sharpe (n=35) | No stop | **With realistic stop** | Δ |
|---|---:|---:|---:|
| naive / long_short_quintile | +0.92 | **−0.99** | −1.91 |
| ew / long_short_quintile | +0.91 | **−0.94** | −1.85 |
| logit / long_short_quintile | +0.24 | −1.54 | −1.78 |
| naive / quintile_short | −0.32 | −1.50 | −1.18 |
| ew / quintile_short | −0.08 | −1.64 | −1.56 |
| logit / quintile_short | −0.20 | −1.68 | −1.48 |

The gap is **∼1.5–1.9 Sharpe points** in OOS — that's the realised cost
of squeeze gap risk on this universe. The no-stop column shows that the
signal IS in the data; the stop column shows that you can't extract it
with a simple per-position stop at realistic execution.

Full data:
[`reports/backtest_summary_monthly.csv`](reports/backtest_summary_monthly.csv) (stop-loss applied) /
[`reports/backtest_summary_monthly_nostop.csv`](reports/backtest_summary_monthly_nostop.csv) (no stop) /
[`reports/headline_table.csv`](reports/headline_table.csv) (chart-table including ASX 200).

---

## Why the stop loss destroys this strategy

1. **The trigger fires too often.** A 20 % adverse move on a small-mid-cap
   short is common. Across all 3 models and all 18,000+ short positions
   over 16 years, **17 % – 26 % of positions fire the stop**. Most of
   those positions would have recovered by month-end — the stop closes
   early and locks in -32 %.
2. **The gap penalty is brutal on the real squeezes.** When a name DOES
   squeeze, it doesn't crawl to +20 %; it gaps from the previous close to
   open up +40 %, +80 %, +200 %. The gap rule covers at
   `max(entry × 1.32, trigger_day_open)` — so the actual loss on a real
   squeeze can be -80 %, -100 %, -150 %, not -32 %.
3. **The L/S quintile loses its long leg's cushion.** Without stops, the
   L/S spread is positive because the long leg absorbs the short-leg
   pain. With stops, the short leg loses MORE than its uncapped self
   (false alarms + gap penalty), and the long leg can't keep up.

See the realistic stop-loss methodology in
[Realistic stop-loss spec](#realistic-stop-loss-spec) below.

---

## How many stocks are picked per month?

**Short answer**: a *fraction* of the universe (top 20 % for the
quintile), not a fixed number. Specifically:

* The eligible universe varies per rebalance (the `investable` flag
  depends on price coverage + fresh fundamentals + market cap ≥ A$100 m
  on that date). Over the OOS window the universe ranges **284 → 306
  names**, averaging 297.
* **Quintile = top / bottom 20 % by score = ~60 names per leg.** Over OOS
  the short basket is exactly **57 – 61 names per month**, mean 59.7.
* **Equal-weight inside each leg** — every short gets weight `-1 / N`
  and every long gets `+1 / N`.
* L/S quintile holds ~120 positions total per month (~60 long + ~60 short).
* 35 OOS months × ~60 shorts = **2,089 OOS short positions** across 224
  unique tickers.
* Full-panel: 191 months × ~50–60 shorts per model = ~10k short positions
  per model; **346 unique tickers** ever appear in any short basket
  across all 3 models.

It scales with the cross-section, never "always 10". If the universe were
100 stocks the basket would be 20; at 500 it'd be 100.

---

## Realistic stop-loss spec

Implemented in
[`scripts/_apply_stop_loss_full.py`](scripts/_apply_stop_loss_full.py),
applied per short position over its monthly holding period:

1. **Stop trigger** = `entry_price × 1.20` (20 % adverse move).
2. **Daily intraday monitoring**: for each trading day strictly *after*
   entry, check if `daily_high ≥ stop_price`. First day that breaches
   fires the stop.
3. **Execution slippage**: nominal cover = `stop_price × 1.10` =
   `entry × 1.32` → a -32 % short return.
4. **Gap rule**: cover at the WORSE of nominal cover and the trigger-day
   open: `cover_price = max(entry × 1.32, trigger_day_open)`. If the
   stock gaps up 80 % overnight, you cover at -80 %, not -32 %.
5. **No further P&L** from a stopped position for the rest of the month.
6. **No look-ahead**: only daily bars *strictly after* the entry date are
   scanned for trigger.

Daily auto-adjusted OHLC pulled from Yahoo via
[`scripts/_pull_ohlc_full.py`](scripts/_pull_ohlc_full.py) — 346 unique
short-basket tickers × up to 16 years = **1,254,249 daily bars**.

### Stop-fire diagnostics (full 16-year panel)

| Model | Total shorts | Stops fired | Stop rate |
|---|---:|---:|---:|
| naive | 8,844 | 1,464 | 16.6 % |
| ew | 8,844 | 2,333 | 26.4 % |
| logit | 7,728 | 1,836 | 23.8 % |

EW fires the most stops because its short basket concentrates on
multi-factor-bearish names (high SI + expensive + low quality), which
also tend to be the names most prone to squeezes.

Full per-position table (OOS):
[`reports/oos_short_stopped.csv`](reports/oos_short_stopped.csv)
with the seven required columns (`stop_triggered`, `stop_trigger_date`,
`entry_price`, `stop_price`, `cover_price`, `raw_short_return`,
`stopped_short_return`).

---

## Information coefficients — IS and OOS

The Spearman rank correlation between each model's score and the realised
**monthly forward return**, computed cross-sectionally each month then
averaged. For shorts, **negative IC is good** — it means the "shortable"
score correlates with names that fell.

| Model | IS IC | IS t-stat | n_IS | OOS IC | OOS t-stat | n_OOS |
|---|---:|---:|---:|---:|---:|---:|
| **ew** | **−8.3 %** | **−6.73** | 156 | **−8.9 %** | **−3.83** | 35 |
| logit | −5.3 % | −5.28 | 119 | −5.5 % | −2.48 | 35 |
| naive | −1.6 % | −1.70 | 156 | −2.1 % | −1.38 | 35 |

All three models have negative IC in both IS and OOS; `ew` and `logit`
clear |t| > 2 in OOS (significant). **The signal is real** — the model
correctly identifies underperformers. The realistic execution costs are
what prevent the signal from being tradeable, not the signal itself.

---

## What do the models actually do? (plain English)

### Score interpretation

Every model produces a single number per (Date, Ticker) called the
**score**, where **higher = more shortable**. Each month we sort the
~290 stocks by score, short the top 20 % (most shortable), and (for the
L/S quintile) buy the bottom 20 % (least shortable = best expected
returns).

### What "long-short quintile" actually means

The model ranks all ~290 stocks from **most shortable** (rank #1, the
model's strongest bearish bet) to **least shortable** (rank #290 — the
opposite extreme, the model's strongest bullish bet). Then:

1. **Short the top 100** (top quintile by score). These are the names
   the model says will fall most.
2. **Buy the bottom 100** (bottom quintile). These are the **opposite
   extreme** — high quality, low SI, positive momentum, low leverage.
3. **Equal-weight inside each leg.** Net dollar exposure = 0 (long $ =
   short $), so market-direction risk cancels.

You're betting on the **spread** between the two ends. The middle 60 %
gets ignored.

### 1. `naive` — sort by reported short interest

**What it does:** For each rebalance month, looks up the ASIC-reported
**short-interest %** of each stock and uses it directly as the score.
The stock with the highest reported SI gets the highest "shortable" rank.

**The math:** Score = `ShortPct`, rank-normalised to 0–1 across the
cross-section. No training, no fitting. One column.

**Why it works:** Professional short-sellers and hedge funds publish
their positions every week via ASIC. If a lot of them are short the same
name, they collectively have an informed view that the price will fall.
Known anomaly with multi-decade academic pedigree.

**Limitation:** It's a popularity contest. Crowded shorts squeeze
occasionally and naive has no defence — it rides the broad signal and
accepts the tail loss.

### 2. `ew` — polarity-aware equal-weight composite of 12 ranked factors

**What it does:** Computes 12 cross-sectional rank columns (short %,
3-month momentum, 1-month vol, market cap, P/E, ROE, ROIC, FCF yield,
debt/equity, revenue growth, etc.) and averages them — with every factor
**pre-oriented so that high rank = more shortable**.

**The math:** For each month, each factor *f* is ranked 0–1 across all
stocks. Six factors are inverted via `1 − rank` because they're naturally
**bullish** (high ROE = quality = LESS shortable etc.):

| Factor | Polarity | Rationale |
|---|:---:|---|
| `ShortPct_rk` / `short_pct_ff_rk` / `si_z_12m_rk` | +1 keep | High SI = more shortable |
| `vol_1m_rk` | +1 keep | High vol = low-vol anomaly says shortable |
| `pe_rk` | +1 keep | Expensive = shortable |
| `debt_equity_rk` | +1 keep | Levered = shortable |
| `mom_3m_rk` | **−1 invert** | High momentum = bullish |
| `log_mktcap_rk` | **−1 invert** | Mega-caps less shortable |
| `fcf_yield_rk` | **−1 invert** | Cash-rich = bullish |
| `roe_rk` / `roic_rk` | **−1 invert** | Quality = bullish |
| `revenue_growth_yoy_rk` | **−1 invert** | Growing = bullish |

Final score = mean of the 12 polarised ranks, re-ranked cross-sectionally.
No parameters, no training — every factor gets equal weight.

**Why it works:** **Best OOS IC (-8.9 %)** of any model — the polarity
spec correctly orients each factor as a short-direction signal. A
hand-built composite of cheap + levered + shrinking + crowded
beats the linear model with the same features.

**Limitation:** Equal weight is arbitrary. A real-money build would
optimise factor weights (e.g. IC-weighted blending) or group-equal-weight
by theme.

> **A note on polarity.** An earlier version of this composite averaged
> the raw ranks *without flipping the bullish factors*, producing a +4.8 %
> IC (i.e. it was actually a long-bias score). The one-line fix —
> wrapping the six bullish columns in `1 − rank` before the mean —
> flipped OOS IC to `−8.9 %`. When you blend factors, the sign of each
> input has to match the sign of the score you're building.

### 3. `logit` — L2 logistic regression on the ranked features

**What it does:** A simple linear classifier that takes all ~570 feature
rank columns as inputs and predicts the probability that the stock's
**monthly forward return** will be **negative**.

**The math:** Fits weights *β* to each feature, then
`prob_down = sigmoid(β · features)`. Score = `prob_down`. Trained on every
walk-forward fold with L2 regularisation; the final fit on all in-sample
data is what scores the OOS holdout.

**Why it's included:** The robust, interpretable workhorse of
quantitative finance. Linear models resist overfitting, and coefficients
are directly readable as "this factor matters this much".

**Limitation:** Real markets aren't linear. If "high short interest"
matters only when "momentum is *also* negative", logit can't capture
that interaction.

### Why these three (and what got dropped)

`naive` is the no-model floor. `ew` is the no-training composite that
encodes a thoughtful human prior. `logit` is the simplest trained
model — interpretable, low-overfit. **The previous LightGBM
classifier and LambdaRank were dropped** after they consistently
underperformed the no-training baselines on OOS Sharpe and on OOS IC
(both gbm variants stayed weaker than `ew` at every horizon), AND
concentrated their picks too tightly into the squeeze tail. May come
back later but not part of this version.

---

## Stats & math glossary

A reader-friendly walkthrough of the metrics on this page.

* **CAGR** — compound annual growth rate. If $1 grew to $1.45 over 3
  years, CAGR = `1.45^(1/3) − 1 = +13.2 %`. Fund factsheet number.
* **Sharpe ratio** — `(mean return) / (std dev of returns)` annualised.
  Risk-adjusted return per unit of total volatility. > 1 good, > 2 great,
  > 3 exceptional. **For dollar-neutral L/S strategies a 0.5+ Sharpe is
  real alpha because there's no market beta riding underneath**.
* **Ann. vol** — monthly stdev × √12. Higher = more risky.
* **Max drawdown (MaxDD)** — worst peak-to-trough fall on the equity
  curve. A -10 % MaxDD means at the worst point you were down 10 % from
  the previous all-time high.
* **Hit-rate** — fraction of rebalance periods with positive return.
* **Information coefficient (IC)** — Spearman rank correlation between
  the model's score and the realised forward return, computed
  cross-sectionally per month then averaged across months. **Negative
  IC is good for shorts**.
* **t-stat** — IC mean / IC standard error. |t-stat| > 2 means the
  average IC is unlikely to be zero by chance.
* **IS vs OOS** — *in-sample* = data the model was developed on;
  *out-of-sample* = held-back data the model never saw during
  development. **OOS Sharpe is the only honest estimate of future
  performance.**
* **Walk-forward CV** — train on 2010–2014, test on 2014–2015; train on
  2010–2015, test on 2015–2016; etc. The model is never trained on data
  from after its test window.
* **Purge + embargo** — extra safety on top of walk-forward. *Purge*
  drops training samples that overlap the test labels' horizon (here,
  ~1 month). *Embargo* drops the following ~1 month after the test
  window. Stops information leakage via the forward-return label.
* **Quintile** — split into 5 equal buckets. Top quintile = top 20 %
  by score.
* **Long-short (L/S)** — long the bottom-quintile names, short the
  top-quintile. Dollar-neutral.
* **Round-trip commission** — exchange + broker fees on a full
  buy-then-sell. 25 bps per side here.
* **Borrow cost** — annualised fee for borrowing shares to short.
  1.5 % p.a. flat in `CostConfig`; **significantly understates the
  small-cap squeeze tail** where actual borrow can be 500-5,000+
  bps/yr (see [Limitations](#limitations)).
* **Stop-loss execution slippage** — fees / market impact paid when the
  stop fires, modelled here as a 10 % penalty on top of the trigger
  (= -32 % loss before the gap rule).
* **Gap rule** — on the trigger day, cover at the worse of the nominal
  fill and the open price. Captures the cost of overnight squeezes.

---

## Universe — what's in / what's out

**The universe includes every ASX 50 mega-cap.** On the latest rebalance
(29 May 2026) the panel spans market caps from **A$9 m to A$268 B** —
CBA, RIO, BHP, all four banks, CSL, MQG, TLS all present:

| # | Ticker | Mkt Cap (A$bn) | Short % |
|---:|---|---:|---:|
| 1 | CBA | 268.5 | 2.1 |
| 2 | RIO | 159.3 | 8.7 |
| 3 | BHP | 154.1 | 1.4 |
| 4 | WBC | 134.8 | 1.8 |
| 5 | NAB | 126.8 | 1.8 |
| 6 | ANZ | 107.8 | 0.8 |
| 7 | WES | 92.0 | 1.1 |
| 8 | CSL | 83.5 | 0.7 |
| 9 | MQG | 76.7 | 0.4 |
| 10 | TLS | 55.3 | 0.9 |

The strategy *can* short these names; they just rarely make it into the
top-quintile short basket because mega-cap SI is low (CBA 2.1 %,
BHP 1.4 %) and their quality metrics rank them as least-shortable.
**RIO at 8.7 % is the exception** — high SI on a mega-cap is the kind
of signal the strategy will pick up.

---

## Methodology

### 1. Universe and rebalance — Friday-release dates (4-BDay lag respected)

* **ASIC daily aggregate short-position reports**: weekly, Friday-
  anchored (the earliest archived report we use is 16 June 2010).
* Each Friday ASIC release covers positions **as of the prior Monday**
  (4 business days earlier). The pipeline rebalances on the Friday
  release using Friday's adjusted close as the entry price. The Monday
  as-of date is preserved as `AsOfDate` for diagnostic purposes only.
* End-of-month rebalance = the last ASIC release in each calendar month.
  192 dates, day-of-week mix: 180 Fridays, 10 Thursdays, 2 Wednesdays
  (Easter / Christmas / King's Birthday shifts).
* Universe = **top 500 ASX tickers by ASIC-report frequency** over the
  full 16 years, gated at ≥ A$100 m market cap on each rebalance.

### 2. Data sources — and why the Yahoo / FMP split

| What | Source | Why |
|---|---|---|
| Short positions | ASIC daily aggregate PDFs | Only public, free, asset-class-complete source for ASX |
| Prices (adjusted close + volume) | **Yahoo Finance** | FMP only ships ~5 yrs of ASX daily history on this plan; Yahoo goes back to ~2000. Cross-checked vs FMP at **median ρ = 0.9996** over the overlap window |
| Daily OHLC for stop-loss logic | **Yahoo Finance** (auto_adjust=True) | 1,254,249 daily bars × 346 short-basket tickers |
| Fundamentals (7 quarterly endpoints) | Financial Modeling Prep (Premium) | Income, balance sheet, cash flow, ratios, key metrics, enterprise values, financial growth |
| Market capitalisation | **FMP `enterprise_values`** (half-yearly, split-adjusted) | Balance-sheet `commonStockSharesOutstanding` isn't back-adjusted across corporate actions; `enterprise_values.marketCapitalization` is. Fixes the Paladin Energy 1:100 reverse-split bug from earlier versions. |

**Could we go Yahoo-only?** No — Yahoo's `quarterly_financials` only
exposes the last ~4 quarters; its `info` snapshot has no `acceptedDate`
(breaks no-look-ahead); its `marketCap` isn't back-adjusted across
reverse splits. FMP is essential for fundamentals + PIT market cap.

### 3. Features

~25 raw factors → **562 cross-sectional rank columns** across short
interest (SI %, SI z-score, days-to-cover, persistence), price (multi-
horizon momentum and vol, drawdown), liquidity (ADV, turnover, Amihud),
valuation (P/E, P/B, EV/EBITDA, FCF yield), quality (ROE, ROIC, margins,
accruals), leverage and growth. **All time-horizon labels are in months**
(`mom_3m`, `vol_1m`, `si_z_12m`, etc.) for consistency with the monthly
rebalance grid — the underlying compute uses weekly bars but the labels
are framed in months because 4 weeks ≈ 1 calendar month here. Every
numeric factor is ranked **within each rebalance date** to 0–1, then NaN
values are imputed to 0.5 (neutral) so linear models don't blow up on
missing fundamentals.

### 4. Targets

* **Forward-return label**: 1 month (the column is called `fwd_ret_1m`
  in the parquet — internally computed as 4 weeks forward, which equals
  one rebalance period on the monthly grid).
* **Binary**: `fwd_ret_1m < 0` for the logit classifier.

### 5. Cross-validation (walk-forward + IS/OOS holdout)

* **Pure OOS holdout**: the last 36 monthly rebalances (2023-06 →
  2026-05) are *reserved* — the trained models never see these rows
  during development.
* On the **156-month in-sample** portion (2010-07 → 2023-05):
  walk-forward expanding window with ~36-month min train, ~6-month
  test, 1-month embargo (≥ label horizon). 20 folds total, ~120 monthly
  OOF observations per trained model.
* After CV, a final model is fit on the entire IS panel and applied to
  the holdout. That single OOS Sharpe is the unbiased estimate.

### 6. Portfolio + costs

* Top-quintile short OR dollar-neutral L/S quintile (long bottom 20 %,
  short top 20 %), equal weight within each leg, monthly rebalance.
* Liquidity gate: investable + ≥ A$100 m mkt cap.
* Frictions: 25 bps round-trip commission per side, 1.5 % p.a. borrow on
  shorts, 5 bps slippage on weight changes.
* **Realistic stop loss applied per spec above** (20 % trigger / 10 %
  slippage / gap rule on daily OHLC).

---

## Limitations

* **Borrow cost flat 1.5 % p.a.** — significantly underestimates the
  small-cap squeeze tail. Real ASX borrow runs ~25–50 bps/yr for
  ASX 50 large-caps, ~100–200 bps for mid-caps, **500–1,500 bps for
  crowded small-cap shorts**, 5,000+ bps (or recall) for squeeze names.
  Wiring in a per-ticker borrow term structure is on the follow-up list.
* **Sector dummies skipped** — FMP `profile` not pulled. Easy follow-up.
* **No market-impact / liquidity-aware sizing.** The backtest
  equal-weights each leg; a real book would have to taper or skip the
  bottom-of-tail names.
* **Capital-raise / squeeze dynamics absent.** The OHLC engine is
  point-in-time bars; no halts, takeover-bid gap modelling, or hard-to-
  borrow recall events.
* **Sample limitations.** OOS = 36 months (3 years). That's a strong
  holdout for a 16-year IS panel but still a single regime
  (2023–2026: post-COVID rally, AI boom, China-shock 2.0).
* **No tax, financing, FX modelling.**
* **Realistic stop loss is harsh but realistic.** The 20 % trigger +
  10 % slippage + gap rule is a faithful model of how a real stop
  would fill on a squeezing small-cap. A more sophisticated risk
  control (signal-driven exits, sector-neutral construction,
  vol-scaled position sizes) might produce a better result — but
  is out of scope here.

---

## How to reproduce

```bash
git clone https://github.com/Taff1887/short-king-2.0.git
cd short-king-2.0
cp .env.example .env       # put your FMP_API_KEY in .env
uv sync --extra dev
```

```bash
# Full pipeline (cold-run total ~30 min)
uv run python scripts/01_pull_asic.py             --weeks 833
uv run python scripts/02_pull_fmp_fundamentals.py --top-tickers 500 --limit 80
uv run python scripts/_refilter_asic.py
uv run python scripts/03_pull_fmp_prices.py
uv run python scripts/04_build_features.py
uv run python scripts/05_train_and_validate.py    --monthly --holdout-months 36
uv run python scripts/06_backtest.py              --monthly
uv run python scripts/_pull_ohlc_full.py
uv run python scripts/_apply_stop_loss_full.py        # << applies realistic stop loss to all backtests
uv run python scripts/_build_headline_chart_and_table.py
uv run python scripts/_current_positions.py       --monthly
uv run python scripts/_data_audit.py
```

Java (any version) on `PATH` for `tabula-py`; `pdfplumber` is the
Java-free fallback.

---

## v1 → v2 delta

| Dimension | v1.0 (original) | v2.0 (this repo) |
|---|---|---|
| Repo structure | one notebook | src-layout package + scripts + tests + docs |
| Window | implicit (~15 yrs, weekly) | 2010-06 → 2026-05 (15.97 yrs), monthly rebalance |
| Universe | implicit, all ASX | explicit top-500 by ASIC frequency, A$100m mcap gate |
| Prices | Yahoo (no validation) | Yahoo, cross-checked vs FMP at ρ = 0.9996 |
| Fundamentals | none | FMP 7-endpoint PIT panel, lagged to `acceptedDate` |
| Market cap | shares × price (breaks on splits) | FMP `enterprise_values` (split-aware) |
| Features | 5 price/SI only | ~25 raw → 562 cross-sectional ranks |
| Cross-validation | single 400-week train / forward test | walk-forward expanding window + 36-month pure OOS holdout |
| Models | one logit | naive + polarity-aware EW + L2 logit (3 models, no GBM) |
| Risk control | hard 10 % stop, no costs | **Realistic 20 % stop + 10 % slippage + daily-OHLC gap rule** |
| Metrics | total $ PnL only | CAGR / Vol / Sharpe / Sortino / MaxDD / Calmar / IS/OOS / hit-rate / turnover |
| Benchmark | none | ASX 200 buy & hold, integrated into the chart and summary tables |
| Reporting | inline plots | publication-quality PNGs + RESULTS.md + data_audit.md + methodology |
| Reproducibility | none | uv lockfile, deterministic on-disk caches, 38-test pytest suite |

---

## License

MIT — see [`LICENSE`](LICENSE).
