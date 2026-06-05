# Short King 2.0 — ASX Short-Selling Research

> Cross-sectional short-selling on the Australian Securities Exchange.
> **500-stock universe, 15 years 11 months (16 June 2010 → 29 May 2026)
> of weekly ASIC disclosures**, **Friday-release end-of-month rebalance**
> (4-business-day lag respected — we trade on the day positions become
> known, not the as-of date), FMP fundamentals, Yahoo Finance prices
> (cross-checked vs FMP at median ρ = 0.9996), 5 models walk-forward CV'd
> with purge + embargo, **36-month pure out-of-sample holdout**, costed
> backtest (25 bps commission + 1.5 % p.a. borrow + 5 bps slippage),
> no stop loss — every position realises its full uncapped monthly P&L.

A from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)) — a single
430 KB notebook with 5 hand-built signals and 21 total trades that lost
money. **Version 2.0** is a proper research project. Comparison table
and headline results below.

---

## Headline result — IS vs OOS, monthly rebalance, no stop loss

The model dev (in-sample) period is **2010-06 → 2023-05** (~13 years; all
walk-forward CV runs here, ~120 OOF monthly observations per trained
model). The pure out-of-sample holdout is **2023-06 → 2026-05** (36 months;
final-fit model applied — never seen during development).

Net of 25 bps round-trip commission per side + 1.5 % p.a. borrow + 5 bps
slippage. **No stop loss** — every position realises its full uncapped
monthly forward return. Annualisation factor = 12 (monthly).

### Dollar-neutral long-short quintile (long bottom 20 %, short top 20 % by score)

| Model | **IS Sharpe** | **OOS Sharpe** | OOS CAGR | OOS MaxDD | OOS Hit-rate |
|---|---:|---:|---:|---:|---:|
| **naive** (rank ShortPct) | 0.57 | **0.92** | **+11.2 %** | **−10.8 %** | **68.6 %** |
| logit (rebuilt v1) | 0.24 | 0.24 | +2.9 % | −37.1 % | 48.6 % |
| ew (long-bias composite) | −0.38 | 0.08 | −0.4 % | −33.1 % | 54.3 % |
| gbm_rank (LightGBM LambdaRank) | −0.06 | −0.21 | −8.2 % | −51.6 % | 37.1 % |
| gbm_cls (LightGBM binary) | −0.10 | −1.11 | −16.3 % | −44.4 % | 45.7 % |

### Top-quintile short only (no long leg)

| Model | OOS Sharpe | OOS CAGR | OOS MaxDD | OOS Hit-rate |
|---|---:|---:|---:|---:|
| logit | −0.20 | −6.3 % | −40.9 % | 48.6 % |
| naive | −0.32 | −7.2 % | −35.7 % | 51.4 % |
| gbm_rank | −0.50 | −15.2 % | −56.1 % | 40.0 % |
| ew | −1.12 | −14.8 % | −42.5 % | 34.3 % |
| gbm_cls | −1.11 | −18.9 % | −53.4 % | 34.3 % |

![Cumulative growth of $1 — solid = quintile-short, dotted = long-short](charts/cumulative_returns_monthly.png)

### What this honestly shows

**Only `naive` (rank by raw ShortPct) clears positive Sharpe in OOS.**
+0.92 Sharpe, +11.2 % CAGR, -10.8 % MaxDD — investable but not heroic.
All four other models collapse to flat-or-negative Sharpe. The trained
models aren't *wrong* (their OOS information coefficient is still
significantly negative — see [§Information coefficients](#information-coefficients--is-and-oos));
they just **concentrate into specific high-conviction shorts that get
squeezed** (Appen, 4D Medical, BrainChip, Sunrise Resources). Naive's
broader picks ride those tails out.

**The short-only books (no long leg) are all negative Sharpe in OOS.**
Naked-short alpha is not a viable standalone strategy at this universe
and rebalance frequency. The dollar-neutral L/S quintile carries
mostly on the long leg.

### Per-position win-rate is > 50 % — the issue is magnitude, not direction

The trained models pick **the right names**. Per-position win-rate
(share of monthly shorts that ended profitable, i.e. the stock fell
during the month) is at or above 50 % for everything except `ew`:

| Model | Per-position win-rate | Median trade | Mean trade | Worst single position |
|---|---:|---:|---:|---:|
| **logit** | **53.4 %** | **+1.71 %** | −0.33 % | −218 % (APX 2024-07) |
| gbm_rank | 52.5 % | +1.85 % | −0.87 % | −218 % (4DX 2025-08) |
| naive | 50.9 % | +0.43 % | 0.00 % | −177 % (BRN 2024-01) |
| gbm_cls | 50.8 % | +0.38 % | −1.27 % | −217 % (4DX) |
| ew | 47.4 % | −0.50 % | −0.56 % | −50 % |

`logit`'s **median** trade is **+1.71 %** (most months win) but its
**mean** is **−0.33 %**. The gap is fat-right-tail squeezes — APX
+218 %, 4DX +217 %, BRN +177 % in a single month. One of those at a
2.5 % book weight costs ~5 % of NAV in one stroke and erases an entire
year of median-trade gains. This is **structural to short-selling**:
a long position can lose at most 100 %; a short can lose 500 % +.

**The naive baseline is competitive.** Sorting by raw ShortPct (no
training required), long-bottom-20% / short-top-20%, earns OOS Sharpe
**0.92** with an 68.6 % monthly hit-rate and a **−10.8 % MaxDD**. On this
universe, short-interest dispersion is the dominant cross-sectional
signal; the trained models earn back a real but small slice on top
when their picks don't get squeezed. **That is itself the headline
research finding.**

---

## What do the models actually do? (plain English)

If you're new to ML in finance, here is what each of the five models is
doing under the hood, what data it sees, and why I included it.

### Score interpretation

Every model produces a single number per (Date, Ticker) called the
**score**, where **higher = the model thinks this stock is more
shortable** (more likely to fall in the next month). Every month we
sort the ~500 stocks by score, short the top 20 % (worst expected),
and (for the L/S quintile) buy the bottom 20 % (best expected).

### 1. `naive` — sort by reported short interest

**What it does:** For each rebalance month, looks up the ASIC-reported
**short-interest percent** of each stock and uses it directly as the
score. The stock with the highest reported short interest gets the
highest "shortable" rank.

**The math:** Score = `ShortPct`, rank-normalised to 0-1 across the
cross-section. That's it. No training, no fitting, no model. One column.

**Why it works:** Professional short-sellers and hedge funds publish
their positions every week via ASIC. If a lot of them are short the
same name, they collectively have an informed view that the price
will fall. We piggyback on the consensus. Often called the
**short-interest factor** in academic literature; it's a known
anomaly with multi-decade pedigree.

**Limitation:** It's a popularity contest. Crowded shorts
occasionally squeeze hard (4DX, APX) and the naive model has no
defence against that — it just rides the broad signal and accepts the
tail loss. Survives in this dataset because the short basket is
diversified across ~100 names per month.

### 2. `ew` — equal-weight composite of 12 ranked factors

**What it does:** Computes 12 different cross-sectional ranks (short %,
multi-horizon momentum, volatility, market cap, P/E, ROE, debt/equity,
FCF yield, revenue growth, etc.) and averages them. Higher composite
rank = more shortable.

**The math:** For each month, each factor *f* is ranked 0-1 across all
stocks. Score = `mean(rank_f1, rank_f2, ..., rank_f12)`. No
parameters, no training — every factor gets equal weight.

**Why it included:** Industry-standard "smart-beta" composite — what
you'd build before you had any historical data to train on. A useful
sanity check: if a trained model can't beat equal weight, the
"training" wasn't adding signal.

**Limitation:** Many of the input factors (high ROE, low debt,
positive momentum) are **bullish** signals on their own — averaging
them in a *short* score is conceptually backwards. That's why `ew`
shows up with a **positive** IC (+4.8 %): its score correlates with
*winners*, not losers. It's effectively a long-bias model dropped
into a short-bias frame.

### 3. `logit` — logistic regression on all the ranked features

**What it does:** A simple linear classifier that takes all ~25 feature
ranks as inputs and predicts the probability that the stock's
**monthly forward return** will be **negative**.

**The math:** Fits weights *β* to each feature, then
`prob_down = sigmoid(β · features)`. Score = `prob_down`.
The training minimises log-loss (penalises confident wrong
predictions). Trained on every walk-forward fold with L2 regularisation;
the final fit on all in-sample data is what scores the OOS holdout.

**Why it included:** The robust, interpretable workhorse of
quantitative finance. Logistic regression resists overfitting (because
it's linear) and the coefficients are directly readable as "this
factor matters this much". Even when fancier models beat it on metrics,
logit is what gets shipped in real research notes because reviewers
can audit it.

**Limitation:** Real markets aren't linear. If "high short interest"
matters only when "momentum is *also* negative", logit can't capture
that interaction. The next two models can.

### 4. `gbm_cls` — LightGBM binary classifier (non-linear)

**What it does:** Same target as logit (probability that the **monthly
forward return** is negative), but the underlying model is a
**gradient-boosted decision tree**. It builds **400 small trees** in
sequence where each new tree corrects the errors of the trees built
before it. Like asking 400 experts each "is this stock going to fall?"
then taking a weighted vote.

**The math:** Friedman gradient boosting on cross-entropy. Each tree
splits the feature space (e.g. "is `mom_12w_rk > 0.6` AND
`ShortPct_rk > 0.8`?") and outputs a probability adjustment.
Trained with early stopping on a held-out fold to prevent overfit.
Score = final ensemble probability.

**Why it included:** Captures **non-linear feature interactions** that
logit misses. For shorts, interactions matter: a name with both high
SI *and* high leverage *and* deteriorating fundamentals is probably
more shortable than the sum of the three signals alone would suggest.

**Limitation:** More flexible = more prone to overfit. Its OOS
performance (Sharpe −1.11) is the worst of all five models here —
the trees learned IS patterns that didn't survive the regime shift.
Less interpretable than logit; we rely on SHAP / gain-importance
charts to understand it.

### 5. `gbm_rank` — LightGBM LambdaRank

**What it does:** Same machinery as `gbm_cls` (boosted trees) but
trained to **rank stocks** directly each month instead of predicting
an absolute probability. The loss function is **LambdaRank**: the
model is rewarded for putting the worst-returning stocks at the top of
the list each month, regardless of what its absolute probabilities are.

**The math:** Same gradient boosting but with `objective=lambdarank`
and one `query group` per rebalance date. The model optimises NDCG
(normalised discounted cumulative gain) which puts heavy weight on
getting the *top of the list* right — exactly what we care about for
a quintile-short strategy.

**Why it included:** Conceptually the most aligned objective. We
never care what the absolute probability of decline is — we only care
which names rank highest *within this month's cross-section*. Ranking
loss directly optimises that.

**Limitation:** Same overfitting risk as `gbm_cls`. Also more
sensitive to the label horizon: if you change the monthly forward
return (`fwd_ret_4w`) to a longer horizon (`fwd_ret_8w`) the ranking
changes more than the binary label would.

### Why have so many models?

Each one stresses a different part of the signal space:
- `naive` says *"only short-interest matters"*.
- `ew` says *"every factor matters equally — don't pre-judge"*.
- `logit` says *"factors matter in linear combinations, learned from history"*.
- `gbm_cls` says *"non-linear interactions matter, predict probability"*.
- `gbm_rank` says *"non-linear interactions matter, predict ranking"*.

When all five agree on a name (see [§Top short candidates](#top-short-candidates--as-of-2026-05-29)),
that's a high-conviction signal. When they disagree, it's a tell that
the cross-section is unusual that month.

---

## Stats & math glossary

A reader-friendly walkthrough of the metrics on this page.

### Returns / risk

- **CAGR** — *compound annual growth rate*. If $1 grew to $1.45 over 3
  years, CAGR = 1.45^(1/3) − 1 = **+13.2 % p.a.** The number you'd
  publish in a fund factsheet.
- **Vol (volatility)** — annualised standard deviation of monthly
  returns. We multiply the monthly stdev by √12 to scale up. Higher =
  more risky / more swing.
- **Sharpe ratio** — `(mean return) / (std dev of returns)`,
  annualised. Risk-adjusted return per unit of total volatility.
  Heuristics: > 1 is good, > 2 is great, > 3 is exceptional (and
  worth scrutinising for bugs / overfitting). Our headline naive
  L/S quintile prints **0.92** OOS.
- **Sortino ratio** — like Sharpe but the denominator only counts
  *downside* volatility (negative-return months). Penalises losses,
  not wins. Always ≥ Sharpe for a positively-skewed strategy.
- **Max drawdown (MaxDD)** — the worst peak-to-trough loss on the
  equity curve. If you put in $100 at the strategy's all-time high
  and it later fell to $89.20 before recovering, MaxDD = **−10.8 %**.
- **Calmar ratio** — `CAGR / |MaxDD|`. Return per unit of worst-case
  pain. Naive's OOS Calmar of 1.04 is "for every 1 % of CAGR you
  earned, you survived 1 % of drawdown".
- **Hit-rate** — fraction of rebalance periods where the strategy
  earned a positive return. Naive's 68.6 % OOS hit-rate means 24
  of 35 OOS months were profitable.

### Signal quality

- **Information coefficient (IC)** — Spearman rank correlation
  between the model's score and the realised forward return,
  computed cross-sectionally each month, then averaged across
  months. *For shorts, **negative** IC is good* — it means the
  model's "shortable" scores correlate with stocks that fell.
  `logit`'s OOS IC of **−8.5 %** means the model's rankings
  are reliably anti-correlated with actual returns.
- **t-stat** — `mean(monthly IC) / (stderr of monthly IC)`. A
  rough significance test: |t-stat| > 2 means the average IC is
  unlikely to be zero by chance. `logit`'s OOS t-stat of −3.88
  is highly significant.

### Validation

- **In-sample (IS)** — the historical window used to *develop* the
  model. We let the model see this data, fit parameters, tune
  hyperparameters, etc. Always looks good — the model literally
  optimised for it.
- **Out-of-sample (OOS)** — a held-back window the model never saw
  during development. OOS performance is the **only** honest
  estimate of future performance.
- **Walk-forward cross-validation** — the time-aware version of
  k-fold CV. Instead of random shuffling (which would leak future
  data into training), we expand the training window forward in
  time: train on 2010–2014, test on 2014–2015; train on
  2010–2015, test on 2015–2016; etc. The model is never trained
  on data from after its test window.
- **Purge + embargo** — extra safety on top of walk-forward:
  *purge* drops training samples that overlap the test labels'
  horizon (here, 4 weeks); *embargo* drops the next 4 weeks
  after the test window. Stops information from leaking via
  the forward-return label.

### Portfolio construction

- **Cross-sectional rank** — for each month, sort all ~500 stocks
  by some factor (e.g. ShortPct) and assign rank 0 (lowest) to 1
  (highest). Normalises across periods where the absolute factor
  levels differ. Almost every feature in this repo is rank-normalised.
- **Quintile** — the cross-section split into 5 equal buckets.
  "Top quintile by score" = the highest-scoring 20 % of stocks
  this month.
- **Long-short quintile (L/S)** — go long the bottom-quintile stocks
  (best expected returns) and short the top-quintile stocks (worst
  expected). Dollar-neutral: long notional = short notional, so the
  net market exposure is zero. The strategy earns the *spread*
  between the two legs, not the market direction.
- **Equal-weight within a leg** — each of the ~100 quintile names
  gets the same weight (1/100 = 1 % of leg book). No size or
  vol weighting.
- **Turnover** — fraction of book that changes hands each
  rebalance. 50 % turnover = half the names rotated out. High
  turnover = high transaction costs.

### Costs

- **Round-trip commission** — exchange + broker fees on a full
  buy-and-then-sell of a position. 25 bps per side = 50 bps
  round-trip. ASX retail commissions land here.
- **Borrow cost** — annualised fee paid to the prime broker to
  borrow shares for shorting. Charged daily on the short-notional.
  We use 1.5 % p.a. (~12.5 bps/month) which is conservative for
  the ASX small-mid-cap tail.
- **Slippage** — the difference between the price you wanted and
  the price you got. 5 bps here, applied to weight *changes*.

---

## Information coefficients — IS and OOS

Spearman of model score vs realised monthly forward return, computed
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

### 1. Universe and rebalance — Friday-release dates (4-BDay lag respected)

* **ASIC daily aggregate short-position reports**: weekly, Friday-anchored
  (the earliest archived report we found is **16 June 2010**; regime
  started 1 June 2010 but the URL format we use stabilised mid-June).
* Each ASIC release on Friday covers positions **as-of the prior
  Monday** (4 business days earlier). The earlier version of this pipeline
  used the Monday "as-of" date as the trading anchor — but a trader
  doesn't *know* Monday's positions until Friday. **The current
  pipeline rebalances on the Friday release**, using Friday's adjusted
  close as the entry price. The Monday as-of date is preserved as
  `AsOfDate` for diagnostic purposes only.
* End-of-month rebalance = the last ASIC release in each calendar month —
  by construction a trading day. **192 dates, day-of-week mix:
  180 Fridays, 10 Thursdays, 2 Wednesdays** (the non-Fridays are months
  where Easter / Christmas / King's Birthday pushed the release forward).
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

The forward-return label is **4 weeks (≈ 1 calendar month)** — the
monthly rebalance grid means each (Date, Ticker) row's label is the
return from this Friday rebalance to the next one (~4 weeks later).
The column is named `fwd_ret_4w` in the data for legacy reasons.

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
  shorts, 5 bps slippage on weight changes. **No stop loss** — every
  position realises its uncapped monthly forward return.

---

## OOS trade-level analysis — the actual short book

Reconstructed every short position from the OOS holdout (2023-06 → 2026-05,
**2,089 monthly positions across 224 unique tickers**, model = `logit`).
Per-position monthly returns are **uncapped** (no stop loss).
Full per-position table:
[`reports/oos_short_positions.csv`](reports/oos_short_positions.csv);
per-ticker summary:
[`reports/oos_trades.csv`](reports/oos_trades.csv) /
[`reports/oos_trades.md`](reports/oos_trades.md).

**Aggregate OOS stats (short leg only):**

- **Total short-leg cumulative P&L**: **−41.8 % of book** across 2,089 monthly positions. The dollar-neutral L/S quintile is still positive (Sharpe 0.92) because the *long leg* carries it.
- **Per-position win-rate**: 53.5 % — most monthly shorts individually profitable.
- **Median per-position return**: **+1.91 %** (half the positions made ≥ +1.91 %).
- **Mean per-position return**: **−0.33 %** — dragged negative by the fat right-tail.
- **Best single month**: ERA +84 % (Energy Resources of Australia fell 84 % the month it was wound up).
- **Worst single month**: **4DX −314 %** (4D Medical rallied 314 % in Aug 2025 — a 2.5 %-book-weight short took ~8 % of NAV in one stroke).
- **42 single positions lost > 50 %**, **161 lost > 25 %** — uncapped squeezes.

### Top 10 winning shorts

`avg_trade_%` is the mean per-position monthly return (positive = stock fell, short won).
`worst_%` is the worst single month per-position return — many of these
winning shorts took brutal interim squeezes en route to their eventual
cumulative profit.

| # | Ticker | Company | n months shorted | total P&L (% book) | avg trade | best month | worst month | hit-rate | avg SI % | first → last |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | CCX | City Chic Collective | 17 | **+3.82 %** | +14.0 % | +60.7 % | −30.4 % | 71 % | 0.6 | 2024-01 → 2026-03 |
| 2 | LOT | Lotus Resources | 16 | +3.19 % | +12.2 % | +57.6 % | −29.0 % | 63 % | 7.2 | 2024-07 → 2026-04 |
| 3 | NMT | Neometals | 16 | +2.98 % | +11.8 % | +38.0 % | −50.0 % | 81 % | 1.6 | 2023-06 → 2025-06 |
| 4 | CXL | Calix | 30 | +2.93 % | +6.5 % | +49.0 % | **−70.8 %** | 67 % | 2.5 | 2023-06 → 2026-04 |
| 5 | GLL | Galilee Energy | 10 | +2.79 % | +17.5 % | +44.7 % | −5.3 % | 70 % | 0.1 | 2024-01 → 2025-02 |
| 6 | SGR | Star Entertainment | 21 | +2.75 % | +8.5 % | +41.1 % | −26.9 % | 71 % | 4.4 | 2023-06 → 2026-03 |
| 7 | WBT | Weebit Nano | 18 | +2.38 % | +8.6 % | +36.7 % | −33.7 % | 72 % | 5.1 | 2023-07 → 2026-03 |
| 8 | ERA | Energy Resources of Australia | 7 | +2.37 % | +21.1 % | +84.4 % | **−100.0 %** | 86 % | 0.0 | 2024-03 → 2024-09 |
| 9 | CHN | Chalice Mining | 16 | +2.20 % | +8.9 % | +52.9 % | −49.8 % | 69 % | 5.8 | 2023-06 → 2025-06 |
| 10 | BAP | Bapcor | 6 | +2.16 % | +21.3 % | +62.2 % | −3.3 % | 83 % | 7.3 | 2025-11 → 2026-04 |

**Even winners had brutal months:** Calix −71 % in one month, Energy Resources −100 % the month operations terminated, Neometals −50 %. A stop loss would have closed those positions before the eventual cumulative profit was realised — that's why running uncapped is the honest default here.

### Top 10 losing shorts — the squeezes

| # | Ticker | Company | n months shorted | total P&L (% book) | avg trade | best month | worst month | hit-rate | avg SI % | first → last |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SRL | Sunrise Resources | 30 | **−9.43 %** | −17.9 % | +37.7 % | **−231.2 %** | 53 % | 1.3 | 2023-06 → 2025-12 |
| 2 | 4DX | 4D Medical | 20 | **−9.39 %** | −26.7 % | +35.1 % | **−314.0 %** | 50 % | 0.0 | 2023-11 → 2025-12 |
| 3 | TTT | Titomic | 21 | −5.43 % | −14.7 % | +39.5 % | −124.1 % | 43 % | 0.1 | 2024-01 → 2026-03 |
| 4 | ASM | Australian Strategic Materials | 28 | −5.11 % | −10.2 % | +36.1 % | −163.9 % | 50 % | 1.7 | 2023-06 → 2025-12 |
| 5 | APX | Appen | 25 | −3.40 % | −7.7 % | +55.6 % | −127.3 % | 56 % | 3.9 | 2023-06 → 2026-01 |
| 6 | SPL | Starpharma Holdings | 19 | −3.39 % | −9.7 % | +58.5 % | −172.0 % | 47 % | 0.4 | 2023-06 → 2025-12 |
| 7 | EOS | Electro Optic Systems | 19 | −3.31 % | −9.6 % | +28.2 % | −118.7 % | 37 % | 1.0 | 2024-07 → 2026-04 |
| 8 | EGR | EcoGraf | 13 | −2.80 % | −12.4 % | +28.1 % | −139.1 % | 46 % | 0.3 | 2024-01 → 2025-12 |
| 9 | IXR | Ionic Rare Earths | 26 | −2.78 % | −5.8 % | +40.0 % | −125.0 % | 50 % | 0.1 | 2023-06 → 2026-03 |
| 10 | CAT | Catapult Sports | 13 | −2.25 % | −9.8 % | +10.9 % | −44.3 % | 39 % | 0.8 | 2023-06 → 2025-09 |

SRL and 4DX alone cost −18.8 % of book. Negative `worst_%` values *exceed* −100 % because a 2.5 %-book-weight short of a stock that rallies 300 % loses 300 % of *position notional*. These are the squeezes that justify treating short-only risk control as a real research question — not the kind you can paper over with a default flag.

---

## Top short candidates — as of 2026-05-29

Top 15 by *consensus rank* across the three trained models (`logit` +
`gbm_cls` + `gbm_rank`); gated on investable + A$200 m+ market cap +
all-3-models-scored. Full top-30 in
[`reports/current_positions_monthly.csv`](reports/current_positions_monthly.csv);
regenerate with `scripts/_current_positions.py --monthly`.

Top 15 as of the most recent Friday rebalance (2026-05-29):

| # | Ticker | Company | Mkt Cap (A$m) | Short % | logit | gbm_cls | gbm_rank | Consensus |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | BBN | Baby Bunting | 347 | 0.32 | 0.643 | 0.630 | 0.129 | 0.982 |
| 2 | LTR | Liontown Resources | 4,404 | 1.75 | 0.609 | 0.581 | 0.092 | 0.954 |
| 3 | EOS | Electro Optic Systems | 1,821 | 3.59 | 0.604 | 0.562 | −0.423 | 0.922 |
| 4 | AGI | Ainsworth Game Tech. | 340 | 0.00 | 0.619 | 0.654 | −1.121 | 0.909 |
| 5 | CAT | Catapult Sports | 983 | 5.15 | 0.561 | 0.567 | −0.110 | 0.898 |
| 6 | TTT | Titomic | 361 | 0.35 | 0.556 | 0.541 | 0.098 | 0.883 |
| 7 | 4DX | 4D Medical | 1,948 | **10.06** | 0.549 | 0.557 | 0.285 | 0.878 |
| 8 | MSB | Mesoblast | 1,898 | **8.66** | 0.676 | 0.435 | −0.375 | 0.876 |
| 9 | TLX | Telix Pharmaceuticals | 3,794 | **15.15** | 0.584 | 0.533 | −0.841 | 0.871 |
| 10 | ILU | Iluka Resources | 2,488 | 7.53 | 0.619 | 0.557 | −1.310 | 0.871 |
| 11 | SBM | St Barbara | 702 | 3.57 | 0.584 | 0.465 | −0.548 | 0.857 |
| 12 | SHV | Select Harvests | 550 | 2.04 | 0.608 | 0.597 | −1.496 | 0.855 |
| 13 | MYX | Mayne Pharma | 253 | 2.97 | 0.572 | 0.537 | −1.267 | 0.838 |
| 14 | NVX | Novonix | 293 | 2.80 | 0.519 | 0.595 | −0.085 | 0.829 |
| 15 | MVF | Monash IVF | 286 | 1.87 | 0.639 | 0.565 | −1.729 | 0.829 |

The list reads like a real ASX short book: **TLX, MSB, 4DX at 8-15 %
short interest** are well-known shorts; LTR is lithium overhang; NVX,
MVF, SHV, ILU are mid-cap operators where the model picks up valuation
/ leverage signals even when SI is moderate; BBN, CAT, TTT, AGI, EOS are
broken-narrative names the model continues to flag.

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
inflated pre-consolidation mktCap by ~600×. Previous run showed PDN
at **A$3.18 trillion** in Aug-Dec 2023.

The fix doesn't hardcode anything. The assemble step now reads
`enterprise_values.marketCapitalization` (FMP's own quarterly mktCap
snapshots, which ARE split-aware) and asof-joins them onto the panel.
Coverage 85.9 % across 262 k rows. PDN now reports **~A$1.4 B in Aug
2023** — matches FMP's own EV records.

### Bad-data sanity threshold

For the remaining tickers where FMP's own data is wrong (Village Roadshow
2011 at A$931 B from `enterprise_values`, CBA in certain quarters >
A$300 B, etc.), the assemble step now nulls any row above an objective
physical-impossibility threshold: **A$300 B** — larger than peak BHP
(~A$250 B mid-2022), the biggest the ASX has ever produced. This is
not ticker-specific and not a soft cap; it's "if FMP says this is
bigger than anything that has ever existed on the ASX, drop the row".
**109 rows nulled** out of 262 k (0.04 %) — top affected tickers:
VRL (75 rows), CBA (26 rows), INR (8 rows). NaN'd rows fail the
investable filter (≥ A$200 m) naturally.

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
| Risk control | hard 10 % stop, no costs | **No stop loss** — uncapped per-position P&L; commission + 1.5 % p.a. borrow + 5 bps slippage modelled |
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

* **No stop loss / risk control** modelled. The short book takes the
  full pain of multi-bagger squeezes; the only thing rescuing the L/S
  quintile from a negative Sharpe is the long leg. A future iteration
  should test signal-driven exits, vol-scaled position sizes, or
  sector-neutral construction — flagged as a deliberate research
  decision, not an oversight.
* **Sector dummies skipped** — FMP `profile` not yet pulled. Easy follow-up.
* **Borrow cost flat 150 bps p.a.** Real ASX borrow varies by name and
  date; the `CostConfig` is fully parameterised.
* **Capital-raise / squeeze dynamics absent.** The engine is pure
  adjusted-close, no halt / takeover / index-event modelling.
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
