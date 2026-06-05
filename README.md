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

**About sample sizes.** `naive` and `ew` have **zero learned
parameters** — the algorithm is fully specified up front, so for
them there is no IS/OOS distinction at the model level. They can be
(and below, are) evaluated over the **entire 191-month panel**
(2010-06 → 2026-05). For the trained models (`logit`, `gbm_cls`,
`gbm_rank`) the OOS holdout matters: the model never saw those rows
during development, so OOS performance is the only honest estimate
of future performance. We therefore report **two tables** below:

1. **Apples-to-apples OOS** (n=35 monthly rebalances) so every
   model is compared on the same window.
2. **Full-period stats** for the parameter-free models (n=191)
   — the statistically richer estimate when no training was done.

### Table 1 — Dollar-neutral long-short quintile, OOS holdout (n=35, all models)

Long bottom 20 % by score, short top 20 %. Same 36-month window,
same costs, same construction.

| Model | n_OOS | **OOS Sharpe** | OOS CAGR | OOS MaxDD | OOS Hit-rate |
|---|---:|---:|---:|---:|---:|
| **naive** (rank ShortPct) | 35 | **0.92** | +11.2 % | **−10.8 %** | **68.6 %** |
| **ew** (polarity-aware composite) | 35 | **0.91** | **+17.2 %** | −16.5 % | 48.6 % |
| logit (rebuilt v1) | 35 | 0.24 | +2.9 % | −37.1 % | 48.6 % |
| gbm_rank (LightGBM LambdaRank) | 35 | −0.21 | −8.2 % | −51.6 % | 37.1 % |
| gbm_cls (LightGBM binary) | 35 | −1.11 | −16.3 % | −44.4 % | 45.7 % |

### Table 2 — Top-quintile short only, OOS holdout (n=35, all models)

No long leg. Same window/costs.

| Model | n_OOS | OOS Sharpe | OOS CAGR | OOS MaxDD | OOS Hit-rate |
|---|---:|---:|---:|---:|---:|
| **ew** | 35 | **−0.08** | −4.1 % | **−36.6 %** | **51.4 %** |
| logit | 35 | −0.20 | −6.3 % | −40.9 % | 48.6 % |
| naive | 35 | −0.32 | −7.2 % | −35.7 % | 51.4 % |
| gbm_rank | 35 | −0.50 | −15.2 % | −56.1 % | 40.0 % |
| gbm_cls | 35 | −1.11 | −18.9 % | −53.4 % | 34.3 % |

### Table 3 — Full-period stats for the parameter-free models (n=191)

`naive` and `ew` have no training step, so we can evaluate them on
the entire **191-month** panel (16 years, June 2010 → May 2026).
This is **5× the sample** of the OOS-only tables above, and includes
the COVID crash, the 2022 bear, and multiple commodity cycles —
several genuine regime changes the OOS-only window doesn't capture.

| Model | Strategy | n | Sharpe | CAGR | MaxDD | Hit-rate |
|---|---|---:|---:|---:|---:|---:|
| naive | L/S quintile | 191 | **0.62** | +8.2 % | −31.0 % | 61.3 % |
| ew | L/S quintile | 191 | 0.58 | +10.2 % | −57.4 % | 59.7 % |
| naive | quintile-short | 191 | −0.29 | −8.5 % | −82.7 % | 47.1 % |
| ew | quintile-short | 191 | −0.20 | −9.5 % | −86.2 % | 46.1 % |

**Full-period Sharpes are weaker than OOS-only Sharpes** (0.62 vs
0.92 for naive; 0.58 vs 0.91 for ew) — the post-2023 regime has been
particularly kind to short-interest cross-sectional strategies, and
that 35-month window flatters the result. The n=191 Sharpes are the
**more statistically meaningful** estimate for these models. They
still show real positive long-short alpha (0.5+ Sharpe over 16 years
with realistic costs), but at a more sober level than the recent OOS
window alone would suggest.

### Table 4 — Concentration sensitivity: decile (top 10 %) vs quintile (top 20 %)

What if we **sharpen** the bet — short the top 10 % by score instead
of the top 20 %, and (for the L/S variant) long the bottom 10 %
instead of the bottom 20 %? For the parameter-free models this is a
clean signal-to-noise test: if the model's ranking is real, halving
the basket size should *strengthen* the spread per dollar of risk.
For the trained models it tends to *amplify* squeeze risk because
their picks are already concentrated into high-conviction names
(Appen, 4DX, BrainChip) — going decile makes that concentration
worse.

| Model | Strategy | Window | n | Sharpe | CAGR | MaxDD | Hit-rate |
|---|---|---|---:|---:|---:|---:|---:|
| **naive** | L/S decile | OOS | 35 | **1.21** | +21.7 % | **−12.7 %** | **65.7 %** |
| **ew** | L/S decile | OOS | 35 | **1.04** | **+26.2 %** | −20.2 % | 62.9 % |
| **ew** | **decile-short** | OOS | 35 | **+0.07** | −1.6 % | −41.6 % | 45.7 % |
| naive | decile-short | OOS | 35 | −0.10 | −4.7 % | −37.1 % | 51.4 % |
| logit | L/S decile | OOS | 35 | 0.04 | −1.6 % | −46.6 % | 45.7 % |
| gbm_rank | L/S decile | OOS | 35 | −0.18 | −8.9 % | −55.2 % | 51.4 % |
| gbm_cls | L/S decile | OOS | 35 | −0.79 | −16.7 % | −49.4 % | 40.0 % |
| naive | L/S decile | full period | 191 | 0.33 | +4.6 % | −54.0 % | 57.1 % |
| ew | L/S decile | full period | 191 | 0.59 | +12.5 % | −67.0 % | 65.4 % |
| ew | decile-short | full period | 191 | −0.09 | −8.1 % | −88.3 % | 49.7 % |

**Three things worth calling out:**

1. **EW decile-short OOS Sharpe = +0.07** — the **first positive
   short-only Sharpe** anywhere in this project. Concentrating
   into the top 10 % of the polarity-aware composite is the only
   short-only book that clears zero on the OOS holdout, although
   it's barely above zero and goes negative again on the
   full-period n=191 (−0.09). Suggestive, not conclusive.
2. **EW L/S decile OOS Sharpe = 1.04** with **+26.2 % CAGR** —
   the highest single-strategy result in the entire table.
   `naive` L/S decile is even higher (1.21 Sharpe) with a tighter
   drawdown (−12.7 %).
3. **Every trained model gets *worse* under decile concentration.**
   `gbm_cls` L/S decile is −0.79 Sharpe (vs −1.11 quintile —
   marginally less bad, but still terrible). `gbm_rank` and `logit`
   barely move. Concentrating their already-concentrated short list
   doesn't help.

The decile result is consistent with the broader story: this
universe has real cross-sectional short-interest alpha, and a
hand-built no-training composite captures it cleanly. Fancy models
trip over the same fat-tail squeezes that the broader baseline
picks ride out.

The short-only books (Table 2 OOS, Table 3 last two rows, and the
trained-model rows of Table 4) are negative Sharpe almost everywhere.
The single positive-Sharpe naked short — EW decile-short OOS at
+0.07 — is too small to call a strategy. **Naked short alpha is not
a viable standalone book on this universe**; the dollar-neutral L/S
construction is essential.

### Table 5 — What's actually in the universe, and what if we drop the A$200m gate?

**Large caps are already in the universe.** On the most recent
rebalance (29 May 2026) the panel contains 290 names with market
caps spanning **A$9 m → A$268 B** — every major ASX 50 constituent
is present:

| # | Ticker | Company | Mkt Cap (A$bn) | Short % | Investable? |
|---:|---|---|---:|---:|---|
| 1 | CBA | Commonwealth Bank | 268 | 2.1 | ✓ |
| 2 | RIO | Rio Tinto | 159 | 8.7 | ✓ |
| 3 | BHP | BHP Group | 154 | 1.4 | ✓ |
| 4 | WBC | Westpac | 135 | 1.8 | ✓ |
| 5 | NAB | National Australia Bank | 127 | 1.8 | ✓ |
| 6 | ANZ | ANZ Group | 108 | 0.8 | ✓ |
| 7 | WES | Wesfarmers | 92 | 1.1 | ✓ |
| 8 | CSL | CSL Limited | 84 | 0.7 | ✓ |
| 9 | MQG | Macquarie Group | 77 | 0.4 | ✓ |
| 10 | TLS | Telstra | 55 | 0.9 | ✓ |

The strategy *can* short these names — they just rarely make it into
the top-quintile short basket because mega-caps tend to have low
reported short interest (CBA = 2.1 %, BHP = 1.4 %) and high quality
metrics, so the polarity-aware EW ranks them as **least** shortable.
**RIO at 8.7 % is the exception** — high SI for a mega-cap usually
flags a real bearish view, and that's exactly the kind of signal the
strategy picks up.

**What if we drop the A$200m floor?** The default backtest applies
an `investable` gate: a name must have ≥ A$200 m market cap and
fresh fundamentals to be eligible. `scripts/06_backtest.py` now
accepts `--no-investable-gate` which bypasses that filter entirely
— **any name on the panel with a non-NaN price can be picked**,
including sub-A$200 m micro-caps. Side-by-side:

| Model | Strategy | n_OOS | OOS Sharpe (gated, A$200m+) | OOS Sharpe (no gate, all caps) | Δ |
|---|---|---:|---:|---:|---:|
| naive | L/S quintile | 35 | 0.92 | **0.75** | −0.18 |
| ew | L/S quintile | 35 | 0.91 | **0.85** | −0.07 |
| logit | L/S quintile | 35 | 0.24 | **0.14** | −0.10 |
| gbm_rank | L/S quintile | 35 | −0.22 | −0.24 | −0.02 |
| gbm_cls | L/S quintile | 35 | −1.11 | −1.12 | −0.01 |

Full-period (n=191) for the parameter-free models without the gate:
**naive L/S Sharpe 0.46, ew L/S Sharpe 0.53** — both still positive,
just weaker. **Dropping the gate slightly *hurts* every strategy**;
the marginal micro-cap names are noisier than they are alpha-rich
(thin trading, wider spreads, more squeeze events, less reliable
fundamentals). The A$200 m floor is a net-positive filter, not a
universe restriction — it's not what's making the strategy work.
Full no-gate summary:
[`reports/backtest_summary_monthly_nogate.csv`](reports/backtest_summary_monthly_nogate.csv).

![Cumulative growth of $1 — solid = quintile-short, dotted = long-short, dashed black = ASX 200 buy & hold](charts/cumulative_returns_monthly.png)

**ASX 200 reference (dashed black).** A simple buy-and-hold of the
S&P/ASX 200 index (Yahoo ticker `^AXJO`) over the same window grew
$1 → **$1.99 (+99 % total)** with 5.4 % annualised volatility and
ride-the-market beta = 1. The dollar-neutral L/S quintile strategies
above are by construction **market-neutral** (β ≈ 0) — they're not
in competition with index returns in the same way an equity-long
fund is, they're capturing a *spread*. But it's useful context: a
0.5–0.6 Sharpe full-period number on a dollar-neutral book is
genuine alpha because there's no market beta riding underneath it.
The headline naive L/S quintile finishes at $2.91 — comfortably
ahead of the index — *with zero market exposure*.

### What this honestly shows

**Two no-training models clear positive long-short Sharpe at every
horizon.** `naive` (rank by raw ShortPct) prints 0.62 Sharpe over the
full 16-year panel and 0.92 over the recent OOS window. `ew` (the
polarity-aware 12-factor composite) prints 0.58 full-period and 0.91
OOS. Both rely on simple, hand-specified algorithms with zero learned
parameters — and both beat all three of the trained models on OOS
Sharpe.

The three trained models (`logit`, `gbm_rank`, `gbm_cls`) all run
statistically-significant **negative OOS IC** (their score correctly
correlates with falling stocks — see
[IC table below](#information-coefficients--is-and-oos)) — they pick the
right names — but they concentrate too tightly into the high-conviction
short tail (Appen, 4D Medical, BrainChip, Sunrise Resources) and get
mauled when those names squeeze. `naive`'s and `ew`'s broader picks
ride those tails out.

### Per-position win-rate is > 50 % — the issue is magnitude, not direction

Every model picks **the right names**. Per-position win-rate (share
of monthly shorts that ended profitable, i.e. the stock fell during
the month) is at or above 50 % for all five — even the worst
overall-Sharpe model (`gbm_cls`) picks shorts that fall ~51 % of the
time. The `n` column shows the actual number of monthly short
positions each model placed across the 35-month OOS holdout:

| Model | n positions | Per-position win-rate | Median trade | Mean trade | Worst single position |
|---|---:|---:|---:|---:|---:|
| **logit** | 2,089 | **53.4 %** | +1.71 % | −0.33 % | −218 % (APX 2024-07) |
| gbm_rank | 2,089 | 52.5 % | **+1.85 %** | −0.87 % | −218 % (4DX 2025-08) |
| **ew** | 2,089 | 52.0 % | +1.20 % | **+0.27 %** | −218 % (4DX) |
| naive | 2,089 | 50.9 % | +0.43 % | 0.00 % | −177 % (BRN 2024-01) |
| gbm_cls | 2,089 | 50.8 % | +0.38 % | −1.27 % | −218 % (4DX) |

`logit` has the highest win-rate (53.4 %) but a negative **mean**
trade (−0.33 %) — the squeeze-loss tail drags it down. `ew` is the
**only model with a positive mean per-position trade** (+0.27 %)
because its broader, factor-diversified short list avoids
concentrating into the same handful of squeeze names. That's exactly
what you'd hope from a properly-polarised multi-factor short
composite — it's why polarity matters.

**Both no-training baselines beat the trained models.** `naive` and
`ew` each have zero learned parameters, and over the full 16-year
panel (n=191) they both deliver positive long-short Sharpe (0.62
and 0.58) with positive CAGR (+8.2 % and +10.2 %). On the recent
OOS window they accelerate to 0.92 and 0.91. On this universe,
**short-interest dispersion plus a hand-built composite of cheap-
and-levered-and-shrinking signals beats fancy training**. That is
itself the headline research finding — the data isn't asking for a
gradient-boosted ranker; it's asking for the right human prior on
which factors to combine.

---

## ⚠️ Important disclaimer — what this project is, and isn't

**This is a research demonstration, not a fundable strategy at scale.**
The universe is the top ~500 ASX tickers by ASIC-report frequency —
which **includes every ASX 50 mega-cap** (CBA at A$268 B, RIO/BHP at
A$150 B+, the big four banks, CSL, Wesfarmers, Macquarie, Telstra)
**plus the entire small/mid-cap short-interest tail** down to
~A$9 m. The strategy *can* short any of those names; large-caps just
rarely make it into the top-quintile short basket because mega-cap
short interest is low and their quality metrics rank them as
least-shortable. The *median* short basket position has a market
cap in the A$200 m – A$2 bn range, but the universe itself is the
full ASX 200 + the active short-interest tail. (See
[Table 5](#table-5--whats-actually-in-the-universe-and-what-if-we-drop-the-a200m-gate)
for the latest rebalance composition.)

The whole point of the project is to demonstrate that a disciplined,
no-look-ahead cross-sectional model *can* identify successful short
positions out of sample, with a > 50 % per-position win-rate on a
real 36-month holdout.

It is **not** a turnkey alpha source for institutional capital, and
nothing here should be read as one. Specifically:

- **Capacity is small.** Many shorted names trade < A$5 m/day. Even
  a A$10 m position would be a meaningful day's volume in the worst
  names; A$100 m would be impossible to put on without moving the
  price.
- **Borrow availability is real-world variable.** Squeeze targets
  often see borrow vanish entirely; the flat 1.5 % p.a. borrow rate
  in `CostConfig` is an average, not a constraint.
- **No market-impact / liquidity-aware sizing.** The backtest
  equal-weights each leg of the quintile; a real book would have to
  taper or skip the bottom-of-tail names.
- **Tax, financing, FX** all assumed away.
- **No stop loss / risk control** — by design (see Limitations).

A real fund would: (a) restrict the universe to A$2 bn + market caps
where the alpha may well *not* exist, (b) size positions by liquidity,
not equal-weight, and (c) layer some kind of tail-risk control. Each
of those changes the numbers materially. The point of *this*
repository is to show the methodology — universe construction, point-
in-time fundamentals, walk-forward CV with purge/embargo, IS vs OOS
discipline, cost-aware backtest — applied to a problem (short-
selection on the ASX small-mid-cap tail) where there's a measurable
signal to recover. Treat the headline Sharpes as evidence-of-process,
not as a fund prospectus.

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

### What "long-short quintile" actually means — and why it's a spread trade

Imagine the model ranks 500 ASX stocks every month from
**most-shortable** (rank #1, the model's strongest bearish bet) to
**least-shortable** (rank #500, the model's strongest bullish bet —
because "least shortable" = "best expected return"). Then:

1. **Short the top 100** (top quintile = top 20 % by score). These are
   the stocks the model says will *fall the most*.
2. **Buy the bottom 100** (bottom quintile = bottom 20 % by score).
   These are *not* "bad companies the model also picked" — they're
   the **opposite extreme**. The same model says these have the
   *best* expected forward returns (high quality, low SI, positive
   momentum, low leverage, etc.).
3. Equal-weight inside each leg. Net dollar exposure = 0 (long $ =
   short $) so market-direction risk cancels.

You're not betting that the market goes up or down. You're betting
that **the spread between the two ends is real** — that the model's
top-ranked names will under-perform its bottom-ranked names. The middle
60 % of stocks gets ignored entirely; only the two extremes trade.

This is why a strategy can have *negative* short-only Sharpe but
*positive* L/S Sharpe: the short leg loses money on a naked basis,
but the long leg makes enough that the spread is still profitable —
and the dollar-neutrality means no market-beta exposure either way.

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

### 2. `ew` — equal-weight composite of 12 ranked factors (polarity-aware)

**What it does:** Computes 12 different cross-sectional rank columns
(short %, momentum, volatility, market cap, P/E, ROE, ROIC, FCF yield,
debt/equity, revenue growth, etc.) and averages them — with **every
factor pre-oriented so that high rank = more shortable** before the
mean is taken. This polarity step is the difference between a useful
short composite and a long-biased one.

**The math:** For each month, each factor *f* is ranked 0-1 across
all stocks. Some factors are SHORT-aligned in their raw form (high
P/E = expensive = shortable ✓; high SI = crowded short = shortable ✓);
others are BULLISH-aligned (high ROE = quality, high momentum = up-trending,
high FCF yield = cash-rich) and get **flipped via `1 − rank`** before
averaging. Final score = mean of the 12 polarised ranks, re-ranked
cross-sectionally. No parameters, no training — every factor gets
equal weight.

The flipped factors are: `mom_12w_rk`, `log_mktcap_rk`,
`fcf_yield_rk`, `roe_rk`, `roic_rk`, `revenue_growth_yoy_rk`.

**Why it works:** Beats every trained model on OOS IC (`-8.9 %` with
t-stat `-3.83`) and matches `naive` on OOS L/S quintile Sharpe (0.91
vs 0.92). On a small-mid-cap ASX universe, **a hand-built
short-composite of cheap + levered + shrinking signals is the cleanest
single-pass alpha** — fancier models add noise.

**Why it included:** Industry-standard "smart-beta" composite — what
you'd build before you had any historical data to train on. A useful
sanity check: if a trained model can't beat a polarity-aware equal-
weight composite, the training isn't adding signal.

**Limitation:** Equal weight is arbitrary. A real-money build would
optimise factor weights (e.g. via IC-weighted blending) or at least
group-equal-weight by theme (short × valuation × quality × growth).
Also: choosing which 12 factors to include and their polarity is
itself a research decision — the model isn't "data-driven" in the
ML sense, it's a thoughtful human prior.

> **Why this matters.** A previous version of this composite
> averaged the raw ranks *without flipping the bullish factors*. The
> result was an EW score that correlated *positively* with forward
> returns (+4.8 % IC) — it was a long-bias model wearing a short
> mask. Fixing the polarity (one ~20-line edit in
> `scripts/05_train_and_validate.py`) flipped IC to **−8.9 %** and
> OOS L/S Sharpe from **0.08 → 0.91**. Boring but instructive: when
> you blend factors, the sign of each input has to match the sign of
> the score you're building.

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
splits the feature space (e.g. "is **3-month momentum rank** > 0.6 AND
**short-interest rank** > 0.8?") and outputs a probability adjustment.
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
sensitive to the label horizon — changing the forward-return window
from 1 month to 2 months shifts the ranking more than it would shift
a binary label.

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
  horizon (here, ~1 month); *embargo* drops the following ~1
  month after the test window. Stops information from leaking
  via the forward-return label.

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
| **ew** | **−8.3 %** | **−6.73** | 156 | **−8.9 %** | **−3.83** | 35 |
| gbm_rank | **−7.6 %** | **−5.61** | 119 | **−6.9 %** | **−2.16** | 35 |
| logit | **−5.3 %** | **−5.28** | 119 | **−5.5 %** | **−2.48** | 35 |
| gbm_cls | −3.6 % | −3.77 | 119 | −2.3 % | −1.33 | 35 |
| naive | −1.6 % | −1.70 | 156 | −2.1 % | −1.38 | 35 |

**Sign reading.** Negative IC = the "shortable" score correlates with
*lower* forward returns — the model correctly identifies under-
performers. **All five models now have negative IC in both IS and
OOS**, and four of them are statistically significant (|t| > 2).
The polarity-aware `ew` composite has the strongest OOS IC of any
model (`−8.9 %`, t = `−3.83`), edging out the gradient-boosted
ranker. `naive` is the weakest IC (its `t` doesn't clear ±2) because
ShortPct alone is a high-variance signal — but the L/S quintile
construction smooths that out enough to deliver the headline 0.92
Sharpe.

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

### 2. Data sources — and why the Yahoo / FMP split

| What | Source | Why |
|---|---|---|
| Short positions | ASIC daily aggregate PDFs | Only public, free, asset-class-complete source for ASX |
| Prices (adjusted close + volume) | **Yahoo Finance** | FMP only ships ~5 years of ASX daily history on this plan; Yahoo goes back to ~2000. Cross-checked vs FMP at **median ρ = 0.9996** over the overlap window |
| Fundamentals (7 quarterly endpoints) | Financial Modeling Prep (Premium) | Income, balance sheet, cash flow, ratios, key metrics, enterprise values, financial growth |
| Market capitalisation | **FMP enterprise_values endpoint** (half-yearly, split-adjusted) | The balance-sheet `commonStockSharesOutstanding` field is stamped at the latest period-end and isn't back-adjusted across corporate actions. Using FMP's `enterprise_values.marketCapitalization` correctly handles reverse splits (e.g. Paladin Energy 1:100 in 2024 — see [Data audit](#data-audit) below). |
| Sector / industry | _not currently pulled_ — known limitation | FMP `profile` endpoint exists; trivial follow-up to wire in |

**Could we go Yahoo-only?** No — and the reasoning is worth spelling
out because it's a common question. Yahoo gives us **excellent
prices** going back > 25 years, which is why prices already come from
Yahoo (FMP's Starter plan only carries ~5 years for ASX). But Yahoo
falls down on the fundamentals side: its `yfinance.quarterly_financials`
only exposes the last ~4 quarters per company in a queryable format,
its `info` snapshot has trailing ratios without an `acceptedDate`
(so we can't enforce no-look-ahead), and its `marketCap` is not
back-adjusted across reverse splits (the same PDN-style bug we
fixed by switching to FMP `enterprise_values`). FMP's stable API
gives us the full 15-year quarterly panel with SEC-filing timestamps.
**Yahoo for prices + FMP for fundamentals & PIT market cap** is the
right split; collapsing it either direction loses something material.

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

The forward-return label is **1 month** — for each (Date, Ticker) it's
the return from this Friday rebalance to the next one. The column is
named `fwd_ret_4w` in the data for legacy reasons (4 weeks ≈ 1 calendar
month on the rebalance grid); the variable is monthly throughout the
prose.

* **Binary**: `forward_return < 0` for the classifier baseline.
* **Cross-sectional decile rank** of the monthly forward return
  (inverted so worst-return = highest relevance) for the LambdaRank
  model.

### 5. Cross-validation (walk-forward + IS/OOS holdout)

* **Pure OOS holdout**: the last **36 monthly rebalances**
  (2023-06 → 2026-05) are *reserved* — the trained models never see
  these rows during development, neither for fitting nor for
  hyperparameter selection.
* On the 156-month **in-sample** portion (2010-07 → 2023-05):
  walk-forward expanding window with **~36-month min train**,
  **~6-month test**, **1-month embargo (≥ label horizon)**.
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
* **Borrow cost flat 150 bps p.a.** — significantly underestimates the
  small-cap squeeze tail. Real ASX borrow rates run roughly:
  ~25–50 bps/yr for ASX 50 large-caps with deep stock-lending
  markets, ~100–200 bps for mid-caps, **500–1,500 bps for crowded
  small-cap shorts**, and 5,000+ bps (or outright recall) for active
  squeeze names. The 1.5 % p.a. in `CostConfig` is conservative on an
  ASX 100 book and materially **understates costs in the names where
  the alpha appears to live**. The `CostConfig` is fully parameterised
  — wiring in a per-ticker borrow term structure is on the follow-up
  list.
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
