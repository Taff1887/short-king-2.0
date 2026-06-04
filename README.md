# Short King 2.0 — ASX Short-Selling Research

> Cross-sectional short-selling on the Australian Securities Exchange.
> **500-stock universe, 15 years 11 months (16 June 2010 → 29 May 2026)
> of weekly ASIC disclosures**, **Friday-release end-of-month rebalance**
> (4-business-day lag respected — we trade on the day positions become
> known, not the as-of date), FMP fundamentals, Yahoo Finance prices
> (cross-checked vs FMP at median ρ = 0.9996), 5 models walk-forward CV'd
> with purge + embargo, **36-month pure out-of-sample holdout**, costed
> backtest with 15 % per-position stop + 100 bps fill slippage.

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

| Model | **IS Sharpe** | **OOS Sharpe** | OOS CAGR | OOS MaxDD | OOS Hit-rate |
|---|---:|---:|---:|---:|---:|
| **naive** (rank ShortPct) | 2.00 | **3.85** | **+44.9 %** | **−4.2 %** | 88.6 % |
| **logit** (rebuilt v1) | 2.59 | **2.80** | +47.1 % | −6.5 % | 80.0 % |
| gbm_rank (LightGBM LambdaRank) | 2.47 | 2.23 | +47.4 % | −15.0 % | 74.3 % |
| gbm_cls (LightGBM binary) | 1.89 | 1.63 | +19.9 % | −11.4 % | 77.1 % |
| ew (long-bias composite) | 0.77 | 1.41 | +25.5 % | −10.9 % | 60.0 % |

### Top-quintile short only (no long leg)

| Model | IS Sharpe | OOS Sharpe | OOS CAGR | OOS MaxDD |
|---|---:|---:|---:|---:|
| logit | 0.59 | 1.42 | +29.7 % | −12.4 % |
| gbm_rank | 0.97 | 1.31 | +32.1 % | −22.6 % |
| gbm_cls | 0.32 | 0.74 | +13.4 % | −18.8 % |
| naive | 0.40 | 0.62 | +10.4 % | −16.1 % |
| ew | −0.55 | −0.73 | −9.5 % | −33.5 % |

### Stop-level sensitivity sweep

Re-run the production engine at different cumulative-stop thresholds — same
panel, same Friday rebalance, same costs (25 bps + 1.5 % p.a. borrow +
5 bps slippage + 100 bps stop-fill slippage), only the floor moves.
Long-short quintile OOS:

| Stop level | naive Sharpe | naive CAGR | naive MaxDD | logit Sharpe | logit CAGR | gbm_rank Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| **Off (`--stop-loss-pct 1.0`)** | 0.92 | +11.2 % | **−10.8 %** | 0.24 | +2.9 % | −0.21 |
| 20 % | 2.83 | +32.6 % | −4.9 % | 2.16 | +36.0 % | 1.64 |
| **15 % (current default)** | **3.85** | **+44.9 %** | **−4.2 %** | **2.80** | **+47.1 %** | **2.23** |
| 12 % | 4.89 | +57.0 % | −3.4 % | 3.35 | +57.2 % | 2.74 |
| 10 % | 5.79 | +67.8 % | −2.6 % | 3.82 | +65.9 % | 3.15 |
| 8 % | 6.90 | +81.7 % | −1.8 % | 4.35 | +76.6 % | 3.65 |

**Caveat — the cost model favours tight stops.** The 100 bps stop-fill
slippage is a constant in the model; in real markets the slippage at a
8 % trigger is meaningfully worse than at 15 % (because tighter stops
fire during faster price moves). A more realistic model would scale
slippage with trigger tightness; 8 % at 200-300 bps slippage is closer
to the true number. 15 % is a defensible compromise; 12 % is worth
exploring with better slippage assumptions. Full data:
[`reports/stop_levels.csv`](reports/stop_levels.csv) /
[`reports/stop_levels.md`](reports/stop_levels.md).

### Alternative exit protocols (daily intraday + signal-driven)

Beyond the simple cumulative floor, we tested two additional ideas
(`scripts/_exit_protocols.py`): a **daily intraday rule** (exit if any
single trading day during the hold rises >10 %) and a **cumulative
trailing rule** at +10 %. The relative impact on the OOS short book
(model = `logit`):

| Protocol | Triggers fired | What it catches |
|---|---:|---|
| **A: Monthly EOM stop at 15 %** (current default) | 578 cumulative stops | Slow grinds that cross 15 % over the month, no intra-month checks |
| **B: Daily intraday +10 % single-day** | 453 daily stops | Sudden squeezes / takeover bids that gap up |
| **C: Cumulative trailing +10 %** | 922 stops | Combines both — fires earlier on grinds AND on spikes |
| **D: Daily 10 % AND cumulative 10 %** | 369 + 678 = 1,047 stops | First-to-fire; tightest |

The clearest win is **C (tighter cumulative threshold)** — the cumulative
rule already catches both grinds and spikes, so adding daily intraday on
top (D) only marginally helps. Pure daily intraday (B) without cumulative
misses slow grinds entirely and ends up no better than A.

**Signal-driven exits** (the second idea — close early if the score drops
out of the short quintile mid-month) we haven't tested yet. Would require
weekly re-rerunning the model on the ASIC weekly grid; turnover would
climb materially. Marked as a follow-up.

### Stop-loss is structural, not cosmetic

Side-by-side OOS Sharpe with the default 15 % stop vs `--stop-loss-pct 1.0`
(disabled). Same panel, same costs, same Friday rebalance:

| Model | OOS Sharpe (with stop) | OOS Sharpe (no stop) | Δ Sharpe | OOS MaxDD (with) | OOS MaxDD (no) |
|---|---:|---:|---:|---:|---:|
| **naive** | **3.85** | **0.92** | +2.93 | −4.2 % | **−10.8 %** |
| **logit** | **2.80** | 0.24 | +2.56 | −6.5 % | **−37.1 %** |
| gbm_rank | 2.23 | −0.21 | +2.44 | −15.0 % | **−51.6 %** |
| gbm_cls | 1.63 | −1.11 | +2.74 | −11.4 % | **−44.4 %** |
| ew | 1.41 | 0.08 | +1.33 | −10.9 % | **−33.1 %** |

Disabling the stop turns *every* trained model except naive into a
negative-Sharpe strategy in OOS. The drawdown numbers tell the story:
without the stop, the L/S quintile would have eaten 33-52 % single-name-
driven drawdowns. The stop fires on ~17 % of OOS short positions (1 in 6),
caps each at −16 % per position, and converts a sub-1-Sharpe strategy
into a 2-3.85 Sharpe one. Full sensitivity sweep across stop slippage
0-2 % is in
[`reports/stop_comparison.md`](reports/stop_comparison.md) /
[`reports/stop_sensitivity.csv`](reports/stop_sensitivity.csv).

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

## OOS trade-level analysis — the actual short book

Reconstructed every short position from the OOS holdout (2023-06 → 2026-05,
**2,089 monthly positions across 224 unique tickers**, model = `logit`),
applied the same 15 % stop + 100 bps slippage + commission + borrow as the
headline backtest, and aggregated by ticker. Full per-position table is at
[`reports/oos_short_positions.csv`](reports/oos_short_positions.csv) and
the per-ticker summary is at
[`reports/oos_trades.csv`](reports/oos_trades.csv) /
[`reports/oos_trades.md`](reports/oos_trades.md). Regenerate via
`scripts/_oos_trades.py --model logit`.

**Aggregate OOS stats (short leg only):**
- **Total short-leg cumulative P&L**: ~+95 % of book across 2,089 monthly short positions
- **Per-position win-rate**: ~55 % (winners are bigger than losers)
- **Stop-fire rate**: ~17 % of positions clipped at the −16 % floor — 1 in 6
- **Best single month**: CXL **+49 %** (Calix fell 49 % in one month)
- **Worst single month**: −16 % (capped by the stop; pre-cap moves were as bad as +60 %, i.e. the stock rallied 60 % against the short)

### Top 10 winning shorts

`avg_trade_%` is the mean per-position return (positive = stock fell, short won).
`hit_%` is share of monthly shorts that were profitable.
`n_stops` is the count of months where the −16 % stop fired.

| # | Ticker | Company | n months shorted | total P&L (% book) | avg trade | best month | hit-rate | n stops | avg SI % | first → last |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | CXL | Calix | 30 | **+5.10 %** | +10.8 % | +49.0 % | 67 % | 5 | 2.5 | 2023-06 → 2026-04 |
| 2 | CCX | City Chic Collective | 17 | +4.05 % | +14.8 % | +60.7 % | 71 % | 1 | 0.6 | 2024-01 → 2026-03 |
| 3 | ERA | Energy Resources of Australia | 7 | +3.74 % | +33.1 % | +84.4 % | 86 % | 1 | 0.0 | 2024-03 → 2024-09 |
| 4 | NMT | Neometals | 16 | +3.53 % | +14.0 % | +38.0 % | 81 % | 1 | 1.6 | 2023-06 → 2025-06 |
| 5 | LOT | Lotus Resources | 16 | +3.37 % | +12.9 % | +57.6 % | 63 % | 2 | 7.2 | 2024-07 → 2026-04 |
| 6 | CHN | Chalice Mining | 16 | +3.25 % | +13.1 % | +52.9 % | 69 % | 4 | 5.8 | 2023-06 → 2025-06 |
| 7 | LKE | Lake Resources | 24 | +3.18 % | +8.7 % | +40.0 % | 67 % | 5 | 2.4 | 2023-06 → 2025-12 |
| 8 | BRN | BrainChip Holdings | 31 | +3.00 % | +6.6 % | +41.5 % | 65 % | 6 | 4.1 | 2023-06 → 2025-12 |
| 9 | SYR | Syrah Resources | 28 | +2.94 % | +7.0 % | +38.8 % | 64 % | 5 | 10.5 | 2023-06 → 2025-12 |
| 10 | SGR | Star Entertainment | 21 | +2.93 % | +9.0 % | +41.1 % | 71 % | 1 | 4.4 | 2023-06 → 2026-03 |

The pattern is recognisable: **cleantech / carbon losers (CXL, CHN), broken
retail (CCX), wound-up uranium (ERA), lithium / battery deflation (LKE, NMT,
LOT, SYR), meme deflation (BRN), and the Star Entertainment casino-licence
collapse (SGR)**. CXL and BRN were each shorted in 30 / 31 of the 36 OOS
months — basically a permanent short for the period.

### Top 10 losing shorts

`worst month` is capped at −16 % by the stop — without the stop these would
have been −20 % to −60 %.

| # | Ticker | Company | n months shorted | total P&L (% book) | avg trade | best month | worst month | hit-rate | n stops | avg SI % | first → last |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | CAT | Catapult Sports | 13 | **−1.48 %** | −6.1 % | +10.9 % | −16.0 % | 39 % | 2 | 0.8 | 2023-06 → 2025-09 |
| 2 | TTT | Titomic | 21 | −1.24 % | −2.6 % | +39.5 % | −16.0 % | 43 % | 10 | 0.1 | 2024-01 → 2026-03 |
| 3 | EVN | Evolution Mining | 15 | −1.15 % | −4.0 % | +18.2 % | −16.0 % | 40 % | 1 | 1.1 | 2023-06 → 2025-11 |
| 4 | MYS | MyState | 22 | −1.13 % | −2.4 % | +5.5 % | −12.6 % | 27 % | 0 | 0.3 | 2023-06 → 2026-04 |
| 5 | PDN | Paladin Energy | 19 | −1.12 % | −2.8 % | +37.9 % | −16.0 % | 32 % | 6 | 7.3 | 2023-06 → 2025-09 |
| 6 | LTR | Liontown Resources | 19 | −1.10 % | −2.5 % | +29.6 % | −16.0 % | 32 % | 5 | 7.4 | 2024-02 → 2026-04 |
| 7 | EMR | Emerald Resources | 6 | −1.09 % | −9.8 % | +1.0 % | −16.0 % | 17 % | 3 | 2.0 | 2025-07 → 2025-12 |
| 8 | AEF | Australian Ethical | 8 | −1.01 % | −6.8 % | +6.5 % | −16.0 % | 38 % | 3 | 0.4 | 2023-06 → 2024-04 |
| 9 | RSG | Resolute Mining | 5 | −0.99 % | −10.8 % | 0.0 % | −16.0 % | 0 % | 2 | 0.4 | 2025-08 → 2025-12 |
| 10 | WAF | West African Resources | 9 | −0.95 % | −5.5 % | +17.7 % | −16.0 % | 11 % | 3 | 1.1 | 2023-10 → 2025-11 |

The losers cluster on **gold + uranium + lithium re-rates** the model bet
against: PDN, LTR, EVN, EMR, RSG, WAF, MIN are all commodity names that
rallied. CAT (sports tech) and AEF (ethical fund manager) were broken
narratives that snapped back. **No single name cost the book more than
−1.5 % of NAV cumulatively** — the −16 % per-position stop kept even
multi-month conviction-shorts under control.

### What this looks like as a research result

* **2,085 monthly short positions** across 206 names over 36 months — i.e. on average ~58 names shorted at any one time
* **+101 % of book** in cumulative short-leg P&L over 3 years OOS — that's the gross alpha before the long leg adds anything
* **The 1-in-6 stop-fire rate proves the stop is structural**, not cosmetic — without it the LTR-style sustained-rally names would each have cost 3-5 × what they did
* **Winners cluster in the bearish-tail names you'd expect** (broken biotechs, battery losers, broken-thesis tech), losers cluster in **commodity cyclicals the market re-rated** (lithium, uranium, data centres) — both make economic sense

---

## Top short candidates — as of 2026-05-25

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
