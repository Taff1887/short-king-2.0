# Short King 2.0 — Research Methodology

This document describes the research methodology behind Short King 2.0, an ASX-focused
short-selling signal project. It is intended for reviewers familiar with empirical
asset pricing and the practical mechanics of equity long–short backtesting. Where the
v1 prototype made simplifying choices (single linear model, ad-hoc feature scaling,
no walk-forward discipline), v2 reframes the problem as a learning-to-rank task with
explicit point-in-time data handling, expanding-window cross-validation, and
SHAP-based interpretability.

## 1. Hypothesis

**Short interest is informative about future returns.** A substantial body of
literature documents that aggregate short interest negatively predicts the
cross-section of subsequent equity returns. Asquith, Pathak and Ritter (2005,
*Journal of Financial Economics*) show that heavily-shorted, low-institutional-
ownership stocks underperform by economically large margins on both equal- and
value-weighted bases. Engelberg, Reed and Ringgenberg (2012, 2018) push the result
further by showing that short sellers are *informed* traders — their predictive
power survives controls for size, book-to-market, momentum and liquidity, and is
concentrated around public information events (earnings, analyst revisions,
short-fee spikes) rather than diffuse private signals. Boehmer, Jones and Zhang
(2008) corroborate from order-flow microstructure. Drechsler and Drechsler (2014)
attribute persistent short-side returns to shorting costs themselves: the
hard-to-borrow names are where mispricing concentrates because limits-to-arbitrage
bind.

On the ASX specifically, the public ASIC short-position regime since 2010 makes
this a rare jurisdiction where the *change* in aggregated short interest can be
observed at the security level with a known reporting lag. That makes the
hypothesis cleanly testable: if Engelberg–Reed–Ringgenberg generalises to
Australia, then a portfolio that shorts the highest-short-interest, weak-
fundamentals names should earn negative abnormal returns over a multi-week
horizon, even after realistic borrow and trading costs.

**What we predict.** The target is the cross-sectional rank of the four-week
forward total return. We are not trying to forecast the absolute market direction
or the level of each stock's return — we are trying to identify, *each week*, the
names that will underperform their peers over the next month.

**What changed from v1 to v2.** The original notebook fit a single
`LogisticRegression` on five features (short-interest level, change in short
interest, P/E, debt/equity and a momentum proxy) with no walk-forward validation
and a binary "stock fell" target. v2 keeps the v1 specification as a baseline but
expands the design along four axes:

1. The feature set grows from five to roughly two dozen across seven economic
   families (short, price, liquidity, valuation, quality, leverage, growth).
2. The target becomes a *rank* rather than a binary class, fit by LightGBM
   LambdaRank in addition to the binary classifier.
3. Training uses an expanding walk-forward window with a purge and embargo equal
   to the label horizon, so no fold ever sees a label that overlaps a future
   training observation.
4. Interpretability is built in via SHAP rather than coefficient inspection, so
   non-linear interactions between (e.g.) leverage and momentum are surfaced.

## 2. Universe

The investable universe is **ASX-listed ordinary equity** that appears in the
ASIC daily aggregated short-position file. ASIC publishes positions for all
reportable securities, including ETFs, hybrids, REITs and stapled securities;
we restrict to common equity via the security description string by requiring it
to end in `ORDINARY` (with `ORDINARY FULLY PAID` and `ORDINARY STAPLED` both
accepted). This filter is applied at every reporting date — universe membership
is time-varying and is reconstructed from the historical files rather than from
the current index constituent list, eliminating survivorship bias.

**Liquidity gate.** A stock enters the modelling panel only if its FMP market
capitalisation, as of the most recent point-in-time observation, exceeds
**A$200 million**. The cutoff is a deliberate compromise: too low and FMP's
fundamental coverage becomes patchy (small ASX names often have stale or
missing key-metrics records, especially pre-2017), too high and the sample
collapses to the ASX 200 and we lose the very part of the cross-section where
short-side mispricing is empirically strongest. A$200m sits roughly at the
ASX-300 boundary and retains around 350–500 names per week historically.

A second liquidity check applied at portfolio-construction time uses **20-day
average dollar volume (ADV)**. The exact threshold is configurable in
`settings` rather than fixed in this document — the reason is that the minimum
viable trade size for the strategy scales with assumed AUM, and the same model
output can be deployed at A$5m or A$50m of book with different ADV floors. For
the headline backtest we use a floor that keeps the rebalance's largest trade
under ~5 % of the name's ADV.

**US robustness sample.** The same pipeline is run end-to-end on the
**S&P 500 point-in-time universe**, using FMP's `historical-sp500-constituent`
endpoint. The U.S. version has no short-position panel comparable to ASIC, so
short-interest features are sourced from FMP's `short-interest` (NASDAQ
biweekly) data and treated as a robustness check rather than the primary
result. Pattern-borrowed from the sister `quant-factor-ranking` repo.

## 3. Data sources

Three providers feed the pipeline. All raw responses are cached to disk under
`data/raw/`, keyed by a hash of the request URL plus the as-of date, so
re-runs are deterministic and offline.

**ASIC daily aggregated short positions.** ASIC publishes a CSV / PDF each
business day at
`https://download.asic.gov.au/short-selling/RR{YYYYMMDD}-001-SSDailyAggShortPos.csv`
showing, per security, the number of shares reported short and the resulting
percentage of issued capital. The headline release is **end-of-week Friday**,
which is the snapshot we model on. Critically, **the file for date T is not
posted until T + 4 business days** — ASIC's required reporting lag exists to
prevent reverse-engineering of individual fund positions. This is the binding
constraint on tradability: a position observed for Friday week-i is only
known publicly on the following Thursday/Friday at the earliest. Backtests
respect this by lagging the signal release by 4 business days and trading on
the *following* Monday (effectively a 1-week signal lag, see §10).

**Financial Modeling Prep (Premium).** All fundamentals come from FMP under
the `https://financialmodelingprep.com/stable` base URL. The endpoints used:

| Endpoint family               | Purpose                                     |
|-------------------------------|---------------------------------------------|
| `income-statement` (Q)        | revenue, EPS, margin trail                  |
| `balance-sheet-statement` (Q) | debt, equity, working capital               |
| `cash-flow-statement` (Q)     | operating + free cash flow                  |
| `ratios` (Q)                  | leverage, profitability, coverage ratios    |
| `key-metrics` (Q)             | EV-based multiples, ROIC, FCF yield         |
| `enterprise-values` (Q)       | point-in-time EV reconstruction             |
| `income-statement-growth`     | YoY growth rates                            |
| `historical-price-eod/full`   | dividend-adjusted prices                    |
| `profile`                     | sector, industry, exchange                  |

All statement endpoints are read in quarterly form and **lagged to
`acceptedDate`** (the SEC/ASX filing acceptance timestamp returned by FMP),
not to fiscal period end. This is point-in-time discipline: a Q1 result with
period end 31-March that was accepted on 15-May only enters the feature set
from 15-May onward. A **stale-filing cap of four quarters** is applied: if a
ticker's most recent accepted filing is older than four quarters relative to
the as-of date, fundamental features are set to NaN rather than forward-filled
indefinitely. Reasoning: a delisted / suspended / restating issuer should not
appear in the cross-section with one-year-old data masquerading as current.

**Yahoo Finance.** Used only as an independent cross-check of FMP's
dividend-adjusted prices for a randomly-drawn audit sample of tickers each
quarter. Yahoo is *not* in the canonical modelling panel — its symbology and
adjustment policy differ subtly from FMP's, and mixing the two introduces
joins on date that look like signals but are actually data-vendor noise.

## 4. Feature construction

Features are computed at each Friday close, jointly for every ticker in the
liquid universe at that date. They fall into seven economic families:

| Family       | Feature                | Formula (sketch)                                                                 |
|--------------|------------------------|----------------------------------------------------------------------------------|
| Short        | `si_pct`               | reported short shares / issued capital (ASIC)                                    |
| Short        | `si_chg_4w`            | `si_pct(t) − si_pct(t−4w)`                                                       |
| Short        | `si_chg_13w`           | `si_pct(t) − si_pct(t−13w)`                                                      |
| Short        | `si_z_52w`             | `(si_pct(t) − mean_52w) / std_52w`                                               |
| Price        | `ret_4w`               | `adjClose(t) / adjClose(t−4w) − 1`                                               |
| Price        | `ret_13w`              | momentum, intermediate horizon                                                   |
| Price        | `ret_52w_ex1m`         | 12-1 momentum (12-month return excluding most-recent month)                      |
| Price        | `vol_13w`              | std of weekly log returns over 13 weeks                                          |
| Price        | `mdd_52w`              | max drawdown over trailing 52 weeks                                              |
| Liquidity    | `mktcap_log`           | `log(market_cap)`                                                                |
| Liquidity    | `adv20_log`            | `log(20-day avg dollar volume)`                                                  |
| Liquidity    | `amihud`               | Amihud illiquidity: `mean(|ret| / dollar_volume)`                                |
| Valuation    | `pe_ttm`               | price / TTM EPS                                                                  |
| Valuation    | `ev_ebitda`            | enterprise value / TTM EBITDA                                                    |
| Valuation    | `pb`                   | price / book                                                                     |
| Valuation    | `fcf_yield`            | TTM free cash flow / market cap                                                  |
| Quality      | `roe_ttm`              | TTM net income / average equity                                                  |
| Quality      | `roic_ttm`             | TTM NOPAT / invested capital                                                     |
| Quality      | `gross_margin`         | TTM gross profit / revenue                                                       |
| Quality      | `accruals`             | (ΔWC − ΔCash) / total assets, Sloan-style                                        |
| Leverage     | `debt_to_equity`       | total debt / total equity                                                        |
| Leverage     | `net_debt_to_ebitda`   | (debt − cash) / TTM EBITDA                                                       |
| Leverage     | `interest_coverage`    | EBIT / interest expense                                                          |
| Growth       | `revenue_growth_yoy`   | TTM revenue / prior-year TTM revenue − 1                                         |
| Growth       | `eps_growth_yoy`       | TTM EPS / prior-year TTM EPS − 1                                                 |
| Growth       | `asset_growth_yoy`     | total assets vs four quarters prior; high values are a known short signal       |

**Cross-sectional ranking.** Many of these are heavy-tailed (P/E and EV/EBITDA
can be negative or astronomical; ADV is power-law). Rather than winsorise to
arbitrary cutoffs, every feature is converted to a **per-Date cross-sectional
percentile rank in [0, 1]** before entering the model. This is the
single most important preprocessing step: it eliminates regime drift in
feature scales, neutralises outliers, and means the model sees a comparable
distribution at every snapshot. For LightGBM the ranked features are fed
directly; for the logistic baseline they are also standardised within fold.

**Sector dummies are optional.** Industry membership is highly imbalanced on
the ASX (resources dominate the long tail), so sector dummies are toggled by
config. The headline result reports both with and without.

## 5. Prediction target

The **primary target** is the per-Date **decile of the four-week forward
total return**:

```
y_rank(i, t) = rank_within_date( ret_{t -> t+4w}(i) ) / N(t)
```

where `N(t)` is the number of names in the cross-section at date `t`. This
maps each name to a number in [0, 1) representing its forward-return rank
*against its contemporaneous peers*. The LambdaRank objective optimises
ranking quality (NDCG) directly. Short-side performance is then measured by
how well the model identifies the *bottom* of this distribution.

A **binary auxiliary target** `y_down = 1{ret_{t -> t+4w} < 0}` is used to
train the logistic and the LightGBM classifier baselines, both of which
correspond directly to the v1 specification. Both targets are computed from
the same dividend-adjusted price series.

**Why rank, not raw return?** Three reasons. (i) Raw cross-sectional forward
returns are dominated by the market factor — a regression target picks up
"the market is down 5 % next month" rather than the relative-value signal we
care about. Demeaning by the cross-section absorbs this. (ii) The
distribution of weekly forward returns has obvious time-varying volatility
(2008, 2020 dwarf 2014, 2017), so a raw-return MSE loss over-fits the volatile
months and ignores the calm ones. Ranks normalise this. (iii) The trading
rule is itself a ranking rule (short the worst decile), so optimising rank
quality is congruent with the downstream decision.

## 6. Train / test scheme

Backtests use **walk-forward expanding-window** cross-validation with
**purging and embargo** matched to the label horizon. The protocol:

```
Initial train window   = 156 weeks (~3 years)
Test window            =  26 weeks (~6 months)
Step (roll-forward)    =  26 weeks
Purge / embargo        =   4 weeks  (= label horizon)
```

Concretely, fold `k` trains on every weekly snapshot from the start of the
sample up to week `156 + 26·(k−1) − 4`, then scores week `156 + 26·(k−1) + 1`
through `156 + 26·k`. The 4-week purge removes any training observation whose
label would have leaked information about the *first* test observation, and a
matching embargo prevents the *last* training observation's label from
overlapping the test window. This is the standard López de Prado purge-and-
embargo construction.

Over a 15-year history this produces around 25 folds and ensures **out-of-fold
predictions for every observation outside the initial 3-year burn-in**. The
backtest is run on those out-of-fold predictions concatenated end-to-end —
not on in-sample fits — which is the only honest way to estimate
deployment-time performance.

## 7. Models

Five models are run for every fold; their out-of-fold predictions are
stacked into the backtest. From simplest to most expressive:

**Naive baseline — short-interest rank.** The portfolio simply shorts the
top decile of `si_pct`. No model. This is the null hypothesis the rest of
the work has to beat.

**Equal-weight composite.** Six features (`si_pct`, `si_chg_13w`, `ret_13w`,
`asset_growth_yoy`, `net_debt_to_ebitda`, negative `fcf_yield`) are rank-
z-scored within each Date and averaged. Equivalent to a hand-tuned linear
factor model; serves as the "smart beta" reference point.

**Logistic regression (v1 reproduction).** Five-feature LogisticRegression
(`si_pct`, `si_chg_4w`, `pe_ttm`, `debt_to_equity`, `ret_13w`) fit on the
binary `y_down` target. Reproduces the v1 notebook exactly, except wrapped in
the walk-forward scheme with `StandardScaler` fitted on train-fold only. This
quantifies the lift attributable to v2's other changes versus the same
underlying linear hypothesis.

**LightGBM classifier.** Full-feature LightGBM with `objective='binary'` on
`y_down`. Hyperparameters (leaves, learning rate, regularisation, min child
weight) are selected once on the first three folds via random search and then
held fixed — re-tuning per fold leaks information and burns time without
materially improving deployment performance.

**LightGBM LambdaRank (headline model).** Full-feature LightGBM with
`objective='lambdarank'`, group sizes equal to per-Date cross-section sizes,
and labels equal to the per-Date forward-return decile (`y_rank` discretised
to 10 bins). The ranking objective is the most appropriate fit to the actual
investment decision; this is the model whose SHAP plots and backtest carry
the headline numbers.

## 8. Interpretability

Tree models earn a reputation as black boxes only when their explanations
are limited to gain-based importance. We use **SHAP TreeExplainer** as the
primary interpretability tool, computed on a stratified sample of out-of-fold
predictions. Committed artefacts include:

- A **bar chart of mean |SHAP|** across features, grouped and coloured by
  economic family. Expectation given the literature: the short-interest and
  growth/leverage clusters dominate; price-momentum contributes secondarily.
- **SHAP dependence plots** for the top six features, coloured by the
  strongest interacting feature (SHAP's `approximate_interaction` heuristic).
  These reveal non-linearities the logistic model cannot — e.g. high short
  interest may only be predictive *conditional* on weak revenue growth.
- A **monotonicity sanity check**: for each feature, the average predicted
  rank as a function of feature decile should be monotone or near-monotone in
  the expected direction (e.g. higher `si_pct` → lower predicted forward
  rank). Sharp non-monotonicity flags either a data artefact or an
  interaction that needs investigation.

**Gain-based importance** from LightGBM is also reported as a coarse sanity
check, but is known to be biased toward high-cardinality features and is
treated as secondary to SHAP.

## 9. Portfolio construction

The signal is translated into two book constructions, both rebalanced weekly:

**Primary book — short-only top decile.** Each Friday close, the bottom-rank
decile (i.e. the names the model predicts will underperform most) is shorted
**equal-weighted within the basket**. Names that fail the 20-day ADV liquidity
floor are dropped and the remaining basket re-weighted up to the cap. This
book has no offsetting long leg — it is a *pure* short-selling strategy and
its returns include the cost of capital required to fund the short. The
unfunded short return is reported alongside the financed return for
transparency.

**Diagnostic book — dollar-neutral long–short.** Long the top decile,
short the bottom decile, equal-weighted within each leg, scaled so the two
legs have equal gross exposure. This is the cleaner academic measure of the
signal's information content — it isolates the cross-sectional effect from
the equity-beta exposure of the short-only book, and is what we report
against the Engelberg-style benchmarks.

**Costs and frictions.** All backtests apply the following, configurable in
`settings`:

- **Commission**: 25 bps round-trip per turnover dollar.
- **Borrow**: 1.5 % annualised flat on the average short notional outstanding
  during the week. This is a stylised average for ASX general-collateral
  names; hard-to-borrow names sit well above and easy-to-borrow names well
  below. Sensitivity to this assumption is shown in §12.
- **Slippage**: 5 bps applied to the *changed* portion of the book at each
  rebalance (i.e. only on names entering, exiting or having their weight
  re-set, not on the held portion).
- **Cash drag**: short proceeds are assumed to earn the RBA cash rate net of
  a 50 bps haircut representing prime-broker margin requirements.

Costs are subtracted from gross weekly returns to produce the headline
net-of-cost equity curve.

## 10. Backtest

Backtests are run at **weekly frequency** on FMP dividend-adjusted closes.
The timeline at week `i` is:

```
Friday week i      :  ASIC short position observed (release at i+4 BD)
Fri week i + 4 BD  :  data publicly available; signal computable
Weekend            :  model predictions generated for week i snapshot
Monday week i+1    :  positions entered at week i+1 close
Friday week i+1    :  weekly return realised at week i+1 close
```

The implementation models this as a **one-week signal lag**: signals computed
on the snapshot for week `i` drive positions held over week `i+1`, with
entry and exit both at adj-close. This respects the 4-business-day ASIC lag
without forcing intraweek execution timing into the backtest. It also
deliberately *under-states* tradability — in practice a portfolio manager
who saw the ASIC release on Tuesday could trade Tuesday afternoon, half a
week earlier than this model assumes — so reported performance is a
conservative estimate.

Position sizing handles fractional-share issues by rounding to whole shares
at the close price, with the residual cash held at the cash rate. Stocks
that delist or are suspended mid-week are marked-to-last-trade and exited at
the next available print; the realised return for the affected week reflects
that print.

## 11. Metrics

The performance dashboard reports:

- **CAGR**: geometric annualised return of the cumulative equity curve.
- **Annualised volatility**: std of weekly returns × √52.
- **Sharpe ratio**: (mean − r_f) / std, annualised; r_f is the RBA cash rate
  for ASX runs, 3-month T-bill for the U.S. robustness sample.
- **Sortino ratio**: as Sharpe but with downside deviation.
- **Maximum drawdown**: largest peak-to-trough decline of the equity curve.
- **Calmar ratio**: CAGR / |max drawdown|.
- **Hit rate**: share of weeks with positive net return.
- **Turnover**: average per-rebalance one-way notional change as a share of
  GMV; the borrow and commission lines scale with this.
- **Monthly heatmap**: a year × month grid of monthly returns, useful for
  spotting regime effects at a glance.
- **CAPM α / β** against the ASX 200 total-return ETF (`STW.AX`, dividend-
  reinvested). For the U.S. sample, against SPY. Both α (annualised) and β
  are reported with Newey–West standard errors at lag 4 to handle the
  overlap-induced autocorrelation in weekly returns.

## 12. Robustness

A signal that survives a single specification but breaks under any nudge to
the design is not deployable. The robustness pack includes:

**Bootstrap Sharpe confidence interval.** 10 000 stationary block bootstrap
resamples (block length ≈ √T weeks) of the weekly net-return series. We
report the 5th-and-95th-percentile Sharpe so a reader knows whether the
headline number is two-standard-error-distinguishable from zero or marginal.

**Subperiod analysis.** Pre-2020 versus 2020-onwards. The COVID drawdown and
the subsequent retail-driven squeeze of 2020–2021 are a known stress test
for short-selling signals; we report the Sharpe and drawdown separately for
each regime and discuss any cliff.

**Sensitivity to top-K.** The headline uses the top decile. We sweep
K ∈ {top 5, top 10, top 15, top 20, top 30} % and chart Sharpe vs K. A robust
signal should degrade smoothly; a Sharpe that peaks at exactly the chosen K
and falls off either side is suspicious.

**Sensitivity to borrow.** Borrow rate is swept over {0, 0.5, 1.5, 3.0, 5.0}
% annualised, holding all else equal. This bounds the effect of a stylised-
versus-realistic borrow assumption.

**U.S. S&P 500 replication.** The full pipeline (excluding ASIC; substituting
FMP biweekly U.S. short-interest data) is run on a 15-year S&P 500 PIT
sample. A signal that works only on the ASX is hard to distinguish from
sample-specific over-fitting; one that works on both jurisdictions is
considerably more credible.

## 13. Limitations

Several limitations should be kept in mind when interpreting results.

- **ASIC reporting lag.** The 4-business-day delay between observation and
  release means the opportunity has been partially arbitraged by faster
  market participants by the time the signal becomes tradable. Realised
  Sharpe is therefore lower than the "instantaneous-information" upper bound
  the literature sometimes computes.
- **Small-cap fundamentals coverage.** FMP coverage thins below A$200m and
  becomes patchy below ~A$100m. The liquidity gate addresses this but
  excludes precisely the slice of the cross-section where short-side
  mispricing is empirically strongest. A bigger pipe (e.g. S&P Capital IQ or
  Refinitiv) would extend the universe materially.
- **Stylised borrow.** A flat 1.5 % annualised borrow underestimates costs
  on hard-to-borrow names — exactly the names the model is most likely to
  flag. The borrow-rate sensitivity in §12 quantifies the effect; deployment
  would require live prime-broker fee data.
- **Squeeze and capital-raise dynamics are not modelled.** Short squeezes,
  rights issues at deep discounts (a frequent ASX small-cap event), and
  index inclusion/exclusion flows can each produce week-long moves that
  swamp the fundamental signal. We measure their realised impact through
  the drawdown statistics but do not attempt to forecast them.
- **Short bias.** The primary book is short-only and therefore wears
  positive equity-beta exposure on the wrong side. In a sustained rally the
  strategy will lose money even if every individual short pick is correctly
  identified as a relative underperformer. The dollar-neutral diagnostic
  book exists partly to demonstrate the signal *is* informative
  cross-sectionally; the short-only book is the deployable version, and
  shows what that costs in realised P&L during equity rallies.
- **4-week label horizon.** Four weeks balances sample size (we want as
  many independent observations as possible) against noise (one-week
  forward returns are extremely noisy). Longer horizons (13 weeks, 26
  weeks) would let the fundamental signal mature but cut effective sample
  size by a factor of 3–6 and increase label overlap.
- **No transaction-cost model for size-dependent impact.** The 5 bps
  slippage is a flat assumption. Real impact is convex in trade size /
  ADV and the strategy's deployable capacity depends on it. The 5 % of
  ADV trade-size cap is the implicit capacity constraint; above this AUM
  the cost model breaks down.
- **No live point-in-time index membership for the ASX.** Unlike the S&P
  500, ASX historical index constituents are not freely available. The
  ASX universe used here is "everything reportable to ASIC that passes
  the liquidity gate" rather than "the ASX 300 as it stood at date T".
  This is a different — broader — universe than the standard ASX
  benchmark, which weakens the CAPM α attribution slightly.

## 14. Future improvements

Specific extensions, in roughly decreasing order of expected value:

- **Live borrow data.** Prime-broker daily borrow rate and availability
  feed. Would let the cost model price hard-to-borrow names accurately and
  allow position sizing to respond to borrow scarcity.
- **News / earnings revision sentiment.** FMP exposes an analyst-grades
  family (`grade`, `analyst-estimates`, `price-target-consensus`) and a
  news endpoint. Combining sell-side downgrades with the existing short
  signal is the Engelberg–Reed–Ringgenberg (2018) "informed-trading-
  around-news" mechanism made operational.
- **Sector-neutral construction.** The current implementation can be
  sector-concentrated (e.g. it may end up short five resources juniors at
  once). A sector-neutral variant — equal weight within each GICS sector
  and equal weights across sectors — would diversify idiosyncratic sector
  risk at some cost to raw Sharpe.
- **Alternative tree models and stacking.** XGBoost and CatBoost would
  serve as additional out-of-sample sanity checks; stacking via a simple
  linear blender on top of LightGBM, XGBoost and the logistic baseline
  would likely add a few basis points of Sharpe.
- **Conformal predictions.** The current pipeline emits a single point
  estimate (predicted rank). Conformal prediction (Vovk et al.; Romano,
  Patterson & Candès) would emit a calibrated prediction interval per
  observation. In a portfolio context this lets the manager size down
  high-uncertainty names rather than treating all top-decile picks as
  equally confident, which is the right Bayesian behaviour but requires
  more infrastructure.
- **Intraweek timing.** The current 1-week signal lag is conservative;
  trading on the Tuesday or Wednesday after the ASIC release (rather than
  the following Monday) would capture an extra few days of signal decay.
  An event-driven backtest at daily frequency would quantify this.
- **Squeeze detector.** A simple "days-to-cover spiking + share price
  spiking + RSI extreme" overlay that vetoes a short position when the
  squeeze risk is elevated. Likely small contribution on average but
  meaningful tail-risk reduction.
