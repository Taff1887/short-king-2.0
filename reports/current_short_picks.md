# Current short picks — as of 2026-05-29

_Latest monthly rebalance: **2026-05-29**. Investable universe size: **272 names** (≥ A$100 m market cap, has fresh fundamentals, has a valid adjusted close)._

**Score columns** (one per model, all 0-1 except gbm_rank which is z-scored):
* `score_naive` — rank of `ShortPct` across the cross-section
* `score_ew` — polarity-aware equal-weight composite of 12 ranks
* `score_logit` — L2 logistic-regression Pr(monthly return < 0)
* `score_gbm_cls` — LightGBM binary classifier probability
* `score_gbm_rank` — LightGBM LambdaRank raw output (higher = ranked closer to the top of the bearish tail)

Higher = more shortable on every model. `consensus_rk` is the **average of each model's cross-sectional percentile rank** on this date — robust to the different output scales.

## Top 10 by the two best signals (gbm_rank + ew, percentile-ranked)

Per the [signal-quality results](#headline-finding), `gbm_rank` and `ew` are the two highest-Sharpe signals OOS. Both columns are cross-sectional percentile ranks within today's universe (1.00 = most shortable, 0.00 = least shortable). `combo` = average of the two.

| # | Ticker | Company | Mkt Cap (A$m) | Short % | gbm_rank pctile | ew pctile | combo |
|---|---|---|---|---|---|---|---|
| 1 | NVX | Novonix Limited | 293 | 2.80 | 0.98 | 0.99 | 0.98 |
| 2 | IMU | Imugene Limited | 107 | 1.47 | 0.94 | 1.00 | 0.97 |
| 3 | CAT | Catapult Sports Ltd | 983 | 5.15 | 0.97 | 0.97 | 0.97 |
| 4 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.94 | 0.99 | 0.96 |
| 5 | VUL | Vulcan Energy | 630 | 4.65 | 0.92 | 0.99 | 0.96 |
| 6 | SBM | St Barbara Limited | 702 | 3.57 | 0.93 | 0.96 | 0.95 |
| 7 | MSB | Mesoblast Limited | 1,898 | 8.66 | 0.95 | 0.94 | 0.94 |
| 8 | WEB | Web Travel Group Ltd | 943 | 5.56 | 0.96 | 0.93 | 0.94 |
| 9 | STX | Strike Energy Ltd | 358 | 0.45 | 0.96 | 0.93 | 0.94 |
| 10 | ASM | Ausstratmaterials | 152 | 1.84 | 0.91 | 0.96 | 0.94 |

## Top 10 by consensus across all 5 models (percentile-ranked)

Each model's cross-sectional percentile rank within today's universe — 1.00 = most shortable on that model, 0.00 = least. Using percentile ranks (not raw scores) puts every model on the same 0-1 scale, so they're directly comparable. `consensus_rk` is just the average across the 5 percentile columns.

| # | Ticker | Company | Mkt Cap (A$m) | Short % | naive p | ew p | logit p | gbm_cls p | gbm_rank p | consensus_rk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TLX | Telix Pharmaceutical | 3,794 | 15.15 | 0.99 | 0.90 | 0.87 | 0.90 | 0.86 | 0.906 |
| 2 | CAT | Catapult Sports Ltd | 983 | 5.15 | 0.84 | 0.97 | 0.78 | 0.96 | 0.97 | 0.904 |
| 3 | MSB | Mesoblast Limited | 1,898 | 8.66 | 0.93 | 0.94 | 0.98 | 0.71 | 0.95 | 0.902 |
| 4 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.73 | 0.99 | 0.90 | 0.94 | 0.94 | 0.900 |
| 5 | ILU | Iluka Resources | 2,488 | 7.53 | 0.89 | 0.94 | 0.93 | 0.93 | 0.77 | 0.896 |
| 6 | 4DX | 4Dmedical Limited | 1,948 | 10.06 | 0.97 | 0.83 | 0.71 | 0.94 | 1.00 | 0.888 |
| 7 | VUL | Vulcan Energy | 630 | 4.65 | 0.82 | 0.99 | 0.85 | 0.72 | 0.92 | 0.860 |
| 8 | PLS | Pls Group Ltd | 13,590 | 11.53 | 0.98 | 0.88 | 0.66 | 0.81 | 0.96 | 0.859 |
| 9 | SBM | St Barbara Limited | 702 | 3.57 | 0.73 | 0.96 | 0.88 | 0.78 | 0.93 | 0.856 |
| 10 | PNV | Polynovo Limited | 850 | 8.02 | 0.91 | 0.98 | 0.86 | 0.64 | 0.87 | 0.851 |

> **Why percentile, not raw score?** Each model's raw output sits on a different scale: naive/ew are 0-1 cross-sectional ranks; logit/gbm_cls are sigmoid probabilities (~0.2 to 0.7); gbm_rank is raw LambdaRank output (~−4 to +0.4 on a normal day, with negative mean from the optimiser, NOT a polarity flip). Comparing the raw scores side-by-side is misleading. The cross-sectional percentile ranks all live on 0-1 and represent the same thing — 'where this name sits within today's universe per this model'.

## Top 5 per individual model

What each model alone says are its highest-conviction shorts. Cross-reference with the consensus list above — names that appear in multiple individual top-5s are broad agreement signals; names that appear in only one model are 'this model alone thinks this' (often the most informative disagreements).

### Top 5 per `naive`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_naive |
|---|---|---|---|---|---|
| 1 | LOT | Lotus Resources Ltd | 450 | 19.54 | 1.000 |
| 2 | DMP | Domino Pizza Enterpr | 1,984 | 15.20 | 0.997 |
| 3 | TLX | Telix Pharmaceutical | 3,794 | 15.15 | 0.993 |
| 4 | BOE | Boss Energy Ltd | 610 | 14.33 | 0.990 |
| 5 | TWE | Treasury Wine Estate | 4,244 | 13.07 | 0.986 |

### Top 5 per `ew`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_ew |
|---|---|---|---|---|---|
| 1 | IMU | Imugene Limited | 107 | 1.47 | 1.000 |
| 2 | LOT | Lotus Resources Ltd | 450 | 19.54 | 0.997 |
| 3 | VUL | Vulcan Energy | 630 | 4.65 | 0.993 |
| 4 | NVX | Novonix Limited | 293 | 2.80 | 0.990 |
| 5 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.986 |

### Top 5 per `logit`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_logit |
|---|---|---|---|---|---|
| 1 | SHL | Sonic Healthcare | 11,171 | 6.38 | 0.740 |
| 2 | WOW | Woolworths Group Ltd | 35,876 | 2.32 | 0.736 |
| 3 | SXL | Sthn Cross Media | 190 | 0.35 | 0.709 |
| 4 | EML | Eml Payments Ltd | 356 | 3.79 | 0.687 |
| 5 | RIC | Ridley Corporation | 993 | 0.87 | 0.678 |

### Top 5 per `gbm_cls`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_gbm_cls |
|---|---|---|---|---|---|
| 1 | MMS | Mcmillan Shakespeare | 1,187 | 5.56 | 0.660 |
| 2 | AGI | Ainsworth Game Tech. | 340 | 0.00 | 0.654 |
| 3 | BBN | Baby Bunting Grp Ltd | 347 | 0.32 | 0.630 |
| 4 | EGR | Ecograf Limited | 171 | 0.02 | 0.615 |
| 5 | NHC | New Hope Corporation | 3,800 | 4.74 | 0.602 |

### Top 5 per `gbm_rank`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_gbm_rank |
|---|---|---|---|---|---|
| 1 | EGR | Ecograf Limited | 171 | 0.02 | 0.443 |
| 2 | 4DX | 4Dmedical Limited | 1,948 | 10.06 | 0.285 |
| 3 | BBN | Baby Bunting Grp Ltd | 347 | 0.32 | 0.129 |
| 4 | TTT | Titomic Limited | 361 | 0.35 | 0.098 |
| 5 | LTR | Liontown Limited | 4,404 | 1.75 | 0.092 |

## Why these names? Factor breakdown for the top 10 consensus picks

Every cell is 0-1; higher = more shortable on that dimension. `(inv)` columns are naturally-bullish ranks flipped via `1 − rank` so the polarity is consistent across the table.

| # | Ticker | SI | SI z | mom (inv) | vol | P/E | FCF-y (inv) | ROE (inv) | D/E | growth (inv) | EW factor avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TLX | 0.99 | 0.70 | 0.09 | 0.42 | 0.50 | 0.79 | 0.72 | 0.81 | 0.42 | 0.60 |
| 2 | CAT | 0.85 | 0.61 | 0.74 | 0.97 | 0.50 | 0.57 | 0.84 | 0.18 | 0.64 | 0.65 |
| 3 | MSB | 0.93 | 0.77 | 0.58 | 0.56 | 0.50 | 0.73 | 0.81 | 0.38 | 0.64 | 0.66 |
| 4 | EOS | 0.75 | 0.68 | 0.18 | 0.95 | 0.50 | 0.74 | 0.88 | 0.29 | 0.07 | 0.56 |
| 5 | ILU | 0.90 | 0.41 | 0.07 | 0.27 | 0.50 | 0.97 | 0.91 | 0.60 | 0.93 | 0.62 |
| 6 | 4DX | 0.97 | 0.99 | 0.61 | 0.99 | 0.50 | 0.71 | 0.00 | 0.01 | 0.06 | 0.54 |
| 7 | VUL | 0.83 | 0.42 | 0.20 | 0.84 | 0.50 | 0.92 | 0.82 | 0.22 | 0.92 | 0.63 |
| 8 | PLS | 0.98 | 0.48 | 0.04 | 0.38 | 0.98 | 0.61 | 0.66 | 0.41 | 0.07 | 0.51 |
| 9 | SBM | 0.74 | 0.93 | 0.87 | 0.91 | 0.50 | 0.86 | 0.69 | 0.11 | 0.34 | 0.66 |
| 10 | PNV | 0.91 | 0.00 | 0.06 | 0.88 | 1.00 | 0.68 | 0.69 | 0.51 | 0.51 | 0.58 |

**How to read it:**
* Rows where most cells are near 1.0 = multi-factor shorts (crowded SI + falling momentum + low quality + high leverage all at once). These are the safest setups.
* Rows where only the SI columns are high but fundamentals are neutral / bullish = pure crowded-short plays. Higher squeeze risk because the signal rests on one dimension only.
* `EW factor avg` is the simple mean across the 9 columns — gives a one-number read of multi-factor agreement.
