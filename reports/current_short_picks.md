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
| 1 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.97 | 0.99 | 0.98 |
| 2 | IMU | Imugene Limited | 107 | 1.47 | 0.96 | 1.00 | 0.98 |
| 3 | VUL | Vulcan Energy | 630 | 4.65 | 0.94 | 0.99 | 0.97 |
| 4 | STX | Strike Energy Ltd | 358 | 0.45 | 0.99 | 0.93 | 0.96 |
| 5 | CAT | Catapult Sports Ltd | 983 | 5.15 | 0.93 | 0.97 | 0.95 |
| 6 | PDI | Predictive Disc Ltd | 1,930 | 1.13 | 0.93 | 0.95 | 0.94 |
| 7 | ASM | Ausstratmaterials | 152 | 1.84 | 0.92 | 0.96 | 0.94 |
| 8 | WBT | Weebit Nano Ltd | 1,065 | 0.32 | 0.98 | 0.90 | 0.94 |
| 9 | LTR | Liontown Limited | 4,404 | 1.75 | 0.99 | 0.86 | 0.93 |
| 10 | PLS | Pls Group Ltd | 13,590 | 11.53 | 0.96 | 0.88 | 0.92 |

## Top 10 by consensus across all 5 models (percentile-ranked)

Each model's cross-sectional percentile rank within today's universe — 1.00 = most shortable on that model, 0.00 = least. Using percentile ranks (not raw scores) puts every model on the same 0-1 scale, so they're directly comparable. `consensus_rk` is just the average across the 5 percentile columns.

| # | Ticker | Company | Mkt Cap (A$m) | Short % | naive p | ew p | logit p | gbm_cls p | gbm_rank p | consensus_rk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TLX | Telix Pharmaceutical | 3,794 | 15.15 | 0.99 | 0.90 | 0.83 | 0.94 | 0.94 | 0.921 |
| 2 | VUL | Vulcan Energy | 630 | 4.65 | 0.82 | 0.99 | 0.87 | 0.96 | 0.94 | 0.918 |
| 3 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.73 | 0.99 | 0.89 | 1.00 | 0.97 | 0.917 |
| 4 | CAT | Catapult Sports Ltd | 983 | 5.15 | 0.84 | 0.97 | 0.87 | 0.96 | 0.93 | 0.912 |
| 5 | ILU | Iluka Resources | 2,488 | 7.53 | 0.89 | 0.94 | 0.90 | 0.99 | 0.79 | 0.904 |
| 6 | 4DX | 4Dmedical Limited | 1,948 | 10.06 | 0.97 | 0.83 | 0.59 | 0.95 | 1.00 | 0.866 |
| 7 | PWH | Pwr Holdings Limited | 790 | 7.80 | 0.90 | 0.85 | 0.92 | 0.99 | 0.60 | 0.852 |
| 8 | NEU | Neuren Pharmaceut. | 2,357 | 6.40 | 0.88 | 0.77 | 0.89 | 0.84 | 0.88 | 0.851 |
| 9 | LOT | Lotus Resources Ltd | 450 | 19.54 | 1.00 | 1.00 | 0.94 | 0.51 | 0.79 | 0.846 |
| 10 | AD8 | Audinategroupltd | 339 | 1.86 | 0.54 | 0.80 | 0.97 | 0.95 | 0.97 | 0.845 |

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
| 1 | WOW | Woolworths Group Ltd | 35,876 | 2.32 | 0.733 |
| 2 | SHL | Sonic Healthcare | 11,171 | 6.38 | 0.718 |
| 3 | BAP | Bapcor Limited | 703 | 9.41 | 0.663 |
| 4 | EML | Eml Payments Ltd | 356 | 3.79 | 0.654 |
| 5 | SXL | Sthn Cross Media | 190 | 0.35 | 0.642 |

### Top 5 per `gbm_cls`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_gbm_cls |
|---|---|---|---|---|---|
| 1 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.609 |
| 2 | AGI | Ainsworth Game Tech. | 340 | 0.00 | 0.604 |
| 3 | PWH | Pwr Holdings Limited | 790 | 7.80 | 0.587 |
| 4 | SPL | Starpharma Holdings | 151 | 0.09 | 0.585 |
| 5 | ILU | Iluka Resources | 2,488 | 7.53 | 0.571 |

### Top 5 per `gbm_rank`

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_gbm_rank |
|---|---|---|---|---|---|
| 1 | BBN | Baby Bunting Grp Ltd | 347 | 0.32 | 0.269 |
| 2 | 4DX | 4Dmedical Limited | 1,948 | 10.06 | 0.269 |
| 3 | LTR | Liontown Limited | 4,404 | 1.75 | 0.174 |
| 4 | STX | Strike Energy Ltd | 358 | 0.45 | 0.094 |
| 5 | EGR | Ecograf Limited | 171 | 0.02 | 0.047 |

## Why these names? Factor breakdown for the top 10 consensus picks

Every cell is 0-1; higher = more shortable on that dimension. `(inv)` columns are naturally-bullish ranks flipped via `1 − rank` so the polarity is consistent across the table.

| # | Ticker | SI | SI z | mom (inv) | vol | P/E | FCF-y (inv) | ROE (inv) | D/E | growth (inv) | EW factor avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TLX | 0.99 | 0.70 | 0.09 | 0.42 | 0.50 | 0.79 | 0.72 | 0.81 | 0.42 | 0.60 |
| 2 | VUL | 0.83 | 0.42 | 0.20 | 0.84 | 0.50 | 0.92 | 0.82 | 0.22 | 0.92 | 0.63 |
| 3 | EOS | 0.75 | 0.68 | 0.18 | 0.95 | 0.50 | 0.74 | 0.88 | 0.29 | 0.07 | 0.56 |
| 4 | CAT | 0.85 | 0.61 | 0.74 | 0.97 | 0.50 | 0.57 | 0.84 | 0.18 | 0.64 | 0.65 |
| 5 | ILU | 0.90 | 0.41 | 0.07 | 0.27 | 0.50 | 0.97 | 0.91 | 0.60 | 0.93 | 0.62 |
| 6 | 4DX | 0.97 | 0.99 | 0.61 | 0.99 | 0.50 | 0.71 | 0.00 | 0.01 | 0.06 | 0.54 |
| 7 | PWH | 0.91 | 0.11 | 0.49 | 0.57 | 0.91 | 0.68 | 0.38 | 0.79 | 0.20 | 0.56 |
| 8 | NEU | 0.89 | 0.86 | 0.12 | 0.30 | 0.92 | 0.67 | 0.44 | 0.02 | 0.14 | 0.48 |
| 9 | LOT | 1.00 | 0.98 | 0.99 | 0.71 | 0.50 | 0.96 | 0.87 | 0.16 | 0.98 | 0.79 |
| 10 | AD8 | 0.57 | 0.02 | 0.85 | 0.56 | 0.50 | 0.68 | 0.83 | 0.16 | 0.79 | 0.55 |

**How to read it:**
* Rows where most cells are near 1.0 = multi-factor shorts (crowded SI + falling momentum + low quality + high leverage all at once). These are the safest setups.
* Rows where only the SI columns are high but fundamentals are neutral / bullish = pure crowded-short plays. Higher squeeze risk because the signal rests on one dimension only.
* `EW factor avg` is the simple mean across the 9 columns — gives a one-number read of multi-factor agreement.
