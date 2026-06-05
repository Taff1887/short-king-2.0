# Current short picks — as of 2026-05-29

_Latest monthly rebalance: **2026-05-29**. Investable universe size: **272 names** (≥ A$100 m market cap, has fresh fundamentals, has a valid adjusted close)._

**Score columns** (one per model, all 0-1 except gbm_rank which is z-scored):
* `score_naive` — rank of `ShortPct` across the cross-section
* `score_ew` — polarity-aware equal-weight composite of 12 ranks
* `score_logit` — L2 logistic-regression Pr(monthly return < 0)
* `score_gbm_cls` — LightGBM binary classifier probability
* `score_gbm_rank` — LightGBM LambdaRank raw output (higher = ranked closer to the top of the bearish tail)

Higher = more shortable on every model. `consensus_rk` is the **average of each model's cross-sectional percentile rank** on this date — robust to the different output scales.

## Top 10 by consensus across all 5 models

The names every model agrees are in the bearish tail (the safest picks — if all 5 models cluster on the same names, the signal is broad-based rather than driven by any single factor).

| # | Ticker | Company | Mkt Cap (A$m) | Short % | score_naive | score_ew | score_logit | score_gbm_cls | score_gbm_rank | consensus_rk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MSB | Mesoblast Limited | 1,898 | 8.66 | 0.934 | 0.972 | 0.676 | 0.435 | -0.375 | 0.909 |
| 2 | CAT | Catapult Sports Ltd | 983 | 5.15 | 0.848 | 0.976 | 0.561 | 0.567 | -0.110 | 0.904 |
| 3 | TLX | Telix Pharmaceutical | 3,794 | 15.15 | 0.993 | 0.845 | 0.584 | 0.533 | -0.841 | 0.896 |
| 4 | ILU | Iluka Resources | 2,488 | 7.53 | 0.900 | 0.914 | 0.619 | 0.557 | -1.310 | 0.890 |
| 5 | 4DX | 4Dmedical Limited | 1,948 | 10.06 | 0.969 | 0.812 | 0.549 | 0.557 | 0.285 | 0.887 |
| 6 | EOS | Electro Optic Sys. | 1,821 | 3.59 | 0.748 | 0.800 | 0.604 | 0.562 | -0.423 | 0.865 |
| 7 | VUL | Vulcan Energy | 630 | 4.65 | 0.831 | 0.969 | 0.578 | 0.438 | -0.632 | 0.855 |
| 8 | SBM | St Barbara Limited | 702 | 3.57 | 0.745 | 0.955 | 0.584 | 0.465 | -0.548 | 0.854 |
| 9 | WEB | Web Travel Group Ltd | 943 | 5.56 | 0.857 | 0.966 | 0.560 | 0.422 | -0.309 | 0.839 |
| 10 | ACL | Au Clinical Labs | 533 | 8.35 | 0.924 | 0.903 | 0.569 | 0.533 | -1.642 | 0.839 |

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
| 1 | LOT | Lotus Resources Ltd | 450 | 19.54 | 1.000 |
| 2 | NVX | Novonix Limited | 293 | 2.80 | 0.997 |
| 3 | HLS | Healius | 657 | 9.55 | 0.993 |
| 4 | IMU | Imugene Limited | 107 | 1.47 | 0.990 |
| 5 | BAP | Bapcor Limited | 703 | 9.41 | 0.986 |

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
| 1 | MSB | 0.93 | 0.77 | 0.58 | 0.56 | 0.50 | 0.73 | 0.81 | 0.38 | 0.64 | 0.66 |
| 2 | CAT | 0.85 | 0.61 | 0.74 | 0.97 | 0.50 | 0.57 | 0.84 | 0.18 | 0.64 | 0.65 |
| 3 | TLX | 0.99 | 0.70 | 0.09 | 0.42 | 0.50 | 0.79 | 0.72 | 0.81 | 0.42 | 0.60 |
| 4 | ILU | 0.90 | 0.41 | 0.07 | 0.27 | 0.50 | 0.97 | 0.91 | 0.60 | 0.93 | 0.62 |
| 5 | 4DX | 0.97 | 0.99 | 0.61 | 0.99 | 0.50 | 0.71 | 0.00 | 0.01 | 0.06 | 0.54 |
| 6 | EOS | 0.75 | 0.68 | 0.18 | 0.95 | 0.50 | 0.74 | 0.88 | 0.29 | 0.07 | 0.56 |
| 7 | VUL | 0.83 | 0.42 | 0.20 | 0.84 | 0.50 | 0.92 | 0.82 | 0.22 | 0.92 | 0.63 |
| 8 | SBM | 0.74 | 0.93 | 0.87 | 0.91 | 0.50 | 0.86 | 0.69 | 0.11 | 0.34 | 0.66 |
| 9 | WEB | 0.86 | 0.84 | 0.64 | 0.90 | 0.87 | 0.51 | 0.62 | 0.54 | 0.64 | 0.71 |
| 10 | ACL | 0.92 | 0.91 | 0.55 | 0.18 | 0.82 | 0.06 | 0.49 | 0.93 | 0.77 | 0.63 |

**How to read it:**
* Rows where most cells are near 1.0 = multi-factor shorts (crowded SI + falling momentum + low quality + high leverage all at once). These are the safest setups.
* Rows where only the SI columns are high but fundamentals are neutral / bullish = pure crowded-short plays. Higher squeeze risk because the signal rests on one dimension only.
* `EW factor avg` is the simple mean across the 9 columns — gives a one-number read of multi-factor agreement.
