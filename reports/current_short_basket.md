# Current short basket — what's being shorted, and why

_As of 2026-05-29 (last ASIC release of the month). Ranked by **naive score** (the highest-Sharpe non-benchmark strategy: rank by reported short interest). Per-factor polarity-aware ranks alongside show **why** each name is shortable across the EW composite's 12 signals -- every cell is 0-1 where 1 = most shortable on that factor._

**Score columns:**
* `score_naive` -- rank of `ShortPct` across the cross-section (higher = more crowded short).
* `score_ew` -- polarity-aware equal-weight composite of 12 ranks (higher = bearish across many dimensions).
* `score_logit` -- L2 logistic regression Pr(monthly return < 0).

**Factor columns (all 0-1, higher = more shortable; `(inv)` = naturally bullish raw rank flipped via `1 - rank`):**
* `SI %` -- raw short-interest %.
* `SI z` -- short interest z-score vs 12-month history.
* `3m-mom (inv)` -- low 3-month momentum.
* `vol` -- 1-month realised volatility.
* `P/E` -- expensive valuation.
* `FCF-yld (inv)` -- low free cash flow yield.
* `ROE (inv)` -- low return on equity.
* `D/E` -- high leverage.
* `rev-gth (inv)` -- low / negative revenue growth.

| # | Ticker | Company | mktCap (A$m) | ShortPct | score_naive | score_ew | score_logit | SI % | SI z | 3m-mom (inv) | vol | P/E | FCF-yld (inv) | ROE (inv) | D/E | rev-gth (inv) | EW factor avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LOT | Lotus Resources Ltd | 450 | 19.54 | 1.000 | 1.000 | 0.606 | 1.00 | 0.98 | 0.99 | 0.71 | 0.50 | 0.96 | 0.87 | 0.16 | 0.98 | 0.794 |
| 2 | DMP | Domino Pizza Enterpr | 1,984 | 15.20 | 0.997 | 0.786 | 0.596 | 1.00 | 0.56 | 0.53 | 0.54 | 0.59 | 0.17 | 0.33 | 0.90 | 0.78 | 0.599 |
| 3 | TLX | Telix Pharmaceutical | 3,794 | 15.15 | 0.993 | 0.845 | 0.584 | 0.99 | 0.70 | 0.09 | 0.42 | 0.50 | 0.79 | 0.72 | 0.81 | 0.42 | 0.604 |
| 4 | BOE | Boss Energy Ltd | 610 | 14.33 | 0.990 | 0.648 | 0.475 | 0.99 | 0.22 | 0.81 | 0.26 | 0.50 | 0.47 | 0.76 | 0.05 | 0.02 | 0.453 |
| 5 | TWE | Treasury Wine Estate | 4,244 | 13.07 | 0.986 | 0.900 | 0.473 | 0.99 | 0.66 | 0.44 | 0.51 | 0.50 | 0.53 | 0.91 | 0.64 | 0.84 | 0.669 |
| 6 | PLS | Pls Group Ltd | 13,590 | 11.53 | 0.983 | 0.597 | 0.539 | 0.98 | 0.48 | 0.04 | 0.38 | 0.98 | 0.61 | 0.66 | 0.41 | 0.07 | 0.514 |
| 7 | CAR | Car Group Limited | 11,632 | 11.24 | 0.979 | 0.728 | 0.474 | 0.98 | 0.91 | 0.52 | 0.33 | 0.79 | 0.41 | 0.40 | 0.59 | 0.51 | 0.606 |
| 8 | FLT | Flight Centre Travel | 3,210 | 10.77 | 0.976 | 0.945 | 0.581 | 0.98 | 0.58 | 0.63 | 0.71 | 0.65 | 0.79 | 0.40 | 0.81 | 0.78 | 0.704 |
| 9 | PDN | Paladin Energy Ltd | 1,869 | 10.63 | 0.972 | 0.852 | 0.508 | 0.97 | 0.27 | 0.57 | 0.86 | 0.50 | 0.64 | 0.81 | 0.38 | 0.13 | 0.571 |
| 10 | 4DX | 4Dmedical Limited | 1,948 | 10.06 | 0.969 | 0.812 | 0.549 | 0.97 | 0.99 | 0.61 | 0.99 | 0.50 | 0.71 | 0.00 | 0.01 | 0.06 | 0.539 |
| 11 | LYC | Lynas Rare Earths | 12,191 | 9.97 | 0.966 | 0.659 | 0.361 | 0.97 | 0.67 | 0.30 | 0.59 | 0.92 | 0.57 | 0.57 | 0.26 | 0.11 | 0.549 |
| 12 | HLS | Healius | 657 | 9.55 | 0.962 | 0.993 | 0.613 | 0.96 | 0.92 | 0.96 | 0.97 | 0.50 | 0.06 | 0.83 | 0.93 | 0.58 | 0.745 |
| 13 | BPT | Beach Energy Limited | 2,691 | 9.50 | 0.959 | 0.607 | 0.482 | 0.96 | 0.83 | 0.55 | 0.56 | 0.10 | 0.14 | 0.43 | 0.39 | 0.74 | 0.522 |
| 14 | BAP | Bapcor Limited | 703 | 9.41 | 0.955 | 0.986 | 0.673 | 0.96 | 0.71 | 0.96 | 0.98 | 0.50 | 0.17 | 0.90 | 0.79 | 0.57 | 0.726 |
| 15 | CUV | Clinuvel Pharmaceut. | 625 | 9.38 | 0.952 | 0.772 | 0.515 | 0.95 | 0.64 | 0.65 | 0.25 | 0.71 | 0.40 | 0.46 | 0.06 | 0.96 | 0.565 |

### How to read this table

* **A row of mostly dark cells (cells near 1.0)** = the name is shortable across multiple dimensions. These are the kind of multi-factor shorts the EW composite is built to find -- crowded SI + falling momentum + low quality + high leverage.
* **A row with high SI columns but low fundamentals columns** = a 'pure crowded-short' play. Naive will rank it highly because of `score_naive` (just SI rank), but the EW composite will weight it less if its quality / valuation / momentum aren't also bearish.
* **A row with low `EW factor avg` despite a top-15 spot** means naive is the only model pushing the name onto the list. These are the names most exposed to squeeze risk -- where the broad market doesn't share the consensus short view.
