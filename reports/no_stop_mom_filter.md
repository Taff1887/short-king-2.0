# No-stop short book with momentum filter

Drop any name in the top X % of 12-week cross-sectional momentum BEFORE shorting. Re-equal-weight survivors. No stop loss applied. `mom_cutoff=1.0` = no filter (every top-quintile name shorted), `mom_cutoff=0.50` = only short below-median-momentum names.

**Key columns**: `win_rate_%` = share of monthly short positions that ended in profit (target > 50 %). `worst_position_%` = single worst monthly outcome (unstopped, so uncapped). `short_Sharpe_OOS` = annualised Sharpe of the short-leg-only monthly book.


### Short-leg OOS Sharpe (no stop, with momentum filter)

model         ew  gbm_cls  gbm_rank  logit  naive
mom_cutoff                                       
0.5        -0.36    -1.08     -0.53  -0.24  -0.28
0.6        -0.37    -1.00     -0.46  -0.30  -0.17
0.7        -0.63    -1.10     -0.47  -0.27  -0.14
0.8        -1.01    -1.17     -0.55  -0.36  -0.40
0.9        -1.17    -1.38     -0.59  -0.50  -0.40
1.0        -1.16    -1.18     -0.58  -0.45  -0.43

### Per-position win-rate %

model         ew  gbm_cls  gbm_rank  logit  naive
mom_cutoff                                       
0.5         50.3     50.6      52.4   52.9   50.2
0.6         51.0     51.2      52.9   53.2   51.1
0.7         50.4     50.9      52.7   53.5   51.6
0.8         48.7     50.9      52.4   53.2   50.7
0.9         47.6     50.1      52.2   52.6   50.6
1.0         47.4     50.9      52.6   53.5   50.9

### Worst single-position % (uncapped)

model         ew  gbm_cls  gbm_rank  logit  naive
mom_cutoff                                       
0.5        -45.7   -177.4    -177.4 -218.2 -177.4
0.6        -45.7   -177.4    -177.4 -218.2 -177.4
0.7        -45.7   -177.4    -177.4 -218.2 -177.4
0.8        -49.7   -177.4    -177.4 -218.2 -177.4
0.9        -49.7   -177.4    -177.4 -218.2 -177.4
1.0        -49.7   -217.5    -217.5 -218.2 -177.4

### Count of positions losing > 50 %

model        ew  gbm_cls  gbm_rank  logit  naive
mom_cutoff                                      
0.5         0.0     18.0      38.0   23.0    5.0
0.6         0.0     19.0      41.0   25.0    7.0
0.7         0.0     20.0      43.0   26.0    7.0
0.8         0.0     21.0      46.0   28.0    8.0
0.9         0.0     26.0      50.0   33.0    8.0
1.0         0.0     39.0      68.0   40.0   11.0
