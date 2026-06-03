# short-king-2.0 - Results

_Generated 2026-06-03T17:00:41_

## Model metrics (out-of-fold)

| model | ic_mean | ic_t | ic_hit_rate | ic_n_periods | decile_spread_mean |
| --- | --- | --- | --- | --- | --- |
| ew | 0.075 | 9.120 | 0.782 | 211 | -0.001 |
| naive | 0.012 | 2.132 | 0.540 | 211 | -0.008 |
| gbm_cls | -0.030 | -2.550 | 0.367 | 98 | -0.004 |
| logit | -0.043 | -4.293 | 0.311 | 90 | -0.007 |
| gbm_rank | -0.045 | -2.505 | 0.367 | 98 | 0.010 |

## Backtest summary

| model | strategy | CAGR | vol | Sharpe | Sortino | MaxDD | Calmar | hit_rate | avg_turnover | n_rebalances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ew | decile_short | -0.183 | 0.182 | -1.016 | -1.307 | -0.666 | -0.275 | 0.448 | 0.185 | 212 |
| ew | long_short_decile | -0.059 | 0.187 | -0.231 | -0.329 | -0.407 | -0.145 | 0.429 | 0.375 | 212 |
| gbm_cls | decile_short | -0.092 | 0.238 | -0.289 | -0.428 | -0.454 | -0.203 | 0.465 | 0.479 | 99 |
| gbm_cls | long_short_decile | -0.018 | 0.192 | -0.002 | -0.002 | -0.393 | -0.047 | 0.515 | 0.926 | 99 |
| gbm_rank | decile_short | -0.241 | 0.334 | -0.659 | -0.933 | -0.605 | -0.398 | 0.424 | 0.400 | 99 |
| gbm_rank | long_short_decile | -0.227 | 0.284 | -0.765 | -1.020 | -0.568 | -0.400 | 0.465 | 0.704 | 99 |
| logit | decile_short | -0.129 | 0.193 | -0.620 | -0.844 | -0.411 | -0.314 | 0.516 | 0.197 | 91 |
| logit | long_short_decile | 0.036 | 0.151 | 0.307 | 0.467 | -0.220 | 0.162 | 0.527 | 0.378 | 91 |
| naive | decile_short | -0.060 | 0.266 | -0.100 | -0.144 | -0.464 | -0.129 | 0.458 | 0.061 | 212 |
| naive | long_short_decile | 0.081 | 0.201 | 0.488 | 0.722 | -0.183 | 0.443 | 0.552 | 0.251 | 212 |

## Charts

![cumulative_returns](charts/cumulative_returns.png)
![drawdowns](charts/drawdowns.png)
![monthly_heatmap](charts/monthly_heatmap.png)
