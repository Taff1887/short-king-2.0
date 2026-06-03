# short-king-2.0 - Results

_Generated 2026-06-03T20:55:31_

## Model metrics (out-of-fold)

| model | ic_mean | ic_t | ic_hit_rate | ic_n_periods | decile_spread_mean |
| --- | --- | --- | --- | --- | --- |
| ew | 0.075 | 9.120 | 0.782 | 211 | -0.001 |
| naive | 0.012 | 2.132 | 0.540 | 211 | -0.008 |
| gbm_cls | -0.030 | -2.550 | 0.367 | 98 | -0.004 |
| logit | -0.043 | -4.293 | 0.311 | 90 | -0.007 |
| gbm_rank | -0.045 | -2.505 | 0.367 | 98 | 0.010 |

## Backtest summary

| model | strategy | CAGR | vol | Sharpe | Sortino | MaxDD | Calmar | hit_rate | avg_turnover | n_rebalances | n_stops_total | stop_loss_savings_total | stop_commission_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ew | quintile_short | -0.109 | 0.166 | -0.613 | -0.856 | -0.535 | -0.204 | 0.415 | 0.133 | 212 | 234 | 0.166 | 0.014 |
| ew | long_short_quintile | 0.089 | 0.152 | 0.636 | 1.015 | -0.265 | 0.336 | 0.500 | 0.261 | 212 | 1003 | 0.732 | 0.059 |
| gbm_cls | quintile_short | 0.045 | 0.205 | 0.315 | 0.530 | -0.349 | 0.129 | 0.495 | 0.390 | 99 | 455 | 0.548 | 0.026 |
| gbm_cls | long_short_quintile | 0.187 | 0.146 | 1.245 | 2.105 | -0.224 | 0.834 | 0.566 | 0.759 | 99 | 610 | 0.661 | 0.035 |
| gbm_rank | quintile_short | 0.174 | 0.253 | 0.759 | 1.314 | -0.352 | 0.495 | 0.505 | 0.302 | 99 | 637 | 0.793 | 0.036 |
| gbm_rank | long_short_quintile | 0.262 | 0.198 | 1.275 | 2.087 | -0.305 | 0.856 | 0.606 | 0.538 | 99 | 687 | 0.832 | 0.039 |
| logit | quintile_short | -0.032 | 0.161 | -0.125 | -0.196 | -0.324 | -0.099 | 0.484 | 0.155 | 91 | 89 | 0.154 | 0.009 |
| logit | long_short_quintile | 0.206 | 0.109 | 1.773 | 3.525 | -0.098 | 2.091 | 0.571 | 0.303 | 91 | 137 | 0.220 | 0.014 |
| naive | quintile_short | 0.100 | 0.217 | 0.546 | 0.868 | -0.294 | 0.340 | 0.495 | 0.052 | 212 | 568 | 0.570 | 0.033 |
| naive | long_short_quintile | 0.345 | 0.134 | 2.286 | 4.522 | -0.064 | 5.383 | 0.608 | 0.170 | 212 | 1172 | 1.061 | 0.068 |

## Charts

![cumulative_returns](charts/cumulative_returns.png)
![decile_returns](charts/decile_returns.png)
![drawdowns](charts/drawdowns.png)
![feature_correlation](charts/feature_correlation.png)
![feature_distributions](charts/feature_distributions.png)
![monthly_heatmap](charts/monthly_heatmap.png)
![shap_summary](charts/shap_summary.png)
![si_distribution](charts/si_distribution.png)
![top_short_candidates](charts/top_short_candidates.png)
![universe_coverage](charts/universe_coverage.png)
