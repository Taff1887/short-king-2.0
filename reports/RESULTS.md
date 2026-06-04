# short-king-2.0 - Results

_Generated 2026-06-04T23:19:49_

## Model metrics (out-of-fold)

| model | ic_mean | ic_t | ic_hit_rate | ic_n_periods | decile_spread_mean |
| --- | --- | --- | --- | --- | --- |
| ew | 0.075 | 9.120 | 0.782 | 211 | -0.001 |
| naive | 0.012 | 2.132 | 0.540 | 211 | -0.008 |
| gbm_cls | -0.030 | -2.550 | 0.367 | 98 | -0.004 |
| logit | -0.043 | -4.293 | 0.311 | 90 | -0.007 |
| gbm_rank | -0.045 | -2.505 | 0.367 | 98 | 0.010 |

## Backtest summary

| model | strategy | CAGR | vol | Sharpe | Sortino | MaxDD | Calmar | hit_rate | avg_turnover | n_rebalances | n_stops_total | stop_loss_savings_total | stop_commission_total | stop_slippage_drag_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ew | quintile_short | -0.116 | 0.167 | -0.651 | -0.903 | -0.547 | -0.211 | 0.415 | 0.133 | 212 | 234 | 0.138 | 0.014 | 0.028 |
| ew | long_short_quintile | 0.058 | 0.154 | 0.443 | 0.688 | -0.290 | 0.200 | 0.491 | 0.261 | 212 | 1003 | 0.615 | 0.059 | 0.117 |
| gbm_cls | quintile_short | 0.017 | 0.206 | 0.180 | 0.296 | -0.367 | 0.045 | 0.475 | 0.390 | 99 | 455 | 0.496 | 0.026 | 0.052 |
| gbm_cls | long_short_quintile | 0.144 | 0.147 | 0.985 | 1.609 | -0.243 | 0.592 | 0.556 | 0.759 | 99 | 610 | 0.592 | 0.035 | 0.070 |
| gbm_rank | quintile_short | 0.129 | 0.256 | 0.600 | 1.013 | -0.377 | 0.343 | 0.505 | 0.302 | 99 | 637 | 0.720 | 0.036 | 0.073 |
| gbm_rank | long_short_quintile | 0.210 | 0.200 | 1.052 | 1.674 | -0.324 | 0.648 | 0.576 | 0.538 | 99 | 687 | 0.754 | 0.039 | 0.079 |
| logit | quintile_short | -0.043 | 0.161 | -0.191 | -0.296 | -0.330 | -0.129 | 0.473 | 0.155 | 91 | 89 | 0.136 | 0.009 | 0.019 |
| logit | long_short_quintile | 0.186 | 0.109 | 1.617 | 3.134 | -0.108 | 1.733 | 0.560 | 0.303 | 91 | 137 | 0.191 | 0.014 | 0.029 |
| naive | quintile_short | 0.082 | 0.218 | 0.468 | 0.734 | -0.305 | 0.268 | 0.491 | 0.052 | 212 | 568 | 0.504 | 0.033 | 0.066 |
| naive | long_short_quintile | 0.301 | 0.135 | 2.018 | 3.839 | -0.069 | 4.368 | 0.590 | 0.170 | 212 | 1172 | 0.924 | 0.068 | 0.137 |

## Charts

![cumulative_returns](charts/cumulative_returns.png)
![cumulative_returns_monthly](charts/cumulative_returns_monthly.png)
![decile_returns](charts/decile_returns.png)
![drawdowns](charts/drawdowns.png)
![drawdowns_monthly](charts/drawdowns_monthly.png)
![feature_correlation](charts/feature_correlation.png)
![feature_distributions](charts/feature_distributions.png)
![monthly_heatmap](charts/monthly_heatmap.png)
![monthly_heatmap_monthly](charts/monthly_heatmap_monthly.png)
![shap_summary](charts/shap_summary.png)
![si_distribution](charts/si_distribution.png)
![top_short_candidates](charts/top_short_candidates.png)
![universe_coverage](charts/universe_coverage.png)
