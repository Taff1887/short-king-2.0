# short-king-2.0 - Results

_Generated 2026-06-05T12:41:22_

## Model metrics (out-of-fold)

| model | ic_mean | ic_t | ic_hit_rate | ic_n_periods | decile_spread_mean |
| --- | --- | --- | --- | --- | --- |
| ew | 0.075 | 9.120 | 0.782 | 211 | -0.001 |
| naive | 0.012 | 2.132 | 0.540 | 211 | -0.008 |
| gbm_cls | -0.030 | -2.550 | 0.367 | 98 | -0.004 |
| logit | -0.043 | -4.293 | 0.311 | 90 | -0.007 |
| gbm_rank | -0.045 | -2.505 | 0.367 | 98 | 0.010 |

## Backtest summary

| model | strategy | period | CAGR | vol | Sharpe | Sortino | MaxDD | Calmar | hit_rate | avg_turnover | n_rebalances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ew | quintile_short | ALL | -0.054 | 0.003 | -22.089 | -6.856 | -0.241 | -0.226 | 0.000 | 0.131 | 259 |
| ew | long_short_quintile | ALL | -0.093 | 0.004 | -23.189 | -6.887 | -0.380 | -0.244 | 0.000 | 0.264 | 259 |
| gbm_cls | quintile_short | ALL | -0.128 | 0.005 | -26.730 | -6.965 | -0.227 | -0.564 | 0.000 | 0.390 | 99 |
| gbm_cls | long_short_quintile | ALL | -0.220 | 0.010 | -24.712 | -6.925 | -0.374 | -0.590 | 0.000 | 0.748 | 99 |
| gbm_rank | quintile_short | ALL | -0.103 | 0.003 | -33.354 | -7.050 | -0.184 | -0.559 | 0.000 | 0.300 | 99 |
| gbm_rank | long_short_quintile | ALL | -0.167 | 0.006 | -28.555 | -6.994 | -0.289 | -0.577 | 0.000 | 0.535 | 99 |
| logit | quintile_short | ALL | -0.062 | 0.005 | -11.827 | -6.166 | -0.102 | -0.602 | 0.000 | 0.156 | 91 |
| logit | long_short_quintile | ALL | -0.105 | 0.010 | -10.743 | -5.998 | -0.171 | -0.613 | 0.000 | 0.305 | 91 |
| naive | quintile_short | ALL | -0.031 | 0.002 | -20.390 | -6.800 | -0.144 | -0.218 | 0.000 | 0.054 | 259 |
| naive | long_short_quintile | ALL | -0.071 | 0.003 | -22.227 | -6.860 | -0.303 | -0.235 | 0.000 | 0.189 | 259 |

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
