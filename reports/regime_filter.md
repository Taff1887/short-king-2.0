# Market-regime filter sweep (skip shorts during ASX 200 rally regimes)

Rule: in any rebalance month where ASX 200 trailing-N-month return exceeded the threshold AT THE REBALANCE DATE, skip the short leg (short-only -> 0 %; L/S quintile -> long-only that month). No look-ahead: trailing return uses only prior closes. **All numbers use the NO-STOP backtests** so the filter effect is isolated from stop-loss noise.


## quintile_short -- ALL Sharpe

| Model | NONE | trail3m>5% | trail3m>10% | trail3m>15% | trail3m>20% | trail6m>5% | trail6m>10% | trail6m>15% | trail6m>20% |
|---|---|---|---|---|---|---|---|---|---|
| naive | -0.294 | -0.175 | -0.282 | -0.290 | -0.290 | -0.258 | -0.200 | -0.284 | -0.287 |
| ew | -0.202 | -0.126 | -0.172 | -0.183 | -0.183 | -0.213 | -0.100 | -0.199 | -0.198 |
| logit | -0.365 | -0.229 | -0.327 | -0.330 | -0.330 | -0.262 | -0.129 | -0.336 | -0.365 |

## quintile_short -- OOS Sharpe

| Model | NONE | trail3m>5% | trail3m>10% | trail3m>15% | trail3m>20% | trail6m>5% | trail6m>10% | trail6m>15% | trail6m>20% |
|---|---|---|---|---|---|---|---|---|---|
| naive | -0.324 | +0.246 | -0.090 | -0.324 | -0.324 | -0.171 | -0.285 | -0.324 | -0.324 |
| ew | -0.079 | +0.471 | +0.090 | -0.079 | -0.079 | -0.117 | -0.008 | -0.079 | -0.079 |
| logit | -0.204 | +0.348 | -0.088 | -0.204 | -0.204 | -0.361 | -0.159 | -0.204 | -0.204 |

## long_short_quintile -- ALL Sharpe

| Model | NONE | trail3m>5% | trail3m>10% | trail3m>15% | trail3m>20% | trail6m>5% | trail6m>10% | trail6m>15% | trail6m>20% |
|---|---|---|---|---|---|---|---|---|---|
| naive | +0.624 | +0.514 | +0.535 | +0.572 | +0.572 | +0.308 | +0.389 | +0.604 | +0.637 |
| ew | +0.577 | +0.480 | +0.568 | +0.594 | +0.594 | +0.398 | +0.631 | +0.558 | +0.576 |
| logit | +0.241 | +0.273 | +0.265 | +0.291 | +0.291 | +0.273 | +0.478 | +0.256 | +0.230 |

## long_short_quintile -- OOS Sharpe

| Model | NONE | trail3m>5% | trail3m>10% | trail3m>15% | trail3m>20% | trail6m>5% | trail6m>10% | trail6m>15% | trail6m>20% |
|---|---|---|---|---|---|---|---|---|---|
| naive | +0.924 | +1.554 | +1.472 | +0.924 | +0.924 | +1.115 | +0.701 | +0.924 | +0.924 |
| ew | +0.913 | +1.289 | +1.092 | +0.913 | +0.913 | +0.743 | +1.082 | +0.913 | +0.913 |
| logit | +0.242 | +0.668 | +0.287 | +0.242 | +0.242 | +0.043 | +0.350 | +0.242 | +0.242 |

## Months skipped (by filter)

| Filter | Months skipped (of ~191 total) |
|---|---:|
| trail3m>5% | 53 |
| trail3m>10% | 10 |
| trail3m>15% | 1 |
| trail3m>20% | 1 |
| trail6m>5% | 69 |
| trail6m>10% | 32 |
| trail6m>15% | 5 |
| trail6m>20% | 1 |
