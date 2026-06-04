# Exit-protocol comparison — OOS short book

Replay of every OOS short position (model = logit, L/S quintile, 36-month holdout) with day-by-day price monitoring. Each protocol can exit early during the monthly hold; whichever trigger fires first wins. Same costs as headline backtest.

_For dollar-neutral L/S impact see the README — the long leg is identical across protocols; only the short-leg exit logic changes._

| protocol | label | OOS_Sharpe | OOS_CAGR_% | OOS_MaxDD_% | win_rate_% | stop_fire_rate_% | daily_stops | cum_stops | held_to_rebal | avg_days_held |
|---|---|---|---|---|---|---|---|---|---|---|
| A_monthly_only | Monthly EOM stop at +15 % cumulative (current default) | -0.24 | -6.64 | -38.91 | 56.1 | 27.7 | 0 | 578 | 1511 | 18.2 |
| B_daily_10 | Daily intraday: exit if any single day rises >10 % | -0.24 | -6.93 | -39.39 | 56.8 | 21.7 | 453 | 0 | 1636 | 18.5 |
| C_trail_10 | Cumulative trailing: exit if stock rises >10 % from entry | -0.07 | -2.65 | -31.71 | 60.2 | 44.1 | 0 | 922 | 1167 | 15.8 |
| D_both_10 | Both: daily 10 % AND cumulative 10 % (first-to-fire) | -0.06 | -2.55 | -29.57 | 61.0 | 50.1 | 369 | 678 | 1042 | 15.2 |
