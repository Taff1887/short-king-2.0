# Stop-loss impact — long-short quintile only

Side-by-side OOS / IS comparison of the same 5 models with the 15 % per-position hard stop (default) vs with the stop disabled (`--stop-loss-pct 1.0`). Everything else is identical: monthly rebalance, 25 bps commission, 1.5 % p.a. borrow, 5 bps slippage, Friday-release rebalance dates.

| model | period | Sharpe (with) | Sharpe (no) | Δ Sharpe | CAGR (with) | CAGR (no) | MaxDD (with) | MaxDD (no) |
|---|---|---|---|---|---|---|---|---|
| naive | IS | 2.0 | 0.57 | 1.43 | +29.8% | +7.5% | -12.2% | -31.0% |
| naive | OOS | 3.85 | 0.92 | 2.93 | +44.9% | +11.2% | -4.2% | -10.8% |
| logit | IS | 2.59 | 0.24 | 2.35 | +34.0% | +2.5% | -9.7% | -37.6% |
| logit | OOS | 2.8 | 0.24 | 2.56 | +47.1% | +2.9% | -6.5% | -37.1% |
| gbm_rank | IS | 2.47 | -0.06 | 2.54 | +37.6% | -3.3% | -7.6% | -66.0% |
| gbm_rank | OOS | 2.23 | -0.21 | 2.44 | +47.4% | -8.2% | -15.0% | -51.6% |
| gbm_cls | IS | 1.89 | -0.1 | 1.99 | +25.3% | -2.3% | -9.3% | -50.9% |
| gbm_cls | OOS | 1.63 | -1.11 | 2.74 | +19.9% | -16.3% | -11.4% | -44.4% |
| ew | IS | 0.77 | -0.38 | 1.15 | +12.5% | -7.6% | -30.9% | -83.5% |
| ew | OOS | 1.41 | 0.08 | 1.33 | +25.5% | -0.4% | -10.9% | -33.1% |
