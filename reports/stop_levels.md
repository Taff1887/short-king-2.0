# Stop-level sweep — long-short quintile OOS

Same engine, same costs, same Friday rebalance, same 100 bps stop-fill slippage. Only the cumulative stop floor moves.

### OOS Sharpe

model     gbm_rank  logit  naive
stop_pct                        
off          -0.21   0.24   0.92
0.20          1.64   2.16   2.83
0.15          2.23   2.80   3.85
0.12          2.74   3.35   4.89
0.10          3.15   3.82   5.79
0.08          3.65   4.35   6.90

### OOS CAGR (%)

model     gbm_rank  logit  naive
stop_pct                        
off           -8.2    2.9   11.2
0.20          34.6   36.0   32.6
0.15          47.4   47.1   44.9
0.12          58.6   57.2   57.0
0.10          67.9   65.9   67.8
0.08          79.3   76.6   81.7

### OOS MaxDD (%)

model     gbm_rank  logit  naive
stop_pct                        
off          -51.6  -37.1  -10.8
0.20         -20.3  -10.1   -4.9
0.15         -15.0   -6.5   -4.2
0.12         -11.2   -5.1   -3.4
0.10          -8.8   -4.0   -2.6
0.08          -5.9   -3.2   -1.8
