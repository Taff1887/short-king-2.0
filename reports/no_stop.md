# Stop-DISABLED OOS short-book analysis

All 5 models re-evaluated on the OOS holdout (2023-06 → 2026-05) with the 15 % per-position stop turned OFF. Per-position returns are the raw monthly forward returns — no floor. The naive baseline still uses raw ShortPct; the trained models (logit / gbm_cls / gbm_rank) use the same final-fit predictions as the headline run.

**The 50 % win-rate question**: what share of monthly short positions ended in profit (= stock fell)?

   model  n_positions  win_rate_%  mean_trade_%  median_trade_%  std_trade_%  worst_position_%  n_loss_>25%  n_loss_>50%  n_win_>25%  short_book_Sharpe_OOS  short_book_MaxDD_%
   logit         2089        53.4         -0.33            1.71        21.68            -218.2          161           42         140                  -0.51               -50.5
gbm_rank         2089        52.5         -0.87            1.85        23.60            -217.5          214           71         149                  -0.63               -62.2
   naive         2089        50.9         -0.00            0.43        15.39            -177.4          104           11          87                  -0.42               -37.8
 gbm_cls         2089        50.8         -1.27            0.38        19.18            -217.5          136           41          77                  -1.21               -58.8
      ew         2089        47.4         -0.56           -0.50        10.13             -49.7           37            0          25                  -1.17               -40.9

### Read this carefully

- **`win_rate_%`** is the per-position monthly win-rate — how   often the stock fell during the month we were short it. The   goal is > 50 %.
- **`worst_position_%`** is the single worst monthly outcome.   Without the stop, this is uncapped — a stock that rallied   40 % in a month gives a 40 % per-position loss.
- **`n_loss_>25%`** and **`n_loss_>50%`** count the catastrophic   single-month positions (un-stopped). These are the squeezes /   takeover bumps / commodity rallies that the 15 % stop normally   caps at −16 %.

## Worst 20 single-position outcomes (no stop)

   model    Date Ticker  SI_%  trade_return_%  stock_ret_%
   logit 2024-07    APX  3.18          -218.2        218.2
   logit 2025-08    4DX  0.01          -217.5        217.5
 gbm_cls 2025-08    4DX  0.01          -217.5        217.5
gbm_rank 2025-08    4DX  0.01          -217.5        217.5
 gbm_cls 2024-01    BRN  4.33          -177.4        177.4
   naive 2024-01    BRN  4.33          -177.4        177.4
   logit 2024-01    BRN  4.33          -177.4        177.4
gbm_rank 2024-01    BRN  4.33          -177.4        177.4
gbm_rank 2025-08    SRL  0.96          -173.9        173.9
   logit 2025-08    SRL  0.96          -173.9        173.9
 gbm_cls 2025-08    SRL  0.96          -173.9        173.9
 gbm_cls 2025-02    EGR  0.02          -169.6        169.6
   logit 2025-02    EGR  0.02          -169.6        169.6
gbm_rank 2025-02    EGR  0.02          -169.6        169.6
gbm_rank 2025-12    ASM  0.01          -139.1        139.1
   logit 2025-12    ASM  0.01          -139.1        139.1
 gbm_cls 2023-11    TTT  0.18          -137.5        137.5
gbm_rank 2023-11    TTT  0.18          -137.5        137.5
   logit 2025-11    4DX  0.03          -124.0        124.0
gbm_rank 2025-11    4DX  0.03          -124.0        124.0
