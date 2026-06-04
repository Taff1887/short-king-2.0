# OOS short trades — model = logit

_Reconstructed from 2,085 OOS short positions across 206 unique tickers in the 36-month holdout (2023-06 → 2026-05). Per-position P&L applies the same 15 % stop + 100 bps fill slippage + commission + borrow as the headline backtest._

**Columns:** `n_months_shorted` = number of monthly rebalances the ticker was in the SHORT quintile. `total_pnl_book_%` = cumulative contribution to book over all those months (positive = the strategy made money shorting this name). `avg_trade_%` = average per-position return (positive = the stock fell, the short made money). `best_%` / `worst_%` = best / worst single month per-position return. `hit_%` = share of monthly shorts that were profitable. `avg_SI_%` = average reported short interest at the time of shorting. `mktCap_$m` = average market cap when shorted (AUD m).

## Top 20 winners (most profitable shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | n_stops_triggered | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IMU | Imugene Limited | 32 | 5.421 | 10.81 | 42.31 | -16.0 | 75.0 | 4 | 4.5 | 397.0 | 2023-07 | 2026-03 |
| CXL | Calix Limited | 32 | 4.967 | 9.96 | 46.32 | -16.0 | 71.9 | 5 | 2.49 | 384.0 | 2023-06 | 2026-04 |
| BRN | Brainchip Ltd | 32 | 3.902 | 8.06 | 47.44 | -16.0 | 62.5 | 3 | 3.82 | 342.0 | 2023-06 | 2026-03 |
| ERA | Energy Resources 'A' | 7 | 3.892 | 34.29 | 85.71 | -9.37 | 85.7 | 0 | 0.0 | 810.0 | 2024-03 | 2024-09 |
| IPD | Impedimed Limited | 16 | 3.789 | 14.52 | 60.0 | -16.0 | 81.2 | 1 | 0.57 | 111.0 | 2024-05 | 2026-04 |
| PPK | Ppk Group Limited | 25 | 3.679 | 9.41 | 29.57 | -16.0 | 72.0 | 1 | 0.22 | 72.0 | 2023-06 | 2026-04 |
| LKE | Lake Resources | 26 | 3.653 | 9.16 | 45.0 | -16.0 | 65.4 | 4 | 1.89 | 174.0 | 2023-06 | 2025-12 |
| BOT | Botanix Pharma Ltd | 17 | 3.606 | 12.95 | 57.6 | -12.5 | 70.6 | 0 | 2.3 | 585.0 | 2024-06 | 2026-02 |
| PEN | Peninsula Energy Ltd | 24 | 3.297 | 8.99 | 45.88 | -16.0 | 50.0 | 2 | 2.12 | 214.0 | 2023-06 | 2025-12 |
| NMT | Neometals Ltd | 16 | 2.868 | 11.45 | 36.11 | -16.0 | 68.8 | 1 | 1.78 | 186.0 | 2023-06 | 2025-01 |
| AD8 | Audinategroupltd | 14 | 2.838 | 12.55 | 36.09 | -6.99 | 78.6 | 0 | 4.58 | 587.0 | 2024-11 | 2026-04 |
| CCX | City Chic Collective | 20 | 2.667 | 8.71 | 56.9 | -16.0 | 70.0 | 3 | 0.68 | 68.0 | 2023-07 | 2026-02 |
| LOT | Lotus Resources Ltd | 26 | 2.606 | 6.51 | 58.36 | -16.0 | 53.8 | 6 | 5.68 | 423.0 | 2023-06 | 2026-04 |
| GEM | G8 Education Limited | 4 | 2.586 | 37.8 | 51.47 | 23.4 | 100.0 | 0 | 4.12 | 523.0 | 2026-01 | 2026-04 |
| BAP | Bapcor Limited | 11 | 2.577 | 14.05 | 66.59 | -8.58 | 72.7 | 0 | 5.55 | 1335.0 | 2024-06 | 2026-04 |
| APX | Appen Limited | 27 | 2.43 | 6.22 | 64.41 | -16.0 | 59.3 | 8 | 3.36 | 192.0 | 2023-06 | 2026-02 |
| NVX | Novonix Limited | 16 | 2.366 | 9.56 | 26.09 | -14.75 | 81.2 | 0 | 4.52 | 402.0 | 2023-06 | 2025-05 |
| SRL | Sunrise | 19 | 2.356 | 8.16 | 33.33 | -16.0 | 68.4 | 4 | 1.48 | 59.0 | 2023-06 | 2025-09 |
| TLG | Talga Group Ltd | 29 | 2.174 | 5.18 | 34.62 | -16.0 | 69.0 | 3 | 1.24 | 263.0 | 2023-06 | 2026-03 |
| DEV | Devex Resources Ltd | 14 | 2.068 | 9.55 | 24.24 | -16.0 | 78.6 | 1 | 0.21 | 98.0 | 2023-07 | 2025-05 |

## Top 20 losers (worst shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | n_stops_triggered | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PLS | Pilbara Min Ltd | 23 | -1.629 | -3.24 | 21.03 | -16.0 | 43.5 | 7 | 15.78 | 9266.0 | 2024-01 | 2026-04 |
| PDN | Paladin Energy Ltd | 21 | -1.352 | -2.99 | 26.06 | -16.0 | 38.1 | 7 | 10.24 | 1897.0 | 2023-06 | 2026-01 |
| CAT | Catapult Grp Int Ltd | 23 | -1.17 | -2.43 | 29.45 | -16.0 | 43.5 | 5 | 1.98 | 1009.0 | 2023-07 | 2026-04 |
| FML | Focus Minerals Ltd | 9 | -1.087 | -6.24 | 15.38 | -16.0 | 33.3 | 4 | 0.0 | 563.0 | 2024-07 | 2025-12 |
| ADH | Adairs Limited | 10 | -1.006 | -5.13 | 16.47 | -16.0 | 30.0 | 5 | 2.41 | 297.0 | 2023-06 | 2024-10 |
| DYL | Deep Yellow Limited | 13 | -0.941 | -3.49 | 26.12 | -16.0 | 38.5 | 6 | 7.16 | 777.0 | 2023-06 | 2025-05 |
| KGN | Kogan.Com Ltd | 28 | -0.875 | -1.15 | 25.0 | -16.0 | 39.3 | 5 | 1.53 | 474.0 | 2023-06 | 2025-12 |
| BET | Betmakers Tech Group | 11 | -0.848 | -3.59 | 18.33 | -16.0 | 36.4 | 5 | 1.73 | 114.0 | 2023-06 | 2025-12 |
| NXT | Nextdc Limited | 4 | -0.826 | -11.41 | -5.79 | -16.0 | 0.0 | 1 | 7.6 | 8579.0 | 2025-04 | 2026-04 |
| ANG | Austin Engineering | 7 | -0.813 | -6.08 | 5.66 | -16.0 | 42.9 | 3 | 0.11 | 161.0 | 2023-06 | 2023-12 |
| SHV | Select Harvests | 21 | -0.8 | -1.52 | 21.41 | -16.0 | 38.1 | 4 | 4.03 | 513.0 | 2023-10 | 2026-04 |
| SGM | Sims Limited | 5 | -0.761 | -7.96 | 12.0 | -16.0 | 20.0 | 2 | 1.72 | 3264.0 | 2025-10 | 2026-04 |
| BMN | Bannerman Energy Ltd | 25 | -0.672 | -0.8 | 26.14 | -16.0 | 44.0 | 8 | 2.92 | 463.0 | 2023-06 | 2025-12 |
| ABY | Adore Beauty | 15 | -0.651 | -1.78 | 25.5 | -16.0 | 40.0 | 6 | 1.19 | 86.0 | 2023-06 | 2026-04 |
| MFG | Magellan Fin Grp Ltd | 4 | -0.644 | -8.8 | 3.29 | -16.0 | 25.0 | 2 | 2.13 | 1724.0 | 2023-07 | 2023-11 |
| EVN | Evolution Mining Ltd | 7 | -0.641 | -4.78 | 12.98 | -16.0 | 42.9 | 2 | 0.71 | 7068.0 | 2023-06 | 2024-06 |
| LTR | Liontown Resources | 28 | -0.629 | -0.49 | 34.56 | -16.0 | 32.1 | 6 | 8.27 | 2515.0 | 2024-01 | 2026-04 |
| MP1 | Megaport Limited | 4 | -0.619 | -8.21 | 15.15 | -16.0 | 25.0 | 3 | 7.49 | 1376.0 | 2023-06 | 2026-04 |
| SFR | Sandfire Resources | 4 | -0.606 | -8.0 | 0.56 | -16.0 | 25.0 | 1 | 4.1 | 5250.0 | 2025-08 | 2025-12 |
| PPS | Praemium Limited | 6 | -0.587 | -5.14 | 7.22 | -16.0 | 16.7 | 1 | 0.58 | 195.0 | 2024-01 | 2024-06 |

## Aggregate OOS stats (short leg only)

- **Total short-leg P&L**: +101.2 % of book (summed over 2,085 monthly positions)
- **Median per-position return**: +2.63 %
- **Win rate** (per-position): 54.9 %
- **Stop-fire rate**: 16.9 % of positions clipped at -16 %
- **Best single month**: `IMU` (+42.31 %)
- **Worst single month**: `PLS` (-16.00 %)