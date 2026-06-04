# OOS short trades — model = logit

_Reconstructed from 2,089 OOS short positions across 224 unique tickers in the 36-month holdout (2023-06 → 2026-05). Per-position P&L applies **no stop loss** (per-position returns uncapped — same as the default headline backtest) + commission + borrow as the headline backtest._

**Columns:** `n_months_shorted` = number of monthly rebalances the ticker was in the SHORT quintile. `total_pnl_book_%` = cumulative contribution to book over all those months (positive = the strategy made money shorting this name). `avg_trade_%` = average per-position return (positive = the stock fell, the short made money). `best_%` / `worst_%` = best / worst single month per-position return. `hit_%` = share of monthly shorts that were profitable. `avg_SI_%` = average reported short interest at the time of shorting. `mktCap_$m` = average market cap when shorted (AUD m).

## Top 20 winners (most profitable shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | n_stops_triggered | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CCX | City Chic Collective | 17 | 3.82 | 13.96 | 60.66 | -30.43 | 70.6 | 0 | 0.59 | 55.0 | 2024-01 | 2026-03 |
| LOT | Lotus Resources Ltd | 16 | 3.186 | 12.15 | 57.58 | -28.95 | 62.5 | 0 | 7.22 | 487.0 | 2024-07 | 2026-04 |
| NMT | Neometals Ltd | 16 | 2.979 | 11.82 | 38.03 | -50.0 | 81.2 | 0 | 1.61 | 157.0 | 2023-06 | 2025-06 |
| CXL | Calix Limited | 30 | 2.926 | 6.47 | 48.98 | -70.77 | 66.7 | 0 | 2.51 | 399.0 | 2023-06 | 2026-04 |
| GLL | Galilee Energy Ltd | 10 | 2.788 | 17.48 | 44.68 | -5.26 | 70.0 | 0 | 0.09 | 13.0 | 2024-01 | 2025-02 |
| SGR | The Star Ent Grp | 21 | 2.752 | 8.5 | 41.11 | -26.92 | 71.4 | 0 | 4.36 | 1111.0 | 2023-06 | 2026-03 |
| WBT | Weebit Nano Ltd | 18 | 2.379 | 8.64 | 36.67 | -33.72 | 72.2 | 0 | 5.05 | 738.0 | 2023-07 | 2026-03 |
| ERA | Energy Resources 'A' | 7 | 2.371 | 21.06 | 84.38 | -100.0 | 85.7 | 0 | 0.0 | 810.0 | 2024-03 | 2024-09 |
| CHN | Chalice Mining Ltd | 16 | 2.198 | 8.94 | 52.94 | -49.78 | 68.8 | 0 | 5.76 | 1316.0 | 2023-06 | 2025-06 |
| BAP | Bapcor Limited | 6 | 2.162 | 21.31 | 62.15 | -3.33 | 83.3 | 0 | 7.26 | 1034.0 | 2025-11 | 2026-04 |
| MVP | Medical Developments | 21 | 1.919 | 6.13 | 34.78 | -34.84 | 61.9 | 0 | 0.32 | 58.0 | 2023-06 | 2025-09 |
| AMS | Atomos | 6 | 1.753 | 18.03 | 44.44 | -22.86 | 66.7 | 0 | 0.16 | 24.0 | 2024-07 | 2024-12 |
| HLS | Healius | 15 | 1.702 | 7.23 | 35.85 | -19.46 | 73.3 | 0 | 5.1 | 728.0 | 2024-01 | 2026-04 |
| BUB | Bubs Aust Ltd | 14 | 1.662 | 7.74 | 27.27 | -12.5 | 78.6 | 0 | 1.12 | 117.0 | 2023-06 | 2026-02 |
| PEN | Peninsula Energy Ltd | 17 | 1.634 | 6.24 | 30.05 | -55.38 | 70.6 | 0 | 1.16 | 221.0 | 2024-01 | 2026-04 |
| LIC | Lifestyle Communit. | 24 | 1.62 | 4.72 | 36.65 | -20.59 | 54.2 | 0 | 8.18 | 1307.0 | 2024-01 | 2025-12 |
| AQZ | Alliance Aviation | 24 | 1.612 | 4.55 | 41.41 | -6.12 | 62.5 | 0 | 0.05 | 442.0 | 2023-07 | 2026-04 |
| BOT | Botanix Pharma Ltd | 12 | 1.607 | 8.57 | 53.97 | -13.79 | 75.0 | 0 | 2.8 | 670.0 | 2024-08 | 2025-12 |
| AD8 | Audinategroupltd | 10 | 1.513 | 9.31 | 34.33 | -11.02 | 70.0 | 0 | 4.21 | 509.0 | 2025-07 | 2026-04 |
| GLN | Galan Lithium Ltd | 13 | 1.508 | 7.57 | 46.38 | -41.67 | 84.6 | 0 | 1.1 | 171.0 | 2023-06 | 2025-04 |

## Top 20 losers (worst shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | n_stops_triggered | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SRL | Sunrise | 30 | -9.427 | -17.9 | 37.66 | -231.23 | 53.3 | 0 | 1.34 | 54.0 | 2023-06 | 2025-12 |
| 4DX | 4Dmedical Limited | 20 | -9.385 | -26.7 | 35.06 | -314.04 | 50.0 | 0 | 0.05 | 218.0 | 2023-11 | 2025-12 |
| TTT | Titomic Limited | 21 | -5.429 | -14.72 | 39.47 | -124.14 | 42.9 | 0 | 0.08 | 7378.0 | 2024-01 | 2026-03 |
| ASM | Ausstratmaterials | 28 | -5.113 | -10.21 | 36.14 | -163.91 | 50.0 | 0 | 1.71 | 142.0 | 2023-06 | 2025-12 |
| APX | Appen Limited | 25 | -3.399 | -7.7 | 55.56 | -127.27 | 56.0 | 0 | 3.93 | 181.0 | 2023-06 | 2026-01 |
| SPL | Starpharma Holdings | 19 | -3.391 | -9.67 | 58.46 | -172.0 | 47.4 | 0 | 0.43 | 70.0 | 2023-06 | 2025-12 |
| EOS | Electro Optic Sys. | 19 | -3.312 | -9.63 | 28.18 | -118.68 | 36.8 | 0 | 1.03 | 523.0 | 2024-07 | 2026-04 |
| EGR | Ecograf Limited | 13 | -2.797 | -12.37 | 28.12 | -139.13 | 46.2 | 0 | 0.29 | 89.0 | 2024-01 | 2025-12 |
| IXR | Ionic Rare Earths | 26 | -2.776 | -5.8 | 40.0 | -125.0 | 50.0 | 0 | 0.09 | 72.0 | 2023-06 | 2026-03 |
| CAT | Catapult Grp Int Ltd | 13 | -2.253 | -9.76 | 10.88 | -44.31 | 38.5 | 0 | 0.77 | 672.0 | 2023-06 | 2025-09 |
| LTR | Liontown Resources | 19 | -2.164 | -5.93 | 29.62 | -49.58 | 31.6 | 0 | 7.37 | 2434.0 | 2024-02 | 2026-04 |
| DYL | Deep Yellow Limited | 13 | -2.107 | -9.11 | 21.64 | -54.13 | 46.2 | 0 | 5.85 | 688.0 | 2023-06 | 2024-06 |
| ABY | Adore Beauty | 22 | -2.078 | -5.17 | 64.09 | -49.73 | 36.4 | 0 | 0.97 | 85.0 | 2023-06 | 2026-04 |
| NXL | Nuix Limited | 8 | -2.056 | -15.04 | 12.5 | -75.29 | 25.0 | 0 | 0.81 | 722.0 | 2023-06 | 2025-05 |
| PPK | Ppk Group Limited | 18 | -2.045 | -6.11 | 27.01 | -112.12 | 61.1 | 0 | 0.24 | 80.0 | 2023-06 | 2026-03 |
| BMN | Bannerman Energy Ltd | 15 | -2.028 | -7.54 | 19.35 | -41.0 | 53.3 | 0 | 2.52 | 478.0 | 2023-06 | 2026-02 |
| MME | Moneyme Limited | 14 | -1.937 | -7.83 | 18.75 | -51.56 | 42.9 | 0 | 0.0 | 98.0 | 2023-07 | 2025-06 |
| PDN | Paladin Energy Ltd | 19 | -1.9 | -5.41 | 37.86 | -39.59 | 31.6 | 0 | 7.3 | 1954.0 | 2023-06 | 2025-09 |
| IMU | Imugene Limited | 17 | -1.847 | -5.94 | 34.0 | -143.9 | 52.9 | 0 | 4.74 | 473.0 | 2023-06 | 2025-12 |
| RSG | Resolute Mining | 5 | -1.791 | -20.41 | -0.0 | -65.12 | 0.0 | 0 | 0.41 | 1299.0 | 2025-08 | 2025-12 |

## Aggregate OOS stats (short leg only)

- **Total short-leg P&L**: -41.8 % of book (summed over 2,089 monthly positions)
- **Median per-position return**: +1.91 %
- **Win rate** (per-position): 53.5 %
- **Stop-fire rate**: 0.0 % of positions clipped at -16 %
- **Best single month**: `CCX` (+60.66 %)
- **Worst single month**: `SRL` (-231.23 %)