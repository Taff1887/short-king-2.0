# OOS short trades — model = logit

_Reconstructed from 2,089 OOS short positions across 230 unique tickers in the 36-month holdout (2023-06 → 2026-05). Per-position P&L is uncapped (no stop loss) — commission + borrow + slippage apply per the headline backtest._

**Columns:** `n_months_shorted` = number of monthly rebalances the ticker was in the SHORT quintile. `total_pnl_book_%` = cumulative contribution to book over all those months (positive = the strategy made money shorting this name). `avg_trade_%` = average per-position return (positive = the stock fell, the short made money). `best_%` / `worst_%` = best / worst single month per-position return. `hit_%` = share of monthly shorts that were profitable. `avg_SI_%` = average reported short interest at the time of shorting. `mktCap_$m` = average market cap when shorted (AUD m).

## Top 20 winners (most profitable shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CHN | Chalice Mining Ltd | 13 | 3.302 | 15.97 | 52.94 | -43.91 | 76.9 | 5.94 | 1373.0 | 2023-07 | 2025-06 |
| CXL | Calix Limited | 33 | 3.242 | 6.51 | 48.98 | -70.77 | 66.7 | 2.5 | 381.0 | 2023-06 | 2026-04 |
| LOT | Lotus Resources Ltd | 16 | 3.186 | 12.15 | 57.58 | -28.95 | 62.5 | 7.22 | 487.0 | 2024-07 | 2026-04 |
| SGR | The Star Ent Grp | 19 | 3.115 | 10.52 | 41.11 | -14.29 | 68.4 | 4.65 | 1283.0 | 2023-06 | 2025-06 |
| CCX | City Chic Collective | 10 | 2.967 | 18.44 | 60.66 | -14.29 | 90.0 | 0.57 | 65.0 | 2024-01 | 2025-08 |
| NMT | Neometals Ltd | 21 | 2.797 | 8.61 | 38.03 | -50.0 | 66.7 | 1.29 | 135.0 | 2023-06 | 2025-12 |
| GLL | Galilee Energy Ltd | 10 | 2.788 | 17.48 | 44.68 | -5.26 | 70.0 | 0.09 | 13.0 | 2024-01 | 2025-02 |
| ERA | Energy Resources 'A' | 7 | 2.371 | 21.06 | 84.38 | -100.0 | 85.7 | 0.0 | 810.0 | 2024-03 | 2024-09 |
| BAP | Bapcor Limited | 4 | 2.114 | 30.97 | 62.15 | 9.45 | 100.0 | 7.78 | 703.0 | 2026-01 | 2026-04 |
| PEN | Peninsula Energy Ltd | 24 | 1.996 | 5.59 | 45.92 | -55.38 | 62.5 | 1.6 | 195.0 | 2024-01 | 2026-03 |
| NVX | Novonix Limited | 14 | 1.876 | 8.56 | 28.17 | -35.29 | 78.6 | 3.32 | 312.0 | 2023-08 | 2026-02 |
| HLS | Healius | 13 | 1.797 | 8.64 | 35.85 | -15.35 | 69.2 | 5.46 | 784.0 | 2024-01 | 2026-04 |
| AQZ | Alliance Aviation | 25 | 1.787 | 4.8 | 41.41 | -6.12 | 64.0 | 0.05 | 441.0 | 2023-07 | 2026-04 |
| IPD | Impedimed Limited | 11 | 1.771 | 9.99 | 50.0 | -42.42 | 81.8 | 0.54 | 95.0 | 2025-01 | 2026-04 |
| AMS | Atomos | 6 | 1.753 | 18.03 | 44.44 | -22.86 | 66.7 | 0.16 | 24.0 | 2024-07 | 2024-12 |
| PAR | Paradigm Bio. | 21 | 1.637 | 5.19 | 38.78 | -45.31 | 66.7 | 0.53 | 127.0 | 2024-01 | 2026-04 |
| LIC | Lifestyle Communit. | 24 | 1.62 | 4.72 | 36.65 | -20.59 | 54.2 | 8.18 | 1307.0 | 2024-01 | 2025-12 |
| DEV | Devex Resources Ltd | 20 | 1.618 | 5.53 | 30.0 | -43.86 | 65.0 | 0.21 | 89.0 | 2024-01 | 2026-03 |
| GEM | G8 Education Limited | 3 | 1.565 | 30.75 | 37.5 | 23.4 | 100.0 | 3.81 | 523.0 | 2026-02 | 2026-04 |
| BOT | Botanix Pharma Ltd | 13 | 1.525 | 7.58 | 53.97 | -13.79 | 69.2 | 2.59 | 640.0 | 2024-06 | 2025-12 |

## Top 20 losers (worst shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SRL | Sunrise | 29 | -8.074 | -15.78 | 37.66 | -231.23 | 55.2 | 1.33 | 55.0 | 2023-06 | 2025-12 |
| 4DX | 4Dmedical Limited | 20 | -7.565 | -21.45 | 35.06 | -314.04 | 45.0 | 0.05 | 228.0 | 2023-11 | 2025-12 |
| ASM | Ausstratmaterials | 28 | -5.849 | -11.8 | 36.14 | -163.91 | 46.4 | 1.68 | 139.0 | 2023-06 | 2025-12 |
| TTT | Titomic Limited | 22 | -5.404 | -13.96 | 39.47 | -124.14 | 45.5 | 0.08 | 7988.0 | 2024-01 | 2026-03 |
| EOS | Electro Optic Sys. | 18 | -4.637 | -14.45 | 28.18 | -118.68 | 33.3 | 1.14 | 593.0 | 2024-07 | 2026-04 |
| APX | Appen Limited | 27 | -3.03 | -6.29 | 55.56 | -127.27 | 55.6 | 3.82 | 180.0 | 2023-06 | 2026-02 |
| EGR | Ecograf Limited | 13 | -2.832 | -12.53 | 28.12 | -139.13 | 46.2 | 0.26 | 87.0 | 2024-01 | 2025-12 |
| LTR | Liontown Resources | 21 | -2.792 | -7.1 | 29.62 | -49.58 | 28.6 | 7.48 | 2551.0 | 2024-01 | 2026-04 |
| IXR | Ionic Rare Earths | 24 | -2.722 | -6.2 | 40.0 | -125.0 | 50.0 | 0.1 | 72.0 | 2023-06 | 2025-12 |
| SLX | Silex Systems | 29 | -2.647 | -4.77 | 36.77 | -71.81 | 44.8 | 6.85 | 1243.0 | 2023-07 | 2026-04 |
| BMN | Bannerman Energy Ltd | 21 | -2.15 | -5.63 | 20.54 | -43.48 | 57.1 | 2.91 | 507.0 | 2023-06 | 2026-04 |
| CAT | Catapult Grp Int Ltd | 12 | -1.994 | -9.36 | 10.88 | -44.31 | 41.7 | 1.1 | 728.0 | 2023-06 | 2026-03 |
| MP1 | Megaport Limited | 6 | -1.965 | -19.35 | 17.54 | -42.52 | 16.7 | 7.13 | 1164.0 | 2023-06 | 2025-04 |
| NXL | Nuix Limited | 10 | -1.924 | -11.12 | 37.34 | -75.29 | 30.0 | 1.6 | 402.0 | 2023-06 | 2026-04 |
| DYL | Deep Yellow Limited | 17 | -1.913 | -6.18 | 31.94 | -54.13 | 47.1 | 6.45 | 864.0 | 2023-06 | 2026-03 |
| TOE | Toro Energy Limited | 18 | -1.693 | -4.82 | 20.0 | -54.84 | 55.6 | 0.32 | 33.0 | 2024-01 | 2026-02 |
| EMR | Emerald Res Nl | 7 | -1.69 | -13.57 | 1.04 | -35.45 | 14.3 | 1.98 | 2530.0 | 2024-07 | 2025-12 |
| ABY | Adore Beauty | 11 | -1.619 | -8.33 | 27.23 | -49.73 | 45.5 | 1.27 | 87.0 | 2023-06 | 2026-04 |
| VUL | Vulcan Energy | 35 | -1.6 | -2.08 | 34.47 | -55.36 | 54.3 | 4.46 | 462.0 | 2023-06 | 2026-04 |
| PDN | Paladin Energy Ltd | 19 | -1.6 | -4.46 | 37.86 | -39.59 | 36.8 | 6.06 | 1975.0 | 2023-06 | 2024-12 |

## Aggregate OOS stats (short leg only)

- **Total short-leg P&L**: -32.4 % of book (summed over 2,089 monthly positions)
- **Median per-position return**: +1.90 %
- **Win rate** (per-position): 53.4 %
- **Best single month**: `CHN` (+52.94 %)
- **Worst single month**: `SRL` (-231.23 %)