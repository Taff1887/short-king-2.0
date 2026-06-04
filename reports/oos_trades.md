# OOS short trades — model = logit

_Reconstructed from 2,089 OOS short positions across 224 unique tickers in the 36-month holdout (2023-06 → 2026-05). Per-position P&L applies the same 15 % stop + 100 bps fill slippage + commission + borrow as the headline backtest._

**Columns:** `n_months_shorted` = number of monthly rebalances the ticker was in the SHORT quintile. `total_pnl_book_%` = cumulative contribution to book over all those months (positive = the strategy made money shorting this name). `avg_trade_%` = average per-position return (positive = the stock fell, the short made money). `best_%` / `worst_%` = best / worst single month per-position return. `hit_%` = share of monthly shorts that were profitable. `avg_SI_%` = average reported short interest at the time of shorting. `mktCap_$m` = average market cap when shorted (AUD m).

## Top 20 winners (most profitable shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | n_stops_triggered | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CXL | Calix Limited | 30 | 5.101 | 10.82 | 48.98 | -16.0 | 66.7 | 5 | 2.51 | 399.0 | 2023-06 | 2026-04 |
| CCX | City Chic Collective | 17 | 4.049 | 14.81 | 60.66 | -16.0 | 70.6 | 1 | 0.59 | 55.0 | 2024-01 | 2026-03 |
| ERA | Energy Resources 'A' | 7 | 3.74 | 33.06 | 84.38 | -16.0 | 85.7 | 1 | 0.0 | 810.0 | 2024-03 | 2024-09 |
| NMT | Neometals Ltd | 16 | 3.529 | 13.95 | 38.03 | -16.0 | 81.2 | 1 | 1.61 | 157.0 | 2023-06 | 2025-06 |
| LOT | Lotus Resources Ltd | 16 | 3.372 | 12.91 | 57.58 | -16.0 | 62.5 | 2 | 7.22 | 487.0 | 2024-07 | 2026-04 |
| CHN | Chalice Mining Ltd | 16 | 3.251 | 13.05 | 52.94 | -16.0 | 68.8 | 4 | 5.76 | 1316.0 | 2023-06 | 2025-06 |
| LKE | Lake Resources | 24 | 3.184 | 8.72 | 40.0 | -16.0 | 66.7 | 5 | 2.41 | 179.0 | 2023-06 | 2025-12 |
| BRN | Brainchip Ltd | 31 | 2.996 | 6.55 | 41.54 | -16.0 | 64.5 | 6 | 4.07 | 339.0 | 2023-06 | 2025-12 |
| SYR | Syrah Resources | 28 | 2.936 | 6.95 | 38.76 | -16.0 | 64.3 | 5 | 10.52 | 259.0 | 2023-06 | 2025-12 |
| SGR | The Star Ent Grp | 21 | 2.929 | 9.02 | 41.11 | -16.0 | 71.4 | 1 | 4.36 | 1111.0 | 2023-06 | 2026-03 |
| GLL | Galilee Energy Ltd | 10 | 2.788 | 17.48 | 44.68 | -5.26 | 70.0 | 0 | 0.09 | 13.0 | 2024-01 | 2025-02 |
| WBT | Weebit Nano Ltd | 18 | 2.715 | 9.81 | 36.67 | -16.0 | 72.2 | 2 | 5.05 | 738.0 | 2023-07 | 2026-03 |
| PEN | Peninsula Energy Ltd | 17 | 2.406 | 9.01 | 30.05 | -16.0 | 70.6 | 3 | 1.16 | 221.0 | 2024-01 | 2026-04 |
| APX | Appen Limited | 25 | 2.374 | 6.47 | 55.56 | -16.0 | 56.0 | 8 | 3.93 | 181.0 | 2023-06 | 2026-01 |
| MVP | Medical Developments | 21 | 2.196 | 6.98 | 34.78 | -16.0 | 61.9 | 2 | 0.32 | 58.0 | 2023-06 | 2025-09 |
| BAP | Bapcor Limited | 6 | 2.162 | 21.31 | 62.15 | -3.33 | 83.3 | 0 | 7.26 | 1034.0 | 2025-11 | 2026-04 |
| AGY | Argosy Minerals Ltd | 14 | 2.068 | 9.82 | 59.78 | -16.0 | 71.4 | 3 | 3.39 | 325.0 | 2023-07 | 2025-09 |
| AIS | Aeris Resources Ltd | 16 | 1.961 | 8.33 | 52.08 | -16.0 | 50.0 | 3 | 1.26 | 241.0 | 2023-06 | 2025-12 |
| DEV | Devex Resources Ltd | 21 | 1.956 | 6.35 | 28.21 | -16.0 | 61.9 | 3 | 0.22 | 88.0 | 2024-01 | 2026-04 |
| GLN | Galan Lithium Ltd | 13 | 1.928 | 9.54 | 46.38 | -16.0 | 84.6 | 1 | 1.1 | 171.0 | 2023-06 | 2025-04 |

## Top 20 losers (worst shorts in OOS)

| Ticker | Company | n_months_shorted | total_pnl_book_% | avg_trade_% | best_% | worst_% | hit_% | n_stops_triggered | avg_SI_% | mktCap_$m | first | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAT | Catapult Grp Int Ltd | 13 | -1.48 | -6.13 | 10.88 | -16.0 | 38.5 | 2 | 0.77 | 672.0 | 2023-06 | 2025-09 |
| TTT | Titomic Limited | 21 | -1.243 | -2.62 | 39.47 | -16.0 | 42.9 | 10 | 0.08 | 7378.0 | 2024-01 | 2026-03 |
| EVN | Evolution Mining Ltd | 15 | -1.151 | -3.95 | 18.18 | -16.0 | 40.0 | 1 | 1.09 | 7767.0 | 2023-06 | 2025-11 |
| MYS | Mystate Limited | 22 | -1.134 | -2.44 | 5.5 | -12.62 | 27.3 | 0 | 0.25 | 443.0 | 2023-06 | 2026-04 |
| PDN | Paladin Energy Ltd | 19 | -1.119 | -2.78 | 37.86 | -16.0 | 31.6 | 6 | 7.3 | 1954.0 | 2023-06 | 2025-09 |
| LTR | Liontown Resources | 19 | -1.097 | -2.55 | 29.62 | -16.0 | 31.6 | 5 | 7.37 | 2434.0 | 2024-02 | 2026-04 |
| EMR | Emerald Res Nl | 6 | -1.093 | -9.78 | 1.04 | -16.0 | 16.7 | 3 | 1.96 | 2601.0 | 2025-07 | 2025-12 |
| AEF | Australian Ethical | 8 | -1.007 | -6.83 | 6.54 | -16.0 | 37.5 | 3 | 0.37 | 436.0 | 2023-06 | 2024-04 |
| RSG | Resolute Mining | 5 | -0.989 | -10.75 | -0.0 | -16.0 | 0.0 | 2 | 0.41 | 1299.0 | 2025-08 | 2025-12 |
| WAF | West African Res Ltd | 9 | -0.954 | -5.47 | 17.67 | -16.0 | 11.1 | 3 | 1.11 | 2014.0 | 2023-10 | 2025-11 |
| ADH | Adairs Limited | 4 | -0.853 | -11.72 | 1.12 | -16.0 | 25.0 | 3 | 2.36 | 317.0 | 2023-11 | 2025-12 |
| MP1 | Megaport Limited | 5 | -0.846 | -9.29 | 17.54 | -16.0 | 20.0 | 4 | 7.33 | 1167.0 | 2023-07 | 2025-04 |
| MIN | Mineral Resources. | 24 | -0.832 | -1.19 | 39.74 | -16.0 | 33.3 | 5 | 9.49 | 8843.0 | 2024-01 | 2025-12 |
| NXL | Nuix Limited | 8 | -0.756 | -4.94 | 12.5 | -16.0 | 25.0 | 3 | 0.81 | 722.0 | 2023-06 | 2025-05 |
| ARU | Arafura Rare Earths | 18 | -0.727 | -1.7 | 15.38 | -16.0 | 44.4 | 3 | 4.23 | 356.0 | 2024-01 | 2025-12 |
| BMN | Bannerman Energy Ltd | 15 | -0.723 | -2.14 | 19.35 | -16.0 | 53.3 | 5 | 2.52 | 478.0 | 2023-06 | 2026-02 |
| BGA | Bega Cheese Ltd | 10 | -0.678 | -3.43 | 5.96 | -16.0 | 40.0 | 2 | 2.8 | 1260.0 | 2023-06 | 2026-04 |
| LYC | Lynas Rare Earths | 16 | -0.67 | -1.83 | 15.55 | -16.0 | 50.0 | 3 | 7.05 | 6852.0 | 2024-01 | 2025-12 |
| ILU | Iluka Resources | 30 | -0.651 | -0.49 | 12.79 | -16.0 | 50.0 | 4 | 4.57 | 2675.0 | 2023-09 | 2026-04 |
| SXL | Sthn Cross Media | 13 | -0.59 | -1.98 | 16.15 | -16.0 | 46.2 | 2 | 0.99 | 176.0 | 2023-09 | 2026-04 |

## Aggregate OOS stats (short leg only)

- **Total short-leg P&L**: +81.7 % of book (summed over 2,089 monthly positions)
- **Median per-position return**: +1.91 %
- **Win rate** (per-position): 53.5 %
- **Stop-fire rate**: 17.3 % of positions clipped at -16 %
- **Best single month**: `CXL` (+48.98 %)
- **Worst single month**: `CAT` (-16.00 %)