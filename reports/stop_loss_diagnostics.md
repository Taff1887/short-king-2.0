# Realistic stop-loss diagnostics -- OOS short book

**Rule**: 20 % adverse-move trigger; 10 % execution slippage on top; gap rule covers at `max(entry * 1.32, trigger_day_open)`. Applied to the 2,089-position OOS short book (logit L/S quintile, 2023-06 -> 2026-05) using daily auto-adjusted OHLC pulled from Yahoo.

## Summary diagnostics

- **Total OOS short positions**: 2,089
- **Positions stopped**: **138** (**6.6 %** of all short positions)
- **Average stopped-position loss**: -65.60 %
- **Median stopped-position loss**: -65.00 %
- **Worst stopped-position loss**: -112.03 % (uncapped raw would have been -314.04 %)

## Per-position return change

|  | Raw (no stop) | After 20 % stop | Δ |
|---|---:|---:|---:|
| Mean trade return | -0.56 % | -1.29 % | -0.73 pp |
| Median trade return | +1.91 % | +1.85 % | -0.06 pp |

## Portfolio-level impact

Three columns side-by-side: **Raw** (no stop), **Realistic** (20 % trigger + 10 % slippage + gap rule), **Simplified** (-32 % monthly cap, no gap handling -- isolates the cost of the gap rule from the cost of the stop itself).

### Short-leg-only book (notional sums to -1)

|  | Raw | Realistic (gap rule) | Simplified (-32 % cap) | Δ realistic | Δ simplified |
|---|---:|---:|---:|---:|---:|
| CAGR | -9.66 % | -16.69 % | +2.56 % | -7.03 pp | +12.22 pp |
| Ann. vol | 25.76 % | 23.60 % | 22.05 % | -2.16 pp | -3.71 pp |
| Sharpe | -0.263 | -0.653 | +0.221 | -0.390 | +0.484 |
| MaxDD | -53.45 % | -54.10 % | -40.32 % | -0.65 pp | +13.13 pp |

### L/S quintile (long leg unchanged, short leg stopped)

|  | Raw | Realistic (gap rule) | Simplified (-32 % cap) | Δ realistic | Δ simplified |
|---|---:|---:|---:|---:|---:|
| CAGR | -13.24 % | -20.80 % | -2.12 % | -7.56 pp | +11.12 pp |
| Ann. vol | 19.11 % | 20.60 % | 17.31 % | +1.49 pp | -1.81 pp |
| Sharpe | -0.645 | -1.018 | -0.040 | -0.373 | +0.605 |
| MaxDD | -48.50 % | -53.22 % | -32.47 % | -4.72 pp | +16.04 pp |

## Stop-status breakdown

- `NOT_TRIGGERED`: 1,951
- `STOPPED`: 138
