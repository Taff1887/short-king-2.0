# Realistic stop-loss diagnostics -- OOS short book

**Rule**: 20 % adverse-move trigger; 10 % execution slippage on top; gap rule covers at `max(entry * 1.32, trigger_day_open)`. Applied to the 2,089-position OOS short book (logit L/S quintile, 2023-06 -> 2026-05) using daily auto-adjusted OHLC pulled from Yahoo.

## Summary diagnostics

- **Total OOS short positions**: 2,089
- **Positions stopped**: **523** (**25.0 %** of all short positions)
- **Average stopped-position loss**: -32.39 %
- **Median stopped-position loss**: -32.00 %
- **Worst stopped-position loss**: -100.00 % (uncapped raw would have been -314.04 %)

## Per-position return change

|  | Raw (no stop) | After 20 % stop | Δ |
|---|---:|---:|---:|
| Mean trade return | -0.56 % | -2.55 % | -1.99 pp |
| Median trade return | +1.91 % | +0.34 % | -1.57 pp |

## Portfolio-level impact

Three columns side-by-side: **Raw** (no stop), **Realistic** (20 % trigger + 10 % slippage + gap rule), **Simplified** (-32 % monthly cap, no gap handling -- isolates the cost of the gap rule from the cost of the stop itself).

### Short-leg-only book (notional sums to -1)

|  | Raw | Realistic (gap rule) | Simplified (-32 % cap) | Δ realistic | Δ simplified |
|---|---:|---:|---:|---:|---:|
| CAGR | -9.66 % | -28.79 % | +18.34 % | -19.13 pp | +28.01 pp |
| Ann. vol | 25.76 % | 24.31 % | 19.99 % | -1.45 pp | -5.76 pp |
| Sharpe | -0.263 | -1.257 | +0.943 | -0.995 | +1.206 |
| MaxDD | -53.45 % | -68.36 % | -29.38 % | -14.92 pp | +24.06 pp |

### L/S quintile (long leg unchanged, short leg stopped)

|  | Raw | Realistic (gap rule) | Simplified (-32 % cap) | Δ realistic | Δ simplified |
|---|---:|---:|---:|---:|---:|
| CAGR | -27.64 % | -43.87 % | -5.61 % | -16.23 pp | +22.04 pp |
| Ann. vol | 19.48 % | 22.26 % | 16.62 % | +2.78 pp | -2.86 pp |
| Sharpe | -1.542 | -2.418 | -0.266 | -0.876 | +1.276 |
| MaxDD | -62.48 % | -79.72 % | -35.04 % | -17.24 pp | +27.44 pp |

## Stop-status breakdown

- `NOT_TRIGGERED`: 1,566
- `STOPPED`: 523
