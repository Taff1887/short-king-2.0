"""Rewrite lines 332-413 of README.md (the OOS trade-level section) into
a clean, accurate no-stop version. Run-once helper."""

from pathlib import Path

readme = Path("README.md")
lines = readme.read_text(encoding="utf-8").splitlines(keepends=True)

new_section = """## OOS trade-level analysis — the actual short book (no stop)

Reconstructed every short position from the OOS holdout (2023-06 → 2026-05,
**2,089 monthly positions across 224 unique tickers**, model = `logit`),
**no stop loss applied** — per-position monthly returns are uncapped.
Full per-position table:
[`reports/oos_short_positions.csv`](reports/oos_short_positions.csv);
per-ticker summary:
[`reports/oos_trades.csv`](reports/oos_trades.csv) /
[`reports/oos_trades.md`](reports/oos_trades.md).

**Aggregate OOS stats (short leg only, no stop):**

- **Total short-leg cumulative P&L**: **−41.8 % of book** across 2,089 monthly positions. The dollar-neutral L/S quintile is still positive (Sharpe 0.92) because the *long leg* carries it.
- **Per-position win-rate**: 53.4 % — most monthly shorts individually profitable.
- **Median per-position return**: **+1.91 %** (half the positions made ≥ +1.91 %).
- **Mean per-position return**: **−0.33 %** — dragged negative by the fat right-tail.
- **Best single month**: ERA +84 % (Energy Resources of Australia fell 84 % the month it was wound up).
- **Worst single month**: **4DX −314 %** (4D Medical rallied 314 % in Aug 2025 — a 2.5 %-book-weight short took ~8 % of NAV in one stroke).
- **42 single positions lost > 50 %**, **161 lost > 25 %** — uncapped squeezes.

### Top 10 winning shorts (no stop)

`avg_trade_%` is the mean per-position monthly return (positive = stock fell, short won).
`worst_%` is **uncapped** — many of these winning shorts took brutal interim squeezes en route to their eventual cumulative profit.

| # | Ticker | Company | n months shorted | total P&L (% book) | avg trade | best month | worst month | hit-rate | avg SI % | first → last |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | CCX | City Chic Collective | 17 | **+3.82 %** | +14.0 % | +60.7 % | −30.4 % | 71 % | 0.6 | 2024-01 → 2026-03 |
| 2 | LOT | Lotus Resources | 16 | +3.19 % | +12.2 % | +57.6 % | −29.0 % | 63 % | 7.2 | 2024-07 → 2026-04 |
| 3 | NMT | Neometals | 16 | +2.98 % | +11.8 % | +38.0 % | −50.0 % | 81 % | 1.6 | 2023-06 → 2025-06 |
| 4 | CXL | Calix | 30 | +2.93 % | +6.5 % | +49.0 % | **−70.8 %** | 67 % | 2.5 | 2023-06 → 2026-04 |
| 5 | GLL | Galilee Energy | 10 | +2.79 % | +17.5 % | +44.7 % | −5.3 % | 70 % | 0.1 | 2024-01 → 2025-02 |
| 6 | SGR | Star Entertainment | 21 | +2.75 % | +8.5 % | +41.1 % | −26.9 % | 71 % | 4.4 | 2023-06 → 2026-03 |
| 7 | WBT | Weebit Nano | 18 | +2.38 % | +8.6 % | +36.7 % | −33.7 % | 72 % | 5.1 | 2023-07 → 2026-03 |
| 8 | ERA | Energy Resources of Australia | 7 | +2.37 % | +21.1 % | +84.4 % | **−100.0 %** | 86 % | 0.0 | 2024-03 → 2024-09 |
| 9 | CHN | Chalice Mining | 16 | +2.20 % | +8.9 % | +52.9 % | −49.8 % | 69 % | 5.8 | 2023-06 → 2025-06 |
| 10 | BAP | Bapcor | 6 | +2.16 % | +21.3 % | +62.2 % | −3.3 % | 83 % | 7.3 | 2025-11 → 2026-04 |

**Even winners had brutal months without the stop:** Calix −71 % in one month, Energy Resources −100 % the month operations terminated, Neometals −50 %. The 15 % stop would have closed those positions before the eventual cumulative profit was realised — a real trade-off, not a free lunch.

### Top 10 losing shorts (no stop) — the squeezes

| # | Ticker | Company | n months shorted | total P&L (% book) | avg trade | best month | worst month | hit-rate | avg SI % | first → last |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SRL | Sunrise Resources | 30 | **−9.43 %** | −17.9 % | +37.7 % | **−231.2 %** | 53 % | 1.3 | 2023-06 → 2025-12 |
| 2 | 4DX | 4D Medical | 20 | **−9.39 %** | −26.7 % | +35.1 % | **−314.0 %** | 50 % | 0.0 | 2023-11 → 2025-12 |
| 3 | TTT | Titomic | 21 | −5.43 % | −14.7 % | +39.5 % | −124.1 % | 43 % | 0.1 | 2024-01 → 2026-03 |
| 4 | ASM | Australian Strategic Materials | 28 | −5.11 % | −10.2 % | +36.1 % | −163.9 % | 50 % | 1.7 | 2023-06 → 2025-12 |
| 5 | APX | Appen | 25 | −3.40 % | −7.7 % | +55.6 % | −127.3 % | 56 % | 3.9 | 2023-06 → 2026-01 |
| 6 | SPL | Starpharma Holdings | 19 | −3.39 % | −9.7 % | +58.5 % | −172.0 % | 47 % | 0.4 | 2023-06 → 2025-12 |
| 7 | EOS | Electro Optic Systems | 19 | −3.31 % | −9.6 % | +28.2 % | −118.7 % | 37 % | 1.0 | 2024-07 → 2026-04 |
| 8 | EGR | EcoGraf | 13 | −2.80 % | −12.4 % | +28.1 % | −139.1 % | 46 % | 0.3 | 2024-01 → 2025-12 |
| 9 | IXR | Ionic Rare Earths | 26 | −2.78 % | −5.8 % | +40.0 % | −125.0 % | 50 % | 0.1 | 2023-06 → 2026-03 |
| 10 | CAT | Catapult Sports | 13 | −2.25 % | −9.8 % | +10.9 % | −44.3 % | 39 % | 0.8 | 2023-06 → 2025-09 |

**Without the stop the worst names cost 6-9 × what they did with the stop on.** SRL and 4DX alone cost −18.8 % of book. The previous (with-stop) version had no single name worse than −1.5 %. Negative `worst_%` values *exceed* −100 % because a 2.5 %-book-weight short of a stock that rallies 300 % loses 300 % of *position notional*. These are exactly the squeezes the user's "shouldn't be shorting it 23 times!" intuition was pointing at — without the stop, the strategy actually does take the full pain of multi-bagger squeezes.

"""

# Replace lines 332..413 (1-indexed) with the new section.
# Python list is 0-indexed, so slice [331:413].
result = lines[:331] + [new_section] + lines[413:]
readme.write_text("".join(result), encoding="utf-8")
print(f"rewrote OOS-trades section ({len(lines)} -> {len(result)} lines)")
