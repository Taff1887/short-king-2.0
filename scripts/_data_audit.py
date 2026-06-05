"""Data-quality audit + FMP-vs-Yahoo cross-check on the long-window panel.

Three sections:
1. Universe + coverage summary (rows, dates, tickers, by year).
2. Look-ahead audit, fundamental non-null %, extreme weekly returns.
3. FMP-vs-Yahoo monthly-return cross-check on a sample of 50 ASX tickers.

Writes reports/data_audit.md + reports/yahoo_crosscheck.csv for the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.data.clean import check_no_lookahead, detect_corrupted_series
from short_king.data.yahoo_xcheck import batch_crosscheck
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

OUT_MD = settings.reports_dir / "data_audit.md"
OUT_YAHOO = settings.reports_dir / "yahoo_crosscheck.csv"
SAMPLE_N = 50

# ---------- Load -----------------------------------------------------------
asic = read_parquet(settings.processed_dir / "asic_long.parquet")
clean = read_parquet(settings.processed_dir / "master_clean.parquet")
prices = read_parquet(settings.processed_dir / "prices_long.parquet")
features = read_parquet(settings.processed_dir / "features.parquet")
fmp_dir = settings.raw_dir / "fmp_raw"
fmp_endpoints = {p.stem: read_parquet(p) for p in sorted(fmp_dir.glob("*.parquet"))}

lines: list[str] = []
def w(s: str = "") -> None:
    lines.append(s)

w(f"# Data audit — short-king-2.0\n")
w(f"_Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")

# ---------- Section 1: universe ------------------------------------------------
w("## 1. Universe coverage")
w(f"- **asic_long**: {len(asic):,} rows | {asic['Ticker'].nunique()} unique tickers | "
  f"{asic['Date'].nunique()} dates | "
  f"window {pd.to_datetime(asic['Date']).min().date()} → {pd.to_datetime(asic['Date']).max().date()}")
w(f"- **prices_long**: {len(prices):,} rows | {prices['symbol'].nunique()} symbols | "
  f"{prices['date'].nunique()} unique daily dates")
w(f"- **master_clean**: {len(clean):,} rows | "
  f"{int(clean['investable'].sum()):,} investable ({100*clean['investable'].mean():.1f}%)")
w(f"- **features**: {len(features):,} rows × {features.shape[1]} cols")
w()

# Coverage by year - how many tickers are present each year?
asic = asic.copy()
asic["year"] = pd.to_datetime(asic["Date"]).dt.year
yr = asic.groupby("year").agg(
    rows=("Ticker", "size"),
    tickers=("Ticker", "nunique"),
    dates=("Date", "nunique"),
).reset_index()
w("### Coverage by calendar year")
w("| Year | Rows | Unique tickers | Weeks |")
w("|---|---:|---:|---:|")
for _, r in yr.iterrows():
    w(f"| {int(r['year'])} | {r['rows']:,} | {r['tickers']} | {r['dates']} |")
w()

# Top-20 tickers by frequency (sense check the universe)
top = asic.groupby("Ticker").size().sort_values(ascending=False).head(20)
w("### Top-20 tickers by ASIC report frequency")
w("| Rank | Ticker | Weeks present |")
w("|---:|---|---:|")
for i, (tkr, n) in enumerate(top.items(), 1):
    w(f"| {i} | {tkr} | {n} |")
w()

# ---------- Section 2: data quality checks ----------------------------------
w("## 2. Data quality checks\n")

# Look-ahead audit on the cleaned panel.
clean_lc = clean.rename(columns={"Date": "date", "Symbol": "symbol"})
lookahead = check_no_lookahead(clean_lc)
w(f"**Look-ahead audit**: {lookahead['n_violations']} violations across "
  f"{lookahead['n_rows']:,} rows "
  f"(max violation = {lookahead['max_violation_days']:.0f} days).")
w()

# Extreme weekly-return detection.
suspects = detect_corrupted_series(clean_lc, max_abs_weekly_return=1.5)
w(f"**Extreme weekly returns** (|ret_1w| > 150%): {len(suspects)} cells "
  f"({suspects['symbol'].nunique() if len(suspects) else 0} symbols).")
w()

# Fundamental non-null % by endpoint
w("**FMP fundamental column non-null %** (selected key fields):\n")
checks = [
    ("income_statement", ["netIncome", "revenue", "operatingIncome", "ebitda"]),
    ("balance_sheet", ["totalAssets", "totalDebt", "totalStockholdersEquity", "commonStockSharesOutstanding"]),
    ("cash_flow", ["operatingCashFlow", "freeCashFlow"]),
    ("ratios", ["priceEarningsRatio", "returnOnEquity", "returnOnInvestedCapital"]),
    ("key_metrics", ["roic", "fcfYield", "earningsYield"]),
    ("financial_growth", ["revenueGrowth", "epsgrowth"]),
]
w("| Endpoint | Field | Non-null % |")
w("|---|---|---:|")
for ep, fields in checks:
    df = fmp_endpoints.get(ep, pd.DataFrame())
    if df.empty:
        w(f"| {ep} | _missing_ | - |")
        continue
    for fld in fields:
        if fld in df.columns:
            pct = 100 * df[fld].notna().mean()
            w(f"| {ep} | {fld} | {pct:.1f} |")
        else:
            w(f"| {ep} | {fld} | _absent_ |")
w()

# Sector + industry coverage
sector_cov = 100 * clean["sector"].notna().mean() if "sector" in clean.columns else 0
industry_cov = 100 * clean["industry"].notna().mean() if "industry" in clean.columns else 0
w(f"**Sector / industry coverage** on the cleaned panel: "
  f"sector {sector_cov:.1f}%, industry {industry_cov:.1f}%. "
  f"(Profile pull is not yet wired into the pipeline — known limitation.)")
w()

# mktCap distribution
if "mktCap" in clean.columns:
    mc = pd.to_numeric(clean["mktCap"], errors="coerce").dropna()
    mc_qs = mc.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) / 1e6
    w("**Market cap distribution** (A$m) on the cleaned panel:\n")
    w(f"- p10: {mc_qs.loc[0.10]:,.0f}  p25: {mc_qs.loc[0.25]:,.0f}  "
      f"p50: {mc_qs.loc[0.50]:,.0f}  p75: {mc_qs.loc[0.75]:,.0f}  "
      f"p90: {mc_qs.loc[0.90]:,.0f}")
    w(f"- min: {mc.min()/1e6:,.0f}  max: {mc.max()/1e6:,.0f}")
    w()

# Forward-return outliers
for col in ("fwd_ret_1w", "fwd_ret_1m", "fwd_ret_3m"):
    if col in clean.columns:
        s = pd.to_numeric(clean[col], errors="coerce").dropna()
        outliers = ((s.abs() > 2.0).sum())   # > 200 %
        w(f"- **{col}**: n={len(s):,}, mean={s.mean():+.4f}, std={s.std():.4f}, "
          f"|ret|>200% outliers={outliers}, max={s.max():.2f}, min={s.min():.2f}")
w()

# ---------- Section 3: FMP vs Yahoo ------------------------------------------
w("## 3. FMP vs Yahoo Finance cross-check (monthly returns)\n")
w(f"Sample {SAMPLE_N} ASX tickers (random, seed 42). For each: fetch Yahoo "
  f"prices, compute month-end returns, compare to FMP's `adjClose` resampled "
  f"to month-end. Reports Spearman correlation + median absolute difference of "
  f"the month-end price levels.\n")

fmp_min = pd.to_datetime(prices["date"]).min()
fmp_max = pd.to_datetime(prices["date"]).max()
try:
    xc = batch_crosscheck(
        prices,
        max_symbols=SAMPLE_N,
        start=fmp_min.strftime("%Y-%m-%d"),
        end=fmp_max.strftime("%Y-%m-%d"),
    )
    xc.to_csv(OUT_YAHOO, index=False)
    if len(xc):
        ok = xc[xc["flag"] == "ok"]
        warn = xc[xc["flag"] == "warn"]
        mism = xc[xc["flag"] == "mismatch"]
        insuff = xc[xc["flag"] == "insufficient"]
        w(f"- **Coverage**: {len(xc)} symbols sampled, "
          f"{len(ok)} ok, {len(warn)} warn, {len(mism)} mismatch, {len(insuff)} insufficient.")
        if len(ok) + len(warn) + len(mism):
            corr_med = xc["corr_monthly"].dropna().median()
            corr_min = xc["corr_monthly"].dropna().quantile(0.05)
            diff_med = xc["median_abs_diff_pct"].dropna().median()
            w(f"- **Median Spearman correlation** (monthly returns): {corr_med:.4f}")
            w(f"- **5th-percentile correlation** (worst-fit): {corr_min:.4f}")
            w(f"- **Median absolute price-level diff** (month-end): {diff_med:.3f}%\n")
        # Worst 5 mismatches (most useful for the README)
        worst = xc.sort_values("corr_monthly", ascending=True, na_position="last").head(5)
        w("Five worst-fit symbols:\n")
        w("| Symbol | n_months | corr | median_abs_diff_% | flag |")
        w("|---|---:|---:|---:|---|")
        for _, r in worst.iterrows():
            corr = f"{r['corr_monthly']:.3f}" if pd.notna(r["corr_monthly"]) else "-"
            diff = f"{r['median_abs_diff_pct']:.2f}" if pd.notna(r["median_abs_diff_pct"]) else "-"
            n = int(r["n_months"]) if pd.notna(r["n_months"]) else 0
            w(f"| {r['symbol']} | {n} | {corr} | {diff} | {r['flag']} |")
        w()
    else:
        w("_batch_crosscheck returned no rows — Yahoo Finance may be unreachable._")
except Exception as exc:
    logger.warning(f"Yahoo crosscheck failed: {exc}")
    w(f"_Yahoo crosscheck failed: {exc}_")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
logger.info(f"wrote {OUT_MD}")
logger.info(f"wrote {OUT_YAHOO}")
