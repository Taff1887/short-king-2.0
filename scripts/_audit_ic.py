"""Audit the EW-composite IC headline — is it real or a leak/bug?

Hypotheses:
  H1: One of the rank columns leaks forward-return info.
  H2: The "rerank within date" step has a bug.
  H3: The IC is genuine, dominated by one or two factors with strong sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
print(f"Panel: {len(df):,} rows, {df.shape[1]} cols, dates={df['Date'].nunique()}")
fwd = df["fwd_ret_1m"]
print(f"fwd_ret_1m: mean={fwd.mean():+.4f}, std={fwd.std():.4f}, "
      f"non-null={fwd.notna().sum():,}")

ew_cols = [
    "short_pct_ff_rk", "ShortPct_rk", "si_z_12m_rk",
    "mom_3m_rk", "vol_1m_rk",
    "log_mktcap_rk",
    "pe_rk", "fcf_yield_rk",
    "roe_rk", "roic_rk",
    "debt_equity_rk", "revenue_growth_yoy_rk",
]
ew_cols = [c for c in ew_cols if c in df.columns]
print(f"\nEW composite uses {len(ew_cols)} columns: {ew_cols}")


def per_date_ic(score: pd.Series, label: pd.Series, dates: pd.Series) -> pd.Series:
    work = pd.DataFrame({"s": score.values, "y": label.values, "d": dates.values})
    work = work.dropna()
    if work.empty:
        return pd.Series(dtype=float)
    rows: list[float] = []
    for _, g in work.groupby("d"):
        if len(g) < 5 or g["s"].std() == 0 or g["y"].std() == 0:
            continue
        rows.append(g[["s", "y"]].corr(method="spearman").iloc[0, 1])
    return pd.Series(rows, dtype=float)


def summarise(name: str, ic: pd.Series) -> None:
    n = len(ic.dropna())
    if n == 0:
        print(f"  {name:32s} no IC observations")
        return
    mean = ic.mean()
    std = ic.std(ddof=1) if n > 1 else float("nan")
    t = mean / std * np.sqrt(n) if std and std > 0 else float("nan")
    hit = (ic > 0).mean()
    print(f"  {name:32s} n={n:4d}  IC={mean:+.4f}  std={std:.4f}  t={t:+.2f}  hit={hit:.2%}")


print("\n--- Per-factor IC of each EW input vs fwd_ret_1m ---")
for c in ew_cols:
    summarise(c, per_date_ic(df[c], fwd, df["Date"]))


print("\n--- Composite (mean of 12 raw rank cols, BEFORE rerank) ---")
raw_score = df[ew_cols].mean(axis=1, skipna=True)
summarise("composite_raw", per_date_ic(raw_score, fwd, df["Date"]))

print("\n--- Composite (rerank-within-date, the actual script path) ---")
work = pd.DataFrame({"d": df["Date"].values, "s": raw_score.values})
work["score"] = work.groupby("d")["s"].rank(pct=True)
summarise("composite_reranked", per_date_ic(work["score"], fwd, df["Date"]))

print("\n--- Sanity: is there any pool-wide leak? Pool Spearman of each EW input vs fwd_ret_1m ---")
for c in ew_cols:
    sub = df[[c, "fwd_ret_1m"]].dropna()
    if len(sub) < 2:
        continue
    pool = sub.corr(method="spearman").iloc[0, 1]
    print(f"  {c:32s} pool spearman = {pool:+.4f}")

print("\n--- IC count by date: how many dates have an IC (i.e. >=5 names with both score & fwd)? ---")
work = pd.DataFrame({"d": df["Date"], "y": fwd}).dropna()
by_date = work.groupby("d").size()
print(f"  n_dates with >=5 names: {(by_date >= 5).sum()}  (total dates: {df['Date'].nunique()})")
print(f"  early dates (first 5): {by_date.head()}")
print(f"  late dates  (last  5): {by_date.tail()}")
