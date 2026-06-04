"""Trace the mktCap outliers the audit flagged."""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

clean = read_parquet(settings.processed_dir / "master_clean.parquet")
mc = pd.to_numeric(clean["mktCap"], errors="coerce")

print(f"rows: {len(clean):,}")
print(f"mktCap > A$500B (sus): {(mc > 5e11).sum():,} rows")
print(f"mktCap > A$1T (clearly wrong): {(mc > 1e12).sum():,} rows")
print(f"mktCap == 0: {(mc == 0).sum():,} rows\n")

# Top 10 worst rows.
worst = clean.loc[mc.sort_values(ascending=False).index[:10],
                  ["Date", "Ticker", "Company", "sharesOutstanding", "adjClose", "mktCap"]]
worst["mktCap_AUDb"] = worst["mktCap"] / 1e9
print("Top 10 by mktCap:")
print(worst.to_string(index=False))

# Distribution of mktCap by ticker for the top offender.
top_ticker = worst.iloc[0]["Ticker"]
print(f"\n{top_ticker} mktCap distribution across the panel:")
sub = clean[clean["Ticker"] == top_ticker][["Date", "sharesOutstanding", "adjClose", "mktCap"]]
sub["mktCap_AUDb"] = sub["mktCap"] / 1e9
print(sub.tail(10).to_string(index=False))
print(f"mktCap stats: min={sub['mktCap'].min()/1e9:.2f}B, "
      f"median={sub['mktCap'].median()/1e9:.2f}B, "
      f"max={sub['mktCap'].max()/1e9:.2f}B")
