"""Trace fwd_ret_4w coverage by date in the monthly IS panel."""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

df = read_parquet(settings.processed_dir / "features_monthly.parquet")
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

cov = (
    df.groupby("Date")
      .agg(n_rows=("Ticker", "size"),
           n_fwd=("fwd_ret_4w", lambda s: s.notna().sum()))
      .reset_index()
)
cov["pct"] = 100 * cov["n_fwd"] / cov["n_rows"]
print(f"Total monthly dates: {len(cov)}")
print(f"Dates with 0 fwd_ret_4w: {(cov['n_fwd'] == 0).sum()}")
print(f"Dates with < 50 fwd_ret_4w: {(cov['n_fwd'] < 50).sum()}")
print()
print("First 20 dates (head):")
print(cov.head(20).to_string(index=False))
print()
print("Tail 10:")
print(cov.tail(10).to_string(index=False))
