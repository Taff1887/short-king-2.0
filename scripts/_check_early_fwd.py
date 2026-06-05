"""Why is fwd_ret_1m all-NaN for 2010-2011 monthly dates?"""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

prices = read_parquet(settings.processed_dir / "prices_long.parquet")
clean = read_parquet(settings.processed_dir / "master_clean.parquet")

# TLS prices in 2010.
tls_px = prices[prices["symbol"] == "TLS.AX"].sort_values("date")
print(f"TLS.AX prices: {len(tls_px)} rows | range {tls_px['date'].min()} -> {tls_px['date'].max()}")
print(f"TLS first 5 prices:\n{tls_px.head(5).to_string(index=False)}")
print()

# Panel rows for TLS on 2010-06-28.
tls_panel = clean[(clean["Ticker"] == "TLS") & (clean["Date"] >= "2010-06-01") & (clean["Date"] <= "2010-10-01")]
print(f"TLS on 2010-06 to 2010-10 panel:\n{tls_panel[['Date', 'Symbol', 'adjClose', 'fwd_ret_1w', 'fwd_ret_1m', 'fwd_ret_3m']].head(10).to_string(index=False)}")
print()

# Aggregate stats on adjClose vs fwd_ret coverage in early 2010.
early = clean[clean["Date"] < "2011-01-01"]
print(f"Early (pre-2011): {len(early)} rows")
print(f"  adjClose non-null: {early['adjClose'].notna().sum():,}")
print(f"  fwd_ret_1w non-null: {early['fwd_ret_1w'].notna().sum():,}")
print(f"  fwd_ret_1m non-null: {early['fwd_ret_1m'].notna().sum():,}")
print(f"  fwd_ret_3m non-null: {early['fwd_ret_3m'].notna().sum():,}")
