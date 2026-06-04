"""Inspect FMP's historical-market-cap data for PDN.AX."""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

mc = read_parquet(settings.processed_dir / "marketcap_long.parquet")
pdn = mc[mc["symbol"] == "PDN.AX"].sort_values("date")
print(f"PDN.AX rows: {len(pdn)}")
print(f"date range: {pdn['date'].min()} -> {pdn['date'].max()}")
print(f"mktCap range: {pdn['marketCap'].min()/1e9:.2f}B -> {pdn['marketCap'].max()/1e9:.2f}B")
print()
print("First 10:")
print(pdn.head(10).to_string(index=False))
print()
print("Last 10:")
print(pdn.tail(10).to_string(index=False))
print()
# Check the panel
panel = read_parquet(settings.processed_dir / "master_clean.parquet")
pdn_p = panel[panel["Ticker"] == "PDN"].sort_values("Date")[["Date", "Symbol", "adjClose", "sharesOutstanding", "mktCap"]]
print(f"\nPanel PDN: {len(pdn_p)} rows")
print(f"mktCap range: {pdn_p['mktCap'].min()/1e9:.2f}B -> {pdn_p['mktCap'].max()/1e9:.2f}B")
print()
print("Last 10 (current PDN should be ~3-5B):")
print(pdn_p.tail(10).to_string(index=False))
print()
print("Around the consolidation (Aug-Dec 2023; should NOT be trillions):")
print(pdn_p[(pdn_p["Date"] >= "2023-08-01") & (pdn_p["Date"] <= "2023-12-31")].head(10).to_string(index=False))
