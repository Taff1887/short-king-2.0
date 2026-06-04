"""Inspect enterprise_values FMP endpoint - does it have PDN history?"""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

ev = read_parquet(settings.raw_dir / "fmp_raw" / "enterprise_values.parquet")
print(f"enterprise_values: {len(ev):,} rows, {ev.shape[1]} cols")
print(f"columns: {list(ev.columns)}")
print()
pdn = ev[ev["symbol"] == "PDN.AX"].sort_values("date")
print(f"PDN.AX rows: {len(pdn)}")
if len(pdn):
    print(f"date range: {pdn['date'].min()} -> {pdn['date'].max()}")
    if "marketCapitalization" in pdn.columns:
        print(f"mktCap range: {pdn['marketCapitalization'].min()/1e9:.2f}B -> {pdn['marketCapitalization'].max()/1e9:.2f}B")
    print()
    print("All PDN rows:")
    print(pdn[["date", "marketCapitalization"]].to_string(index=False))
