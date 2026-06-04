"""Probe how far back FMP and Yahoo go for ASX prices."""

import pandas as pd
import yfinance as yf

from short_king.data.fmp_client import FMPClient
from short_king.utils.logging import logger

client = FMPClient()

# FMP dividend-adjusted (what we use today).
rows = client.historical_price_eod_adjusted("TLS.AX")
fmp_adj = pd.DataFrame(rows)
fmp_adj["date"] = pd.to_datetime(fmp_adj["date"])
print(f"FMP adjusted TLS.AX: {len(fmp_adj)} rows | "
      f"range {fmp_adj['date'].min()} -> {fmp_adj['date'].max()}")

# FMP raw (split-adjusted only, unadjusted for divs).
rows = client.historical_price_eod_full("TLS.AX")
fmp_full = pd.DataFrame(rows) if rows else pd.DataFrame()
if not fmp_full.empty and "date" in fmp_full.columns:
    fmp_full["date"] = pd.to_datetime(fmp_full["date"])
    print(f"FMP full TLS.AX: {len(fmp_full)} rows | "
          f"range {fmp_full['date'].min()} -> {fmp_full['date'].max()}")
else:
    print(f"FMP full TLS.AX: empty payload")

# Yahoo
yh = yf.download("TLS.AX", start="2000-01-01", progress=False, auto_adjust=False, threads=False)
print(f"Yahoo TLS.AX: {len(yh)} rows | range {yh.index.min()} -> {yh.index.max()}")
