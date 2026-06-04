"""Compare the 5-yr vs 16-yr top-500 ASIC universe."""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

asic = read_parquet(settings.processed_dir / "asic_long.parquet")
print(f"asic_long: {len(asic):,} rows | {asic['Ticker'].nunique()} unique tickers | "
      f"{asic['Date'].nunique()} dates")

freq = asic.groupby("Ticker").size().sort_values(ascending=False)
top500 = freq.head(500).index.tolist()
print(f"top-500 cutoff: tickers ranked #500 has {freq.iloc[499]} weekly reports; "
      f"median weeks/ticker in top-500 = {int(freq.head(500).median())}")

old_path = settings.processed_dir / "universe_tickers.txt"
old = set(old_path.read_text(encoding="utf-8").split()) if old_path.exists() else set()
new = set(top500)
print(f"old (5-yr) top-500: {len(old)}")
print(f"new (16-yr) top-500: {len(new)}")
print(f"overlap: {len(old & new)} | new-only: {len(new - old)} | dropped: {len(old - new)}")
