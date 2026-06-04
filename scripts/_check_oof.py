"""Inspect the monthly OOF panel to understand IS / OOS coverage."""

import pandas as pd
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
print(f"OOF rows: {len(oof):,}")
print(f"models: {oof['model'].unique().tolist()}")
print(f"period values: {oof['period'].value_counts().to_dict()}")
print(f"date range: {oof['Date'].min()} -> {oof['Date'].max()}")
print()
print("Score coverage by model x period (non-null counts):")
for model in oof["model"].unique():
    sub = oof[oof["model"] == model]
    for period in ("IS", "OOS"):
        ss = sub[sub["period"] == period]
        n_total = len(ss)
        n_scored = ss["score"].notna().sum()
        n_dates_scored = ss[ss["score"].notna()]["Date"].nunique()
        print(f"  {model:10s} {period:3s}  rows={n_total:>6,}  scored={n_scored:>6,}  unique_dates_scored={n_dates_scored}")
