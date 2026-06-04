"""Trace walk_forward behaviour on the monthly IS panel."""

import pandas as pd
from short_king.models.walk_forward import walk_forward_splits
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

df = read_parquet(settings.processed_dir / "features_monthly.parquet")
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
print(f"Total monthly rows: {len(df):,}, dates: {df['Date'].nunique()}, "
      f"range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

# IS subset = first 156 dates (holdout = last 36).
dates_unique = sorted(df["Date"].unique())
holdout_start = dates_unique[-36]
is_mask = df["Date"] < holdout_start
df_is = df.loc[is_mask].reset_index(drop=True)
print(f"\nIS panel: {len(df_is):,} rows, {df_is['Date'].nunique()} dates, "
      f"range: {df_is['Date'].min().date()} -> {df_is['Date'].max().date()}")

# Walk-forward with the monthly defaults (in calendar weeks).
splits = walk_forward_splits(
    df_is["Date"],
    min_train_weeks=156,
    test_weeks=26,
    embargo_weeks=4,
)
print(f"\nWalk-forward splits: {len(splits)} folds")
all_test_dates = set()
for i, sp in enumerate(splits):
    test_dates = pd.to_datetime(df_is.iloc[sp.test_idx]["Date"]).dt.date.unique()
    all_test_dates.update(test_dates)
    print(f"  fold {i}: train [{sp.train_start.date()}..{sp.train_end.date()}] "
          f"test [{sp.test_start.date()}..{sp.test_end.date()}] "
          f"train_rows={len(sp.train_idx):,} test_rows={len(sp.test_idx):,} "
          f"test_dates={len(test_dates)}")
print(f"\nTotal unique test dates across all folds: {len(all_test_dates)}")
