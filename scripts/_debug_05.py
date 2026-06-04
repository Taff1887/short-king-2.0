"""Run a minimal version of 05's CV pipeline and trace where scores get lost."""

import numpy as np
import pandas as pd

from short_king.models.baselines import fit_logit_baseline, predict_logit_baseline
from short_king.models.walk_forward import walk_forward_splits
from short_king.utils.config import settings
from short_king.utils.io import read_parquet

df = read_parquet(settings.processed_dir / "features_monthly.parquet")
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# IS subset (same logic as 05).
dates_unique = sorted(df["Date"].unique())
holdout_start = dates_unique[-36]
is_mask = df["Date"] < holdout_start
df_is = df.loc[is_mask].reset_index(drop=True)
print(f"df_is rows: {len(df_is):,}, dates: {df_is['Date'].nunique()}")

# Build feature cols + label.
feat_cols = [c for c in df_is.columns if c.endswith("_rk") or c.startswith("sec_")]
print(f"feature cols: {len(feat_cols)}")
fwd_col = "fwd_ret_4w"
y_bin = (df_is[fwd_col] < 0).astype("float64").where(df_is[fwd_col].notna())

# Walk-forward.
splits = walk_forward_splits(
    df_is["Date"],
    min_train_weeks=156, test_weeks=26, embargo_weeks=4,
)
print(f"splits: {len(splits)}")

out = pd.Series(np.nan, index=df_is.index, name="score", dtype="float64")
skipped = 0
for i, sp in enumerate(splits):
    X_tr = df_is.iloc[sp.train_idx][feat_cols].astype("float64", copy=False)
    X_te = df_is.iloc[sp.test_idx][feat_cols].astype("float64", copy=False)
    y_tr = y_bin.iloc[sp.train_idx]
    tr_mask = y_tr.notna() & np.isfinite(X_tr.to_numpy()).all(axis=1)
    if int(tr_mask.sum()) == 0:
        print(f"  fold {i}: NO usable training rows ({len(sp.train_idx)} input, all NaN after masking)")
        skipped += 1
        continue
    try:
        model, used_cols = fit_logit_baseline(
            X_tr.loc[tr_mask], y_tr.loc[tr_mask].astype(int),
            feature_cols=feat_cols,
        )
        scores = predict_logit_baseline(model, X_te, used_cols).to_numpy()
    except Exception as exc:
        print(f"  fold {i}: FAILED -- {exc!r}")
        skipped += 1
        continue
    out.iloc[sp.test_idx] = scores
    n_finite = int(np.isfinite(scores).sum())
    print(f"  fold {i}: wrote {n_finite}/{len(scores)} scores  "
          f"(train kept {int(tr_mask.sum())}/{len(sp.train_idx)})")
print(f"\nTotal non-null IS scores: {int(out.notna().sum()):,} / {len(out):,}  "
      f"(skipped {skipped} folds)")
print(f"Unique IS dates with scores: {df_is.loc[out.notna(), 'Date'].nunique()}")
