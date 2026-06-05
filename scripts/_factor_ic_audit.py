"""Factor IC audit -- which individual ranks actually predict forward returns?

For every ``*_rk`` column on the feature panel, compute the per-month
cross-sectional Spearman IC vs the next monthly forward return. Aggregate
to mean IC, IC t-stat, IC hit-rate. **IS-only** to avoid look-ahead --
this is feature selection, and using OOS rows would bias the EW spec
toward whatever happens to work post-2023.

Output: reports/factor_ic_audit.csv with one row per (raw_feature,
mean_ic, t_stat, hit_rate, n_periods, sign_match, decision).

Decision rule (kept conservative):
  * |t_stat| >= 2.0 -> statistically significant signal
  * Within sig group, prefer factors with intuitive economic story
    (we'll filter manually before re-specifying EW)

The output table is what drives the new EW spec in 05_train_and_validate.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

HOLDOUT_START = pd.Timestamp("2023-06-01")


def _per_month_ic(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.Series:
    """Spearman cross-sectional IC per month."""
    return (df.dropna(subset=[score_col, ret_col])
              .groupby("Date")
              .apply(lambda g: g[score_col].corr(g[ret_col], method="spearman"),
                     include_groups=False)
              .dropna())


def main() -> int:
    settings.ensure_dirs()
    feat = read_parquet(settings.processed_dir / "features_monthly.parquet")
    if "fwd_ret_1m" not in feat.columns:
        logger.error("features_monthly.parquet missing fwd_ret_1m")
        return 1

    # IS only -- exclude holdout window so any EW spec choice we make based
    # on this table is itself OOS-validated downstream.
    is_df = feat[feat["Date"] < HOLDOUT_START].copy()
    logger.info(f"IS rows: {len(is_df):,} across {is_df['Date'].nunique()} months")

    rk_cols = [c for c in is_df.columns if c.endswith("_rk")]
    logger.info(f"auditing {len(rk_cols)} *_rk columns")

    rows: list[dict] = []
    for c in rk_cols:
        ic = _per_month_ic(is_df, c, "fwd_ret_1m")
        if len(ic) < 12:
            continue
        mean_ic = float(ic.mean())
        std_ic = float(ic.std(ddof=1))
        t_stat = float(mean_ic / std_ic * np.sqrt(len(ic))) if std_ic > 0 else np.nan
        hit_rate = float((ic < 0).mean())  # share of months where IC was negative (= short signal)
        rows.append({
            "feature": c,
            "n_months": int(len(ic)),
            "mean_ic": round(mean_ic, 5),
            "t_stat": round(t_stat, 3) if pd.notna(t_stat) else np.nan,
            "ic_negative_share": round(hit_rate, 3),
            "is_significant": abs(t_stat) >= 2.0 if pd.notna(t_stat) else False,
            "is_short_aligned": mean_ic < 0,  # negative IC = good for shorts
        })

    df = pd.DataFrame(rows).sort_values("t_stat")
    csv_path = settings.reports_dir / "factor_ic_audit.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Print the cleanest short-aligned signals (negative IC, significant).
    sig_short = df[(df.is_significant) & (df.is_short_aligned)].sort_values("t_stat")
    print()
    print("=" * 90)
    print(f"TOP-30 SHORT-ALIGNED FACTORS (negative IC, |t| >= 2.0) on IS data only")
    print("=" * 90)
    print(sig_short.head(30).to_string(index=False))

    sig_long = df[(df.is_significant) & (~df.is_short_aligned)].sort_values("t_stat", ascending=False)
    print()
    print("=" * 90)
    print(f"TOP-20 LONG-ALIGNED FACTORS (positive IC, |t| >= 2.0) -- if used in EW, INVERT")
    print("=" * 90)
    print(sig_long.head(20).to_string(index=False))

    # Audit the CURRENT EW spec specifically.
    EW_CURRENT = [
        ("short_pct_ff_rk",        +1),
        ("ShortPct_rk",            +1),
        ("si_z_12m_rk",            +1),
        ("mom_3m_rk",              -1),
        ("vol_1m_rk",              +1),
        ("log_mktcap_rk",          -1),
        ("pe_rk",                  +1),
        ("fcf_yield_rk",           -1),
        ("roe_rk",                 -1),
        ("roic_rk",                -1),
        ("debt_equity_rk",         +1),
        ("revenue_growth_yoy_rk",  -1),
    ]
    print()
    print("=" * 90)
    print("AUDIT OF CURRENT EW SPEC (the 12 factors EW is built from)")
    print("=" * 90)
    print(f"{'feature':<28} {'EW pol':>6} {'mean_ic':>9} {'t_stat':>8} "
          f"{'sig?':>6} {'sign_match?':>11} {'verdict':>15}")
    print('-' * 90)
    for col, pol in EW_CURRENT:
        row = df[df.feature == col]
        if row.empty:
            print(f"{col:<28}  {pol:>+5d}  {'MISSING':>9}")
            continue
        r = row.iloc[0]
        ic = r['mean_ic']; t = r['t_stat']
        sig = abs(t) >= 2.0
        # Sign match: if EW polarity is +1, we expect mean_ic NEGATIVE
        # (positive raw factor -> shortable -> negative future return).
        # If EW polarity is -1 (we invert before averaging), then the
        # FLIPPED rank should have negative IC, so the RAW rank should
        # have POSITIVE IC.
        expected_ic_sign = -pol  # raw rank's expected IC sign
        sign_match = (np.sign(ic) == np.sign(expected_ic_sign)) or abs(t) < 1
        verdict = "KEEP" if sig and sign_match else ("WEAK" if sign_match else "WRONG SIGN")
        print(f"{col:<28}  {pol:>+5d}  {ic:>+9.4f}  {t:>+7.2f}  "
              f"{'YES' if sig else 'no':>6}  {'YES' if sign_match else 'NO':>11}  {verdict:>15}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
