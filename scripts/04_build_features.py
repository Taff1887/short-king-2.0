"""Assemble the PIT panel, clean it, then build the feature matrix.

Three-stage pipeline:
1. ``assemble_pit_panel`` joins the ASIC weekly short panel with FMP
   fundamentals (lagged to their ``filingDate``) and adjusted prices on a
   common Friday grid -> ``data/processed/master_pit.parquet``.
2. ``conservative_clean`` nulls out demonstrably bad (symbol, week) cells
   (e.g. unadjusted splits with abs-weekly-return > 150 %) and refreshes
   ``investable`` -> ``data/processed/master_clean.parquet``.
3. ``build_feature_panel`` stitches the per-family feature modules + adds
   cross-sectional ranks and sector dummies -> ``data/processed/features.parquet``.

Quality summaries are logged at each stage; lookahead violations abort.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from short_king.data.assemble import (
    assemble_pit_panel,
    panel_quality_summary,
    write_panel,
)
from short_king.data.clean import (
    check_no_lookahead,
    conservative_clean,
    detect_corrupted_series,
    quality_report,
    write_clean,
)
from short_king.features.build import build_feature_panel, write_feature_panel
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


# Map between PascalCase keys emitted by assemble and the lowercase keys
# the clean module operates on. We bridge at the script layer so the two
# modules stay simple and locally-consistent.
_TO_CLEAN = {"Date": "date", "Symbol": "symbol"}
_FROM_CLEAN = {v: k for k, v in _TO_CLEAN.items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-sector-dummies", action="store_true",
                   help="Skip one-hot sector dummies (saves columns for linear baselines).")
    p.add_argument("--no-cross-sectional", action="store_true",
                   help="Skip within-date percentile ranking (debugging only).")
    p.add_argument("--allow-lookahead", action="store_true",
                   help="Do not abort on lookahead-check failures (debugging only).")
    p.add_argument("--corrupt-threshold", type=float, default=1.5,
                   help="Abs weekly-return threshold above which a (symbol, week) "
                        "cell is treated as corrupted (default 1.5 = 150 %).")
    return p.parse_args()


def _require(path) -> bool:
    if not path.exists():
        logger.error(f"{path} not found - required input is missing.")
        return False
    return True


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    asic_path = settings.processed_dir / "asic_long.parquet"
    prices_path = settings.processed_dir / "prices_long.parquet"
    fmp_raw_dir = settings.raw_dir / "fmp_raw"
    if not _require(asic_path):
        logger.error("must run 01_pull_asic.py first")
        return 1
    if not _require(prices_path):
        logger.error("must run 03_pull_fmp_prices.py first")
        return 1
    if not fmp_raw_dir.exists() or not any(fmp_raw_dir.glob("*.parquet")):
        logger.error(f"{fmp_raw_dir} empty - must run 02_pull_fmp_fundamentals.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"04_build_features: start {t0.isoformat(timespec='seconds')}")

    asic = read_parquet(asic_path)
    prices = read_parquet(prices_path)
    # FMP historical-market-cap (split-adjusted at the share-count level) is
    # the correct mktCap source; pulled by 03_pull_fmp_prices.py.
    mcap_path = settings.processed_dir / "marketcap_long.parquet"
    market_cap_long = read_parquet(mcap_path) if mcap_path.exists() else None
    if market_cap_long is None:
        logger.warning(f"{mcap_path} not found - assemble will use shares*price fallback")
    fmp_tables = {p.stem: read_parquet(p) for p in sorted(fmp_raw_dir.glob("*.parquet"))}
    # Make sure all 7 canonical endpoints are keyed even if any were empty / missing.
    from short_king.data.fundamentals import STATEMENT_ENDPOINTS
    import pandas as _pd
    for ep in STATEMENT_ENDPOINTS:
        fmp_tables.setdefault(ep, _pd.DataFrame())
    logger.info(
        f"inputs: asic_long={len(asic):,} rows | prices_long={len(prices):,} rows "
        f"| fmp_raw endpoints={list(fmp_tables)}"
    )

    # 1) Assemble the PIT panel.
    panel = assemble_pit_panel(
        asic_long=asic,
        fundamentals=fmp_tables,
        prices_long=prices,
        market_cap_long=market_cap_long,
    )
    write_panel(panel, settings.processed_dir / "master_pit.parquet")
    logger.info(f"panel_quality_summary:\n{panel_quality_summary(panel)}")

    # 2) Conservative clean. The clean module operates on lowercase keys
    #    (date, symbol, adjClose) - bridge in/out at the script layer.
    panel_lc = panel.rename(columns=_TO_CLEAN)
    suspects = detect_corrupted_series(panel_lc, max_abs_weekly_return=args.corrupt_threshold)
    clean_lc = conservative_clean(panel_lc, suspects)

    # Look-ahead audit on the lowercase frame (clean filled filing_lag_days).
    lookahead = check_no_lookahead(clean_lc)
    if lookahead["n_violations"] > 0 and not args.allow_lookahead:
        logger.error(f"check_no_lookahead FAILED: {lookahead}")
        return 2
    logger.info(f"check_no_lookahead passed: {lookahead}")

    # Side-by-side coverage report (raw vs clean).
    logger.info(f"quality_report:\n{quality_report(panel_lc, clean_lc)}")

    # Bridge back to PascalCase for downstream features/models/backtest.
    clean = clean_lc.rename(columns=_FROM_CLEAN)
    write_clean(clean, settings.processed_dir / "master_clean.parquet")

    # 3) Feature matrix (computed weekly so rolling windows like vol_4w stay
    # meaningful even when we later downsample to monthly for modelling).
    features = build_feature_panel(
        clean,
        cross_sectional=not args.no_cross_sectional,
        add_sector_dummies=not args.no_sector_dummies,
    )
    out_features = write_feature_panel(features, settings.processed_dir / "features.parquet")

    # 3b) End-of-month snapshot - the LAST ASIC report in each calendar month.
    # Because ASIC only publishes on trading days, every row's Date is by
    # construction a valid ASX trading day. Most are Fridays; in months where
    # Good Friday or Christmas falls on the last Friday, ASIC publishes the
    # preceding Thursday (or following Monday) so the panel can include those
    # day-of-week values - we audit the distribution below and log a warning
    # if any month-end falls on a weekend (which would indicate a data bug).
    monthly = features.copy()
    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly["_ym"] = monthly["Date"].dt.to_period("M")
    eom_per_month = monthly.groupby("_ym")["Date"].transform("max")
    monthly = monthly[monthly["Date"] == eom_per_month].drop(columns="_ym")
    monthly = monthly.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Trading-day audit: every EOM rebalance date must be Mon-Fri.
    eom_dates = monthly["Date"].drop_duplicates().sort_values()
    dow_dist = eom_dates.dt.day_name().value_counts().to_dict()
    weekend = eom_dates[eom_dates.dt.dayofweek > 4]
    logger.info(f"EOM rebalance day-of-week distribution: {dow_dist}")
    if len(weekend):
        logger.error(f"EOM trading-day check FAILED: {len(weekend)} weekend dates found - data bug")
    # How far from calendar month-end is each EOM Friday on average? A typical
    # last-Friday-of-month is 0-6 days before the real month-end.
    cal_eom = eom_dates.dt.to_period("M").dt.end_time.dt.normalize()
    gap_days = (cal_eom - eom_dates).dt.days
    logger.info(
        f"EOM gap-to-calendar-month-end: median={int(gap_days.median())}d, "
        f"max={int(gap_days.max())}d (lower is closer to true month-end)"
    )

    out_monthly = write_feature_panel(
        monthly, settings.processed_dir / "features_monthly.parquet"
    )

    t1 = dt.datetime.now()
    logger.info(
        f"04_build_features: weekly {out_features} | rows={len(features):,} cols={features.shape[1]}\n"
        f"                   monthly {out_monthly} | rows={len(monthly):,} dates={monthly['Date'].nunique()}\n"
        f"                   took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
