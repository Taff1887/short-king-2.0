"""Assemble the PIT panel, clean it, then build the feature matrix.

Three-stage pipeline:

1. ``assemble_pit_panel`` joins the ASIC weekly short panel with FMP
   fundamentals (lagged to their ``filingDate``) and adjusted prices on a
   common Friday grid -> ``data/processed/master_pit.parquet``.
2. ``conservative_clean`` drops illiquid / corrupted series, then a strict
   ``check_no_lookahead`` audit runs -> ``data/processed/master_clean.parquet``.
3. ``build_feature_panel`` stitches the per-family feature modules + adds
   cross-sectional ranks and sector dummies -> ``data/processed/features.parquet``.

Quality summaries are logged at each stage; lookahead violations abort.
"""

from __future__ import annotations

import argparse
import datetime as dt

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--min-mcap",
        type=float,
        default=2e8,
        help="Min market cap filter passed to conservative_clean (default A$200m).",
    )
    p.add_argument(
        "--no-sector-dummies",
        action="store_true",
        help="Skip one-hot sector dummies (saves columns for linear baselines).",
    )
    p.add_argument(
        "--no-cross-sectional",
        action="store_true",
        help="Skip within-date percentile ranking (debugging only).",
    )
    p.add_argument(
        "--allow-lookahead",
        action="store_true",
        help="Do not abort on lookahead-check failures (debugging only).",
    )
    return p.parse_args()


def _require(path):
    if not path.exists():
        logger.error(f"{path} not found — required input is missing.")
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
        logger.error(f"{fmp_raw_dir} empty — must run 02_pull_fmp_fundamentals.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"04_build_features: start {t0.isoformat(timespec='seconds')}")

    asic = read_parquet(asic_path)
    prices = read_parquet(prices_path)
    fmp_tables = {
        p.stem: read_parquet(p) for p in sorted(fmp_raw_dir.glob("*.parquet"))
    }
    logger.info(
        f"inputs: asic_long={len(asic):,} rows | prices_long={len(prices):,} rows "
        f"| fmp_raw endpoints={list(fmp_tables)}"
    )

    # 1) Assemble the PIT panel.
    panel = assemble_pit_panel(asic=asic, fmp_tables=fmp_tables, prices=prices)
    write_panel(panel, settings.processed_dir / "master_pit.parquet")
    summary = panel_quality_summary(panel)
    logger.info(f"panel_quality_summary:\n{summary}")

    # 2) Clean + audit for look-ahead bias.
    corrupted = detect_corrupted_series(panel)
    if corrupted is not None and len(corrupted):
        logger.warning(f"detect_corrupted_series flagged {len(corrupted)} series")
    clean = conservative_clean(panel, min_mcap=args.min_mcap)
    write_clean(clean, settings.processed_dir / "master_clean.parquet")
    logger.info(f"quality_report (clean):\n{quality_report(clean)}")

    leak = check_no_lookahead(clean)
    if leak is not None:
        # Allow a (bool, msg/df) or a non-empty DataFrame to signal a violation.
        violated = (
            (isinstance(leak, tuple) and not bool(leak[0]))
            or (hasattr(leak, "empty") and not leak.empty)
        )
        if violated:
            logger.error(f"check_no_lookahead FAILED: {leak}")
            if not args.allow_lookahead:
                return 2
        else:
            logger.info(f"check_no_lookahead passed: {leak}")

    # 3) Build the feature matrix.
    features = build_feature_panel(
        clean,
        cross_sectional=not args.no_cross_sectional,
        add_sector_dummies=not args.no_sector_dummies,
    )
    out_features = write_feature_panel(features, settings.processed_dir / "features.parquet")

    t1 = dt.datetime.now()
    logger.info(
        f"04_build_features: wrote {out_features} | rows={len(features):,} "
        f"cols={features.shape[1]} | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
