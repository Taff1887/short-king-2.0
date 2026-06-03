"""Pull weekly ASIC daily aggregate short-position reports into a long panel.

ASIC publishes a Friday-anchored PDF of every reported short position on the ASX
(see :mod:`short_king.data.asic` for the as-of vs release-date mechanics). This
script drives ``fetch_weeks_back`` for ``--weeks`` Fridays, restricts the result
to ORDINARY shares, and persists the canonical long panel to
``data/processed/asic_long.parquet`` — the seed for the rest of the pipeline.

Re-runs are cheap: every parsed release is cached as parquet under
``data/raw/asic_cache/`` so only new Fridays hit the network.
"""

from __future__ import annotations

import argparse
import datetime as dt

from short_king.data.asic import COLUMNS_KEPT, fetch_weeks_back, filter_ordinary_only
from short_king.utils.config import settings
from short_king.utils.io import write_parquet
from short_king.utils.logging import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weeks",
        type=int,
        default=260,
        help="Number of weekly ASIC releases to fetch, anchored on the most recent Friday "
        "(default 260 ~ five years).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override settings.asic_max_workers for the parallel fetch.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    t0 = dt.datetime.now()
    logger.info(f"01_pull_asic: start {t0.isoformat(timespec='seconds')} | weeks={args.weeks}")

    asic = fetch_weeks_back(args.weeks, max_workers=args.max_workers)
    if asic.empty:
        logger.error("ASIC fetch returned zero rows — aborting; check network / cache.")
        return 1

    # Restrict to ordinary common stock (drop bonds/hybrids/options/ETFs).
    asic = filter_ordinary_only(asic)
    asic = asic[COLUMNS_KEPT].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    out_path = settings.processed_dir / "asic_long.parquet"
    write_parquet(asic, out_path)

    n_dates = asic["Date"].dt.date.nunique() if not asic.empty else 0
    n_tickers = asic["Ticker"].nunique() if not asic.empty else 0
    t1 = dt.datetime.now()
    logger.info(
        f"01_pull_asic: wrote {out_path} | rows={len(asic):,} dates={n_dates} "
        f"tickers={n_tickers} | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
