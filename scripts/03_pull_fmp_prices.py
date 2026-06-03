"""Pull total-return-adjusted daily prices from FMP for every ticker in ASIC.

The date range is derived from the ASIC panel itself: the earliest as-of date
minus a buffer (for trailing returns/momentum windows) up to today. Prices are
written long to ``data/processed/prices_long.parquet`` with columns
``[symbol, date, adjClose, volume]``.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from short_king.data.fmp_client import FMPClient
from short_king.data.prices import fetch_many_adjusted
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--buffer-days",
        type=int,
        default=400,
        help="Pre-roll the price start date by this many calendar days, so "
        "trailing windows (e.g. 252d momentum) are populated at the earliest "
        "ASIC as-of date (default 400).",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override start date YYYY-MM-DD (default: ASIC min - buffer-days).",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Override end date YYYY-MM-DD (default: today).",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore FMP JSON cache and re-fetch.",
    )
    p.add_argument(
        "--limit-symbols",
        type=int,
        default=None,
        help="Optional cap on number of symbols (smoke-tests / dev).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    asic_path = settings.processed_dir / "asic_long.parquet"
    if not asic_path.exists():
        logger.error(f"{asic_path} not found — must run 01_pull_asic.py first.")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"03_pull_fmp_prices: start {t0.isoformat(timespec='seconds')}")

    asic = read_parquet(asic_path)
    tickers = (
        asic["Ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    tickers.sort()
    symbols = [f"{t}.AX" for t in tickers]
    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]

    # Derive the price window from ASIC unless overridden.
    asic_min = pd.to_datetime(asic["Date"]).min()
    start = (
        args.start
        if args.start is not None
        else (asic_min - pd.Timedelta(days=args.buffer_days)).strftime("%Y-%m-%d")
    )
    end = args.end if args.end is not None else dt.date.today().strftime("%Y-%m-%d")
    logger.info(
        f"03_pull_fmp_prices: {len(symbols)} symbols | window {start} -> {end}"
    )

    client = FMPClient(force_refresh=args.force_refresh)
    prices = fetch_many_adjusted(symbols, start=start, end=end, client=client)
    if prices.empty:
        logger.error("Prices fetch returned zero rows — aborting.")
        return 1

    out_path = settings.processed_dir / "prices_long.parquet"
    write_parquet(prices, out_path)

    n_sym = prices["symbol"].nunique()
    n_dates = prices["date"].nunique()
    t1 = dt.datetime.now()
    logger.info(
        f"03_pull_fmp_prices: wrote {out_path} | rows={len(prices):,} "
        f"symbols={n_sym} dates={n_dates} | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
