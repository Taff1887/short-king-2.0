"""Pull quarterly fundamentals from FMP for every ticker observed in ASIC.

Reads ``data/processed/asic_long.parquet`` (produced by 01_pull_asic.py),
maps each unique ASX ticker to its ``.AX`` FMP symbol, and calls
:func:`short_king.data.fundamentals.fetch_many` to retrieve the seven canonical
statement / metric endpoints. Each endpoint is persisted as a separate parquet
file under ``data/raw/fmp_raw/`` so the assembly step can join them by
``symbol`` + ``period``-key without re-fetching.
"""

from __future__ import annotations

import argparse
import datetime as dt

from short_king.data.fmp_client import FMPClient
from short_king.data.fundamentals import fetch_many, write_raw_tables
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Max quarters per endpoint per symbol (default 80 ~ 20 years).",
    )
    p.add_argument(
        "--period",
        choices=("quarter", "annual"),
        default="quarter",
        help="FMP statement period grain (default 'quarter').",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore on-disk FMP JSON cache and re-fetch every endpoint.",
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
    logger.info(f"02_pull_fmp_fundamentals: start {t0.isoformat(timespec='seconds')}")

    asic = read_parquet(asic_path)
    tickers = (
        asic["Ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    tickers.sort()
    symbols = [f"{t}.AX" for t in tickers]
    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]
    logger.info(
        f"02_pull_fmp_fundamentals: {len(symbols)} unique ASX symbols (limit={args.limit})"
    )

    client = FMPClient(force_refresh=args.force_refresh)
    tables = fetch_many(symbols, period=args.period, limit=args.limit, client=client)

    outdir = settings.raw_dir / "fmp_raw"
    write_raw_tables(tables, outdir=outdir)

    t1 = dt.datetime.now()
    nonempty = sum(1 for _, df in tables.items() if not df.empty)
    logger.info(
        f"02_pull_fmp_fundamentals: wrote {nonempty}/{len(tables)} endpoints "
        f"under {outdir} | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
