"""Pull total-return-adjusted daily prices from FMP for every kept ticker.

Reads ``data/processed/asic_long.parquet`` (already pruned to the kept universe
by 02_pull_fmp_fundamentals.py, if --top-tickers was used). Date range is
derived from the ASIC panel itself: the earliest as-of date minus a buffer
(for trailing returns/momentum windows) up to today. Prices are written long
to ``data/processed/prices_long.parquet``.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from short_king.data.fmp_client import FMPClient
from short_king.data.prices import fetch_many_adjusted, fetch_many_market_cap
from short_king.data.yahoo_xcheck import fetch_many_yahoo_adjusted
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buffer-days", type=int, default=400,
                   help="Pre-roll start date so trailing windows are populated (default 400).")
    p.add_argument("--start", type=str, default=None,
                   help="Override start date YYYY-MM-DD.")
    p.add_argument("--end", type=str, default=None,
                   help="Override end date YYYY-MM-DD.")
    p.add_argument("--force-refresh", action="store_true",
                   help="Ignore FMP JSON cache and re-fetch.")
    p.add_argument("--limit-symbols", type=int, default=None,
                   help="Cap on symbols (dev / smoke).")
    p.add_argument("--price-source", choices=("yahoo", "fmp"), default="yahoo",
                   help="Primary price source. FMP only ships ~5 years of ASX daily "
                        "data per call on the current plan, so 'yahoo' is default and "
                        "is what unlocks the full 16-year ASIC window. The Yahoo "
                        "cross-check (yahoo_crosscheck.csv) shows median monthly-return "
                        "correlation of 0.9996 vs FMP in the overlap window.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    asic_path = settings.processed_dir / "asic_long.parquet"
    if not asic_path.exists():
        logger.error(f"{asic_path} not found - must run 01_pull_asic.py first.")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"03_pull_fmp_prices: start {t0.isoformat(timespec='seconds')}")

    asic = read_parquet(asic_path)
    tickers = sorted(asic["Ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    symbols = [f"{t}.AX" for t in tickers]
    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]

    asic_min = pd.to_datetime(asic["Date"]).min()
    start = args.start or (asic_min - pd.Timedelta(days=args.buffer_days)).strftime("%Y-%m-%d")
    end = args.end or dt.date.today().strftime("%Y-%m-%d")
    logger.info(f"03_pull_fmp_prices: {len(symbols)} symbols | window {start} -> {end}")

    client = FMPClient(force_refresh=args.force_refresh)
    if args.price_source == "yahoo":
        logger.info(f"price-source=yahoo (16-year history; FMP only has ~5y on this plan)")
        prices = fetch_many_yahoo_adjusted(symbols, start=start, end=end)
    else:
        logger.info("price-source=fmp")
        prices = fetch_many_adjusted(symbols, start=start, end=end, client=client)
    if prices.empty:
        logger.error("Prices fetch returned zero rows - aborting.")
        return 1

    out_path = settings.processed_dir / "prices_long.parquet"
    write_parquet(prices, out_path)

    # FMP historical-market-cap is the correct daily mktCap (split-adjusted
    # share count), which the assemble step prefers over the broken
    # `sharesOutstanding * adjClose` derivation. Persisted separately so the
    # assembly step can do a clean as-of join.
    logger.info(f"fetching FMP historical-market-cap for {len(symbols)} symbols")
    mcap = fetch_many_market_cap(symbols, start=start, end=end, client=client)
    mcap_path = settings.processed_dir / "marketcap_long.parquet"
    if not mcap.empty:
        write_parquet(mcap, mcap_path)
    else:
        logger.warning("mktCap fetch returned zero rows - assemble step will fall back to shares*price")

    n_sym = prices["symbol"].nunique()
    n_dates = prices["date"].nunique()
    n_mcap = mcap["symbol"].nunique() if not mcap.empty else 0
    t1 = dt.datetime.now()
    logger.info(
        f"03_pull_fmp_prices: wrote {out_path} | prices={len(prices):,} ({n_sym} symbols, {n_dates} dates) "
        f"| mktCap={len(mcap):,} ({n_mcap} symbols) | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
