"""Pull daily OHLC (auto-adjusted) for every ticker that appears in the OOS
short basket, covering the OOS window 2023-05-15 to 2026-06-10.

Output: data/processed/prices_ohlc_oos.parquet with columns
[Ticker, date, open, high, low, close]. Used only by
``scripts/_stop_loss_realistic.py``; the main backtest still runs off
``prices_long.parquet`` (adjClose only) for compatibility.

Run once before ``_stop_loss_realistic.py``. Idempotent -- safe to re-run.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from short_king.utils.config import settings
from short_king.utils.logging import logger


def main() -> int:
    settings.ensure_dirs()
    pos_path = settings.reports_dir / "oos_short_positions.csv"
    if not pos_path.exists():
        logger.error(f"{pos_path} not found - run _oos_trades.py first")
        return 1

    pos = pd.read_csv(pos_path)
    tickers = sorted(pos["Ticker"].dropna().unique().tolist())
    logger.info(f"pulling daily OHLC for {len(tickers)} OOS-short tickers from Yahoo")

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return 1

    # Yahoo expects '<ticker>.AX' for ASX.
    yahoo_syms = [f"{t}.AX" for t in tickers]

    # Pull in batches of 50 so a single 503 doesn't kill everything; auto_adjust
    # gives split/dividend-adjusted OHLC, which is what we want to compare to
    # the entry_price (already adjClose-based) on a like-for-like basis.
    out: list[pd.DataFrame] = []
    BATCH = 50
    for i in range(0, len(yahoo_syms), BATCH):
        batch = yahoo_syms[i:i + BATCH]
        attempts = 0
        while attempts < 3:
            try:
                raw = yf.download(
                    batch, start="2023-05-15", end="2026-06-10",
                    auto_adjust=True, progress=False, threads=True,
                    group_by="ticker",
                )
                break
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                logger.warning(f"batch {i//BATCH+1} attempt {attempts} failed: {exc}; retry")
                time.sleep(2)
        else:
            logger.error(f"batch {i//BATCH+1} failed after 3 attempts; skipping")
            continue

        if raw is None or len(raw) == 0:
            continue

        # yfinance returns a multi-column frame keyed by (ticker, OHLC). Pivot
        # to long form.
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in batch:
                if sym not in raw.columns.get_level_values(0):
                    continue
                sub = raw[sym].dropna(how="all").copy()
                if sub.empty:
                    continue
                sub.columns = [c.lower() for c in sub.columns]
                sub = sub.reset_index().rename(columns={"Date": "date"})
                sub["Ticker"] = sym.removesuffix(".AX")
                # Keep only the columns we need.
                keep = [c for c in ("Ticker", "date", "open", "high", "low", "close") if c in sub.columns]
                out.append(sub[keep])
        else:
            sub = raw.dropna(how="all").copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.reset_index().rename(columns={"Date": "date"})
            sub["Ticker"] = batch[0].removesuffix(".AX")
            keep = [c for c in ("Ticker", "date", "open", "high", "low", "close") if c in sub.columns]
            out.append(sub[keep])

        logger.info(f"batch {i//BATCH+1}/{(len(yahoo_syms)+BATCH-1)//BATCH}: pulled {len(batch)} symbols")

    if not out:
        logger.error("no OHLC data pulled")
        return 2

    ohlc = pd.concat(out, ignore_index=True)
    ohlc["date"] = pd.to_datetime(ohlc["date"]).dt.tz_localize(None).dt.normalize()
    ohlc = ohlc.dropna(subset=["high", "low", "open"]).sort_values(["Ticker", "date"])
    ohlc = ohlc.drop_duplicates(subset=["Ticker", "date"], keep="last").reset_index(drop=True)

    out_path = settings.processed_dir / "prices_ohlc_oos.parquet"
    ohlc.to_parquet(out_path, index=False)
    logger.info(
        f"wrote {out_path} | {len(ohlc):,} rows | "
        f"{ohlc['Ticker'].nunique()} tickers | "
        f"{ohlc['date'].min().date()} -> {ohlc['date'].max().date()}"
    )
    # Coverage report.
    by_t = ohlc.groupby("Ticker").size()
    logger.info(f"per-ticker OHLC day count: min={by_t.min()}, "
                f"median={int(by_t.median())}, max={by_t.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
