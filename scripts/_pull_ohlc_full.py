"""Pull daily auto-adjusted OHLC for every ticker that ever appears in any
short basket across the FULL IS + OOS panel (2010-2026). Used by
``_apply_stop_loss_full.py`` to compute realistic stop-loss-applied
portfolio returns for every (model, strategy) backtest.

Output: data/processed/prices_ohlc_full.parquet with columns
[Ticker, date, open, high, low, close]. Roughly 346 tickers x 16 years
= ~1M daily bars.

Run once after _strategy_explore / 06_backtest produce OOF predictions and
the full short-basket ticker list. Idempotent.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from short_king.utils.config import settings
from short_king.utils.logging import logger


def main() -> int:
    settings.ensure_dirs()
    pos_path = settings.reports_dir / "_full_short_tickers.txt"
    if not pos_path.exists():
        logger.error(f"{pos_path} not found - generate the ticker list first")
        return 1

    tickers = [t.strip() for t in pos_path.read_text(encoding="utf-8").splitlines() if t.strip()]
    logger.info(f"pulling daily OHLC for {len(tickers)} short-basket tickers from Yahoo")

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return 1

    yahoo_syms = [f"{t}.AX" for t in tickers]

    # Batched download. 16 years of daily data x 346 symbols would saturate a
    # single batch -- split into 50-symbol batches and retry on failure.
    out: list[pd.DataFrame] = []
    BATCH = 50
    for i in range(0, len(yahoo_syms), BATCH):
        batch = yahoo_syms[i:i + BATCH]
        attempts = 0
        raw = None
        while attempts < 3:
            try:
                raw = yf.download(
                    batch, start="2010-05-01", end="2026-06-10",
                    auto_adjust=True, progress=False, threads=True,
                    group_by="ticker",
                )
                break
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                logger.warning(f"batch {i//BATCH+1} attempt {attempts} failed: {exc}; retry")
                time.sleep(2)
        if raw is None or len(raw) == 0:
            logger.warning(f"batch {i//BATCH+1} skipped (no data)")
            continue

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
                keep = [c for c in ("Ticker", "date", "open", "high", "low", "close") if c in sub.columns]
                out.append(sub[keep])
        else:
            sub = raw.dropna(how="all").copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.reset_index().rename(columns={"Date": "date"})
            sub["Ticker"] = batch[0].removesuffix(".AX")
            keep = [c for c in ("Ticker", "date", "open", "high", "low", "close") if c in sub.columns]
            out.append(sub[keep])

        logger.info(f"batch {i//BATCH+1}/{(len(yahoo_syms)+BATCH-1)//BATCH}: {len(batch)} symbols")

    if not out:
        logger.error("no OHLC data pulled")
        return 2

    ohlc = pd.concat(out, ignore_index=True)
    ohlc["date"] = pd.to_datetime(ohlc["date"]).dt.tz_localize(None).dt.normalize()
    ohlc = ohlc.dropna(subset=["high", "low", "open"]).sort_values(["Ticker", "date"])
    ohlc = ohlc.drop_duplicates(subset=["Ticker", "date"], keep="last").reset_index(drop=True)

    out_path = settings.processed_dir / "prices_ohlc_full.parquet"
    ohlc.to_parquet(out_path, index=False)
    logger.info(
        f"wrote {out_path} | {len(ohlc):,} rows | "
        f"{ohlc['Ticker'].nunique()} tickers | "
        f"{ohlc['date'].min().date()} -> {ohlc['date'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
