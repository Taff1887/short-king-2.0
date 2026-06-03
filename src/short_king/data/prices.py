"""FMP price-history fetchers for ASX short-selling research.

Two FMP endpoints feed the rest of the pipeline:

* ``historical-price-eod/dividend-adjusted`` -> ``adjClose`` is a *total-return
  index* (split + dividend adjusted). This is the correct basis for computing
  returns, momentum and volatility — anything where compounded total return
  matters.
* ``historical-price-eod/full`` -> ``close`` is *split-adjusted only* (back to
  today's basis) but **not** dividend-adjusted, i.e. an actual price level
  comparable across splits. We use it to refresh value ratios to the live
  rebalance-date price instead of the stale period-end price FMP bakes into
  its ratios.

For short-king we only need recent history (post-2010 is plenty), so a single
request per ticker stays well inside FMP's ~5,000-bar-per-call cap and we skip
the cursor-pagination logic used in the sister ``qfr`` package.
"""

from __future__ import annotations

import pandas as pd

from short_king.data.fmp_client import FMPClient
from short_king.utils.logging import logger


def _to_frame(rows: list[dict], *, price_col: str, symbol: str) -> pd.DataFrame:
    """Normalise an FMP price payload into ``[date, <price_col>, volume]``."""
    if not rows:
        return pd.DataFrame(columns=["date", price_col, "volume"])
    df = pd.DataFrame(rows)
    if price_col not in df.columns:
        logger.warning(f"prices {symbol}: missing '{price_col}' column in FMP response")
        return pd.DataFrame(columns=["date", price_col, "volume"])
    keep = ["date", price_col]
    if "volume" in df.columns:
        keep.append("volume")
    else:
        df["volume"] = pd.NA
        keep.append("volume")
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Restrict to ``[start, end]`` inclusive (no-op for ``None`` bounds)."""
    if df.empty:
        return df
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def fetch_prices_adjusted(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    client: FMPClient | None = None,
) -> pd.DataFrame:
    """Total-return-adjusted daily bars for one symbol.

    Returns ``[date, adjClose, volume]`` ascending, optionally clipped to
    ``[start, end]``. ``adjClose`` is split- and dividend-adjusted, so it is
    the right input for computing returns, momentum and realised volatility.
    """
    client = client or FMPClient()
    rows = client.historical_prices(symbol, from_date=start, to_date=end, series="dividend-adjusted")
    df = _to_frame(rows, price_col="adjClose", symbol=symbol)
    return _clip(df, start, end)


def fetch_prices_full(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    client: FMPClient | None = None,
) -> pd.DataFrame:
    """Split-adjusted (dividend-UNadjusted) daily bars for one symbol.

    Returns ``[date, close, volume]`` ascending, optionally clipped to
    ``[start, end]``. ``close`` is an actual price *level* (comparable to FMP's
    per-share fundamentals); use it to refresh value ratios to the live
    rebalance-date price.
    """
    client = client or FMPClient()
    rows = client.historical_prices(symbol, from_date=start, to_date=end, series="full")
    df = _to_frame(rows, price_col="close", symbol=symbol)
    return _clip(df, start, end)


def fetch_many_adjusted(
    symbols: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    client: FMPClient | None = None,
) -> pd.DataFrame:
    """Total-return-adjusted bars for many symbols, stacked long.

    Loops sequentially — the underlying ``FMPClient`` throttles itself, so we
    don't need our own concurrency here. Per-symbol failures are logged and
    skipped rather than aborting the bulk pull. Returns
    ``[symbol, date, adjClose, volume]`` sorted by ``[symbol, date]``.
    """
    client = client or FMPClient()
    frames: list[pd.DataFrame] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_prices_adjusted(sym, start=start, end=end, client=client)
        except Exception as e:  # noqa: BLE001 - log and continue the bulk pull
            logger.warning(f"prices {sym}: {e}")
            df = pd.DataFrame(columns=["date", "adjClose", "volume"])
        if not df.empty:
            df = df.copy()
            df["symbol"] = sym
            frames.append(df)
        if i % 50 == 0:
            logger.info(f"prices: {i}/{n} symbols")
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "adjClose", "volume"])
    out = pd.concat(frames, ignore_index=True)
    return (
        out[["symbol", "date", "adjClose", "volume"]]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
