"""Quarterly fundamentals from the FMP stable API.

Pulls the seven datasets used downstream for factor construction:

* ``income_statement``    — revenue, margins, EPS  (carries ``filingDate``)
* ``balance_sheet``       — assets, debt, equity   (carries ``filingDate``)
* ``cash_flow``           — operating / FCF        (carries ``filingDate``)
* ``ratios``              — pre-computed ratios    (period-end ``date`` only)
* ``key_metrics``         — multiples & sizing     (period-end ``date`` only)
* ``enterprise_values``   — EV / market cap        (period-end ``date`` only)
* ``financial_growth``    — YoY/QoQ growth rates   (period-end ``date`` only)

The three core statements expose ``filingDate`` / ``acceptedDate`` — the
public-availability timestamps that drive point-in-time lagging. The four
derived datasets only carry the period-end ``date``; their availability is
mapped to the parent income filing during the panel-assembly step (joined on
``symbol`` + ``fiscalYear`` + ``period``), keeping the pipeline free of
look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from short_king.data.fmp_client import FMPClient
from short_king.utils.config import settings
from short_king.utils.io import ensure_dir, write_parquet
from short_king.utils.logging import logger

# Logical endpoint names, in canonical order. The tuple is part of the public
# API — downstream code iterates it to know which tables exist on disk.
STATEMENT_ENDPOINTS: tuple[str, ...] = (
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "ratios",
    "key_metrics",
    "enterprise_values",
    "financial_growth",
)

# Logical endpoint -> FMPClient method name. Kept private so callers always go
# through the iterable above.
_ENDPOINT_METHODS: dict[str, str] = {
    "income_statement": "income_statement",
    "balance_sheet": "balance_sheet",
    "cash_flow": "cash_flow",
    "ratios": "ratios",
    "key_metrics": "key_metrics",
    "enterprise_values": "enterprise_values",
    "financial_growth": "financial_growth",
}

# Columns coerced to datetime when present. ``filingDate`` / ``acceptedDate``
# only appear on the core statements; ``date`` is always the period-end.
_DATE_COLS: tuple[str, ...] = ("date", "filingDate", "acceptedDate")


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known date columns to ``datetime64[ns]`` in place; tolerate junk."""
    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Stable per-symbol chronological order for easy diffing and inspection."""
    sort_cols = [c for c in ("symbol", "date") if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols).reset_index(drop=True)


def fetch_one(
    symbol: str,
    *,
    period: str = "quarter",
    limit: int = 80,
    client: FMPClient | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch all seven statement endpoints for a single ``symbol``.

    Returns a dict keyed by ``STATEMENT_ENDPOINTS``; each value is a per-symbol
    DataFrame (rows = filings, columns = raw FMP fields). Empty endpoints are
    returned as empty DataFrames so callers can rely on the key set.
    """
    client = client or FMPClient()
    out: dict[str, pd.DataFrame] = {}
    for name in STATEMENT_ENDPOINTS:
        method = getattr(client, _ENDPOINT_METHODS[name])
        try:
            rows = method(symbol, period=period, limit=limit)
        except Exception as exc:  # noqa: BLE001 - one bad endpoint shouldn't kill the rest
            logger.warning(f"{name} {symbol}: {exc}")
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            df = _coerce_dates(df)
            out[name] = _sort_frame(df)
        else:
            out[name] = pd.DataFrame()
    return out


def fetch_many(
    symbols: list[str],
    *,
    period: str = "quarter",
    limit: int = 80,
    client: FMPClient | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch all seven endpoints for many symbols and concatenate per endpoint.

    Returns a dict keyed by ``STATEMENT_ENDPOINTS``; each value is one long
    DataFrame with a ``symbol`` column. Per-symbol failures are logged and
    skipped so a single bad ticker doesn't poison the bulk pull. Progress is
    logged every 50 symbols.
    """
    client = client or FMPClient()
    symbols = list(symbols)
    n = len(symbols)
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in STATEMENT_ENDPOINTS}

    for i, sym in enumerate(symbols, 1):
        per_symbol = fetch_one(sym, period=period, limit=limit, client=client)
        for name, df in per_symbol.items():
            if not df.empty:
                frames[name].append(df)
        if i % 50 == 0 or i == n:
            logger.info(f"fundamentals: {i}/{n} symbols")

    out: dict[str, pd.DataFrame] = {}
    for name in STATEMENT_ENDPOINTS:
        parts = frames[name]
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            out[name] = _sort_frame(_coerce_dates(combined))
        else:
            out[name] = pd.DataFrame()
            logger.warning(f"fundamentals: {name} returned no rows for any symbol")
    return out


def write_raw_tables(
    tables: dict[str, pd.DataFrame],
    outdir: Path | None = None,
) -> None:
    """Write each endpoint DataFrame to ``<outdir>/<endpoint>.parquet``.

    Defaults to ``data/raw/fmp_raw/``. Empty frames are skipped (with a warning)
    so we don't litter the cache with placeholder files.
    """
    outdir = ensure_dir(outdir or (settings.raw_dir / "fmp_raw"))
    for name, df in tables.items():
        if df.empty:
            logger.warning(f"fundamentals: skipping write for empty table {name!r}")
            continue
        path = outdir / f"{name}.parquet"
        write_parquet(df, path)
        logger.info(f"fundamentals: wrote {len(df):,} rows -> {path}")


__all__ = [
    "STATEMENT_ENDPOINTS",
    "fetch_one",
    "fetch_many",
    "write_raw_tables",
]
