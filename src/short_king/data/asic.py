"""ASIC daily aggregate short-position PDF scraper.

ASIC publishes a daily PDF of every reported short position on the ASX at
``download.asic.gov.au/short-selling/RR{YYYYMMDD}-001-SSDailyAggShortPos.pdf``.
The position reported on release-day ``R`` is the *as-of* position 4 business
days earlier (``R - 4 BDay``), so a Friday release describes the prior Monday.

This module parses one or many of those PDFs into a tidy weekly panel:

    Date | ReleaseDate | Ticker | Company | ShortPositions | TotalVolume | ShortPct

Improvements over the v1 notebook prototype:

* On-disk parquet cache per release — re-runs are deterministic and cheap.
* Dual parser: ``tabula-py`` first (lattice mode handles the report's ruled
  table cleanly), ``pdfplumber`` as a pure-Python fallback when Java is
  unavailable or tabula returns nothing parseable.
* Explicit row validation via :func:`is_ticker` — bonds and option codes that
  leak past header detection are dropped.
* Concurrent fetch via ``ThreadPoolExecutor`` for full-history pulls.
* PythonCase column names so downstream code never has to quote them.
"""

from __future__ import annotations

import datetime as dt
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from short_king.utils.config import settings
from short_king.utils.dates import asof_date, fridays_back, most_recent_friday
from short_king.utils.io import ensure_dir, read_parquet, write_parquet
from short_king.utils.logging import logger

# Public column contract — used by every downstream consumer of this module.
COLUMNS_KEPT: list[str] = [
    "Date",
    "ReleaseDate",
    "Ticker",
    "Company",
    "ShortPositions",
    "TotalVolume",
    "ShortPct",
]

# How many calendar days back from the target Friday we'll probe for a PDF
# (ASIC occasionally releases the Friday file on the following Mon/Tue, or
# skips a public holiday).
_MAX_FALLBACK_DAYS = 6

# A request timeout that's generous enough for a ~1MB PDF on a slow link.
_HTTP_TIMEOUT_S = 30

# Minimum row count for a parsed table fragment to be considered real data
# (the report's header/legend tables are tiny; the data table has thousands).
_MIN_ROWS_PER_FRAGMENT = 4


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def is_ticker(x: Any) -> bool:
    """True if ``x`` looks like an ASX ticker (1-6 alphanumerics, optional dot).

    Used to drop header rows, footnotes, and stray text that bleeds into the
    Ticker column when tabula mis-segments a row.
    """
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    if not s or " " in s:
        return False
    if s.lower().startswith("product code"):
        return False
    core = s.replace(".", "")
    return 1 <= len(core) <= 6 and core.isalnum()


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def _release_url(release_date: dt.date) -> str:
    return settings.asic_base_url.format(datestr=release_date.strftime("%Y%m%d"))


def _cache_path(release_date: dt.date) -> Path:
    ensure_dir(settings.asic_cache_dir)
    return settings.asic_cache_dir / f"asic_{release_date.strftime('%Y%m%d')}.parquet"


def _norm_cols(cols: list[str]) -> list[str]:
    return [str(c).strip().lower().replace("\n", " ").replace("  ", " ") for c in cols]


def _to_numeric(s: pd.Series) -> pd.Series:
    """Strip commas/whitespace then coerce; non-numeric -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _download_pdf(url: str) -> bytes | None:
    """GET the PDF; return raw bytes, or None on 404 / network error."""
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        logger.debug(f"PDF GET failed for {url}: {exc}")
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        logger.debug(f"PDF GET status={resp.status_code} for {url}")
        return None
    if not resp.content.startswith(b"%PDF"):
        logger.debug(f"Non-PDF body for {url} (len={len(resp.content)})")
        return None
    return resp.content


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def _parse_with_tabula(pdf_bytes: bytes) -> list[pd.DataFrame]:
    """Best-effort lattice-then-stream parse with tabula. Empty list on failure.

    Lattice exploits the ruled table in the ASIC report and is dramatically
    cleaner than stream mode; stream is the fallback when ruling is missing
    (some older releases).
    """
    try:
        import tabula  # noqa: WPS433 - optional heavy dep
    except ImportError:
        logger.debug("tabula-py not installed; skipping tabula parser.")
        return []

    # tabula needs a path; write to a tempfile.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(pdf_bytes)
        tmp_path = fh.name

    tables: list[pd.DataFrame] = []
    for kwargs in (dict(lattice=True), dict(stream=True)):
        try:
            result = tabula.read_pdf(tmp_path, pages="all", multiple_tables=True, **kwargs)
        except Exception as exc:  # tabula wraps Java errors as plain Exception
            logger.debug(f"tabula parse failed ({kwargs}): {exc}")
            continue
        if result:
            tables = [t for t in result if isinstance(t, pd.DataFrame) and not t.empty]
            if tables:
                break

    # Best-effort cleanup; safe to ignore on Windows file-locks.
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except OSError:
        pass

    return tables


def _parse_with_pdfplumber(pdf_bytes: bytes) -> list[pd.DataFrame]:
    """Pure-Python fallback using pdfplumber. Empty list on failure."""
    try:
        import pdfplumber  # noqa: WPS433
    except ImportError:
        logger.debug("pdfplumber not installed; skipping pdfplumber parser.")
        return []

    tables: list[pd.DataFrame] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                for raw in page.extract_tables() or []:
                    if not raw or len(raw) < 2:
                        continue
                    header, *rows = raw
                    df = pd.DataFrame(rows, columns=header)
                    tables.append(df)
    except Exception as exc:
        logger.debug(f"pdfplumber parse failed: {exc}")
        return []
    return tables


def _rename_clean_one(df: pd.DataFrame) -> pd.DataFrame:
    """Map heterogenous PDF headers into the canonical PythonCase column set.

    The ASIC report header text drifts between releases ("Product Code"
    vs "Reported Short Position vs Total Product in Issue Reported as Short
    Positions"). Match on normalized substrings rather than exact strings.
    """
    raw_cols = list(df.columns)
    normed = _norm_cols([str(c) for c in raw_cols])
    rename: dict[Any, str] = {}
    for i, c in enumerate(normed):
        if c.startswith("product code"):
            rename[raw_cols[i]] = "Ticker"
        elif c.startswith("product"):
            rename[raw_cols[i]] = "Company"
        elif "reported short position" in c and "reported as short" not in c:
            rename[raw_cols[i]] = "ShortPositions"
        elif "total product in issue" in c and "reported as short" not in c:
            rename[raw_cols[i]] = "TotalVolume"
        elif (
            "% of total product in issue reported as short positions" in c
            or c.endswith("short positions")
        ):
            rename[raw_cols[i]] = "ShortPct"

    df = df.rename(columns=rename)
    keep = ["Company", "Ticker", "ShortPositions", "TotalVolume", "ShortPct"]
    df = df[[c for c in keep if c in df.columns]].copy()

    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=keep)

    df = df[df["Ticker"].apply(is_ticker)]

    if "ShortPositions" in df.columns:
        df["ShortPositions"] = _to_numeric(df["ShortPositions"])
    if "TotalVolume" in df.columns:
        df["TotalVolume"] = _to_numeric(df["TotalVolume"])
    if "ShortPct" in df.columns:
        df["ShortPct"] = pd.to_numeric(df["ShortPct"], errors="coerce")

    # Derive ShortPct if missing (older releases occasionally omitted the column).
    if "ShortPct" not in df.columns and {"ShortPositions", "TotalVolume"}.issubset(df.columns):
        df["ShortPct"] = (df["ShortPositions"] / df["TotalVolume"]) * 100.0

    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep]


def _parse_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    """Try tabula then pdfplumber. Return a cleaned, deduped, ticker-only df.

    Returns an empty DataFrame (with the 5 data columns) if neither parser
    yields anything usable.
    """
    fragments = _parse_with_tabula(pdf_bytes)
    if not fragments:
        fragments = _parse_with_pdfplumber(pdf_bytes)
    if not fragments:
        return pd.DataFrame(
            columns=["Company", "Ticker", "ShortPositions", "TotalVolume", "ShortPct"]
        )

    cleaned = [
        _rename_clean_one(t.dropna(how="all"))
        for t in fragments
        if len(t) >= _MIN_ROWS_PER_FRAGMENT
    ]
    cleaned = [c for c in cleaned if not c.empty]
    if not cleaned:
        return pd.DataFrame(
            columns=["Company", "Ticker", "ShortPositions", "TotalVolume", "ShortPct"]
        )

    out = pd.concat(cleaned, ignore_index=True).drop_duplicates(subset=["Ticker"])
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def fetch_release(
    release_date: dt.date | pd.Timestamp,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Parse one ASIC release into a clean DataFrame with :data:`COLUMNS_KEPT`.

    The function probes the target date and up to 6 calendar days earlier
    (matches v1 behaviour — ASIC occasionally publishes a missing Friday
    file on the following business day). On success the result is cached as
    ``settings.asic_cache_dir / asic_{YYYYMMDD}.parquet`` keyed by the
    release date that actually had a PDF (not the target).

    Returns an empty DataFrame with the canonical columns when no release in
    the fallback window is available.
    """
    target = pd.Timestamp(release_date).normalize().date()

    # Cache hit on the *target* date itself short-circuits everything.
    target_cache = _cache_path(target)
    if not force_refresh and target_cache.exists():
        try:
            cached = read_parquet(target_cache)
            logger.debug(f"ASIC cache hit: {target_cache.name} ({len(cached)} rows)")
            return cached
        except Exception as exc:
            logger.warning(f"Corrupt cache {target_cache.name}, refetching: {exc}")

    for delta in range(_MAX_FALLBACK_DAYS + 1):
        probe = target - dt.timedelta(days=delta)
        probe_cache = _cache_path(probe)

        # Even when target wasn't cached, a fallback date may have been
        # cached on a previous run — honour that.
        if not force_refresh and probe_cache.exists():
            try:
                cached = read_parquet(probe_cache)
                logger.debug(f"ASIC cache hit (fallback): {probe_cache.name}")
                return cached
            except Exception as exc:
                logger.warning(f"Corrupt cache {probe_cache.name}, refetching: {exc}")

        url = _release_url(probe)
        pdf_bytes = _download_pdf(url)
        if pdf_bytes is None:
            continue

        body = _parse_pdf(pdf_bytes)
        if body.empty:
            logger.debug(f"PDF parsed but no usable rows: {url}")
            continue

        release_ts = pd.Timestamp(probe)
        asof_ts = asof_date(release_ts)
        body.insert(0, "ReleaseDate", release_ts)
        body.insert(0, "Date", asof_ts)
        body = body[COLUMNS_KEPT].reset_index(drop=True)

        kept = len(body)
        # Drop rows where the two core numeric columns are entirely missing.
        body = body.dropna(subset=["ShortPositions", "TotalVolume"], how="all").reset_index(
            drop=True
        )
        dropped = kept - len(body)

        # Cast counts to nullable Int64 (preserves NaN, avoids float in IDs).
        body["ShortPositions"] = body["ShortPositions"].astype("Int64")
        body["TotalVolume"] = body["TotalVolume"].astype("Int64")
        body["ShortPct"] = body["ShortPct"].astype("float64")

        write_parquet(body, probe_cache)
        logger.info(
            f"ASIC {probe.strftime('%Y-%m-%d')}: kept={len(body)} dropped={dropped} "
            f"asof={asof_ts.strftime('%Y-%m-%d')}"
        )
        return body

    logger.warning(
        f"No ASIC release found within {_MAX_FALLBACK_DAYS} days of {target.isoformat()}"
    )
    return pd.DataFrame(columns=COLUMNS_KEPT)


def fetch_weeks_back(
    weeks: int,
    *,
    anchor: dt.date | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Parallel fetch of ``weeks`` most recent Friday releases (back from anchor).

    The result is a long DataFrame keyed by ``(Date, Ticker)``, sorted, with
    duplicates resolved by keeping the first occurrence (in practice ASIC
    never duplicates a ticker within a release).
    """
    if weeks <= 0:
        return pd.DataFrame(columns=COLUMNS_KEPT)

    anchor = anchor or most_recent_friday()
    fridays = fridays_back(weeks, anchor=anchor)
    workers = max_workers or settings.asic_max_workers

    logger.info(
        f"Fetching {weeks} ASIC weekly releases (anchor={anchor.isoformat()}, "
        f"workers={workers})"
    )

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_friday = {ex.submit(fetch_release, f): f for f in fridays}
        for fut in as_completed(future_to_friday):
            friday = future_to_friday[fut]
            try:
                df = fut.result()
            except Exception as exc:
                logger.error(f"fetch_release({friday.isoformat()}) raised: {exc}")
                continue
            if not df.empty:
                frames.append(df)

    if not frames:
        logger.warning("fetch_weeks_back: no releases retrieved.")
        return pd.DataFrame(columns=COLUMNS_KEPT)

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Date", "Ticker"], keep="first")
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    logger.info(
        f"fetch_weeks_back: {out['Date'].dt.date.nunique()} as-of dates, "
        f"{out['Ticker'].nunique()} unique tickers, {len(out)} rows."
    )
    return out


def filter_ordinary_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose Company description contains 'ORDINARY'.

    ASIC's daily file mixes ordinary shares with bonds, hybrids, ETFs,
    options and stapled securities; for an equity short-interest model
    we only want the underlying common stock.
    """
    if df.empty or "Company" not in df.columns:
        return df.copy()
    before = len(df)
    out = df[df["Company"].astype(str).str.contains("ORDINARY", case=False, na=False)].copy()
    logger.info(
        f"filter_ordinary_only: kept {len(out)}/{before} rows "
        f"(dropped {before - len(out)} non-ordinary securities)"
    )
    return out.reset_index(drop=True)


__all__ = [
    "COLUMNS_KEPT",
    "is_ticker",
    "fetch_release",
    "fetch_weeks_back",
    "filter_ordinary_only",
]
