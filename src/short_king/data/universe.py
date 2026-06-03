"""Universe construction for short-king.

Two universes are built side-by-side:

* **ASX equities (primary)** — derived *from* the ASIC short-position panel.
  ASIC inherently defines the set of names that ever carried a reportable short
  position, so taking the unique tickers (after restricting to ORDINARY shares)
  gives a closed, defensible study population for short-research. Each ticker is
  enriched with a static FMP profile (sector / industry / market cap) and the
  first/last week it was observed in ASIC.

* **US S&P 500 (robustness)** — a true point-in-time membership panel, rebuilt
  by walking FMP's chronological change-log *backwards* from the current
  constituents. Identical mechanics to ``qfr/data/universe.py``: undoing every
  change recorded after the as-of date — additions are dropped, removals are
  restored — yields the survivorship-bias-free roster on each rebalance date.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from short_king.data.fmp_client import FMPClient
from short_king.utils.logging import logger


# ---------------------------------------------------------------------------
# ASX universe (from ASIC)
# ---------------------------------------------------------------------------
_PROFILE_COLS: tuple[str, ...] = (
    "companyName",
    "sector",
    "industry",
    "mktCap",
    "averageVolume",
    "currency",
    "isin",
)


def _company_mode(series: pd.Series) -> str:
    """Most common company name for a ticker (ASIC names drift slightly week-to-week)."""
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    mode = s.mode()
    return mode.iloc[0] if not mode.empty else s.iloc[0]


def _fetch_one_profile(client: FMPClient, ticker: str) -> dict:
    """Fetch a single ASX profile (``.AX`` suffix); empty dict on failure."""
    symbol = f"{ticker}.AX"
    try:
        profile = client.profile(symbol) or {}
    except Exception as exc:  # noqa: BLE001 — log + skip is the right behaviour here
        logger.warning(f"profile {symbol}: {exc}")
        profile = {}
    return {"ticker": ticker, "symbol": symbol, **profile}


def build_asx_universe_from_asic(
    asic_long: pd.DataFrame,
    client: FMPClient | None = None,
    *,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Assemble the ASX research universe from an ASIC short-position panel.

    The input must already be filtered to ordinary shares (``filter_ordinary_only``)
    — every unique ``Ticker`` becomes a universe member. FMP profile metadata is
    attached symbol-by-symbol with on-disk JSON caching, so re-runs are free.
    Static metadata (sector / industry / mktCap) is intentionally a point-in-time
    snapshot taken once; for time-varying sector exposures use the dated panels.

    Parameters
    ----------
    asic_long : DataFrame
        Long ASIC table after ``filter_ordinary_only``, with at least columns
        ``Date``, ``Ticker``, ``Company``.
    client : FMPClient | None
        Reuse a client to share the rate limiter; created on demand otherwise.
    max_workers : int
        Concurrent profile fetches. Defaults to 1 (single-threaded) so the
        client's polite rate-limiter is the binding constraint; raise it to
        parallelise after warming the cache.

    Returns
    -------
    DataFrame with columns
        ``['symbol', 'ticker', 'company', 'sector', 'industry',
           'mktCap', 'averageVolume', 'currency', 'isin',
           'first_seen', 'last_seen']``
        one row per ticker, sorted by ``ticker``.
    """
    if asic_long is None or asic_long.empty:
        logger.warning("build_asx_universe_from_asic: empty ASIC input")
        return pd.DataFrame(
            columns=[
                "symbol", "ticker", "company", "sector", "industry",
                "mktCap", "averageVolume", "currency", "isin",
                "first_seen", "last_seen",
            ]
        )

    # 1) Per-ticker presence window + canonical company name from ASIC itself.
    seen = (
        asic_long.dropna(subset=["Ticker"])
        .groupby("Ticker", as_index=False)
        .agg(
            company=("Company", _company_mode),
            first_seen=("Date", "min"),
            last_seen=("Date", "max"),
        )
        .rename(columns={"Ticker": "ticker"})
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    tickers: list[str] = seen["ticker"].astype(str).str.strip().tolist()
    n = len(tickers)
    logger.info(f"ASX universe seed: {n} unique tickers from ASIC")

    # 2) Pull FMP profiles (cached). Single-thread when max_workers<=1 to keep
    #    the client's rate-limiter the sole gate; otherwise fan out.
    client = client or FMPClient()
    profiles: list[dict] = []
    if max_workers <= 1:
        for i, tkr in enumerate(tickers, 1):
            profiles.append(_fetch_one_profile(client, tkr))
            if i % 100 == 0 or i == n:
                logger.info(f"profiles: {i}/{n}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one_profile, client, t): t for t in tickers}
            done = 0
            for fut in as_completed(futures):
                profiles.append(fut.result())
                done += 1
                if done % 100 == 0 or done == n:
                    logger.info(f"profiles: {done}/{n}")

    prof_df = pd.DataFrame(profiles)

    # 3) Join the static profile onto the ASIC-derived seed.
    keep = ["ticker", "symbol"] + [c for c in _PROFILE_COLS if c in prof_df.columns]
    prof_df = prof_df.reindex(columns=keep)

    out = seen.merge(prof_df, on="ticker", how="left")
    # Synthesise symbol for any rows where the profile call returned nothing.
    out["symbol"] = out["symbol"].fillna(out["ticker"].astype(str) + ".AX")

    # 4) Reorder and rename to the documented schema.
    if "companyName" in out.columns:
        # Prefer FMP's official name where available; fall back to ASIC name.
        out["company"] = out["companyName"].where(
            out["companyName"].notna() & (out["companyName"].astype(str).str.len() > 0),
            out["company"],
        )
        out = out.drop(columns=["companyName"])
    for col in ("sector", "industry", "currency", "isin"):
        if col not in out.columns:
            out[col] = pd.NA
    for col in ("mktCap", "averageVolume"):
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[
        [
            "symbol", "ticker", "company", "sector", "industry",
            "mktCap", "averageVolume", "currency", "isin",
            "first_seen", "last_seen",
        ]
    ].sort_values("ticker").reset_index(drop=True)

    n_with_sector = out["sector"].notna().sum()
    n_with_mcap = out["mktCap"].notna().sum()
    logger.info(
        f"ASX universe: {len(out)} tickers | "
        f"sector coverage {n_with_sector}/{len(out)} | "
        f"mktCap coverage {n_with_mcap}/{len(out)}"
    )
    return out


# ---------------------------------------------------------------------------
# S&P 500 point-in-time membership (robustness check)
# ---------------------------------------------------------------------------
def _parse_sp500_changes(raw: list[dict]) -> pd.DataFrame:
    """Normalise the FMP change-log into ``[date, added, removed]`` rows."""
    rows = []
    for rec in raw or []:
        date = rec.get("date")
        if not date:
            continue
        added = (rec.get("symbol") or "").strip() or None
        removed = (rec.get("removedTicker") or "").strip() or None
        if added is None and removed is None:
            continue
        rows.append({"date": pd.Timestamp(date), "added": added, "removed": removed})
    if not rows:
        return pd.DataFrame(columns=["date", "added", "removed"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _sp500_current_set(raw: list[dict]) -> set[str]:
    """Current S&P 500 constituents as a set of symbols."""
    if not raw:
        return set()
    df = pd.DataFrame(raw)
    if "symbol" not in df.columns:
        return set()
    return set(df["symbol"].astype(str).str.strip())


def _reconstruct_sp500(
    today: set[str], changes: pd.DataFrame, as_of: pd.Timestamp
) -> set[str]:
    """Members as-of ``as_of`` — undo every change recorded strictly after that date."""
    members = set(today)
    future = changes[changes["date"] > as_of].sort_values("date", ascending=False)
    for _, ch in future.iterrows():
        added, removed = ch["added"], ch["removed"]
        if isinstance(added, str) and added:
            members.discard(added)
        if isinstance(removed, str) and removed:
            members.add(removed)
    return members


def build_sp500_pit(
    start: str,
    end: str,
    client: FMPClient | None = None,
) -> pd.DataFrame:
    """Long point-in-time S&P 500 membership panel at month-ends in ``[start, end]``.

    Returns
    -------
    DataFrame ``['date', 'symbol']`` — one row per (date, member), sorted.
    """
    client = client or FMPClient()
    raw_current = client.sp500_constituents()
    raw_changes = client.historical_sp500_constituents()

    today = _sp500_current_set(raw_current)
    changes = _parse_sp500_changes(raw_changes)

    dates: Iterable[pd.Timestamp] = pd.date_range(start=start, end=end, freq="ME")
    logger.info(
        f"SP500 PIT: current={len(today)} | changes={len(changes)} | "
        f"rebalances={len(dates)}"
    )

    records: list[tuple[pd.Timestamp, str]] = []
    for d in dates:
        ts = pd.Timestamp(d)
        for sym in _reconstruct_sp500(today, changes, ts):
            records.append((ts, sym))

    panel = (
        pd.DataFrame(records, columns=["date", "symbol"])
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    logger.info(
        f"SP500 PIT panel: {len(panel):,} rows | "
        f"{panel['symbol'].nunique()} unique symbols ever in-universe"
    )
    return panel


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def universe_summary(asx_uni: pd.DataFrame) -> pd.DataFrame:
    """One-column diagnostic table for the ASX universe (counts, mcap quantiles, top sectors).

    The shape (a single ``value`` column with descriptive row labels) is chosen
    to print cleanly in notebooks and reports without further formatting.
    """
    rows: list[tuple[str, object]] = []
    n_total = len(asx_uni)
    rows.append(("n_total", n_total))

    if n_total == 0:
        return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")

    n_with_sector = int(asx_uni["sector"].notna().sum()) if "sector" in asx_uni else 0
    n_with_mcap = int(asx_uni["mktCap"].notna().sum()) if "mktCap" in asx_uni else 0
    rows.append(("n_with_sector", n_with_sector))
    rows.append(("n_with_mcap", n_with_mcap))
    rows.append(
        ("pct_with_sector", round(100.0 * n_with_sector / n_total, 1) if n_total else 0.0)
    )
    rows.append(
        ("pct_with_mcap", round(100.0 * n_with_mcap / n_total, 1) if n_total else 0.0)
    )

    if "mktCap" in asx_uni:
        mcap = pd.to_numeric(asx_uni["mktCap"], errors="coerce").dropna()
        if not mcap.empty:
            for q in (0.10, 0.25, 0.50, 0.75, 0.90):
                rows.append((f"mktCap_q{int(q * 100):02d}", float(mcap.quantile(q))))
            rows.append(("mktCap_mean", float(mcap.mean())))
            rows.append(("mktCap_sum", float(mcap.sum())))

    if "sector" in asx_uni:
        top5 = (
            asx_uni["sector"]
            .dropna()
            .astype(str)
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(5)
        )
        for i, (sector, count) in enumerate(top5.items(), 1):
            rows.append((f"top{i}_sector", f"{sector} ({int(count)})"))

    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
