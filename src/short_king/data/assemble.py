"""Assemble the point-in-time master panel for short-selling research.

This is the methodological core of the data layer. It produces one tidy long
frame keyed by ``[Date, Ticker]`` over the ASIC Friday rebalance grid, where
every value is knowable *as of* that Date:

* **ASIC short-interest** — the cleaned short-positions snapshot for the week.
* **Price** — last adjusted close on/just-before Date (within a calendar-week
  tolerance), the entry price for any trade.
* **Fundamentals** — for each Date we take the most recent quarterly filing
  whose ``acceptedDate`` (the public-availability timestamp) is on or before
  Date. Derived endpoints (ratios, key_metrics, enterprise_values, growth) are
  joined to the same fiscal period as the income filing, so their availability
  inherits the income filing's lag.
* **Market-cap** — ``sharesOutstanding`` from the most-recent PIT balance sheet
  multiplied by the as-of adjusted close (rather than vendor's marketCap, which
  is keyed on period-end and would back-fill).
* **Forward returns** — 1w/4w/12w labels computed off ``prices_long`` (so the
  weekly grid does not have to be balanced per ticker). These peek into the
  future *by design* and are LABELS only — never inputs.
* **Investable flag** — drops rows with no price, stale filings (>4Q lag), or a
  market cap below the liquidity gate (default A$200m).

Caveat: no look-ahead. Filings require ``acceptedDate <= Date``; price requires
trade-date ``<= Date``. Forward-return columns are the only future-peeking
fields and are clearly named ``fwd_ret_*`` so callers cannot mistake them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger

# --- Knobs ----------------------------------------------------------------
DEFAULT_MIN_MKT_CAP_AUD = 100_000_000  # A$100m liquidity gate
DEFAULT_MAX_FILING_STALE_QUARTERS = 4

# Calendar-week tolerance for the price as-of join (ASIC Date is a Friday; the
# market may have been shut, so the previous trading day is accepted).
PRICE_TOLERANCE_DAYS = 7

# Approximate days per quarter -- used to translate the staleness cap into a
# day-based comparison (a quarterly filing accepted ~92 days ago is "1 quarter
# stale", with a generous buffer for vendor lag).
DAYS_PER_QUARTER = 92

# Endpoint shorthands the caller is expected to pass in the ``fundamentals``
# dict. The first three carry ``acceptedDate``; the remaining four carry only
# the period-end ``date`` and are joined by fiscal period.
_CORE_ENDPOINTS = ("income_statement", "balance_sheet", "cash_flow")
_DERIVED_ENDPOINTS = ("ratios", "key_metrics", "enterprise_values", "financial_growth")

# Columns we explicitly do NOT carry through (link/cik/CIK noise) -- everything
# else from the seven endpoints is preserved with a column prefix to keep
# namespaces disjoint and the panel self-describing.
_DROP_COLS = frozenset(
    {
        "link",
        "finalLink",
        "cik",
        "calendarYear",
        "fillingDate",
        "reportedCurrency",
    }
)

# The income statement is the staleness anchor (richest filing schedule).
_KEYS = ["symbol", "fiscalYear", "period"]


# --- helpers --------------------------------------------------------------
def _normalise_dates(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def _prefix_cols(df: pd.DataFrame, prefix: str, keep: tuple[str, ...]) -> pd.DataFrame:
    """Prefix non-key columns to keep endpoint namespaces disjoint."""
    renames = {
        c: f"{prefix}_{c}" for c in df.columns if c not in keep and c not in _DROP_COLS
    }
    return df.drop(columns=[c for c in _DROP_COLS if c in df.columns]).rename(columns=renames)


def _filings_with_availability(income: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol (fiscalYear, period, acceptedDate, period_end) anchor frame."""
    cols = [c for c in (*_KEYS, "acceptedDate", "date") if c in income.columns]
    out = income[cols].copy()
    out = _normalise_dates(out, ("acceptedDate", "date"))
    out = out.dropna(subset=["acceptedDate"])
    out = out.rename(columns={"date": "period_end"})
    return out.drop_duplicates(subset=_KEYS).sort_values("acceptedDate")


def _asof_join_price(
    asic: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    tol_days: int = PRICE_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Attach last adjClose / volume on or before Date per Symbol (within tolerance)."""
    px = prices.loc[:, ["symbol", "date", "adjClose", "volume"]].dropna(subset=["adjClose"])
    px = px[px["adjClose"] > 0].copy()
    # Force nanosecond precision on BOTH sides — parquet round-trips can leave
    # ms/us precision which makes pandas' merge_asof refuse to merge.
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    # ``merge_asof`` requires the ``on`` key globally sorted (not just sorted
    # within each ``by`` group).
    px = px.sort_values("date").reset_index(drop=True)

    left = asic[["Symbol", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize().astype("datetime64[ns]")
    left = left.sort_values("Date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        px.rename(columns={"symbol": "Symbol", "date": "Date"}),
        on="Date",
        by="Symbol",
        direction="backward",
        tolerance=pd.Timedelta(days=tol_days),
    )
    return out  # columns: Symbol, Date, adjClose, volume


def _asof_join_fundamentals(
    asic: pd.DataFrame,
    filings: pd.DataFrame,
) -> pd.DataFrame:
    """Latest filing per (Symbol, Date) with ``acceptedDate <= Date``.

    Returns the join keys plus ``period_end`` and ``filing_lag_days``.
    """
    left = asic[["Symbol", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize().astype("datetime64[ns]")
    # Both sides must be globally sorted on their respective ``on`` columns.
    left = left.sort_values("Date").reset_index(drop=True)

    right = filings.rename(columns={"symbol": "Symbol"}).copy()
    right["acceptedDate"] = pd.to_datetime(right["acceptedDate"]).astype("datetime64[ns]")
    right = right.sort_values("acceptedDate").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="Date",
        right_on="acceptedDate",
        by="Symbol",
        direction="backward",
    )
    out["filing_lag_days"] = (out["Date"] - out["acceptedDate"]).dt.days
    return out  # Symbol, Date, fiscalYear, period, acceptedDate, period_end, filing_lag_days


def _forward_returns_from_prices(
    prices: pd.DataFrame,
    asic: pd.DataFrame,
    *,
    horizons_weeks: tuple[int, ...],
) -> pd.DataFrame:
    """Compute forward returns labeled at each (Symbol, Date) in the ASIC grid.

    Implementation note: rather than shift inside a sparse weekly panel that
    may be missing Fridays per symbol, we attach the future price by
    independently asof-joining ``Date + horizon_weeks*7`` against the dense
    daily price series. The asof is *backward* with a 7-day tolerance, so a
    market holiday on the target Friday picks up the preceding trading day --
    the same convention used for the entry price.
    """
    px = prices.loc[:, ["symbol", "date", "adjClose"]].dropna(subset=["adjClose"])
    px = px[px["adjClose"] > 0].copy()
    # Force ns precision on the date column - parquet round-trips can produce
    # ms/us which breaks pandas merge_asof's dtype check.
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    # Global sort on the ``on`` key (required by ``merge_asof`` with ``by``).
    px = px.sort_values("date").reset_index(drop=True)
    px_renamed = px.rename(columns={"symbol": "Symbol", "date": "_target", "adjClose": "_fwd_px"})

    base = asic[["Symbol", "Date", "adjClose"]].copy()
    base["Date"] = pd.to_datetime(base["Date"]).dt.normalize().astype("datetime64[ns]")

    tol = pd.Timedelta(days=PRICE_TOLERANCE_DAYS)
    out = base[["Symbol", "Date"]].copy()
    for h in horizons_weeks:
        target = base[["Symbol", "Date"]].copy()
        target["_target"] = (target["Date"] + pd.Timedelta(weeks=h)).astype("datetime64[ns]")
        target = target.sort_values("_target").reset_index(drop=True)
        joined = pd.merge_asof(
            target,
            px_renamed,
            on="_target",
            by="Symbol",
            direction="backward",
            tolerance=tol,
        )
        joined = joined[["Symbol", "Date", "_fwd_px"]]
        merged = base.merge(joined, on=["Symbol", "Date"], how="left")
        out[f"fwd_ret_{h}w"] = merged["_fwd_px"] / merged["adjClose"] - 1.0

    ret_cols = [f"fwd_ret_{h}w" for h in horizons_weeks]
    out[ret_cols] = out[ret_cols].replace([np.inf, -np.inf], np.nan)
    return out


# --- main API -------------------------------------------------------------
def assemble_pit_panel(
    asic_long: pd.DataFrame,
    fundamentals: dict[str, pd.DataFrame],
    prices_long: pd.DataFrame,
    *,
    market_cap_long: pd.DataFrame | None = None,
    min_mkt_cap: float = DEFAULT_MIN_MKT_CAP_AUD,
    max_stale_quarters: int = DEFAULT_MAX_FILING_STALE_QUARTERS,
    forward_horizons_weeks: tuple[int, ...] = (1, 4, 12),
) -> pd.DataFrame:
    """Return the long PIT panel keyed by (Date, Ticker).

    See module docstring for the lookahead-control guarantees.

    The ``fundamentals`` dict must have the keys
    ``{'income_statement','balance_sheet','cash_flow','ratios','key_metrics',
    'enterprise_values','financial_growth'}``. Any missing endpoint is treated
    as an empty frame (the affected columns are simply absent in the output).
    """
    missing_core = [k for k in _CORE_ENDPOINTS if k not in fundamentals]
    if missing_core:
        raise ValueError(f"fundamentals dict missing required endpoints: {missing_core}")

    # --- ASIC frame: derive Symbol, normalise Date, keep declared schema ----
    asic = asic_long.copy()
    asic["Date"] = pd.to_datetime(asic["Date"]).dt.normalize()
    if "ReleaseDate" in asic.columns:
        asic["ReleaseDate"] = pd.to_datetime(asic["ReleaseDate"]).dt.normalize()
    asic["Symbol"] = asic["Ticker"].astype(str).str.upper().str.strip() + ".AX"

    # --- 4-business-day lag: swap Date and ReleaseDate ----------------------
    # ASIC publishes the Friday "release" reporting positions as-of the
    # PRIOR MONDAY (4 BDays earlier). The previous version of the pipeline
    # used the Monday "as-of" date as the trading anchor - meaning the
    # backtest implicitly assumed it could trade on Monday's adjClose
    # knowing the Monday position. That's a 4-BDay look-ahead: in real life
    # we only learn the Monday position when the report drops on Friday.
    #
    # Fix: use Friday (ReleaseDate) as the rebalance date. Prices and
    # forward returns join on Friday's close; positions are the as-of
    # Monday snapshot but tradable only at Friday. The original Monday
    # date is preserved as ``AsOfDate`` for diagnostic / audit purposes.
    if "ReleaseDate" in asic.columns:
        asic["AsOfDate"] = asic["Date"]      # Monday positions snapshot
        asic["Date"] = asic["ReleaseDate"]   # Friday tradable date

    # --- Prices: as-of attach ---------------------------------------------
    px_attach = _asof_join_price(asic, prices_long)
    panel = asic.merge(px_attach, on=["Symbol", "Date"], how="left")

    # --- Forward returns (labels only) ------------------------------------
    fwd = _forward_returns_from_prices(
        prices_long, panel, horizons_weeks=forward_horizons_weeks
    )
    panel = panel.merge(fwd, on=["Symbol", "Date"], how="left")

    # --- Fundamentals: anchor = income acceptedDate -----------------------
    income = fundamentals["income_statement"]
    income = _normalise_dates(income, ("date", "acceptedDate", "filingDate"))
    filings = _filings_with_availability(income)

    pit_keys = _asof_join_fundamentals(panel, filings)
    panel = panel.merge(
        pit_keys[
            [
                "Symbol",
                "Date",
                "fiscalYear",
                "period",
                "acceptedDate",
                "period_end",
                "filing_lag_days",
            ]
        ],
        on=["Symbol", "Date"],
        how="left",
    )

    # Attach each endpoint's payload by (symbol, fiscalYear, period) -- this
    # ensures derived endpoints (which lack acceptedDate) inherit the parent
    # filing's availability.
    panel = panel.rename(columns={"Symbol": "symbol"})
    for ep in (*_CORE_ENDPOINTS, *_DERIVED_ENDPOINTS):
        df = fundamentals.get(ep)
        if df is None or df.empty:
            continue
        df = _normalise_dates(df.copy(), ("date", "acceptedDate", "filingDate"))
        keep_cols = ("symbol", "fiscalYear", "period")
        missing = [k for k in keep_cols if k not in df.columns]
        if missing:
            # Endpoints lacking fiscalYear/period (e.g. enterprise_values) are
            # keyed only by period-end ``date``. As-of-join them on date instead
            # of skipping - this is how we pick up enterprise_values's
            # ``marketCapitalization`` field across history (the primary
            # mktCap source, since balance-sheet ``commonStockSharesOutstanding``
            # is stamped-at-latest-period-end and breaks across reverse splits).
            if "date" not in df.columns:
                logger.warning(
                    f"assemble: skipping endpoint '{ep}' - no fiscalYear/period AND no date column"
                )
                continue
            logger.info(
                f"assemble: as-of-joining endpoint '{ep}' on (symbol, date) since "
                f"it lacks {missing}"
            )
            df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
            df = df.drop(
                columns=[c for c in ("acceptedDate", "filingDate") if c in df.columns]
            )
            # merge_asof requires globally-sorted ``on`` keys on BOTH sides.
            # ``by`` (symbol) is only used to bucket; the on-key must be sorted.
            df = df.drop_duplicates(subset=["symbol", "date"]).sort_values("date").reset_index(drop=True)
            df = _prefix_cols(df, prefix=ep, keep=("symbol", "date"))
            # Build (symbol, date) sorted left side for merge_asof.
            left = panel[["symbol", "Date"]].copy()
            left["Date"] = pd.to_datetime(left["Date"]).dt.normalize().astype("datetime64[ns]")
            left["_row_order"] = np.arange(len(left))
            left_sorted = left.sort_values("Date").reset_index(drop=True)
            joined = pd.merge_asof(
                left_sorted, df.rename(columns={"date": "Date"}),
                on="Date", by="symbol", direction="backward",
            )
            joined = joined.sort_values("_row_order").reset_index(drop=True)
            # Append the new columns onto the panel (keyed by original row order).
            new_cols = [c for c in joined.columns if c.startswith(f"{ep}_")]
            for c in new_cols:
                panel[c] = joined[c].to_numpy()
            continue
        # Endpoints with the fiscal join key.
        df = df.drop(columns=[c for c in ("date", "acceptedDate", "filingDate") if c in df.columns])
        df = df.drop_duplicates(subset=list(keep_cols))
        df = _prefix_cols(df, prefix=ep, keep=keep_cols)
        panel = panel.merge(df, on=list(keep_cols), how="left")
    panel = panel.rename(columns={"symbol": "Symbol"})

    # --- Market cap (PIT) -------------------------------------------------
    # Primary source: FMP's `historical-market-capitalization` endpoint, which
    # is split-adjusted at the share-count level. This is the correct mktCap
    # for issuers that have been through reverse splits (e.g. Paladin Energy's
    # 100:1 in 2024) — the balance-sheet `commonStockSharesOutstanding` field
    # is stamped at period-end and is NOT back-adjusted, so combining it with
    # `adjClose` (which IS split-adjusted to today) produces a mktCap that is
    # off by the split factor for any pre-split period.
    bs_shares_col = "balance_sheet_commonStockSharesOutstanding"
    alt_shares_col = "balance_sheet_weightedAverageShsOut"
    if bs_shares_col in panel.columns:
        shares = panel[bs_shares_col]
    elif "income_statement_weightedAverageShsOut" in panel.columns:
        shares = panel["income_statement_weightedAverageShsOut"]
    elif alt_shares_col in panel.columns:
        shares = panel[alt_shares_col]
    else:
        shares = pd.Series(np.nan, index=panel.index, name="shares")
    panel["sharesOutstanding"] = pd.to_numeric(shares, errors="coerce")

    # Compute the shares-times-price fallback (last resort).
    mc_fallback = panel["sharesOutstanding"] * panel["adjClose"]

    # PRIMARY source: enterprise_values_marketCapitalization (just attached by
    # the as-of-join above). FMP records this at each fiscal period-end and it
    # is split-aware (FMP recomputes when the issuer reverse-splits), so it is
    # the correct historical mktCap for issuers like Paladin Energy that the
    # balance-sheet share-count approach mangles.
    ev_mcap_col = "enterprise_values_marketCapitalization"
    if ev_mcap_col in panel.columns:
        mc_ev = pd.to_numeric(panel[ev_mcap_col], errors="coerce")
        n_ev = int(mc_ev.notna().sum())
        logger.info(
            f"mktCap: enterprise_values.marketCapitalization covers {n_ev:,}/{len(panel):,} rows "
            f"({100*n_ev/max(len(panel),1):.1f}%)"
        )
    else:
        mc_ev = pd.Series(np.nan, index=panel.index, dtype="float64")

    # SECONDARY source: FMP's daily historical-market-capitalization endpoint
    # (only covers ~90d of history on most plans). Useful for the very latest
    # rebalances where enterprise_values is staler than the live mktCap.
    if market_cap_long is not None and not market_cap_long.empty:
        mc = market_cap_long.copy()
        # Normalise to the same (Symbol, Date) keys as the panel.
        if "symbol" in mc.columns:
            mc = mc.rename(columns={"symbol": "Symbol"})
        mc["Date"] = pd.to_datetime(mc["date"]).dt.normalize().astype("datetime64[ns]")
        mc = mc[["Symbol", "Date", "marketCap"]].dropna()
        mc = mc.sort_values(["Symbol", "Date"]).reset_index(drop=True)

        left = panel[["Symbol", "Date"]].copy()
        left["Date"] = pd.to_datetime(left["Date"]).dt.normalize().astype("datetime64[ns]")
        left = left.sort_values("Date").reset_index(drop=True)
        joined = pd.merge_asof(
            left,
            mc.sort_values("Date").reset_index(drop=True),
            on="Date",
            by="Symbol",
            direction="backward",
            tolerance=pd.Timedelta(days=PRICE_TOLERANCE_DAYS),
        )
        mc_fmp = joined.set_index(["Symbol", "Date"])["marketCap"]
        # Realign to the panel's row order.
        panel_idx = pd.MultiIndex.from_arrays([panel["Symbol"].values, panel["Date"].values])
        mc_hmc_aligned = pd.to_numeric(
            mc_fmp.reindex(panel_idx).reset_index(drop=True), errors="coerce"
        )
        n_hmc = int(mc_hmc_aligned.notna().sum())
        logger.info(
            f"mktCap: historical-market-cap endpoint covers {n_hmc:,}/{len(panel):,} rows "
            f"({100*n_hmc/max(len(panel),1):.1f}%) - daily, recent-history only"
        )
    else:
        mc_hmc_aligned = pd.Series(np.nan, index=panel.index, dtype="float64")

    # Compose: enterprise_values (primary, full history, split-aware) → daily
    # historical-market-cap (newer, more current) → shares*price (last resort).
    panel["mktCap"] = mc_ev.combine_first(mc_hmc_aligned).combine_first(mc_fallback)
    n_ev_used = int(mc_ev.notna().sum())
    n_hmc_used = int((mc_ev.isna() & mc_hmc_aligned.notna()).sum())
    n_fb_used = int((mc_ev.isna() & mc_hmc_aligned.isna() & mc_fallback.notna()).sum())
    logger.info(
        f"mktCap composition: enterprise_values={n_ev_used:,} | historical-market-cap={n_hmc_used:,} "
        f"| shares*price fallback={n_fb_used:,}"
    )

    # Physical-impossibility sanity: no ASX-listed company has ever reached
    # A$300B (peak BHP ≈ A$250B mid-2022). Any row above that is FMP source-
    # data error (e.g. enterprise_values reports pre-consolidation shares for
    # certain tickers in certain quarters). Null these rows rather than
    # silently feeding them into log_mktcap / liquidity / portfolio
    # construction. No ticker-specific hardcoding.
    sanity_cap_aud = 3e11  # A$300B
    bad_mask = panel["mktCap"] > sanity_cap_aud
    n_bad = int(bad_mask.sum())
    if n_bad:
        bad_tickers = panel.loc[bad_mask, "Ticker"].value_counts().head(5).to_dict()
        logger.warning(
            f"mktCap sanity: NaN'd {n_bad:,} rows > A$300B "
            f"(top affected tickers: {bad_tickers})"
        )
        panel.loc[bad_mask, "mktCap"] = np.nan

    # --- Investable flag --------------------------------------------------
    panel["filing_stale_quarters"] = panel["filing_lag_days"] / DAYS_PER_QUARTER
    has_price = panel["adjClose"].notna()
    fresh = panel["filing_lag_days"].notna() & (
        panel["filing_stale_quarters"] <= max_stale_quarters
    )
    big_enough = panel["mktCap"].notna() & (panel["mktCap"] >= min_mkt_cap)
    panel["investable"] = has_price & fresh & big_enough

    # --- Final column order: declared schema first, then fundamentals -----
    panel = panel.rename(columns={"Symbol": "Symbol"})
    declared = [
        "Date",          # Friday (release date) - the tradable rebalance date
        "AsOfDate",      # Monday (positions snapshot) - 4 BDays before Date
        "ReleaseDate",
        "Ticker",
        "Symbol",
        "Company",
        "sector",
        "industry",
        "ShortPositions",
        "TotalVolume",
        "ShortPct",
        "adjClose",
        "volume",
        "mktCap",
        "sharesOutstanding",
        *[f"fwd_ret_{h}w" for h in forward_horizons_weeks],
    ]
    # Add any declared cols that the input did not carry (e.g. sector if a
    # profile attach hasn't happened yet) as NaN so the schema is stable.
    for c in declared:
        if c not in panel.columns:
            panel[c] = np.nan

    trailing = [
        "fiscalYear",
        "period",
        "acceptedDate",
        "period_end",
        "filing_lag_days",
        "filing_stale_quarters",
        "investable",
    ]
    fundamental_cols = [
        c
        for c in panel.columns
        if c not in declared
        and c not in trailing
        and c
        not in {
            "Date",
            "Ticker",
            "Symbol",
        }
    ]
    col_order = declared + fundamental_cols + trailing
    # De-dupe while preserving order in case of accidental overlap.
    seen: set[str] = set()
    col_order = [c for c in col_order if not (c in seen or seen.add(c))]
    panel = panel[col_order]

    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    logger.info(
        f"PIT panel: {len(panel):,} rows | "
        f"{panel['Ticker'].nunique():,} tickers | "
        f"{panel['Date'].nunique():,} dates | "
        f"investable {panel['investable'].mean():.1%}"
    )
    return panel


def write_panel(panel: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the panel to parquet (default: ``data/processed/master_pit.parquet``)."""
    path = path or (settings.processed_dir / "master_pit.parquet")
    write_parquet(panel, path)
    logger.info(f"wrote {path} ({len(panel):,} rows, {panel.shape[1]} cols)")
    return path


def panel_quality_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Concise quality stats for the PIT panel.

    Returns a 2-column DataFrame ``[metric, value]`` covering row/date/ticker
    counts, investable share, median filing lag, per-year coverage and a sector
    breakdown. Values are stringified so heterogeneous metrics share one frame
    (the result is for logging / a README data-dictionary table, not for
    downstream modelling).
    """
    rows: list[tuple[str, str]] = []
    add = lambda k, v: rows.append((k, str(v)))  # noqa: E731 -- terse local helper

    add("n_rows", f"{len(panel):,}")
    add("n_dates", f"{panel['Date'].nunique():,}")
    add("n_tickers", f"{panel['Ticker'].nunique():,}")
    if "investable" in panel.columns and len(panel):
        add("pct_investable", f"{panel['investable'].mean():.1%}")
    if "adjClose" in panel.columns and len(panel):
        add("pct_with_price", f"{panel['adjClose'].notna().mean():.1%}")
    if "filing_lag_days" in panel.columns:
        med = panel["filing_lag_days"].median()
        add("median_filing_lag_days", f"{med:.0f}" if pd.notna(med) else "NaN")
    if "mktCap" in panel.columns and len(panel):
        med_mc = panel.loc[panel["investable"].fillna(False), "mktCap"].median()
        add(
            "median_investable_mktCap_AUD",
            f"{med_mc:,.0f}" if pd.notna(med_mc) else "NaN",
        )

    # Coverage by year.
    if "Date" in panel.columns and len(panel):
        years = panel.assign(year=panel["Date"].dt.year)
        by_year = years.groupby("year").size()
        for y, n in by_year.items():
            add(f"rows_{y}", f"{n:,}")

    # Sector breakdown (top 12 + 'Other').
    if "sector" in panel.columns and panel["sector"].notna().any():
        by_sec = (
            panel.dropna(subset=["sector"])
            .groupby("sector")
            .size()
            .sort_values(ascending=False)
        )
        head = by_sec.head(12)
        for s, n in head.items():
            add(f"sector_{s}", f"{n:,}")
        rest = int(by_sec.iloc[12:].sum())
        if rest:
            add("sector_other", f"{rest:,}")

    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:  # pragma: no cover -- glue, exercised by the pipeline driver
    """Driver: load inputs from disk, assemble, write, log quality summary."""
    settings.ensure_dirs()
    proc, raw = settings.processed_dir, settings.raw_dir

    asic = read_parquet(proc / "asic_long.parquet")
    prices = read_parquet(proc / "prices_long.parquet")

    fmp_dir = raw / "fmp_raw"
    fundamentals: dict[str, pd.DataFrame] = {}
    for ep in (*_CORE_ENDPOINTS, *_DERIVED_ENDPOINTS):
        path = fmp_dir / f"{ep}.parquet"
        if path.exists():
            fundamentals[ep] = read_parquet(path)
        else:
            logger.warning(f"{path} missing -- endpoint '{ep}' will be skipped")

    panel = assemble_pit_panel(asic, fundamentals, prices)
    write_panel(panel)
    logger.info("\n" + panel_quality_summary(panel).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
