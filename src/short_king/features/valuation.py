"""Valuation features from the lagged-fundamentals PIT panel.

These are level measures of how richly a name is priced — earnings yield, FCF
yield, EV/EBITDA, P/E, P/B, P/S, sales yield. For shorts, expensive (low yield /
high multiple) is the bearish tilt, but the *direction* is the model's call;
this module just emits the raw measure with no sign flips. Where FMP already
ships a canonical ratio column we use it; where it doesn't, we derive from
primitives (e.g. EV/EBITDA from ``enterpriseValue`` and ``ebitda``).

Inputs are read directly off the PIT panel (filing-date-lagged), so there is no
look-ahead. Missing columns degrade gracefully to an all-NaN series with a
single warning rather than a flood of per-row logs.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from short_king.utils.logging import logger

# Canonical FMP column names. Centralised so a vendor rename is a one-line fix.
COL_ENTERPRISE_VALUE = "enterpriseValue"
COL_EBITDA = "ebitda"
COL_MARKET_CAP = "marketCap"
COL_NET_INCOME = "netIncome"
COL_REVENUE = "revenue"
COL_FREE_CASH_FLOW = "freeCashFlow"

COL_PE = "priceEarningsRatio"
COL_PB = "priceToBookRatio"
COL_PS = "priceToSalesRatio"
COL_EV_EBITDA = "evToEBITDA"
COL_FCF_YIELD = "freeCashFlowYield"
COL_EARNINGS_YIELD = "earningsYield"

# Track which (feature, missing-cols) warnings we've already logged so we don't
# spam the log on each call within a session.
_WARNED: set[tuple[str, tuple[str, ...]]] = set()


def _nan_like(df: pd.DataFrame, name: str) -> pd.Series:
    """Return an all-NaN float Series aligned to ``df.index`` with a stable name."""
    return pd.Series(np.nan, index=df.index, dtype="float64", name=name)


def _warn_missing(feature: str, missing: Iterable[str]) -> None:
    """Log a single warning per (feature, missing-set) — quiet on re-entry."""
    key = (feature, tuple(sorted(missing)))
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(f"valuation.{feature}: missing column(s) {sorted(missing)} — returning NaN")


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise divide that maps zero/negative denominators and infinities to NaN.

    Negative denominators are masked because most valuation ratios are
    economically meaningless (or worse, sign-flipped) when the denominator goes
    through zero — e.g. EV/EBITDA when EBITDA<0, or earnings yield when net
    income<0 we keep (loss-makers are valid shorts), but the *price* primitives
    (market cap) we don't expect to ever be non-positive.
    """
    d = denom.where(denom > 0)
    return (numer / d).replace([np.inf, -np.inf], np.nan)


# --- Individual measures ---------------------------------------------------
def ev_ebitda(df: pd.DataFrame) -> pd.Series:
    """EV / EBITDA. Prefer FMP's column, else derive from ``enterpriseValue/ebitda``."""
    if COL_EV_EBITDA in df.columns:
        return df[COL_EV_EBITDA].replace([np.inf, -np.inf], np.nan).rename("ev_ebitda")
    missing = [c for c in (COL_ENTERPRISE_VALUE, COL_EBITDA) if c not in df.columns]
    if missing:
        _warn_missing("ev_ebitda", missing)
        return _nan_like(df, "ev_ebitda")
    return _safe_div(df[COL_ENTERPRISE_VALUE], df[COL_EBITDA]).rename("ev_ebitda")


def pe(df: pd.DataFrame) -> pd.Series:
    """Price-to-earnings. Prefer FMP's column, else marketCap / netIncome."""
    if COL_PE in df.columns:
        return df[COL_PE].replace([np.inf, -np.inf], np.nan).rename("pe")
    missing = [c for c in (COL_MARKET_CAP, COL_NET_INCOME) if c not in df.columns]
    if missing:
        _warn_missing("pe", missing)
        return _nan_like(df, "pe")
    return _safe_div(df[COL_MARKET_CAP], df[COL_NET_INCOME]).rename("pe")


def pb(df: pd.DataFrame) -> pd.Series:
    """Price-to-book. Direct column only — no clean primitive fallback in the panel."""
    if COL_PB in df.columns:
        return df[COL_PB].replace([np.inf, -np.inf], np.nan).rename("pb")
    _warn_missing("pb", [COL_PB])
    return _nan_like(df, "pb")


def ps(df: pd.DataFrame) -> pd.Series:
    """Price-to-sales. Prefer FMP's column, else marketCap / revenue."""
    if COL_PS in df.columns:
        return df[COL_PS].replace([np.inf, -np.inf], np.nan).rename("ps")
    missing = [c for c in (COL_MARKET_CAP, COL_REVENUE) if c not in df.columns]
    if missing:
        _warn_missing("ps", missing)
        return _nan_like(df, "ps")
    return _safe_div(df[COL_MARKET_CAP], df[COL_REVENUE]).rename("ps")


def fcf_yield(df: pd.DataFrame) -> pd.Series:
    """Free-cash-flow yield = freeCashFlow / marketCap."""
    if COL_FCF_YIELD in df.columns:
        return df[COL_FCF_YIELD].replace([np.inf, -np.inf], np.nan).rename("fcf_yield")
    missing = [c for c in (COL_FREE_CASH_FLOW, COL_MARKET_CAP) if c not in df.columns]
    if missing:
        _warn_missing("fcf_yield", missing)
        return _nan_like(df, "fcf_yield")
    return _safe_div(df[COL_FREE_CASH_FLOW], df[COL_MARKET_CAP]).rename("fcf_yield")


def earnings_yield(df: pd.DataFrame) -> pd.Series:
    """Earnings yield = netIncome / marketCap (signed; loss-makers remain negative)."""
    if COL_EARNINGS_YIELD in df.columns:
        return df[COL_EARNINGS_YIELD].replace([np.inf, -np.inf], np.nan).rename("earnings_yield")
    missing = [c for c in (COL_NET_INCOME, COL_MARKET_CAP) if c not in df.columns]
    if missing:
        _warn_missing("earnings_yield", missing)
        return _nan_like(df, "earnings_yield")
    # netIncome can be negative; we still keep the sign (don't use _safe_div on numerator).
    mcap = df[COL_MARKET_CAP].where(df[COL_MARKET_CAP] > 0)
    return (df[COL_NET_INCOME] / mcap).replace([np.inf, -np.inf], np.nan).rename("earnings_yield")


def sales_yield(df: pd.DataFrame) -> pd.Series:
    """Sales yield = revenue / marketCap (the inverse of P/S)."""
    missing = [c for c in (COL_REVENUE, COL_MARKET_CAP) if c not in df.columns]
    if missing:
        _warn_missing("sales_yield", missing)
        return _nan_like(df, "sales_yield")
    return _safe_div(df[COL_REVENUE], df[COL_MARKET_CAP]).rename("sales_yield")


# --- Panel builder ---------------------------------------------------------
def valuation_features_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``ev_ebitda, pe, pb, ps, fcf_yield, earnings_yield, sales_yield`` to ``df``.

    Returns a *new* DataFrame (the input is not mutated). Each feature column is
    computed independently, so a missing input only knocks out the columns it
    feeds — the rest still populate.
    """
    out = df.copy()
    out["ev_ebitda"] = ev_ebitda(df).values
    out["pe"] = pe(df).values
    out["pb"] = pb(df).values
    out["ps"] = ps(df).values
    out["fcf_yield"] = fcf_yield(df).values
    out["earnings_yield"] = earnings_yield(df).values
    out["sales_yield"] = sales_yield(df).values
    return out


__all__ = [
    "ev_ebitda",
    "pe",
    "pb",
    "ps",
    "fcf_yield",
    "earnings_yield",
    "sales_yield",
    "valuation_features_panel",
]
