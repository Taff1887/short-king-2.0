"""Balance-sheet leverage and growth features.

Short-sellers gravitate to *fragile* balance sheets and *unsustainable* growth.
Each function here turns one raw fundamental concept into a single Series that
is later cross-sectionally ranked alongside valuation / quality / momentum
inputs in the short-candidacy composite.

All functions operate on a tidy PIT panel keyed by ``[date, symbol]`` (the
output of ``data.assemble.assemble_master``-style joining) and return a
``pd.Series`` aligned to the input DataFrame's index. Where FMP already
publishes the canonical ratio (``debtToEquityRatio``, ``currentRatio``,
``interestCoverageRatio``, ``revenueGrowth``, ``epsgrowth`` ...) we prefer
that vendor value — it is computed from the same filing and avoids divergence
between equivalent definitions; otherwise we compute from primitives.

Orientation convention (for downstream ranking): each helper returns the
*natural* value of the ratio. The short-side scoring layer is responsible for
flipping signs (high leverage / low coverage / high asset growth -> high
short score). Keeping signs natural here means the same Series is reusable in
long/short and long-only contexts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Fetch a column or a NaN-filled Series of the right length/index."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that masks zero/near-zero denominators."""
    den = den.where(den.abs() > 1e-12)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _prefer_vendor(vendor: pd.Series, computed: pd.Series) -> pd.Series:
    """Use the vendor column where present; fall back to ``computed`` otherwise."""
    return vendor.where(vendor.notna(), computed)


def _yoy_growth(df: pd.DataFrame, level_col: str) -> pd.Series:
    """Compute trailing-4-quarter YoY growth from a level series, in-panel.

    Used only when the FMP financial-growth endpoint did not publish the
    corresponding ``*Growth`` field. Groups by ``symbol`` and shifts 4 quarterly
    filings back; the panel must already be ordered chronologically per symbol
    (the PIT join in ``assemble.assemble_master`` preserves filing order).
    """
    if level_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    s = pd.to_numeric(df[level_col], errors="coerce")
    # Order within symbol by the panel's date column (or period_end) so the
    # shift(4) is a true t-4 quarters back even if the caller didn't pre-sort.
    sort_key = "date" if "date" in df.columns else ("period_end" if "period_end" in df.columns else None)
    if sort_key is not None:
        order = df.sort_values(["symbol", sort_key]).index
        s = s.loc[order]
    prev = s.groupby(df.loc[s.index, "symbol"]).shift(4)
    out = _safe_div(s - prev, prev.abs())
    return out.reindex(df.index)


# --------------------------------------------------------------------------
# Leverage
# --------------------------------------------------------------------------
def debt_equity(df: pd.DataFrame) -> pd.Series:
    """Total debt / total stockholders' equity. Prefer FMP ``debtToEquityRatio``.

    High D/E means a small earnings shock wipes out a large slice of equity —
    classic short-side fragility. Falls back to ``totalDebt / totalStockhold-
    ersEquity`` when the vendor ratio is missing.
    """
    computed = _safe_div(_col(df, "totalDebt"), _col(df, "totalStockholdersEquity"))
    return _prefer_vendor(_col(df, "debtToEquityRatio"), computed).rename("debt_equity")


def net_debt_to_ebitda(df: pd.DataFrame) -> pd.Series:
    """Net debt / EBITDA — how many years of EBITDA service the net obligation.

    FMP does not publish this directly on ``ratios``, so it is computed from
    ``netDebt`` (balance-sheet) and ``ebitda`` (income-statement). When
    ``netDebt`` is absent we approximate it as ``totalDebt - cashAndCash-
    Equivalents`` from the same filing.
    """
    nd_direct = _col(df, "netDebt")
    nd_fallback = _col(df, "totalDebt") - _col(df, "cashAndCashEquivalents")
    net_debt = nd_direct.where(nd_direct.notna(), nd_fallback)
    return _safe_div(net_debt, _col(df, "ebitda")).rename("net_debt_to_ebitda")


def interest_coverage(df: pd.DataFrame) -> pd.Series:
    """EBIT / interest expense. Prefer FMP ``interestCoverageRatio``.

    Below ~1.5x is a classic distress signal. ``operatingIncome`` is the EBIT
    proxy used by FMP itself; ``interestExpense`` is taken positive (it is
    reported as a positive number on income statements). When both fundamen-
    tals exist but the vendor ratio is missing we compute it.
    """
    ebit = _col(df, "operatingIncome")
    intexp = _col(df, "interestExpense").abs()
    computed = _safe_div(ebit, intexp)
    return _prefer_vendor(_col(df, "interestCoverageRatio"), computed).rename("interest_coverage")


def current_ratio(df: pd.DataFrame) -> pd.Series:
    """Current assets / current liabilities. Prefer FMP ``currentRatio``.

    Below 1.0 means working-capital fragility — a near-term cash crunch is
    one shock away. Falls back to ``totalCurrentAssets / totalCurrentLia-
    bilities`` when the vendor ratio is missing.
    """
    computed = _safe_div(_col(df, "totalCurrentAssets"), _col(df, "totalCurrentLiabilities"))
    return _prefer_vendor(_col(df, "currentRatio"), computed).rename("current_ratio")


# --------------------------------------------------------------------------
# Growth
# --------------------------------------------------------------------------
def revenue_growth_yoy(df: pd.DataFrame) -> pd.Series:
    """Year-over-year revenue growth. Prefer FMP ``revenueGrowth``.

    Computed in-panel from quarterly ``revenue`` only as a fallback (FMP's
    financial-growth endpoint already does this from the underlying filings).
    """
    return _prefer_vendor(_col(df, "revenueGrowth"), _yoy_growth(df, "revenue")).rename(
        "revenue_growth_yoy"
    )


def eps_growth_yoy(df: pd.DataFrame) -> pd.Series:
    """Year-over-year diluted-EPS growth. Prefer FMP ``epsgrowth``.

    EPS is a signed quantity that can cross zero (a swing from a small loss
    to a small profit gives a deceptive growth-rate), so the rank layer
    downstream applies winsorisation before composing.
    """
    return _prefer_vendor(_col(df, "epsgrowth"), _yoy_growth(df, "epsDiluted")).rename(
        "eps_growth_yoy"
    )


def asset_growth_yoy(df: pd.DataFrame) -> pd.Series:
    """Year-over-year total-asset growth — high values are a known short signal.

    Cooper, Gulen & Schill (2008) document the *asset growth anomaly*: firms
    that aggressively expand their balance sheet (via M&A, capex, working-
    capital build) systematically underperform. FMP does not publish a
    canonical ``totalAssetsGrowth`` on the financial-growth endpoint we use,
    so this is computed from quarterly ``totalAssets`` levels.

    TODO: when/if FMP exposes ``growthTotalAssets`` (legacy v3 endpoint), prefer
    it the same way the other growth helpers do.
    """
    vendor = _col(df, "growthTotalAssets")  # not in our standard pull, but harmless
    return _prefer_vendor(vendor, _yoy_growth(df, "totalAssets")).rename("asset_growth_yoy")


# --------------------------------------------------------------------------
# Panel builder
# --------------------------------------------------------------------------
_FEATURE_BUILDERS = {
    "debt_equity": debt_equity,
    "net_debt_to_ebitda": net_debt_to_ebitda,
    "interest_coverage": interest_coverage,
    "current_ratio": current_ratio,
    "revenue_growth_yoy": revenue_growth_yoy,
    "eps_growth_yoy": eps_growth_yoy,
    "asset_growth_yoy": asset_growth_yoy,
}


def leverage_growth_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all leverage + growth features in one pass over a PIT panel.

    Expects a tidy panel keyed by ``[date, symbol]`` (the output of the data-
    assembly stage). Returns a frame carrying the two keys plus each named
    feature; rows are returned in input order so the caller can ``merge`` on
    ``[date, symbol]`` without re-sorting the master panel.
    """
    keys = [k for k in ("date", "symbol") if k in df.columns]
    if {"date", "symbol"} - set(keys):
        raise KeyError(
            "leverage_growth_panel requires a panel keyed by ['date', 'symbol']; "
            f"got columns {list(df.columns)[:10]}..."
        )

    out = df[keys].copy()
    for name, fn in _FEATURE_BUILDERS.items():
        out[name] = fn(df).to_numpy()

    cov = (out[list(_FEATURE_BUILDERS)].notna().mean() * 100).round(1)
    logger.info(
        f"leverage_growth_panel: {len(out):,} rows | feature coverage (%):\n{cov.to_string()}"
    )
    return out.reset_index(drop=True)
