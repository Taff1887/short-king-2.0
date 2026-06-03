"""Quality / profitability features.

Quality is the structural counterpart to value: a stock can look cheap *because*
its fundamentals are deteriorating. For a short book this matters double — the
canonical short setup is a name whose accruals are bloated, whose cash flow no
longer supports reported earnings, and whose margins are eroding. Each function
here computes one such signal as a clean ``pd.Series`` aligned to ``df.index``.

All signals are oriented **higher = better fundamental quality** so they can be
flipped uniformly into a short signal (low quality -> short). FMP exposes most
of these directly (``returnOnEquity``, ``grossProfitMargin``, ...); we prefer
those when present and fall back to first-principles construction from the raw
income / balance / cash-flow line items so the panel stays usable even on the
subset of rows where the derived datasets are missing.

The input DataFrame is the PIT master panel keyed by ``[date, symbol]``: each
row is a rebalance date with the most recent quarterly filing already as-of-
joined (no look-ahead). All ratios are therefore PIT by construction.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from short_king.utils.logging import logger


# --- internal helpers -----------------------------------------------------
def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise num/den with zero-denominator -> NaN, inf -> NaN."""
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series | None:
    """Return the first column in ``candidates`` that exists in ``df``, else None.

    FMP sometimes renames or duplicates fields across statement vs key-metrics
    vs ratios datasets; this lets callers list the preferred source first and
    fall back gracefully.
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


# --- profitability ratios -------------------------------------------------
def roe(df: pd.DataFrame) -> pd.Series:
    """Return on equity. Prefer FMP's pre-computed ``returnOnEquity``."""
    pre = _first_present(df, ("returnOnEquity",))
    if pre is not None:
        return pre.astype(float).replace([np.inf, -np.inf], np.nan)
    ni = _first_present(df, ("netIncome", "netIncomeCommon"))
    eq = _first_present(df, ("totalStockholdersEquity", "totalEquity"))
    if ni is None or eq is None:
        logger.warning("roe: missing netIncome or totalStockholdersEquity")
        return pd.Series(np.nan, index=df.index, name="roe")
    return _safe_div(ni.astype(float), eq.astype(float)).rename("roe")


def roic(df: pd.DataFrame) -> pd.Series:
    """Return on invested capital. Prefer FMP's pre-computed value.

    Falls back to NOPAT / (equity + total debt - cash), where NOPAT is
    approximated as operating income net of a 21% statutory tax. This is the
    conventional definition (e.g. Damodaran) and a sensible default when the
    vendor's precise effective-tax-rate computation is unavailable.
    """
    pre = _first_present(df, ("returnOnInvestedCapital", "roic",
                              "returnOnCapitalEmployed"))
    if pre is not None:
        return pre.astype(float).replace([np.inf, -np.inf], np.nan)
    op = _first_present(df, ("operatingIncome", "ebit"))
    eq = _first_present(df, ("totalStockholdersEquity", "totalEquity"))
    debt = _first_present(df, ("totalDebt",))
    cash = _first_present(df, ("cashAndCashEquivalents", "cashAndShortTermInvestments"))
    if op is None or eq is None or debt is None:
        logger.warning("roic: missing operatingIncome / equity / totalDebt")
        return pd.Series(np.nan, index=df.index, name="roic")
    nopat = op.astype(float) * (1.0 - 0.21)
    invested = eq.astype(float) + debt.astype(float)
    if cash is not None:
        invested = invested - cash.astype(float)
    return _safe_div(nopat, invested).rename("roic")


def roa(df: pd.DataFrame) -> pd.Series:
    """Return on assets. Prefer FMP's pre-computed ``returnOnAssets``."""
    pre = _first_present(df, ("returnOnAssets",))
    if pre is not None:
        return pre.astype(float).replace([np.inf, -np.inf], np.nan)
    ni = _first_present(df, ("netIncome", "netIncomeCommon"))
    ta = _first_present(df, ("totalAssets",))
    if ni is None or ta is None:
        logger.warning("roa: missing netIncome or totalAssets")
        return pd.Series(np.nan, index=df.index, name="roa")
    return _safe_div(ni.astype(float), ta.astype(float)).rename("roa")


def gross_margin(df: pd.DataFrame) -> pd.Series:
    """Gross profit / revenue. Prefer FMP's pre-computed margin."""
    pre = _first_present(df, ("grossProfitMargin", "grossMargin"))
    if pre is not None:
        return pre.astype(float).replace([np.inf, -np.inf], np.nan)
    gp = _first_present(df, ("grossProfit",))
    rev = _first_present(df, ("revenue",))
    if gp is None or rev is None:
        logger.warning("gross_margin: missing grossProfit or revenue")
        return pd.Series(np.nan, index=df.index, name="gross_margin")
    return _safe_div(gp.astype(float), rev.astype(float)).rename("gross_margin")


def operating_margin(df: pd.DataFrame) -> pd.Series:
    """Operating income / revenue. Prefer FMP's pre-computed margin."""
    pre = _first_present(df, ("operatingProfitMargin", "operatingMargin"))
    if pre is not None:
        return pre.astype(float).replace([np.inf, -np.inf], np.nan)
    op = _first_present(df, ("operatingIncome", "ebit"))
    rev = _first_present(df, ("revenue",))
    if op is None or rev is None:
        logger.warning("operating_margin: missing operatingIncome or revenue")
        return pd.Series(np.nan, index=df.index, name="operating_margin")
    return _safe_div(op.astype(float), rev.astype(float)).rename("operating_margin")


def net_margin(df: pd.DataFrame) -> pd.Series:
    """Net income / revenue. Prefer FMP's pre-computed margin."""
    pre = _first_present(df, ("netProfitMargin", "netMargin"))
    if pre is not None:
        return pre.astype(float).replace([np.inf, -np.inf], np.nan)
    ni = _first_present(df, ("netIncome", "netIncomeCommon"))
    rev = _first_present(df, ("revenue",))
    if ni is None or rev is None:
        logger.warning("net_margin: missing netIncome or revenue")
        return pd.Series(np.nan, index=df.index, name="net_margin")
    return _safe_div(ni.astype(float), rev.astype(float)).rename("net_margin")


# --- earnings quality -----------------------------------------------------
def accruals(df: pd.DataFrame) -> pd.Series:
    """Sloan-style accruals: (netIncome - cashFromOps) / totalAssets.

    Earnings that aren't backed by cash flow are a classic short signal — the
    accounting recognises revenue/earnings the business hasn't yet collected.
    High positive accruals = lower quality (and so this signal is oriented
    higher = WORSE; the panel below flips it to keep the convention
    higher = better).
    """
    ni = _first_present(df, ("netIncome", "netIncomeCommon"))
    cfo = _first_present(df, ("operatingCashFlow", "cashFromOps", "netCashProvidedByOperatingActivities"))
    ta = _first_present(df, ("totalAssets",))
    if ni is None or cfo is None or ta is None:
        logger.warning("accruals: missing netIncome / operatingCashFlow / totalAssets")
        return pd.Series(np.nan, index=df.index, name="accruals")
    return _safe_div(ni.astype(float) - cfo.astype(float), ta.astype(float)).rename("accruals")


def cfo_to_ni(df: pd.DataFrame) -> pd.Series:
    """Cash-quality of earnings: operatingCashFlow / netIncome.

    Healthy businesses convert reported earnings into cash; persistent CFO/NI
    well below 1 is a hallmark of earnings management. We return NaN where net
    income <= 0 because the ratio is meaningless (and unstable) for loss-makers
    — those names should be diagnosed by other signals (e.g. ROA, accruals)
    rather than by a noisy negative ratio.
    """
    cfo = _first_present(df, ("operatingCashFlow", "cashFromOps", "netCashProvidedByOperatingActivities"))
    ni = _first_present(df, ("netIncome", "netIncomeCommon"))
    if cfo is None or ni is None:
        logger.warning("cfo_to_ni: missing operatingCashFlow or netIncome")
        return pd.Series(np.nan, index=df.index, name="cfo_to_ni")
    ni_f = ni.astype(float)
    den = ni_f.where(ni_f > 0)
    return _safe_div(cfo.astype(float), den).rename("cfo_to_ni")


# --- panel ----------------------------------------------------------------
QUALITY_COMPONENTS: tuple[str, ...] = (
    "roe", "roic", "roa",
    "gross_margin", "operating_margin", "net_margin",
    "accruals", "cfo_to_ni",
)


def quality_features_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble all quality signals into one frame keyed like ``df``.

    Carries ``date`` and ``symbol`` (if present) so the result can be merged or
    concatenated alongside other feature families. ``accruals`` is flipped sign
    so that, like every other column, higher = better quality (less accrual
    distortion). All values are PIT — staleness control happens upstream when
    the master panel is assembled, not here.
    """
    key_cols = [c for c in ("date", "symbol") if c in df.columns]
    out = df[key_cols].copy() if key_cols else pd.DataFrame(index=df.index)

    out["roe"] = roe(df)
    out["roic"] = roic(df)
    out["roa"] = roa(df)
    out["gross_margin"] = gross_margin(df)
    out["operating_margin"] = operating_margin(df)
    out["net_margin"] = net_margin(df)
    out["accruals"] = -accruals(df)  # flip so higher = better (low accruals)
    out["cfo_to_ni"] = cfo_to_ni(df)

    feat_cols = list(QUALITY_COMPONENTS)
    cov = (out[feat_cols].notna().mean() * 100).round(1)
    logger.info(f"quality_features_panel: {len(out):,} rows | non-null %:\n{cov.to_string()}")
    return out
