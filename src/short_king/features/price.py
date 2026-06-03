"""Price-derived features for the weekly ASX short-selling panel.

Computed per ticker on the long point-in-time panel (one row per Ticker x Date).
The panel cadence is weekly (Friday-anchored), so every window argument is in
*weeks*. All functions group by ``Ticker`` to guarantee that rolling windows
never leak across names, and the panel is sorted by ``[Ticker, Date]`` first so
``shift`` / ``rolling`` operate on a clean time-ordered series.

Public API:
    momentum(df, weeks)              -> Series indexed like df
    momentum_skip_1w(df, weeks)      -> Series (skip-1w momentum, used to neutralise
                                       1-week microstructure reversal)
    realised_vol(df, window)         -> Series (std of weekly returns)
    max_drawdown_52w(df, window=52)  -> Series (rolling 52w max drawdown, <= 0)
    beta_to_market(df, market_col, window) -> Series (rolling beta on weekly returns)
    price_features_panel(df)         -> DataFrame copy of df with the full feature set
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger

# Required input columns. ``mkt_ret_1w`` is optional; if missing, beta is skipped.
_TICKER = "Ticker"
_DATE = "Date"
_PRICE = "adjClose"


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Internal helper: stable sort by [Ticker, Date]. Returns a view (no copy)."""
    return df.sort_values([_TICKER, _DATE], kind="mergesort")


def momentum(df: pd.DataFrame, *, weeks: int) -> pd.Series:
    """``weeks``-week trailing return per ticker (``adjClose.pct_change(weeks)``).

    Result is reindexed to ``df.index`` so the caller can assign it as a column
    without surprises from the groupby reordering.
    """
    s = _sorted(df).groupby(_TICKER, sort=False)[_PRICE].pct_change(weeks)
    return s.reindex(df.index)


def momentum_skip_1w(df: pd.DataFrame, *, weeks: int) -> pd.Series:
    """Skip-1-week momentum: ``adjClose[t-1] / adjClose[t-weeks-1] - 1``.

    Skipping the most recent week removes the well-known short-term reversal
    effect that contaminates raw 12-week momentum.
    """
    g = _sorted(df).groupby(_TICKER, sort=False)[_PRICE]
    s = g.shift(1) / g.shift(weeks + 1) - 1.0
    return s.reindex(df.index)


def realised_vol(df: pd.DataFrame, *, window: int = 4) -> pd.Series:
    """Standard deviation of weekly returns over a trailing ``window`` per ticker.

    Uses ``min_periods = max(2, window // 2)`` so very short histories still
    yield a value when half the window is observed.
    """
    sdf = _sorted(df)
    rets = sdf.groupby(_TICKER, sort=False)[_PRICE].pct_change(1)
    min_periods = max(2, window // 2)
    vol = rets.groupby(sdf[_TICKER], sort=False).transform(
        lambda x: x.rolling(window, min_periods=min_periods).std()
    )
    return vol.reindex(df.index)


def max_drawdown_52w(df: pd.DataFrame, *, window: int = 52) -> pd.Series:
    """Rolling drawdown over a trailing ``window`` weeks: ``price / running_max - 1``.

    Values are in [-1, 0]: 0 means at a window high, -0.40 means 40% off the
    trailing high. Reported as a *level* (not the min over the window), which
    is what callers want for a same-week feature.
    """
    sdf = _sorted(df)
    min_periods = max(2, window // 4)

    def _dd(s: pd.Series) -> pd.Series:
        roll_max = s.rolling(window, min_periods=min_periods).max()
        return s / roll_max - 1.0

    out = sdf.groupby(_TICKER, sort=False)[_PRICE].transform(_dd)
    return out.reindex(df.index)


def beta_to_market(
    df: pd.DataFrame,
    *,
    market_col: str = "mkt_ret_1w",
    window: int = 52,
) -> pd.Series:
    """Rolling beta of weekly returns vs. a market-return column on the panel.

    Beta = Cov(stock_ret, mkt_ret) / Var(mkt_ret) over the trailing ``window``.
    The market series must already be joined onto the panel as ``market_col``
    (same weekly index per row). If absent, returns an all-NaN series.
    """
    if market_col not in df.columns:
        logger.debug(f"beta_to_market: '{market_col}' missing - returning NaN series")
        return pd.Series(np.nan, index=df.index, name=f"beta_{window}w")

    sdf = _sorted(df).copy()
    sdf["_ret"] = sdf.groupby(_TICKER, sort=False)[_PRICE].pct_change(1)
    sdf["_mkt"] = pd.to_numeric(sdf[market_col], errors="coerce")
    min_periods = max(8, window // 2)

    def _beta(block: pd.DataFrame) -> pd.Series:
        cov = block["_ret"].rolling(window, min_periods=min_periods).cov(block["_mkt"])
        var = block["_mkt"].rolling(window, min_periods=min_periods).var()
        return cov / var.replace(0.0, np.nan)

    beta = (
        sdf.groupby(_TICKER, sort=False, group_keys=False)[["_ret", "_mkt"]]
        .apply(_beta)
        .replace([np.inf, -np.inf], np.nan)
    )
    return beta.reindex(df.index)


def price_features_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all price-derived features attached.

    Adds: ``mom_1w``, ``mom_4w``, ``mom_12w``, ``mom_26w``, ``mom_52w``,
    ``mom_12w_skip1``, ``vol_4w``, ``vol_12w``, ``drawdown_52w``, and
    ``beta_52w`` (only if ``mkt_ret_1w`` is present on the input panel).

    The input must carry ``Ticker``, ``Date`` and ``adjClose``. The output
    preserves the caller's row order; sorting is done internally for the
    window calculations only.
    """
    missing = {_TICKER, _DATE, _PRICE} - set(df.columns)
    if missing:
        raise KeyError(f"price_features_panel: panel missing required columns {sorted(missing)}")

    out = df.copy()
    out["mom_1w"] = momentum(out, weeks=1)
    out["mom_4w"] = momentum(out, weeks=4)
    out["mom_12w"] = momentum(out, weeks=12)
    out["mom_26w"] = momentum(out, weeks=26)
    out["mom_52w"] = momentum(out, weeks=52)
    out["mom_12w_skip1"] = momentum_skip_1w(out, weeks=12)
    out["vol_4w"] = realised_vol(out, window=4)
    out["vol_12w"] = realised_vol(out, window=12)
    out["drawdown_52w"] = max_drawdown_52w(out, window=52)

    if "mkt_ret_1w" in out.columns:
        out["beta_52w"] = beta_to_market(out, market_col="mkt_ret_1w", window=52)
    else:
        logger.info("price_features_panel: 'mkt_ret_1w' absent - skipping beta_52w")

    n_tkr = out[_TICKER].nunique()
    logger.info(
        f"price features: {len(out):,} rows x {n_tkr} tickers | "
        f"added mom/vol/drawdown" + (" + beta_52w" if "beta_52w" in out.columns else "")
    )
    return out


__all__ = [
    "momentum",
    "momentum_skip_1w",
    "realised_vol",
    "max_drawdown_52w",
    "beta_to_market",
    "price_features_panel",
]
