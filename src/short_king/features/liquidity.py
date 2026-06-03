"""Liquidity and size features for the weekly ASX short-selling panel.

Liquidity matters in a short book for two reasons that the EDA in v1 made obvious:
* you can only borrow what others trade, so cheap-to-borrow names cluster in the
  liquid tail of the universe;
* the Amihud (2002) price-impact proxy is the cleanest single measure of how much
  a dollar of flow moves the tape, and it dominates simple turnover once size is
  controlled for.

Each function is a pure transform of the assembled weekly panel keyed by
``[date, symbol]``. By convention (see ``data/assemble.py``) the panel carries
``volume`` as the **trailing-period average daily share volume** (not a weekly
sum), so ``adv_aud = close * volume`` is already in *AUD/day* units. If a panel
predates that convention and only carries a weekly sum (``weekly_volume``), we
fall back to ``weekly_volume / 5`` to recover the daily rate.

All rolling windows are computed *per ticker*, look back only, and never cross
the symbol boundary - no look-ahead, no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger

__all__ = [
    "adv_aud",
    "turnover_pct",
    "amihud_illiquidity",
    "log_mktcap",
    "liquidity_features_panel",
]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _require(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"liquidity features require columns: {missing}")


def _adv_shares(df: pd.DataFrame) -> pd.Series:
    """Average daily share volume.

    Prefers ``volume`` (already the trailing daily average per the panel
    convention); falls back to ``weekly_volume / 5`` for legacy panels.
    """
    if "volume" in df.columns:
        return pd.to_numeric(df["volume"], errors="coerce")
    if "weekly_volume" in df.columns:
        return pd.to_numeric(df["weekly_volume"], errors="coerce") / 5.0
    raise KeyError("liquidity features need either 'volume' or 'weekly_volume'")


def _weekly_return(df: pd.DataFrame) -> pd.Series:
    """The weekly return column. Prefers a precomputed ``ret_1w`` else derives it."""
    if "ret_1w" in df.columns:
        return pd.to_numeric(df["ret_1w"], errors="coerce")
    if "close" in df.columns:
        return (
            df.sort_values(["symbol", "date"])
            .groupby("symbol")["close"]
            .pct_change()
        )
    raise KeyError("amihud needs either 'ret_1w' or 'close' to derive weekly returns")


def _per_ticker_rolling_mean(
    df: pd.DataFrame, series: pd.Series, window: int
) -> pd.Series:
    """Rolling mean over ``window`` weeks per ticker, no cross-ticker contamination.

    ``min_periods=max(2, window // 4)`` keeps short histories usable while still
    requiring enough observations to avoid noisy single-week values.
    """
    s = series.copy()
    s.index = df.index  # ensure alignment after groupby.transform
    minp = max(2, window // 4)
    return (
        s.groupby(df["symbol"])
        .transform(lambda x: x.rolling(window, min_periods=minp).mean())
    )


# --------------------------------------------------------------------------
# Public features
# --------------------------------------------------------------------------
def adv_aud(df: pd.DataFrame, *, window: int = 20) -> pd.Series:
    """Average daily dollar volume in AUD, trailing ``window`` weeks per ticker.

    Built as ``close * average_daily_shares`` then smoothed. The smoothing
    suppresses one-off block-trade spikes that would otherwise let a single
    crossing print dominate the liquidity score for a month.
    """
    _require(df, ("symbol", "date", "close"))
    daily_aud = pd.to_numeric(df["close"], errors="coerce") * _adv_shares(df)
    daily_aud = daily_aud.replace([np.inf, -np.inf], np.nan)
    out = _per_ticker_rolling_mean(df, daily_aud, window)
    out.name = "adv_aud"
    return out


def turnover_pct(df: pd.DataFrame, *, window: int = 20) -> pd.Series:
    """Average daily turnover as a fraction of shares outstanding.

    ``adv_shares / sharesOutstanding`` smoothed over ``window`` weeks. Size-free
    by construction - useful for separating "small but actively traded" from
    "small and ignored".
    """
    _require(df, ("symbol", "sharesOutstanding"))
    shares = _adv_shares(df)
    so = pd.to_numeric(df["sharesOutstanding"], errors="coerce").replace(0, np.nan)
    ratio = (shares / so).replace([np.inf, -np.inf], np.nan)
    out = _per_ticker_rolling_mean(df, ratio, window)
    out.name = "turnover_pct"
    return out


def amihud_illiquidity(df: pd.DataFrame, *, window: int = 20) -> pd.Series:
    """Amihud (2002) illiquidity: ``|ret| / dollar_volume`` averaged over ``window`` weeks.

    Higher = more price impact per dollar traded = less liquid. We compute the
    weekly ratio first then average, which matches Amihud's daily-then-average
    construction at weekly resolution. Zero-volume weeks blow the ratio up; we
    map those to NaN so they don't poison the rolling mean.
    """
    _require(df, ("symbol", "date", "close"))
    ret = _weekly_return(df).abs()
    daily_aud = pd.to_numeric(df["close"], errors="coerce") * _adv_shares(df)
    dv = daily_aud.where(daily_aud > 0)  # 0 -> NaN to avoid inf
    weekly_amihud = (ret / dv).replace([np.inf, -np.inf], np.nan)
    out = _per_ticker_rolling_mean(df, weekly_amihud, window)
    out.name = "amihud"
    return out


def log_mktcap(df: pd.DataFrame) -> pd.Series:
    """``log(marketCap)`` - the standard size feature.

    Logging is mandatory: ASX market caps span ~five orders of magnitude, so the
    raw value would let mega-caps dominate any linear model. Non-positive caps
    (rare data glitches) become NaN rather than -inf.
    """
    _require(df, ("marketCap",))
    mc = pd.to_numeric(df["marketCap"], errors="coerce")
    mc = mc.where(mc > 0)
    out = np.log(mc)
    out.name = "log_mktcap"
    return out


def liquidity_features_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``adv_aud``, ``turnover_pct``, ``amihud`` and ``log_mktcap`` to the panel.

    Returns a copy with the four new columns added. The input is expected to be
    the weekly ``[date, symbol]`` panel produced by ``data.assemble``; ordering
    is preserved on the way out.
    """
    _require(df, ("symbol", "date"))
    out = df.sort_values(["symbol", "date"]).copy()

    out["adv_aud"] = adv_aud(out)
    out["turnover_pct"] = turnover_pct(out)
    out["amihud"] = amihud_illiquidity(out)
    out["log_mktcap"] = log_mktcap(out)

    cov = out[["adv_aud", "turnover_pct", "amihud", "log_mktcap"]].notna().mean()
    logger.info(
        "liquidity features attached | non-null %: "
        + ", ".join(f"{c}={cov[c] * 100:.1f}" for c in cov.index)
    )
    # Restore caller's row order (sort by date then symbol is the panel convention).
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
