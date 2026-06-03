"""Short-interest derived features.

This module turns the raw ASIC short-interest series (``ShortPct``,
``ShortPositions``) plus price/volume context into the family of signals the
v1 short-king prototype relied on:

* a clean **free-float-based** short interest (denominator = freely tradeable
  shares, not just shares outstanding) — institutionally the standard,
* **days-to-cover** — the squeeze proxy traders actually quote,
* a rolling **52-week z-score and percentile** of short interest — captures
  how stretched the current short book is *relative to that name's own
  history*, which matters far more than the raw level (some names are
  structurally heavily-shorted),
* a short-term **persistence** count and a **4-week build rate** — capture
  *direction and conviction* of the short book, not just level.

All rolling/diff features assume the input is sorted by ``[Ticker, Date]`` and
always groupby ``Ticker`` so we never bleed information across names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger

__all__ = [
    "short_pct_free_float",
    "days_to_cover",
    "si_z_52w",
    "si_percentile_52w",
    "si_persistence",
    "si_build_rate_4w",
    "short_signals_panel",
]

# Canonical column names on master_clean.parquet.
_TICKER = "Ticker"
_DATE = "Date"
_SHORT_POS = "ShortPositions"
_SHORT_PCT = "ShortPct"
_VOLUME = "volume"
_SHARES_OUT = "sharesOutstanding"
# Optional float / insider columns. May or may not be present depending on
# vendor coverage (FMP doesn't always carry these for ASX names). The
# free-float estimator handles absence cleanly.
_FLOAT_SHARES = "floatShares"
_INSIDER_PCT = "heldPercentInsiders"


def _require(df: pd.DataFrame, cols: list[str], func_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{func_name}: input DataFrame missing required column(s) {missing}"
        )


# --- 1. Short Interest as % of Free Float --------------------------------
def short_pct_free_float(df: pd.DataFrame) -> pd.Series:
    """``ShortPositions / floatShares`` (%), with sharesOutstanding fallback.

    Free float — the portion of shares that actually changes hands — is the
    economically correct denominator for short interest: a 5% short of the
    *whole* register can be a 15% short of the *float* when insiders hold a
    third of the company. We prefer ``floatShares`` when available and fall
    back to ``sharesOutstanding * (1 - insider %)``, then to plain
    ``sharesOutstanding`` if no insider data exists either.

    Returned series is in percent (consistent with the ASIC ``ShortPct``
    convention used everywhere else in the project) and indexed like the
    input. Division-by-zero / missing-denominator rows are NaN.
    """
    _require(df, [_SHORT_POS], "short_pct_free_float")

    short_pos = pd.to_numeric(df[_SHORT_POS], errors="coerce")

    float_shares = (
        pd.to_numeric(df[_FLOAT_SHARES], errors="coerce")
        if _FLOAT_SHARES in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    shares_out = (
        pd.to_numeric(df[_SHARES_OUT], errors="coerce")
        if _SHARES_OUT in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    insider_pct = (
        pd.to_numeric(df[_INSIDER_PCT], errors="coerce")
        if _INSIDER_PCT in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Fallback 1: shares outstanding * (1 - insider %). Insider % is treated
    # as a fraction in [0, 1]; clip to that range to defend against bad data.
    insider_frac = insider_pct.clip(lower=0.0, upper=1.0).fillna(0.0)
    fallback_float = shares_out * (1.0 - insider_frac)

    denom = float_shares.where(float_shares.gt(0), fallback_float)
    # Fallback 2: bare shares outstanding if neither float nor insider info.
    denom = denom.where(denom.gt(0), shares_out)
    denom = denom.where(denom.gt(0))  # zeros -> NaN to avoid div-by-zero.

    out = (short_pos / denom) * 100.0
    return out.rename("short_pct_ff")


# --- 2. Days to Cover ----------------------------------------------------
def days_to_cover(df: pd.DataFrame, *, window_days: int = 20) -> pd.Series:
    """``ShortPositions / mean(daily volume)`` over a trailing window (per ticker).

    Volume in the master panel is *daily share volume on the as-of date*; the
    rolling mean over ``window_days`` proxies the average daily turnover that
    shorts would have to buy back to cover. NaN when volume is missing,
    zero, or the window has insufficient observations.
    """
    _require(df, [_TICKER, _SHORT_POS, _VOLUME], "days_to_cover")

    short_pos = pd.to_numeric(df[_SHORT_POS], errors="coerce")
    volume = pd.to_numeric(df[_VOLUME], errors="coerce")
    # A zero-volume bar shouldn't drag the mean to zero (would explode the
    # ratio); treat zeros as NaN before averaging.
    volume = volume.where(volume.gt(0))

    avg_vol = (
        volume.groupby(df[_TICKER])
        .transform(lambda s: s.rolling(window_days, min_periods=max(1, window_days // 2)).mean())
    )
    avg_vol = avg_vol.where(avg_vol.gt(0))

    out = short_pos / avg_vol
    return out.rename("days_to_cover")


# --- 3. Rolling 52-week z-score of ShortPct ------------------------------
def si_z_52w(df: pd.DataFrame, *, window: int = 52) -> pd.Series:
    """Per-ticker rolling z-score of ``ShortPct`` over ``window`` periods.

    ``(x_t - mean_window) / std_window``. A positive z says the current short
    book is *stretched* relative to that name's own recent history — which
    historically marks crowded-short setups vulnerable to short-squeeze
    reversals. ``min_periods`` is set to roughly a quarter of the window so
    we don't emit volatile estimates on too little data.
    """
    _require(df, [_TICKER, _SHORT_PCT], "si_z_52w")
    min_periods = max(2, window // 4)

    si = pd.to_numeric(df[_SHORT_PCT], errors="coerce")

    def _z(s: pd.Series) -> pd.Series:
        roll = s.rolling(window, min_periods=min_periods)
        mu = roll.mean()
        sd = roll.std(ddof=0)
        z = (s - mu) / sd
        return z.replace([np.inf, -np.inf], np.nan)

    out = si.groupby(df[_TICKER]).transform(_z)
    return out.rename("si_z_52w")


# --- 4. Rolling percentile of ShortPct -----------------------------------
def si_percentile_52w(
    df: pd.DataFrame, *, window: int = 52, min_periods: int = 12
) -> pd.Series:
    """Per-ticker rolling percentile of ``ShortPct`` over ``window``.

    Where current short interest sits in its own 52-week distribution —
    distribution-free (so it's robust to fat tails and outliers, unlike the
    z-score) and the v1 prototype's preferred crowding signal.
    Returned values are in [0, 1].
    """
    _require(df, [_TICKER, _SHORT_PCT], "si_percentile_52w")

    si = pd.to_numeric(df[_SHORT_PCT], errors="coerce")

    def _pct(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=min_periods).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
        )

    out = si.groupby(df[_TICKER]).transform(_pct)
    return out.rename("si_percentile_52w")


# --- 5. Persistence: number of up weeks in a trailing window -------------
def si_persistence(df: pd.DataFrame, *, window: int = 5) -> pd.Series:
    """Per-ticker count of weeks in trailing ``window`` where ``ShortPct`` rose w/w.

    A value of ``window`` means short interest increased every week in the
    window — i.e. an unambiguous, conviction-led build. The v1 prototype
    found this far more predictive than the raw level: a single spike
    is noise, four-out-of-five rising weeks is a campaign.
    """
    _require(df, [_TICKER, _SHORT_PCT], "si_persistence")
    min_periods = max(2, (window + 1) // 2)

    si = pd.to_numeric(df[_SHORT_PCT], errors="coerce")

    def _count_up(s: pd.Series) -> pd.Series:
        # Boolean → int so .rolling().sum() yields counts.
        ups = s.diff().gt(0).astype("float64")
        # Restore NaNs where ``diff`` is NaN so we don't count the first row
        # of every ticker as "not up" (it's actually undefined).
        ups = ups.where(s.diff().notna())
        return ups.rolling(window, min_periods=min_periods).sum()

    out = si.groupby(df[_TICKER]).transform(_count_up)
    return out.rename("si_persistence")


# --- 6. 4-week build rate ------------------------------------------------
def si_build_rate_4w(df: pd.DataFrame, *, window: int = 4) -> pd.Series:
    """Per-ticker ``ShortPct - ShortPct.shift(window)``.

    The raw 4-week change in short interest (in percentage points). Pairs
    naturally with ``si_persistence``: build rate gives *magnitude*,
    persistence gives *conviction*. NaN for the first ``window`` rows of
    each ticker.
    """
    _require(df, [_TICKER, _SHORT_PCT], "si_build_rate_4w")

    si = pd.to_numeric(df[_SHORT_PCT], errors="coerce")
    out = si.groupby(df[_TICKER]).transform(lambda s: s - s.shift(window))
    return out.rename("si_build_rate_4w")


# --- 7. Panel assembler --------------------------------------------------
def short_signals_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every short-interest signal attached.

    Order is preserved (we never re-sort the caller's frame). Assumes
    ``df`` is already sorted by ``[Ticker, Date]`` — the rolling/diff
    helpers depend on that to be valid.
    """
    _require(df, [_TICKER, _DATE, _SHORT_PCT], "short_signals_panel")

    out = df.copy()
    out["short_pct_ff"] = short_pct_free_float(out)
    out["days_to_cover"] = days_to_cover(out)
    out["si_z_52w"] = si_z_52w(out)
    out["si_percentile_52w"] = si_percentile_52w(out)
    out["si_persistence"] = si_persistence(out)
    out["si_build_rate_4w"] = si_build_rate_4w(out)

    new_cols = [
        "short_pct_ff",
        "days_to_cover",
        "si_z_52w",
        "si_percentile_52w",
        "si_persistence",
        "si_build_rate_4w",
    ]
    coverage = (out[new_cols].notna().mean() * 100).round(1)
    logger.info(
        f"short_signals_panel: built {len(new_cols)} signals over {len(out):,} rows; "
        f"non-null coverage (%): {coverage.to_dict()}"
    )
    return out
