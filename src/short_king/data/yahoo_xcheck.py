"""Yahoo Finance cross-check for FMP ASX prices.

Two purposes only:
  1. **Validation** — cross-correlate monthly returns vs FMP and report the
     median absolute level difference. Used to sanity-check the canonical
     FMP price panel before it feeds features and the backtest.
  2. **Diagnostic fallback** — pull a Yahoo series for a single symbol when
     FMP returns empty. *Never* folded into the canonical panel: mixing
     vendors silently introduces splits/adjustment-method drift that would
     contaminate factor research.

Why ``Adj Close``: total-return correlation requires dividend-adjusted prices
on both sides. FMP's ``adjClose`` is already split- and dividend-adjusted, so
Yahoo's ``Adj Close`` is the like-for-like input.

Why monthly resampling: daily prices have non-trivial cross-vendor timestamp
noise (close-snap differences, half-day handling, suspensions) that drown the
signal we care about — agreement on level and direction of returns.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from short_king.utils.logging import logger


# Reasonable upper bound on the percentage gap we'd ever accept between two
# clean vendors for the same security. Anything above this almost certainly
# means a corporate-action mis-adjustment on one side.
_SANITY_DIFF_PCT = 50.0


def fetch_yahoo_prices(
    symbol_yh: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Daily total-return prices for one Yahoo-formatted ticker (e.g. ``BHP.AX``).

    Returns ``['date','close']`` ascending. Empty DataFrame on any failure —
    caller decides whether absence is fatal. ``close`` here is Yahoo's
    ``Adj Close`` so it is comparable to FMP's ``adjClose``.
    """
    cols = ["date", "close"]
    try:
        raw = yf.download(
            symbol_yh,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
            actions=False,
        )
    except Exception as exc:  # noqa: BLE001 — yfinance raises a zoo of errors
        logger.warning(f"yfinance download failed for {symbol_yh}: {exc!r}")
        return pd.DataFrame(columns=cols)

    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)

    # yf.download returns a MultiIndex on the columns when threads is False but
    # the schema flips between versions and between single/multi-symbol calls.
    # Normalise by selecting the 'Adj Close' field defensively.
    series = _extract_adj_close(raw, symbol_yh)
    if series is None or series.empty:
        return pd.DataFrame(columns=cols)

    out = (
        series.rename("close")
        .to_frame()
        .reset_index()
        .rename(columns={"Date": "date", "index": "date"})
    )
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return out[cols]


def _extract_adj_close(raw: pd.DataFrame, symbol_yh: str) -> pd.Series | None:
    """Pull ``Adj Close`` out of a yf.download frame regardless of column shape."""
    if isinstance(raw.columns, pd.MultiIndex):
        # Two common shapes:
        #   level 0 = field ('Adj Close', 'Close', ...), level 1 = symbol
        #   level 0 = symbol,                            level 1 = field
        for field_level in (0, 1):
            try:
                fields = raw.columns.get_level_values(field_level).unique()
            except IndexError:
                continue
            if "Adj Close" in fields:
                idx: Any = ("Adj Close", symbol_yh) if field_level == 0 else (symbol_yh, "Adj Close")
                try:
                    return raw[idx].squeeze("columns") if isinstance(raw[idx], pd.DataFrame) else raw[idx]
                except KeyError:
                    # Single-symbol calls sometimes drop the symbol level entirely.
                    flat = raw.xs("Adj Close", axis=1, level=field_level, drop_level=True)
                    return flat.squeeze("columns") if isinstance(flat, pd.DataFrame) else flat
        return None

    if "Adj Close" in raw.columns:
        return raw["Adj Close"]
    if "Close" in raw.columns:
        # Last-resort fallback; correlation may suffer for high-yield names.
        logger.debug(f"{symbol_yh}: 'Adj Close' missing, falling back to 'Close'")
        return raw["Close"]
    return None


def crosscheck_monthly_returns(fmp: pd.DataFrame, yh: pd.DataFrame) -> dict:
    """Compare FMP vs Yahoo on month-end resampled total-return prices.

    ``fmp``: ``['date','adjClose']``; ``yh``: ``['date','close']``. Returns
    ``{'n_months', 'corr_monthly', 'median_abs_diff_pct'}``. ``corr_monthly``
    is Pearson on month-over-month returns; ``median_abs_diff_pct`` is the
    median of ``|fmp - yh| / yh * 100`` over the overlapping month-ends and is
    a coarse calibration check (different vendors round splits differently).
    """
    empty = {"n_months": 0, "corr_monthly": float("nan"), "median_abs_diff_pct": float("nan")}
    if fmp is None or yh is None or fmp.empty or yh.empty:
        return empty

    f = _to_month_end(fmp, value_col="adjClose").rename("fmp")
    y = _to_month_end(yh, value_col="close").rename("yh")
    if f.empty or y.empty:
        return empty

    joined = pd.concat([f, y], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return {"n_months": int(len(joined)), "corr_monthly": float("nan"),
                "median_abs_diff_pct": float("nan")}

    rets = joined.pct_change().dropna()
    if rets.empty or rets["fmp"].std() == 0 or rets["yh"].std() == 0:
        corr = float("nan")
    else:
        corr = float(rets["fmp"].corr(rets["yh"]))

    diff_pct = (joined["fmp"] - joined["yh"]).abs() / joined["yh"].replace(0, np.nan) * 100.0
    median_abs_diff = float(diff_pct.median())

    return {
        "n_months": int(len(joined)),
        "corr_monthly": corr,
        "median_abs_diff_pct": median_abs_diff,
    }


def _to_month_end(df: pd.DataFrame, *, value_col: str) -> pd.Series:
    """Daily prices -> last observed price per month-end (no look-ahead)."""
    if value_col not in df.columns or "date" not in df.columns:
        return pd.Series(dtype=float)
    s = (
        df.assign(date=pd.to_datetime(df["date"]).dt.tz_localize(None))
        .set_index("date")[value_col]
        .sort_index()
        .astype(float)
    )
    return s.resample("ME").last().dropna()


def batch_crosscheck(
    fmp_long: pd.DataFrame,
    *,
    max_symbols: int = 50,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Cross-check a sample of symbols from a long FMP price panel.

    ``fmp_long`` must contain ``['symbol','date','adjClose']``. ``symbol`` is
    the canonical ASX ticker (e.g. ``BHP``); Yahoo's ``BHP.AX`` form is
    derived inside. Sampling is seeded (42) so the diagnostic is reproducible
    across runs without re-hitting Yahoo for every name.
    """
    required = {"symbol", "date", "adjClose"}
    missing = required - set(fmp_long.columns)
    if missing:
        raise ValueError(f"fmp_long missing required columns: {sorted(missing)}")

    symbols = sorted(fmp_long["symbol"].dropna().unique().tolist())
    if not symbols:
        logger.warning("batch_crosscheck: fmp_long has no symbols")
        return pd.DataFrame(
            columns=["symbol", "symbol_yh", "n_months", "corr_monthly",
                     "median_abs_diff_pct", "flag"]
        )

    rng = random.Random(42)
    sample = symbols if len(symbols) <= max_symbols else rng.sample(symbols, max_symbols)
    logger.info(f"batch_crosscheck: sampling {len(sample)}/{len(symbols)} symbols")

    rows: list[dict] = []
    for sym in sample:
        sym_yh = _to_yahoo_symbol(sym)
        fmp_sym = fmp_long.loc[fmp_long["symbol"] == sym, ["date", "adjClose"]].copy()
        yh_df = fetch_yahoo_prices(sym_yh, start=start, end=end)
        stats = crosscheck_monthly_returns(fmp_sym, yh_df)
        rows.append({
            "symbol": sym,
            "symbol_yh": sym_yh,
            "n_months": stats["n_months"],
            "corr_monthly": stats["corr_monthly"],
            "median_abs_diff_pct": stats["median_abs_diff_pct"],
            "flag": _flag(stats),
        })

    out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    logger.info(
        f"batch_crosscheck done: median corr={out['corr_monthly'].median():.3f}, "
        f"median diff%={out['median_abs_diff_pct'].median():.2f}"
    )
    return out


def _to_yahoo_symbol(symbol: str) -> str:
    """Canonical ASX ticker -> Yahoo form. Idempotent if already suffixed."""
    s = symbol.strip().upper()
    return s if s.endswith(".AX") else f"{s}.AX"


def _flag(stats: dict) -> str:
    """Single-word diagnostic label so downstream filtering is trivial."""
    if stats["n_months"] < 6 or pd.isna(stats["corr_monthly"]):
        return "insufficient"
    if stats["corr_monthly"] >= 0.95 and stats["median_abs_diff_pct"] < 5.0:
        return "ok"
    if stats["corr_monthly"] >= 0.80 and stats["median_abs_diff_pct"] < _SANITY_DIFF_PCT:
        return "warn"
    return "mismatch"


def fetch_many_yahoo_adjusted(
    symbols: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Bulk Yahoo Finance dividend-adjusted price fetch, stacked long.

    Returns ``[symbol, date, adjClose, volume]`` keyed identically to the
    long FMP price panel so the assemble step can swap sources transparently.
    Used as the primary price source for the 16-year ASIC window because FMP
    on the current plan only ships ~5 years of ASX daily history per call.
    """
    frames: list[pd.DataFrame] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        try:
            yh = fetch_yahoo_prices(sym, start=start, end=end)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"yh {sym}: {e}")
            yh = pd.DataFrame(columns=["date", "close"])
        if not yh.empty:
            yh = yh.rename(columns={"close": "adjClose"})
            yh["symbol"] = sym
            yh["volume"] = pd.NA  # yfinance volume isn't preserved through Adj Close path
            frames.append(yh[["symbol", "date", "adjClose", "volume"]])
        if i % 50 == 0:
            logger.info(f"yh prices: {i}/{n} symbols")
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "adjClose", "volume"])
    out = pd.concat(frames, ignore_index=True)
    return (
        out.sort_values(["symbol", "date"])
           .reset_index(drop=True)
    )


__all__ = [
    "fetch_yahoo_prices",
    "fetch_many_yahoo_adjusted",
    "crosscheck_monthly_returns",
    "batch_crosscheck",
]
