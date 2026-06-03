"""Performance metrics for the short-selling strategy returns.

All functions operate on a ``pd.Series`` of periodic returns indexed by a
``DatetimeIndex`` (weekly Friday-stamped by convention for this project, but
the ``periods_per_year`` arg lets the same code annualise monthly, daily, or
any other frequency). Returns are *arithmetic* simple returns (not log), so
multi-period compounding goes through ``(1 + r).prod()``.

The headline entry point is :func:`summary_table`, which collapses a wide
returns DataFrame (one column per strategy, optional ``turnover`` column)
into a single ``pd.Series`` of summary statistics suitable for the report
front page. The individual metric functions are exported for ad-hoc use in
notebooks and unit tests.

Design notes
------------
- ``max_drawdown`` is reported as a **negative** number (e.g. ``-0.23``) so
  drawdowns compose intuitively with returns on plots and in tables.
- ``sortino`` uses periodic downside semideviation (``r < target``, default
  target = 0) annualised by ``sqrt(periods_per_year)`` — matches the
  conventional Sortino, not the rarer "MAR"-style variant.
- ``calmar`` = CAGR / |MaxDD|. Undefined (NaN) when MaxDD is zero.
- ``avg_turnover`` lives here rather than in the engine module because it is
  treated as a *performance metric* of a strategy (rate of capital recycling
  per period), and is reported alongside Sharpe/CAGR on the summary page.
- ``monthly_returns_table`` is the standard "calendar heatmap" wide layout
  with a trailing ``YEAR`` column. We compound *within* the year using the
  observed monthly returns (not 12 partial-month annualisation) so partial
  years (start/end of sample) are reported correctly.
- ``ann_alpha`` is the annualised intercept from a periodic OLS regression
  ``r_strategy ~ alpha + beta * r_bench``. We bypass scipy/statsmodels and
  use ``np.polyfit`` for a zero-dependency CAPM line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger

__all__ = [
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "avg_turnover",
    "to_monthly",
    "to_annual",
    "monthly_returns_table",
    "summary_table",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clean(returns: pd.Series) -> pd.Series:
    """Return a finite-valued, NaN-dropped copy. Empty in, empty out."""
    if returns is None:
        return pd.Series(dtype=float)
    s = pd.Series(returns).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _ensure_datetime_index(returns: pd.Series) -> pd.Series:
    """Coerce the index to a DatetimeIndex so resample()/groupby work."""
    s = pd.Series(returns).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    return s.dropna()


def _compound(returns: pd.Series) -> float:
    """Cumulative gross multiple ``prod(1 + r)``. Empty -> 1.0 (identity)."""
    s = _clean(returns)
    if s.empty:
        return 1.0
    return float((1.0 + s).prod())


# ---------------------------------------------------------------------------
# Single-series metrics
# ---------------------------------------------------------------------------
def cagr(returns: pd.Series, *, periods_per_year: int = 52) -> float:
    """Compound annual growth rate from ``periods_per_year`` periodic returns.

    Uses ``(prod(1+r))**(periods_per_year / n_periods) - 1`` — no calendar
    timestamps required, so a 52-period weekly series annualises identically
    whether or not we have a real DatetimeIndex.
    """
    s = _clean(returns)
    if s.empty:
        return float("nan")
    gross = (1.0 + s).prod()
    if gross <= 0:
        # Strategy blew up - CAGR is mathematically undefined (no real root).
        return float("nan")
    return float(gross ** (periods_per_year / len(s)) - 1.0)


def sharpe(returns: pd.Series, *, periods_per_year: int = 52, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio. ``rf`` is the annual risk-free rate (decimal)."""
    s = _clean(returns)
    if s.empty:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = s - rf_per
    sd = float(excess.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, *, periods_per_year: int = 52, rf: float = 0.0) -> float:
    """Annualised Sortino ratio (semideviation below the periodic rf target)."""
    s = _clean(returns)
    if s.empty:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = s - rf_per
    downside = excess[excess < 0]
    if downside.empty:
        return float("nan")
    # Population-style semideviation: sqrt(mean(min(excess, 0)**2)) over the
    # full sample, not just the loss subset - standard Sortino definition.
    semi = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if semi == 0 or not np.isfinite(semi):
        return float("nan")
    return float(excess.mean() / semi * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the equity curve, as a negative number.

    Returns 0.0 for monotonically non-decreasing equity. NaN for empty input.
    """
    s = _clean(returns)
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(returns: pd.Series, *, periods_per_year: int = 52) -> float:
    """CAGR / |MaxDD|. NaN when MaxDD is zero (no drawdown -> ratio undefined)."""
    c = cagr(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(returns)
    if not np.isfinite(c) or not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(c / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Share of periods with strictly positive return."""
    s = _clean(returns)
    if s.empty:
        return float("nan")
    return float((s > 0).mean())


# ---------------------------------------------------------------------------
# DataFrame metrics
# ---------------------------------------------------------------------------
def avg_turnover(returns: pd.DataFrame, col: str = "turnover") -> float:
    """Average periodic turnover from a DataFrame column.

    ``returns`` is the strategy run-log (one row per period) with a
    ``turnover`` column carrying the per-period one-way notional traded
    (e.g. 0.20 = 20% of book turned over this period). If the column is
    missing we log a warning and return NaN rather than raising — turnover
    is informational, not load-bearing for the rest of summary_table.
    """
    if returns is None or not isinstance(returns, pd.DataFrame) or col not in returns.columns:
        logger.warning(f"avg_turnover: column '{col}' not found - returning NaN")
        return float("nan")
    s = pd.Series(returns[col]).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())


# ---------------------------------------------------------------------------
# Frequency conversion
# ---------------------------------------------------------------------------
def to_monthly(returns: pd.Series, *, date_col: pd.Series | None = None) -> pd.Series:
    """Resample periodic (typically weekly) returns to monthly compounded.

    ``date_col`` lets the caller pass an external date column when the input
    series carries an integer / positional index instead of a DatetimeIndex
    (e.g. straight off a parquet read). When provided it overrides the
    series' own index.
    """
    s = pd.Series(returns).astype(float)
    if date_col is not None:
        s = pd.Series(s.values, index=pd.to_datetime(date_col), name=s.name)
    s = _ensure_datetime_index(s)
    if s.empty:
        return pd.Series(dtype=float)
    # Compound within each month: (1+r1)(1+r2)... - 1, using period-end labels.
    return s.resample("ME").apply(lambda x: float((1.0 + x).prod() - 1.0)).rename(s.name)


def to_annual(returns: pd.Series) -> pd.Series:
    """Compound periodic returns into calendar-year totals."""
    s = _ensure_datetime_index(pd.Series(returns).astype(float))
    if s.empty:
        return pd.Series(dtype=float)
    return s.resample("YE").apply(lambda x: float((1.0 + x).prod() - 1.0)).rename(s.name)


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Calendar-style wide table: rows = year, cols = Jan..Dec + YEAR.

    The Jan..Dec cells carry monthly compounded returns; the ``YEAR`` column
    is each row's full-year compounded return computed from the observed
    monthly cells (so a partial year is correctly reported as a partial-year
    return, not annualised).
    """
    monthly = to_monthly(returns)
    if monthly.empty:
        cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "YEAR"]
        return pd.DataFrame(columns=cols, dtype=float)

    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    wide = df.pivot_table(index="year", columns="month", values="ret", aggfunc="first")

    # Reindex to all 12 months so the calendar layout is stable even for
    # samples with missing-month rows.
    wide = wide.reindex(columns=range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    wide.columns = month_names

    # Annual = compounded over the (possibly partial) monthly cells in that row.
    def _row_year(row: pd.Series) -> float:
        vals = row.dropna()
        if vals.empty:
            return float("nan")
        return float((1.0 + vals).prod() - 1.0)

    wide["YEAR"] = wide.apply(_row_year, axis=1)
    wide.index.name = "year"
    return wide


# ---------------------------------------------------------------------------
# CAPM-style alpha / beta vs a benchmark series
# ---------------------------------------------------------------------------
def _alpha_beta(strategy: pd.Series, bench: pd.Series, periods_per_year: int) -> tuple[float, float]:
    """Periodic OLS ``r_s = a + b * r_b``; return (annual_alpha, beta).

    Both series are inner-joined on index before fitting. NaN/Inf rows are
    dropped. ``annual_alpha = a * periods_per_year`` (simple-arithmetic
    convention to match the rest of this module).
    """
    if strategy is None or bench is None:
        return float("nan"), float("nan")
    df = pd.concat(
        [pd.Series(strategy).astype(float), pd.Series(bench).astype(float)],
        axis=1, join="inner",
    )
    df.columns = ["r", "b"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["b"].std(ddof=1) == 0:
        return float("nan"), float("nan")
    beta, intercept = np.polyfit(df["b"].to_numpy(), df["r"].to_numpy(), 1)
    return float(intercept * periods_per_year), float(beta)


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------
def summary_table(
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 52,
    bench: pd.Series | None = None,
) -> pd.Series:
    """Collapse a strategy returns frame into the headline summary statistics.

    ``returns`` must contain a ``ret`` (or single non-turnover) column with
    the strategy's periodic return. An optional ``turnover`` column feeds
    :func:`avg_turnover`. When ``bench`` is provided, ``ann_alpha`` and
    ``beta`` are computed via CAPM regression against it.

    Returned ``pd.Series`` keys (always present, NaN when undefined):
    ``cagr, ann_vol, sharpe, sortino, max_drawdown, calmar, hit_rate,
    avg_turnover, ann_alpha, beta, n_periods``.
    """
    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)

    # Resolve the return column. Prefer 'ret', else the first non-turnover
    # numeric column. This keeps the API forgiving for one-strategy frames.
    if isinstance(returns, pd.Series):
        ret = pd.Series(returns).astype(float)
        df = pd.DataFrame({"ret": ret})
    else:
        df = returns.copy()
        if "ret" in df.columns:
            ret = pd.Series(df["ret"]).astype(float)
        else:
            candidates = [
                c for c in df.columns
                if c != "turnover" and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not candidates:
                raise KeyError(
                    "summary_table: no return column found - expected 'ret' "
                    "or a numeric column other than 'turnover'"
                )
            ret = pd.Series(df[candidates[0]]).astype(float)

    ret_clean = _clean(ret)
    ann_vol = (
        float(ret_clean.std(ddof=1) * np.sqrt(periods_per_year))
        if len(ret_clean) > 1 else float("nan")
    )
    ann_alpha, beta_ = _alpha_beta(ret, bench, periods_per_year) if bench is not None \
        else (float("nan"), float("nan"))

    return pd.Series(
        {
            "cagr": cagr(ret, periods_per_year=periods_per_year),
            "ann_vol": ann_vol,
            "sharpe": sharpe(ret, periods_per_year=periods_per_year),
            "sortino": sortino(ret, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(ret),
            "calmar": calmar(ret, periods_per_year=periods_per_year),
            "hit_rate": hit_rate(ret),
            "avg_turnover": avg_turnover(df) if isinstance(df, pd.DataFrame) else float("nan"),
            "ann_alpha": ann_alpha,
            "beta": beta_,
            "n_periods": int(len(ret_clean)),
        }
    )
