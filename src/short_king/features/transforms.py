"""Cross-sectional and time-series transforms for factor construction.

These are the building blocks for EDA, factor signals, and model inputs:
winsorisation tames outliers, z-scoring standardises scale, rank normalisation
is robust to fat tails (and is the basis for ranking-loss models), and
neutralisation strips out sector/industry effects so the residual factor is
what actually carries the alpha. Cross-sectional ops operate *within each
rebalance date*, so no information ever leaks across time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger


def winsorize(series: pd.Series, *, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series to its lower/upper sample quantiles."""
    if series.empty:
        return series.copy()
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN where the denominator is 0 or NaN.

    Avoids ``inf`` / ``-inf`` polluting downstream z-scores and rank stats.
    """
    den_safe = den.where((den != 0) & den.notna())
    return num / den_safe


def _select_numeric_cols(df: pd.DataFrame, by: str, cols: list[str] | None) -> list[str]:
    """Resolve the target columns: explicit list, or all numeric columns excluding ``by``."""
    if cols is not None:
        return list(cols)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c != by]


def cross_sectional_zscore(
    df: pd.DataFrame,
    *,
    by: str = "Date",
    cols: list[str] | None = None,
    winsor: tuple[float, float] | None = (0.01, 0.99),
) -> pd.DataFrame:
    """Winsorise and standardise selected columns *within each rebalance date*.

    Returns a copy of ``df`` with new ``{col}_z`` columns appended. The original
    columns are left untouched so downstream code can still see raw values.
    """
    if by not in df.columns:
        raise KeyError(f"groupby column '{by}' not in DataFrame")

    target_cols = _select_numeric_cols(df, by, cols)
    out = df.copy()

    def _transform(s: pd.Series) -> pd.Series:
        x = winsorize(s, lower=winsor[0], upper=winsor[1]) if winsor is not None else s
        std = x.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.nan, index=s.index)
        return (x - x.mean()) / std

    g = out.groupby(by, sort=False, observed=True)
    for c in target_cols:
        if c not in out.columns:
            logger.warning("cross_sectional_zscore: column {} missing, skipping", c)
            continue
        out[f"{c}_z"] = g[c].transform(_transform)
    return out


def cross_sectional_rank(
    df: pd.DataFrame,
    *,
    by: str = "Date",
    cols: list[str] | None = None,
    pct: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """Cross-sectional rank within each ``by`` group; adds ``{col}_rk`` columns.

    With ``pct=True`` (default) ranks are in ``[0, 1]`` — convenient for blending
    factors of different scales. ``ascending=False`` puts the largest raw value
    at rank 1 (or pct 1.0), which is the natural orientation for "higher is
    better" signals like momentum.
    """
    if by not in df.columns:
        raise KeyError(f"groupby column '{by}' not in DataFrame")

    target_cols = _select_numeric_cols(df, by, cols)
    out = df.copy()
    g = out.groupby(by, sort=False, observed=True)
    for c in target_cols:
        if c not in out.columns:
            logger.warning("cross_sectional_rank: column {} missing, skipping", c)
            continue
        out[f"{c}_rk"] = g[c].rank(pct=pct, ascending=ascending, method="average")
    return out


def rolling_zscore(
    series: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Time-series z-score within a single ticker's history.

    Use this for "is today's value extreme relative to its own recent past?"
    style signals (e.g. short-interest spike vs trailing 26 weeks). Caller is
    responsible for sorting by date and applying *per ticker* (e.g. via
    ``groupby('Ticker').transform``).
    """
    if window <= 1:
        raise ValueError(f"rolling_zscore window must be > 1, got {window}")
    mp = min_periods if min_periods is not None else window
    roll = series.rolling(window=window, min_periods=mp)
    mean = roll.mean()
    std = roll.std(ddof=0)
    std_safe = std.where(std > 0)
    return (series - mean) / std_safe


def neutralise(
    df: pd.DataFrame,
    *,
    factor: str,
    neutralise_by: list[str],
    by: str = "Date",
) -> pd.DataFrame:
    """Cross-sectional residualisation of ``factor`` against group dummies.

    Within each rebalance date, regress ``df[factor]`` on one-hot dummies of
    ``neutralise_by`` (typically ``['Sector']`` or ``['Sector', 'Industry']``)
    and store the residuals as ``{factor}_neutral``. This removes the part of
    the factor that is explained purely by sector membership, leaving the
    idiosyncratic signal that's actually worth betting on.

    Rows with NaN in ``factor`` or any neutraliser are passed through as NaN.
    Groups too small to fit (n <= number of dummies) fall back to demeaning
    within that date.
    """
    missing = [c for c in [factor, by, *neutralise_by] if c not in df.columns]
    if missing:
        raise KeyError(f"neutralise: missing required columns {missing}")

    out = df.copy()
    resid_col = f"{factor}_neutral"
    out[resid_col] = np.nan

    valid_mask = out[factor].notna()
    for c in neutralise_by:
        valid_mask &= out[c].notna()

    def _residualise(block: pd.DataFrame) -> pd.Series:
        y = block[factor].astype(float)
        if len(block) < 2 or y.std(ddof=0) == 0:
            return y - y.mean()
        # Dummy-encode the neutralisers; drop_first avoids the dummy trap.
        X = pd.get_dummies(
            block[neutralise_by].astype("category"),
            drop_first=True,
            dtype=float,
        )
        # If after encoding there's nothing to regress on, just demean.
        if X.shape[1] == 0 or X.shape[0] <= X.shape[1]:
            return y - y.mean()
        X.insert(0, "_const", 1.0)
        X_mat = X.to_numpy()
        y_vec = y.to_numpy()
        try:
            beta, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        except np.linalg.LinAlgError:
            logger.warning("neutralise: lstsq failed for group, falling back to demean")
            return y - y.mean()
        resid = y_vec - X_mat @ beta
        return pd.Series(resid, index=block.index)

    valid = out.loc[valid_mask, [by, factor, *neutralise_by]]
    if valid.empty:
        return out

    residuals = (
        valid.groupby(by, sort=False, observed=True, group_keys=False)
        .apply(_residualise, include_groups=False)
    )
    # ``apply`` may return a Series with the original index; align before assignment.
    out.loc[residuals.index, resid_col] = residuals.values
    return out
