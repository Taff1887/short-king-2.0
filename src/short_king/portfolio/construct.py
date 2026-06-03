"""Portfolio construction — turn a per-date short-conviction score into target weights.

Three constructors that all consume a long-format ``(Date, Ticker, score, ...)``
panel and emit a long-format ``(Date, Ticker, weight, side)`` target-book:

* ``top_k_short``       — pick the top-K names by score on each date, short them
                          equal-weight (each = -1/K). The default "short-only
                          conviction book" used by the headline backtest.
* ``decile_short``      — short the entire top decile equal-weight. Tracks the
                          economic decile spread without imposing a hard cap on
                          breadth, so it scales with the investable universe.
* ``long_short_decile`` — dollar-neutral diagnostic: short the top decile, long
                          the bottom decile, each leg equal-weighted to sum to
                          -1 / +1 respectively. Used to confirm the score has
                          cross-sectional information, not just a short-side
                          tail.

All three are PURE functions of the input frame — they do not read or write disk,
do not call out to other modules, and do not mutate the input. Outputs are
sorted by [Date, Ticker], have NaN weights dropped, and use a signed-weight
convention (negative weight ⇒ short position) plus an explicit ``side`` column
(+1 long / -1 short) so downstream code can filter on either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _validate_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise KeyError naming any missing columns so the message points the caller
    straight at the offending input rather than failing deep inside a groupby."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"portfolio construction: input frame is missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def _filter_eligible(
    scores: pd.DataFrame,
    *,
    score_col: str,
    investable_col: str,
    liquidity_col: str,
    liquidity_floor: float | None,
) -> pd.DataFrame:
    """Apply the cross-sectional eligibility filter shared by every constructor.

    A name is eligible on a given date iff (a) ``investable_col`` is truthy,
    (b) the score is non-NaN, and (c) ``liquidity_col >= liquidity_floor`` when
    a floor is set. The liquidity column is only required to exist when a floor
    is actually being enforced — this lets callers who don't have ADV data run
    the constructors without forging a dummy column.
    """
    if investable_col in scores.columns:
        elig = scores[scores[investable_col].astype(bool)].copy()
    else:
        # If the panel doesn't carry an explicit investable flag we treat every
        # row as eligible. This is the common case during exploratory runs.
        logger.warning(
            f"investable column '{investable_col}' not found — treating all rows as investable"
        )
        elig = scores.copy()

    elig = elig[elig[score_col].notna()]

    if liquidity_floor is not None:
        if liquidity_col not in elig.columns:
            raise KeyError(
                f"liquidity_floor={liquidity_floor} requested but liquidity column "
                f"'{liquidity_col}' is missing from the input frame"
            )
        elig = elig[elig[liquidity_col].notna() & (elig[liquidity_col] >= liquidity_floor)]

    return elig


def _finalise(book: pd.DataFrame, *, date_col: str, ticker_col: str) -> pd.DataFrame:
    """Standard output contract: keep only the four canonical columns, drop NaN
    weights, sort by (Date, Ticker), reset index. Applied as the last step of
    every constructor so all three share an identical schema."""
    out = book[[date_col, ticker_col, "weight", "side"]].copy()
    out = out.dropna(subset=["weight"])
    out = out.sort_values([date_col, ticker_col]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------
def top_k_short(
    scores: pd.DataFrame,
    *,
    k: int = 20,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    score_col: str = "score",
    investable_col: str = "investable",
    liquidity_floor: float | None = None,
    liquidity_col: str = "adv_aud",
) -> pd.DataFrame:
    """Equal-weighted short book of the top-K most-shortable names per date.

    On each date, pick the K eligible names with the highest ``score_col`` and
    assign them ``weight = -1/K`` (so the book sums to -1, i.e. $1 short
    notional). If fewer than K names are eligible on a date that date is dropped
    rather than under-filled — an under-filled book would silently rescale the
    realised exposure and mask data-coverage holes.

    Parameters
    ----------
    scores:
        Long-format panel keyed by (``date_col``, ``ticker_col``) with a
        numeric ``score_col``. Higher score = more shortable.
    k:
        Number of names to short per date.
    investable_col:
        Boolean column flagging tradeable names. If absent every row is treated
        as investable (with a warning).
    liquidity_floor, liquidity_col:
        Optional cross-sectional liquidity gate. When ``liquidity_floor`` is
        set, names with ``liquidity_col < floor`` are dropped before the top-K
        pick — this prevents the book from concentrating into micro-caps that
        can't actually be borrowed.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    _validate_cols(scores, [date_col, ticker_col, score_col])

    elig = _filter_eligible(
        scores,
        score_col=score_col,
        investable_col=investable_col,
        liquidity_col=liquidity_col,
        liquidity_floor=liquidity_floor,
    )
    if elig.empty:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight", "side"])

    # Rank within date by descending score; use 'first' tie-breaker so the pick
    # is deterministic when the score has duplicates (otherwise pandas would
    # return an averaged rank that doesn't slice cleanly).
    elig = elig.assign(
        _rank=elig.groupby(date_col)[score_col].rank(method="first", ascending=False)
    )

    # Only keep dates that have at least K eligible names — see docstring.
    counts = elig.groupby(date_col)[ticker_col].size()
    full_dates = counts[counts >= k].index
    dropped = counts[counts < k]
    if len(dropped):
        logger.warning(
            f"top_k_short: {len(dropped)} date(s) had < k={k} eligible names and were dropped "
            f"(e.g. {dropped.head(3).to_dict()})"
        )
    elig = elig[elig[date_col].isin(full_dates)]

    picks = elig[elig["_rank"] <= k].copy()
    picks["weight"] = -1.0 / float(k)
    picks["side"] = -1

    return _finalise(picks, date_col=date_col, ticker_col=ticker_col)


def decile_short(
    scores: pd.DataFrame,
    *,
    n_deciles: int = 10,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    score_col: str = "score",
    investable_col: str = "investable",
    liquidity_floor: float | None = None,
    liquidity_col: str = "adv_aud",
) -> pd.DataFrame:
    """Equal-weighted short book of the top decile (or other top-quantile slice).

    Unlike ``top_k_short`` the breadth here scales with the size of the
    investable universe on each date — useful when the universe shrinks
    materially (e.g. early in the sample) and a fixed K would over-concentrate.
    Each picked name receives ``weight = -1 / n_in_decile`` so the book still
    sums to -1.

    ``n_deciles`` defaults to 10 (top decile = top 10 %). Set to 5 for top
    quintile, 4 for top quartile, etc.
    """
    if n_deciles < 2:
        raise ValueError(f"n_deciles must be >= 2, got {n_deciles}")
    _validate_cols(scores, [date_col, ticker_col, score_col])

    elig = _filter_eligible(
        scores,
        score_col=score_col,
        investable_col=investable_col,
        liquidity_col=liquidity_col,
        liquidity_floor=liquidity_floor,
    )
    if elig.empty:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight", "side"])

    # qcut needs at least n_deciles distinct ranks per group, else it errors.
    # We rank-then-qcut so ties don't blow up duplicates="raise"; 'first' gives
    # a strict total order and 'duplicates="drop"' is belt-and-braces.
    def _top_decile_mask(g: pd.DataFrame) -> pd.Series:
        if len(g) < n_deciles:
            return pd.Series(False, index=g.index)
        ranks = g[score_col].rank(method="first", ascending=True)
        try:
            buckets = pd.qcut(ranks, n_deciles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(False, index=g.index)
        # Top decile = highest score = highest rank = bucket index (n_deciles - 1).
        # When qcut drops duplicate edges the max realised bucket may be lower;
        # use the actual max so we still grab the genuine top slice.
        top_bucket = int(np.nanmax(buckets)) if np.isfinite(np.nanmax(buckets)) else None
        if top_bucket is None:
            return pd.Series(False, index=g.index)
        return buckets == top_bucket

    elig = elig.assign(
        _in_top=elig.groupby(date_col, group_keys=False).apply(_top_decile_mask)
    )
    picks = elig[elig["_in_top"]].copy()

    if picks.empty:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight", "side"])

    # Equal-weight within each date so the per-date short notional = -1.
    n_per_date = picks.groupby(date_col)[ticker_col].transform("size")
    picks["weight"] = -1.0 / n_per_date.astype(float)
    picks["side"] = -1

    return _finalise(picks, date_col=date_col, ticker_col=ticker_col)


def long_short_decile(
    scores: pd.DataFrame,
    *,
    n_deciles: int = 10,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    score_col: str = "score",
    investable_col: str = "investable",
    liquidity_floor: float | None = None,
    liquidity_col: str = "adv_aud",
) -> pd.DataFrame:
    """Dollar-neutral long-short: short the top decile, long the bottom decile.

    Each leg is equal-weighted and sized so that the long leg sums to +1 and
    the short leg sums to -1 (gross exposure = 2, net = 0). This is the
    canonical academic decile spread — its sign and significance are the
    cleanest read on whether the score contains cross-sectional information
    rather than just a one-sided tail.

    Names that fall into neither extreme decile on a given date are omitted
    from the output (they don't appear with weight = 0). Dates where ``qcut``
    can't form ``n_deciles`` buckets (fewer than ``n_deciles`` eligible names)
    are dropped entirely rather than produce a degenerate spread.
    """
    if n_deciles < 2:
        raise ValueError(f"n_deciles must be >= 2, got {n_deciles}")
    _validate_cols(scores, [date_col, ticker_col, score_col])

    elig = _filter_eligible(
        scores,
        score_col=score_col,
        investable_col=investable_col,
        liquidity_col=liquidity_col,
        liquidity_floor=liquidity_floor,
    )
    if elig.empty:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight", "side"])

    def _assign_bucket(g: pd.DataFrame) -> pd.Series:
        if len(g) < n_deciles:
            return pd.Series(np.nan, index=g.index)
        ranks = g[score_col].rank(method="first", ascending=True)
        try:
            buckets = pd.qcut(ranks, n_deciles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=g.index)
        return buckets

    elig = elig.assign(
        _bucket=elig.groupby(date_col, group_keys=False).apply(_assign_bucket)
    )

    # Resolve top / bottom buckets per date — under heavy ties qcut may drop
    # edges and the realised max bucket can be < n_deciles - 1. We use the
    # observed min and max within each date so the spread is always genuinely
    # the extreme slices of the realised distribution.
    bucket_bounds = elig.groupby(date_col)["_bucket"].agg(["min", "max"])
    elig = elig.merge(bucket_bounds, left_on=date_col, right_index=True, how="left")

    # Drop dates where min == max (i.e. only one realised bucket — can't form a spread).
    degen = bucket_bounds[bucket_bounds["min"] == bucket_bounds["max"]].index
    if len(degen):
        logger.warning(
            f"long_short_decile: {len(degen)} date(s) had a degenerate score distribution "
            f"and were dropped"
        )
    elig = elig[~elig[date_col].isin(degen)]
    if elig.empty:
        return pd.DataFrame(columns=[date_col, ticker_col, "weight", "side"])

    is_top = elig["_bucket"] == elig["max"]
    is_bot = elig["_bucket"] == elig["min"]

    legs = elig[is_top | is_bot].copy()
    legs["side"] = np.where(legs["_bucket"] == legs["max"], -1, 1)

    # Equal-weight WITHIN each (date, side) so each leg sums to +/- 1.
    # Count per (date, side); weight = side / count.
    counts = legs.groupby([date_col, "side"])[ticker_col].transform("size")
    legs["weight"] = legs["side"].astype(float) / counts.astype(float)

    return _finalise(legs, date_col=date_col, ticker_col=ticker_col)
