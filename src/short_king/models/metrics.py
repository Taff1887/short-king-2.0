"""Predictive-model evaluation metrics: IC, rank-IC, hit-rate, decile spread, calibration.

These are the standard quant tools for judging whether a cross-sectional model's
predictions actually rank names correctly. The Pearson IC tells you raw linear
agreement; the Spearman (rank) IC is robust to outliers and is the metric of
choice for ranking-loss models. Grouping by Date gives one IC per rebalance,
whose mean / t-stat / hit-rate is the institutional signal-quality summary.
Decile spread is the natural portfolio-level read of monotonicity, and the
calibration table answers "when the model says 60%, does it actually happen 60%
of the time?" — essential for classification heads (e.g. crash-probability).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.logging import logger


def _align_drop_na(predictions: pd.Series, targets: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two series on a common index and drop rows where either is NaN."""
    df = pd.concat([predictions.rename("p"), targets.rename("t")], axis=1, join="inner")
    df = df.dropna()
    return df["p"], df["t"]


def _corr_one(pred: pd.Series, tgt: pd.Series, method: str) -> float:
    """Single-group correlation with NaN-safe alignment and constant-input guard."""
    p, t = _align_drop_na(pred, tgt)
    if len(p) < 2 or p.std(ddof=0) == 0 or t.std(ddof=0) == 0:
        return float("nan")
    return float(p.corr(t, method=method))


def _grouped_corr(
    predictions: pd.Series,
    targets: pd.Series,
    by: pd.Series,
    method: str,
) -> pd.Series:
    """Compute one correlation per group of ``by``, returning a Series indexed by group."""
    df = pd.concat(
        [predictions.rename("p"), targets.rename("t"), by.rename("g")],
        axis=1, join="inner",
    ).dropna(subset=["p", "t", "g"])
    if df.empty:
        return pd.Series(dtype=float, name=f"{method}_ic")

    out: dict = {}
    for key, block in df.groupby("g", sort=True, observed=True):
        out[key] = _corr_one(block["p"], block["t"], method=method)
    s = pd.Series(out, name=f"{method}_ic")
    s.index.name = by.name if by.name is not None else "group"
    return s


def rank_ic(
    predictions: pd.Series,
    targets: pd.Series,
    *,
    by: pd.Series | None = None,
) -> float | pd.Series:
    """Spearman (rank) correlation of predictions vs targets.

    If ``by`` is given (typically a Date series aligned to ``predictions``),
    returns one IC per group as a Series; otherwise a single float.
    """
    if by is None:
        return _corr_one(predictions, targets, method="spearman")
    return _grouped_corr(predictions, targets, by, method="spearman")


def ic(
    predictions: pd.Series,
    targets: pd.Series,
    *,
    by: pd.Series | None = None,
) -> float | pd.Series:
    """Pearson correlation of predictions vs targets, optionally per group."""
    if by is None:
        return _corr_one(predictions, targets, method="pearson")
    return _grouped_corr(predictions, targets, by, method="pearson")


def ic_summary(ic_series: pd.Series) -> dict:
    """Aggregate a per-period IC series into the standard signal-quality summary.

    Reports the mean IC, its sample std, the t-stat ``mean / (std / sqrt(n))``
    (i.e. ``mean / std * sqrt(n)``), the hit-rate (fraction of periods with
    positive IC), and the sample size.
    """
    s = pd.Series(ic_series, dtype=float).dropna()
    n = int(len(s))
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "t_stat": float("nan"),
                "hit_rate": float("nan"), "n": 0}

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    if np.isfinite(std) and std > 0:
        t_stat = mean / (std / np.sqrt(n))
    else:
        t_stat = float("nan")
    return {
        "mean": mean,
        "std": std,
        "t_stat": float(t_stat),
        "hit_rate": float((s > 0).mean()),
        "n": n,
    }


def hit_rate(
    predictions: pd.Series,
    targets: pd.Series,
    *,
    threshold: float = 0.5,
) -> float:
    """Classification hit-rate: fraction of rows where ``predictions >= threshold``
    matches the (0/1) label in ``targets``. NaNs in either series are dropped.
    """
    p, t = _align_drop_na(predictions, targets)
    if p.empty:
        return float("nan")
    pred_cls = (p >= threshold).astype(int)
    label = (t.astype(float) > 0).astype(int)  # treat any positive label as 1
    return float((pred_cls == label).mean())


def decile_spread(
    predictions: pd.Series,
    targets: pd.Series,
    *,
    by: pd.Series,
    n: int = 10,
) -> pd.Series:
    """Per-group mean target in the top ``n``-tile minus the bottom ``n``-tile.

    Within each ``by`` group (typically Date), names are sorted by prediction
    and split into ``n`` equal-count buckets; the spread is the mean target in
    the highest-prediction bucket minus the lowest. This is the rank-based,
    monotonicity-focused cousin of IC: high spread = the model separates winners
    from losers, even if the linear correlation is modest.

    Groups with fewer than ``n`` valid observations yield NaN for that period.
    """
    if n < 2:
        raise ValueError(f"decile_spread: n must be >= 2, got {n}")

    df = pd.concat(
        [predictions.rename("p"), targets.rename("t"), by.rename("g")],
        axis=1, join="inner",
    ).dropna(subset=["p", "t", "g"])
    if df.empty:
        return pd.Series(dtype=float, name=f"top_minus_bot_q{n}")

    spreads: dict = {}
    for key, block in df.groupby("g", sort=True, observed=True):
        if len(block) < n:
            spreads[key] = float("nan")
            continue
        # Rank-based bucketing — ``method='first'`` breaks ties deterministically so
        # qcut doesn't choke on duplicate-edge bins (common with sparse predictions).
        try:
            buckets = pd.qcut(
                block["p"].rank(method="first"),
                q=n,
                labels=False,
            )
        except ValueError as exc:
            logger.warning("decile_spread: qcut failed for group {}: {}", key, exc)
            spreads[key] = float("nan")
            continue
        top = block.loc[buckets == n - 1, "t"].mean()
        bot = block.loc[buckets == 0, "t"].mean()
        spreads[key] = float(top - bot)

    s = pd.Series(spreads, name=f"top_minus_bot_q{n}")
    s.index.name = by.name if by.name is not None else "group"
    return s


def calibration_table(
    probs: pd.Series,
    labels: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability table for a probabilistic classifier.

    Bins predicted probabilities into ``n_bins`` equal-width buckets on ``[0, 1]``
    and reports, per bucket: mean predicted probability, observed positive rate,
    and count. Use this to diagnose over/under-confidence — a well-calibrated
    model has ``mean_prob == observed_rate`` (the y=x line in a reliability plot).
    """
    if n_bins < 2:
        raise ValueError(f"calibration_table: n_bins must be >= 2, got {n_bins}")

    p, y = _align_drop_na(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``include_lowest`` so a probability of exactly 0 lands in bin 0.
    bins = pd.cut(p, bins=edges, include_lowest=True, labels=False)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({
                "bin": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_prob": float("nan"),
                "observed_rate": float("nan"),
                "count": 0,
            })
            continue
        rows.append({
            "bin": b,
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "mean_prob": float(p[mask].mean()),
            "observed_rate": float((y[mask].astype(float) > 0).mean()),
            "count": cnt,
        })
    return pd.DataFrame(rows)
