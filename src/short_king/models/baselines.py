"""Baseline short-attractiveness models.

Three baselines in increasing sophistication, all operating on the
*rank-normalised feature panel* (columns suffixed ``_rk``, in ``[0, 1]``,
already constructed cross-sectionally per ``Date`` by
``features.transforms.cross_sectional_rank``):

1. ``naive_si_rank``      — "no model": just the ShortPct cross-sectional rank.
2. ``ew_composite``       — Equal-weighted blend of theme-level ranks, then
                            re-ranked. The simplest interpretable composite.
3. ``fit_logit_baseline`` — Regularised L2 logistic regression on the same
                            rank features, trained to predict
                            *Pr(forward return < 0)*. Reproduces the v1
                            prototype but wrapped in a proper
                            ``StandardScaler -> LogisticRegression`` Pipeline,
                            with a parallel ``statsmodels`` fit that dumps a
                            human-readable coefficient table to
                            ``reports/baseline_logit_summary.txt`` for the
                            README write-up.

Design notes:

* All three return a score *aligned to the input row index* so they slot
  straight into the per-date ranking / portfolio construction code without
  re-joining.
* Signs are aligned for the SHORT direction: higher score = more attractive to
  short. Because the underlying ``*_rk`` columns are already oriented that way
  by the feature engineering layer (e.g. expensive valuation -> high rank), the
  composite is a plain mean, no sign-flips.
* There is **no look-ahead**: the rank features are built per ``Date``, and the
  logit is fit on whatever (train) rows the caller passes in. Walk-forward
  splitting is the caller's responsibility (see ``models.walkforward``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from short_king.utils.config import settings
from short_king.utils.logging import logger

# Canonical theme-level rank columns the equal-weight composite blends. These
# names match what the features layer emits (cross_sectional_rank -> *_rk).
# Anything missing from the panel is silently skipped (with a warning) so the
# composite still produces a usable score on partial data.
DEFAULT_EW_COLS: tuple[str, ...] = (
    "short_rk",
    "momentum_rk",
    "vol_rk",
    "valuation_rk",
    "quality_rk",
    "growth_rk",
)

# Column on the FEATURE PANEL holding the raw weekly short percentage. The
# naive baseline turns this into a cross-sectional rank within each rebalance
# date.
SHORTPCT_COL = "ShortPct"
DATE_COL = "Date"

# Where the statsmodels Logit summary lands for the README. Kept as a module
# constant rather than buried in the fit function so callers / tests can
# override or assert on the path.
LOGIT_SUMMARY_PATH = settings.reports_dir / "baseline_logit_summary.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_cols(features: pd.DataFrame, cols: Iterable[str] | None) -> list[str]:
    """Resolve which ``_rk`` columns to use, warning loudly on any missing."""
    requested = list(cols) if cols is not None else list(DEFAULT_EW_COLS)
    present = [c for c in requested if c in features.columns]
    missing = [c for c in requested if c not in features.columns]
    if missing:
        logger.warning(
            "baselines: missing rank columns {} — proceeding with {}",
            missing,
            present,
        )
    if not present:
        raise ValueError(
            f"baselines: none of the requested rank columns are present: {requested}"
        )
    return present


def _rerank_within_date(score: pd.Series, dates: pd.Series) -> pd.Series:
    """Rank a score cross-sectionally per ``Date``, returning percentile ranks in ``[0, 1]``.

    Re-ranking after a blend gives a clean uniform distribution per date, which
    keeps downstream portfolio sizing and IC calculations on a common footing.
    """
    tmp = pd.DataFrame({"_d": dates.values, "_s": score.values}, index=score.index)
    out = tmp.groupby("_d", sort=False, observed=True)["_s"].rank(
        pct=True, ascending=True, method="average"
    )
    return out.astype("float64")


# ---------------------------------------------------------------------------
# 1) Naive baseline: rank by raw ShortPct
# ---------------------------------------------------------------------------
def naive_si_rank(features: pd.DataFrame) -> pd.Series:
    """Cross-sectional rank of ``ShortPct`` within each ``Date``.

    Higher rank == more shorted == more attractive short candidate. This is the
    "no model" benchmark every fancier signal has to beat.
    """
    for col in (DATE_COL, SHORTPCT_COL):
        if col not in features.columns:
            raise KeyError(f"naive_si_rank: required column '{col}' missing from features")
    score = (
        features.groupby(DATE_COL, sort=False, observed=True)[SHORTPCT_COL]
        .rank(pct=True, ascending=True, method="average")
        .astype("float64")
    )
    score.name = "naive_si_rank"
    return score.reindex(features.index)


# ---------------------------------------------------------------------------
# 2) Equal-weight composite over theme-level rank columns
# ---------------------------------------------------------------------------
def ew_composite(
    features: pd.DataFrame,
    *,
    cols: list[str] | None = None,
) -> pd.Series:
    """Equal-weight average of theme ``_rk`` columns, re-ranked per ``Date``.

    The inputs are already SHORT-oriented (e.g. ``valuation_rk`` is high for
    *expensive* names), so a plain mean is the right blend — no sign flips.
    Missing values are handled by ``DataFrame.mean(skipna=True)``: a name with
    only a subset of themes still gets a score, just from fewer pillars.
    """
    if DATE_COL not in features.columns:
        raise KeyError(f"ew_composite: required column '{DATE_COL}' missing")
    use_cols = _resolve_cols(features, cols)
    raw = features[use_cols].mean(axis=1, skipna=True)
    out = _rerank_within_date(raw, features[DATE_COL])
    out.name = "ew_composite"
    return out.reindex(features.index)


# ---------------------------------------------------------------------------
# 3) Logistic-regression baseline
# ---------------------------------------------------------------------------
def _default_logit_feature_cols(features: pd.DataFrame) -> list[str]:
    """Pick every ``_rk`` column on the panel — the natural feature space for v1's logit."""
    cols = [c for c in features.columns if c.endswith("_rk")]
    if not cols:
        raise ValueError(
            "fit_logit_baseline: no '*_rk' feature columns found and none were supplied"
        )
    return cols


def _write_statsmodels_summary(
    X: pd.DataFrame,
    y: pd.Series,
    out_path: Path,
) -> None:
    """Fit a statsmodels Logit (with intercept) and dump ``summary2()`` to disk.

    Statsmodels gives the inferential table — coefficients, SEs, z-stats, p-values,
    pseudo-R^2 — that the README leans on. Kept in a separate fit from sklearn so
    sklearn's regularised path stays the prediction-time model and statsmodels
    stays the *interpretation* model.
    """
    try:
        import statsmodels.api as sm
    except ImportError:  # pragma: no cover - statsmodels is a project dep
        logger.warning(
            "baselines: statsmodels not importable, skipping summary table at {}",
            out_path,
        )
        return

    X_sm = sm.add_constant(X, has_constant="add")
    try:
        res = sm.Logit(y.astype(int).values, X_sm.values).fit(disp=False, maxiter=200)
    except Exception as exc:  # noqa: BLE001 — broad on purpose; never block training
        logger.warning("baselines: statsmodels Logit fit failed ({}); skipping summary", exc)
        return

    # Re-attach feature names so the printed table is human-readable.
    res.model.exog_names = list(X_sm.columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_str = res.summary2(xname=list(X_sm.columns)).as_text()
    except Exception:  # noqa: BLE001 — summary2 can choke on edge cases
        summary_str = res.summary().as_text()

    header = (
        "Short King v2 — baseline logistic regression (statsmodels)\n"
        f"N obs = {len(y)}    target = Pr(forward return < 0)\n"
        f"Features ({len(X.columns)}): {', '.join(X.columns)}\n"
        + "=" * 78 + "\n"
    )
    out_path.write_text(header + summary_str, encoding="utf-8")
    logger.info("baselines: wrote statsmodels Logit summary -> {}", out_path)


def fit_logit_baseline(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    feature_cols: list[str] | None = None,
    scale: bool = True,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
) -> tuple[Pipeline, list[str]]:
    """Fit a regularised logistic regression to predict *Pr(forward return < 0)*.

    Reproduces the v1 prototype's classifier but with a proper preprocessing
    Pipeline so the same object can be re-used at inference time without
    re-fitting any scaler. The ``StandardScaler`` step is the difference
    between "ranks already on [0,1]" and "z-scaled to unit variance" — keeping
    it on by default makes the L2 penalty operate on a like-for-like scale
    across features.

    The companion ``statsmodels`` fit writes a coefficient table to
    ``reports/baseline_logit_summary.txt`` for the README write-up.

    Returns the fitted ``Pipeline`` and the list of feature names used (in the
    same column order the model saw at fit time — pass that exact list to
    ``predict_logit_baseline``).
    """
    cols = list(feature_cols) if feature_cols is not None else _default_logit_feature_cols(features)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"fit_logit_baseline: feature columns missing from panel: {missing}")

    # Align target to features.index. The caller may pass a Series indexed
    # differently (e.g. on a multi-index); reindex to the panel rows, then drop
    # any rows where the target is NaN or any feature is non-finite.
    y_full = pd.Series(target).reindex(features.index)
    X_full = features[cols].apply(pd.to_numeric, errors="coerce")

    finite_mask = np.isfinite(X_full.to_numpy()).all(axis=1)
    label_mask = y_full.notna().to_numpy()
    keep = finite_mask & label_mask
    n_drop = (~keep).sum()
    if n_drop:
        logger.info(
            "fit_logit_baseline: dropping {} rows with missing label or non-finite features",
            int(n_drop),
        )
    X = X_full.loc[keep].copy()
    y = y_full.loc[keep].astype(int)

    if X.empty:
        raise ValueError("fit_logit_baseline: no usable training rows after filtering NaNs")
    classes = np.unique(y.values)
    if len(classes) < 2:
        raise ValueError(
            f"fit_logit_baseline: target has only one class after filtering ({classes!r}); "
            "need both 0 and 1 to fit a classifier"
        )

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "logit",
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=C,
                max_iter=1000,
                fit_intercept=True,
                class_weight=class_weight,
            ),
        )
    )
    pipe = Pipeline(steps)
    pipe.fit(X.values, y.values)
    logger.info(
        "fit_logit_baseline: trained on {} rows x {} features (C={}, class_weight={})",
        len(X),
        len(cols),
        C,
        class_weight,
    )

    # Companion statsmodels fit for the human-readable coefficient table. Uses
    # the same (possibly scaled) design matrix so the printed coefficients live
    # in the same space as sklearn's, modulo the regularisation.
    if scale:
        # Fit a fresh scaler on the kept rows (the pipeline already did this
        # internally but we need access to the transformed matrix here).
        sm_scaler = StandardScaler().fit(X.values)
        X_for_sm = pd.DataFrame(sm_scaler.transform(X.values), columns=cols, index=X.index)
    else:
        X_for_sm = X
    _write_statsmodels_summary(X_for_sm, y, LOGIT_SUMMARY_PATH)

    return pipe, cols


def predict_logit_baseline(
    model: Pipeline,
    features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Predict ``Pr(forward return < 0)`` from a fitted ``fit_logit_baseline`` pipeline.

    Returns a Series aligned to ``features.index``. Rows with any non-finite
    feature are returned as NaN rather than silently coerced — downstream
    portfolio code should treat them as un-rankable on that date.
    """
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise KeyError(f"predict_logit_baseline: feature columns missing from panel: {missing}")

    X = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(X.to_numpy()).all(axis=1)

    out = pd.Series(np.nan, index=features.index, name="prob_down", dtype="float64")
    if finite_mask.any():
        probs = model.predict_proba(X.loc[finite_mask].values)[:, 1]
        out.loc[finite_mask] = probs
    return out


__all__ = [
    "DEFAULT_EW_COLS",
    "LOGIT_SUMMARY_PATH",
    "naive_si_rank",
    "ew_composite",
    "fit_logit_baseline",
    "predict_logit_baseline",
]
