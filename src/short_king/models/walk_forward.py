"""Purged + embargoed expanding-window cross-validation for cross-sectional panels.

For weekly ASIC short-selling panels, each row's label is the forward N-week
return. That means a training row dated ``t`` has a label that overlaps with
rows dated ``t+1 ... t+N`` — using those rows in the test set leaks future
information backward into training. The fix is the standard de Prado recipe:

  * Expanding training window: train always starts at the first available
    date and ends at a moving cutoff. New folds add data, never discard it,
    which matches how a live model is actually retrained.
  * Embargo gap: at least ``embargo_weeks >= label_horizon`` weeks are
    excised between the training cutoff and the test start. This removes
    rows whose forward label would otherwise overlap with the test window.
  * Disjoint test windows: by default each test window is ``test_weeks``
    long and the next fold steps forward by the same amount, so OOF
    predictions across folds cover the time axis without overlap.

The output of :func:`fit_predict_walkforward` is a long OOF prediction frame
suitable for downstream IC, decile, and portfolio analytics — the same data
contract used by ``qfr.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from short_king.utils.logging import logger


@dataclass
class WalkForwardSplit:
    """One fold of the expanding-window walk-forward CV.

    ``train_idx`` / ``test_idx`` are integer row positions into the *original*
    ``dates`` array passed to :func:`walk_forward_splits` — ready for
    ``X.iloc[split.train_idx]`` style slicing.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def _as_datetime_array(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Normalise the date input to a ``datetime64[ns]`` ndarray aligned to row order."""
    if isinstance(dates, pd.DatetimeIndex):
        arr = dates.to_numpy()
    else:
        arr = pd.to_datetime(dates, errors="raise").to_numpy()
    if arr.dtype.kind != "M":
        raise TypeError(f"dates must be datetime-like, got dtype={arr.dtype}")
    return arr


def walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    min_train_weeks: int = 156,
    test_weeks: int = 26,
    embargo_weeks: int = 4,
    step_weeks: int | None = None,
) -> list[WalkForwardSplit]:
    """Build expanding-window train/test splits with a leakage embargo.

    Parameters
    ----------
    dates
        Row-aligned timestamp series (typically the ``Date`` column of a long
        cross-sectional panel). Need not be sorted; rows are assigned to a
        fold purely by their date.
    min_train_weeks
        Length of the initial training window before the first test fold —
        in weeks of calendar time, not number of rows.
    test_weeks
        Length of each test window in weeks.
    embargo_weeks
        Calendar gap between the train cutoff and the test start. Must be
        ``>=`` the label horizon to prevent leakage from overlapping forward
        returns.
    step_weeks
        How far to advance ``test_start`` between folds. Defaults to
        ``test_weeks`` (non-overlapping disjoint test windows).

    Returns
    -------
    list[WalkForwardSplit]
        One split per fold, each carrying explicit integer row indices for
        fast slicing.
    """
    if min_train_weeks <= 0 or test_weeks <= 0 or embargo_weeks < 0:
        raise ValueError(
            "min_train_weeks/test_weeks must be > 0 and embargo_weeks >= 0; "
            f"got {min_train_weeks=}, {test_weeks=}, {embargo_weeks=}"
        )
    step = step_weeks if step_weeks is not None else test_weeks
    if step <= 0:
        raise ValueError(f"step_weeks must be > 0, got {step}")

    arr = _as_datetime_array(dates)
    if arr.size == 0:
        return []

    t0 = pd.Timestamp(arr.min()).normalize()
    t_end = pd.Timestamp(arr.max()).normalize()
    week = pd.Timedelta(weeks=1)

    splits: list[WalkForwardSplit] = []
    train_start = t0
    train_end = t0 + min_train_weeks * week - pd.Timedelta(days=1)
    fold = 0
    while True:
        test_start = train_end + embargo_weeks * week + pd.Timedelta(days=1)
        test_end = test_start + test_weeks * week - pd.Timedelta(days=1)
        if test_start > t_end:
            break  # no more data to test against
        # Allow the final fold to be truncated by the end of the panel.
        test_end_eff = min(test_end, t_end)

        train_mask = (arr >= np.datetime64(train_start)) & (arr <= np.datetime64(train_end))
        test_mask = (arr >= np.datetime64(test_start)) & (arr <= np.datetime64(test_end_eff))
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        if train_idx.size == 0 or test_idx.size == 0:
            # Empty windows can happen at edges or in panels with date gaps;
            # advance to the next fold rather than emit a degenerate split.
            logger.debug(
                "walk_forward: skipping empty fold {} train=[{}, {}] test=[{}, {}]",
                fold, train_start.date(), train_end.date(),
                test_start.date(), test_end_eff.date(),
            )
        else:
            splits.append(
                WalkForwardSplit(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end_eff,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )
            fold += 1

        # Expanding window: train_start is fixed, train_end advances by step.
        train_end = train_end + step * week
        if train_end >= t_end:
            break

    logger.info(
        "walk_forward: {} folds | dates [{} .. {}] | min_train={}w test={}w embargo={}w step={}w",
        len(splits), t0.date(), t_end.date(),
        min_train_weeks, test_weeks, embargo_weeks, step,
    )
    return splits


def _emit_predictions(estimator, X_te: pd.DataFrame, predict_proba: bool) -> np.ndarray:
    """Pick the right scoring head: probabilities for classifiers, raw output otherwise."""
    if predict_proba and hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_te)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            # Binary or multi-class: take the probability of the positive (largest) class.
            return proba[:, -1]
        return proba.ravel()
    if predict_proba and hasattr(estimator, "decision_function"):
        # Some classifiers (e.g. LinearSVC) expose decision_function but not proba.
        return np.asarray(estimator.decision_function(X_te)).ravel()
    return np.asarray(estimator.predict(X_te)).ravel()


def fit_predict_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_factory,
    *,
    min_train_weeks: int = 156,
    test_weeks: int = 26,
    embargo_weeks: int = 4,
    predict_proba: bool = True,
) -> pd.DataFrame:
    """Fit ``model_factory()`` per fold; concat OOF predictions across all test windows.

    Parameters
    ----------
    X, y, dates
        Row-aligned feature matrix, label series and timestamp series. All
        three must share the same length and index ordering.
    model_factory
        Zero-argument callable that returns a *fresh* sklearn-style estimator
        (i.e. supports ``.fit(X, y)`` and one of ``predict_proba``,
        ``decision_function``, ``predict``). A factory is used so each fold
        trains an independent model — no parameter leakage across folds.
    predict_proba
        If True, prefer ``predict_proba`` / ``decision_function`` so the
        output is a ranking score. Set False for regression targets.

    Returns
    -------
    pd.DataFrame
        Long frame with columns ``['date', 'idx', 'y_true', 'y_pred',
        'fold']``. ``idx`` is the original integer row position into the
        input ``X`` so the caller can rejoin metadata (symbol, sector, ...).
    """
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError(
            f"length mismatch: X={len(X)}, y={len(y)}, dates={len(dates)}"
        )

    splits = walk_forward_splits(
        dates,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
        embargo_weeks=embargo_weeks,
    )
    if not splits:
        logger.warning("walk_forward: no folds produced; returning empty predictions")
        return pd.DataFrame(columns=["date", "idx", "y_true", "y_pred", "fold"])

    # Reset to positional indexing so np.ndarray indices line up regardless of
    # whatever index the caller passed in.
    X_pos = X.reset_index(drop=True)
    y_pos = y.reset_index(drop=True)
    d_pos = pd.to_datetime(pd.Series(dates).reset_index(drop=True))

    out_frames: list[pd.DataFrame] = []
    for fold_id, sp in enumerate(splits):
        X_tr, y_tr = X_pos.iloc[sp.train_idx], y_pos.iloc[sp.train_idx]
        X_te, y_te = X_pos.iloc[sp.test_idx], y_pos.iloc[sp.test_idx]

        # Drop rows with NaN labels from training so the fit doesn't blow up;
        # leave the test set untouched and let NaN labels pass through as-is
        # (caller may want to inspect coverage).
        tr_mask = y_tr.notna()
        if tr_mask.sum() == 0:
            logger.warning("walk_forward: fold {} has no usable training labels, skipping", fold_id)
            continue

        estimator = model_factory()
        estimator.fit(X_tr.loc[tr_mask], y_tr.loc[tr_mask])
        y_pred = _emit_predictions(estimator, X_te, predict_proba=predict_proba)

        out_frames.append(
            pd.DataFrame(
                {
                    "date": d_pos.iloc[sp.test_idx].to_numpy(),
                    "idx": sp.test_idx,
                    "y_true": y_te.to_numpy(),
                    "y_pred": np.asarray(y_pred),
                    "fold": fold_id,
                }
            )
        )
        logger.info(
            "walk_forward: fold {} train n={} [{} .. {}] -> test n={} [{} .. {}]",
            fold_id, int(tr_mask.sum()), sp.train_start.date(), sp.train_end.date(),
            len(sp.test_idx), sp.test_start.date(), sp.test_end.date(),
        )

    if not out_frames:
        return pd.DataFrame(columns=["date", "idx", "y_true", "y_pred", "fold"])
    return pd.concat(out_frames, ignore_index=True)
