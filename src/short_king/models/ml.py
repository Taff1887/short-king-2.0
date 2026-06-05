"""LightGBM classifier + LambdaRank learning-to-rank models.

The v1 prototype trained a single global logistic regression on a binary
"will this name be a top-decile short next week?" target. That works as a
sanity baseline but throws away two things v2 wants back:

* **Non-linearity and interactions.** Short setups are conjunctive (e.g. high
  short interest *and* deteriorating quality *and* weak momentum). A linear
  model can only add main effects; a GBM splits on combinations.
* **Cross-sectional structure.** The economically meaningful question is not
  "is this name a short in absolute terms" but "where does this name sit in
  the cross section *today*". LambdaRank optimises exactly that: a pairwise
  ranking loss aware of NDCG-style position weighting, with groups equal to
  the rebalance dates.

This module exposes a tiny, deliberate surface: build/fit a classifier, build/
fit a ranker, score either with a single ``predict_score`` helper, plus a
``decile_target`` utility that produces the 0..9 within-date decile labels the
ranker is trained against. All training is point-in-time by construction —
``fit_gbm_ranker`` requires the caller to supply ``group_dates``, and we
verify the rows are already grouped contiguously so LightGBM's ``group``
argument lines up correctly with the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker

from short_king.utils.logging import logger


# --- Configuration --------------------------------------------------------
@dataclass
class GBMConfig:
    """Hyper-parameters shared by the classifier and the ranker.

    Defaults are conservative: a modest tree count and large ``min_data_in_leaf``
    keep variance under control on a few-hundred-stock weekly panel, while
    feature/bagging fractions provide stochastic regularisation. Override per
    experiment rather than editing this dataclass.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = -1
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    random_state: int = 42


def _cfg_to_kwargs(cfg: GBMConfig) -> dict[str, object]:
    """Translate ``GBMConfig`` into LightGBM sklearn-style kwargs.

    The sklearn wrapper renames a few LightGBM core params (``feature_fraction``
    -> ``colsample_bytree``, ``bagging_fraction`` -> ``subsample``,
    ``bagging_freq`` -> ``subsample_freq``, ``min_data_in_leaf`` ->
    ``min_child_samples``); we translate once here so the dataclass stays
    LightGBM-native and readable for anyone scanning the config.
    """
    return {
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_data_in_leaf,
        "colsample_bytree": cfg.feature_fraction,
        "subsample": cfg.bagging_fraction,
        "subsample_freq": cfg.bagging_freq,
        "random_state": cfg.random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


# --- Classifier -----------------------------------------------------------
def make_gbm_classifier(cfg: GBMConfig | None = None) -> LGBMClassifier:
    """Return an unfitted ``LGBMClassifier`` configured from ``cfg``.

    No ``class_weight`` is set here: the decision to balance is made by
    ``fit_gbm_classifier`` based on the actual target distribution at fit
    time, so this constructor stays a pure object-with-hyperparams.
    """
    cfg = cfg or GBMConfig()
    return LGBMClassifier(**_cfg_to_kwargs(cfg))


def _slice_features(
    features: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Subset ``features`` to ``feature_cols`` and validate the request.

    LightGBM happily trains on a DataFrame with extra columns, but doing so
    silently leaks identifiers / labels into the model. We force the caller to
    name the features explicitly and fail loudly if any are missing.
    """
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise KeyError(
            f"feature_cols not in DataFrame: {missing[:10]}"
            f"{'... (+more)' if len(missing) > 10 else ''}"
        )
    return features.loc[:, feature_cols]


def fit_gbm_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    feature_cols: list[str],
    cfg: GBMConfig | None = None,
    early_stopping_rounds: int | None = None,
    eval_features: pd.DataFrame | None = None,
    eval_target: pd.Series | None = None,
) -> LGBMClassifier:
    """Fit a binary classifier; auto-balances class weights for skewed targets.

    The "short worth taking" label is structurally imbalanced (top decile is
    ~10% positive), so unless the target turns out approximately 50/50 we use
    ``class_weight='balanced'`` so the loss isn't dominated by the majority
    class. ``early_stopping_rounds`` is wired through LightGBM's standard
    sklearn API and is a no-op if the caller doesn't pass eval data.
    """
    cfg = cfg or GBMConfig()
    X = _slice_features(features, feature_cols)
    y = target.loc[X.index] if target.index.equals(X.index) else target.values

    # Class-balance heuristic: anything that strays more than 10pp from 50/50
    # gets the 'balanced' class_weight; perfectly balanced data is left alone
    # so we don't pay the weight-vector overhead for nothing.
    pos_rate = float(np.mean(np.asarray(y) == 1)) if len(y) else 0.0
    class_weight: str | None
    if 0.0 < pos_rate < 1.0 and abs(pos_rate - 0.5) > 0.10:
        class_weight = "balanced"
        logger.info(
            f"fit_gbm_classifier: imbalanced target (pos_rate={pos_rate:.3f}) "
            f"-> class_weight='balanced'"
        )
    else:
        class_weight = None

    model = LGBMClassifier(class_weight=class_weight, **_cfg_to_kwargs(cfg))

    fit_kwargs: dict[str, object] = {}
    if eval_features is not None and eval_target is not None:
        X_eval = _slice_features(eval_features, feature_cols)
        y_eval = (
            eval_target.loc[X_eval.index]
            if eval_target.index.equals(X_eval.index)
            else eval_target.values
        )
        fit_kwargs["eval_set"] = [(X_eval, y_eval)]
        if early_stopping_rounds is not None:
            # LightGBM 4.x routes early stopping through a callback rather than
            # a fit-time kwarg.
            from lightgbm import early_stopping

            fit_kwargs["callbacks"] = [early_stopping(early_stopping_rounds, verbose=False)]
    elif early_stopping_rounds is not None:
        logger.warning(
            "fit_gbm_classifier: early_stopping_rounds set but no eval data supplied -- ignoring"
        )

    logger.info(
        f"fit_gbm_classifier: rows={len(X):,} features={len(feature_cols)} "
        f"pos_rate={pos_rate:.3f} early_stop={early_stopping_rounds}"
    )
    model.fit(X, y, **fit_kwargs)
    return model


# --- Ranker ---------------------------------------------------------------
def _group_sizes(group_dates: pd.Series) -> np.ndarray:
    """Return contiguous-group sizes for LightGBM's ``group`` argument.

    LightGBM expects rows to already be sorted so that every group's rows are
    contiguous; passing scrambled rows produces silently wrong NDCG without
    raising. We don't reshuffle the caller's data (that would risk
    de-aligning ``X`` and ``y``); instead we verify contiguity and raise.
    """
    arr = group_dates.to_numpy()
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    # Boundaries where the value changes; +1 to convert from index to count.
    change_points = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, arr.size]
    sizes = ends - starts
    # Sanity check: every date should appear exactly once as a contiguous block.
    unique_first = arr[starts]
    if len(set(unique_first.tolist())) != len(unique_first):
        raise ValueError(
            "fit_gbm_ranker: group_dates is not sorted into contiguous blocks. "
            "Sort features/target/group_dates by Date before calling."
        )
    return sizes.astype(np.int64)


def fit_gbm_ranker(
    features: pd.DataFrame,
    target_decile: pd.Series,
    group_dates: pd.Series,
    *,
    feature_cols: list[str],
    cfg: GBMConfig | None = None,
) -> LGBMRanker:
    """Fit a cross-sectional LambdaRank model over ``group_dates`` blocks.

    ``target_decile`` must be an integer 0..9 produced by ``decile_target``;
    LambdaRank's pairwise loss treats those as ordinal relevance grades, so
    decile 9 (top-decile forward return = worst-performing names = ideal short
    candidates) is "most relevant" and gets pulled to the top of the within-
    date sort. ``group_dates`` aligns 1:1 with ``features`` and must already be
    sorted into contiguous date blocks; we verify and fail fast if not.
    """
    cfg = cfg or GBMConfig()
    if len(features) != len(target_decile) or len(features) != len(group_dates):
        raise ValueError(
            f"fit_gbm_ranker: length mismatch features={len(features)} "
            f"target={len(target_decile)} group_dates={len(group_dates)}"
        )

    X = _slice_features(features, feature_cols)
    y = target_decile.to_numpy() if hasattr(target_decile, "to_numpy") else np.asarray(target_decile)
    sizes = _group_sizes(group_dates.reset_index(drop=True))

    # LambdaRank treats relevance as a non-negative integer ordinal; coerce
    # explicitly so a stray float decile (e.g. NaN-filled with 0.0) doesn't
    # surface a cryptic LightGBM error during training.
    if np.isnan(y).any():
        raise ValueError(
            "fit_gbm_ranker: target_decile contains NaN -- drop those rows "
            "(and the corresponding feature rows) before fitting."
        )
    y_int = y.astype(np.int32)

    model = LGBMRanker(objective="lambdarank", **_cfg_to_kwargs(cfg))
    logger.info(
        f"fit_gbm_ranker: rows={len(X):,} groups={len(sizes)} "
        f"avg_group_size={sizes.mean():.1f} features={len(feature_cols)}"
    )
    model.fit(X, y_int, group=sizes)
    return model


# --- Scoring --------------------------------------------------------------
def predict_score(
    model: LGBMClassifier | LGBMRanker,
    features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Return a model score aligned with ``features.index``.

    For a classifier the score is the positive-class probability (the natural
    "is this a short" probability used by the portfolio layer); for a ranker
    it's the raw real-valued LambdaRank score (only the within-date *order*
    is meaningful, which is exactly what the portfolio layer consumes).
    """
    X = _slice_features(features, feature_cols)
    if isinstance(model, LGBMClassifier):
        proba = model.predict_proba(X)
        # Binary classifier: pick the positive-class column. We don't assume
        # the column order — look it up on the fitted model.
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            pos_idx = classes.index(1)
        elif True in classes:
            pos_idx = classes.index(True)
        else:
            # Multi-class or unusual labels: default to the last column, which
            # by sklearn convention is the largest class label.
            pos_idx = proba.shape[1] - 1
        scores = proba[:, pos_idx]
    elif isinstance(model, LGBMRanker):
        scores = model.predict(X)
    else:
        # Duck-typed fallback: anything that exposes predict_proba uses it,
        # otherwise predict. Keeps the helper useful for sklearn pipelines.
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, -1]
        else:
            scores = model.predict(X)
    return pd.Series(np.asarray(scores), index=X.index, name="score")


# --- Labels ---------------------------------------------------------------
def decile_target(forward_return: pd.Series, *, by: pd.Series) -> pd.Series:
    """Cross-sectional 0..9 decile rank of ``forward_return`` within each ``by``.

    Decile 9 is the *highest* forward return — for a short book we expect the
    portfolio layer to invert and short the *lowest* forward-return names,
    which by the ranker's training convention are decile 0. Keeping the
    orientation "higher = better forward return" makes the label trivially
    interpretable in cross-asset comparisons; downstream code should flip
    when constructing the short signal.

    Returns an integer Series (NaN where the within-date group has fewer than
    10 valid observations, since deciles aren't well defined there).
    """
    if len(forward_return) != len(by):
        raise ValueError(
            f"decile_target: length mismatch forward_return={len(forward_return)} by={len(by)}"
        )

    # qcut with duplicates='drop' degrades gracefully on small / tied groups;
    # we then mask groups with too few valid points so the ranker isn't fed
    # noise from 3-stock dates.
    def _qcut_group(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 10:
            return pd.Series(np.nan, index=s.index)
        try:
            ranks = pd.qcut(valid, q=10, labels=False, duplicates="drop")
        except ValueError:
            # All values identical: every name is "decile 0" by convention.
            return pd.Series(0.0, index=s.index).where(s.notna())
        out = pd.Series(np.nan, index=s.index)
        out.loc[valid.index] = ranks.astype(float).values
        return out

    result = (
        forward_return.groupby(by, sort=False, observed=True, group_keys=False)
        .apply(_qcut_group)
    )
    # Coerce to nullable Int8: deciles are bounded small ints, NaN-preserving
    # so the caller can filter rows the ranker shouldn't see.
    return result.astype("Int8")


__all__ = [
    "GBMConfig",
    "make_gbm_classifier",
    "fit_gbm_classifier",
    "fit_gbm_ranker",
    "predict_score",
    "decile_target",
]
