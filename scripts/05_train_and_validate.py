"""Walk-forward train + validate every model on the feature panel.

Models compared (all walk-forward, no look-ahead):

* ``naive``       - rank by raw short-interest percentile (industry baseline).
* ``ew``          - equal-weight composite of the ``*_rk`` feature columns.
* ``logit``       - logistic regression on the rank features (sklearn).
* ``gbm_cls``     - LightGBM binary classifier on ``fwd_ret_4w < 0``.
* ``gbm_rank``    - LightGBM cross-sectional LambdaRank on inverted decile target.

Out-of-fold predictions land in ``reports/oof_predictions.parquet`` and the
per-model metrics table in ``reports/model_metrics.csv``. SHAP and gain-based
feature-importance artefacts for the GBM classifier are saved under
``reports/`` and ``charts/``.
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker

from short_king.models.baselines import (
    ew_composite,
    fit_logit_baseline,
    naive_si_rank,
    predict_logit_baseline,
)
from short_king.models.interpret import (
    gain_importance,
    mean_abs_shap,
    plot_shap_summary,
    shap_values_sampled,
)
from short_king.models.metrics import (
    calibration_table,
    decile_spread,
    ic_summary,
    rank_ic,
)
from short_king.models.ml import (
    GBMConfig,
    decile_target,
    fit_gbm_classifier,
    fit_gbm_ranker,
    predict_score,
)
from short_king.models.walk_forward import walk_forward_splits
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger

_TICKER_COL = "Ticker"
_DATE_COL = "Date"
_FWD_RET_COL = "fwd_ret_4w"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon-weeks", type=int, default=4)
    p.add_argument("--min-train-weeks", type=int, default=156, help="~3y minimum training window")
    p.add_argument("--test-weeks", type=int, default=4)
    p.add_argument("--embargo-weeks", type=int, default=4)
    p.add_argument("--shap-n-sample", type=int, default=2000)
    p.add_argument(
        "--monthly", action="store_true",
        help="Use features_monthly.parquet (last-Friday-of-month panel). "
             "Sets defaults to min-train=36 / test=6 / embargo=1 months "
             "unless explicitly overridden. Adjusts OOF parquet name to "
             "reports/oof_predictions_monthly.parquet.",
    )
    return p.parse_args()


def _resolve_fwd_ret(features: pd.DataFrame, horizon_weeks: int) -> str:
    """Pick the forward-return column to use as the label source."""
    candidates = [
        f"fwd_ret_{horizon_weeks}w",
        _FWD_RET_COL,
        f"ret_fwd_{horizon_weeks}w",
        "ret_fwd_4w",
    ]
    for c in candidates:
        if c in features.columns:
            return c
    raise KeyError(
        f"no forward-return column in features (looked for {candidates}). "
        "Build features with a label column before training."
    )


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """The model-input columns: every ``*_rk`` and every ``sec_*`` dummy."""
    return [c for c in df.columns if c.endswith("_rk") or c.startswith("sec_")]


def _rk_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_rk")]


def _select_rows_by_date(
    df: pd.DataFrame,
    idx: np.ndarray,
) -> pd.DataFrame:
    """Positional row slice that preserves the original row order on a reset-index frame."""
    return df.iloc[idx]


def _walkforward_predict(
    df: pd.DataFrame,
    feat_cols: list[str],
    y: pd.Series,
    fit_predict: callable,
    *,
    min_train_weeks: int,
    test_weeks: int,
    embargo_weeks: int,
    model_name: str,
) -> pd.Series:
    """Generic walk-forward driver returning an OOF score Series aligned to ``df.index``.

    ``fit_predict(X_tr, y_tr, X_te) -> np.ndarray`` is responsible for fitting on
    the training fold (which may need extra context like ``group_dates`` for the
    ranker) and producing a score vector for the test rows.
    """
    splits = walk_forward_splits(
        df[_DATE_COL],
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
        embargo_weeks=embargo_weeks,
    )
    if not splits:
        logger.warning(f"{model_name}: no walk-forward folds produced — returning all-NaN scores")
        return pd.Series(np.nan, index=df.index, name="score", dtype="float64")

    out = pd.Series(np.nan, index=df.index, name="score", dtype="float64")
    for fold_id, sp in enumerate(splits):
        # Cast feature matrix to plain float64 ndarray — some *_rk columns
        # are pandas extension Float64 (nullable), which breaks np.isfinite
        # and tree-model native arrays.
        X_tr = df.iloc[sp.train_idx][feat_cols].astype("float64", copy=False)
        X_te = df.iloc[sp.test_idx][feat_cols].astype("float64", copy=False)
        y_tr = y.iloc[sp.train_idx]

        # Drop NaN labels from training, leave test predictions unconstrained.
        tr_mask = y_tr.notna() & np.isfinite(X_tr.to_numpy()).all(axis=1)
        if int(tr_mask.sum()) == 0:
            logger.warning(f"{model_name}: fold {fold_id} has no usable training rows, skipping")
            continue
        X_tr_f = X_tr.loc[tr_mask]
        y_tr_f = y_tr.loc[tr_mask]
        train_idx_for_groups = np.asarray(sp.train_idx)[tr_mask.to_numpy()]

        try:
            scores = fit_predict(X_tr_f, y_tr_f, X_te, train_idx_for_groups)
        except Exception as exc:  # noqa: BLE001 - log + continue, walk-forward should be resilient
            logger.warning(f"{model_name}: fold {fold_id} failed: {exc}")
            continue

        out.iloc[sp.test_idx] = np.asarray(scores, dtype="float64")
        logger.info(
            f"{model_name}: fold {fold_id} train n={int(tr_mask.sum())} "
            f"[{sp.train_start.date()}..{sp.train_end.date()}] -> "
            f"test n={len(sp.test_idx)} [{sp.test_start.date()}..{sp.test_end.date()}]"
        )

    return out


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    # ---- Monthly vs weekly source -------------------------------------------
    # The --monthly switch picks features_monthly.parquet (last-Friday-of-month
    # rows), retunes the walk-forward defaults to monthly units, and routes
    # the OOF parquet to a separate filename so weekly + monthly runs don't
    # clobber each other.
    if args.monthly:
        feat_filename = "features_monthly.parquet"
        oof_filename = "oof_predictions_monthly.parquet"
        metrics_filename = "model_metrics_monthly.csv"
        # Override the defaults only when the user didn't pass an explicit
        # value (default sentinels are the weekly numbers).
        if args.min_train_weeks == 156:
            args.min_train_weeks = 36
        if args.test_weeks == 4:
            args.test_weeks = 6
        if args.embargo_weeks == 4:
            args.embargo_weeks = 1
        logger.info(
            "05: --monthly enabled | "
            f"min_train={args.min_train_weeks}m test={args.test_weeks}m embargo={args.embargo_weeks}m"
        )
    else:
        feat_filename = "features.parquet"
        oof_filename = "oof_predictions.parquet"
        metrics_filename = "model_metrics.csv"

    feat_path = settings.processed_dir / feat_filename
    if not feat_path.exists():
        logger.error(f"{feat_path} not found — must run 04_build_features.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"05_train_and_validate: start {t0.isoformat(timespec='seconds')}")

    features = read_parquet(feat_path)
    # Stable row identity throughout: positional integer index aligns with
    # walk_forward_splits' integer row positions.
    df = features.sort_values([_DATE_COL, _TICKER_COL]).reset_index(drop=True)
    fwd_col = _resolve_fwd_ret(df, args.horizon_weeks)
    logger.info(f"using forward-return column: {fwd_col}")

    feat_cols = _feature_cols(df)
    rk_cols = _rk_cols(df)
    if not feat_cols:
        logger.error("feature panel has no '*_rk' or 'sec_*' columns; nothing to train on")
        return 2
    logger.info(f"model inputs: {len(rk_cols)} *_rk + {len(feat_cols) - len(rk_cols)} sec_* = {len(feat_cols)} cols")

    # Binary + decile targets, per-date.
    y_bin = (df[fwd_col] < 0).astype("float64").where(df[fwd_col].notna())
    # decile_target returns 0..9 where 9 == highest forward return. For a SHORT
    # signal we want the ranker to pull "worst forward return" (decile 0) to the
    # top, so we invert: y_rank = 9 - decile  =>  decile 0 becomes relevance 9.
    decile = decile_target(df[fwd_col], by=df[_DATE_COL])
    y_rank = (9 - decile.astype("float64")).where(decile.notna())

    wf_common = dict(
        min_train_weeks=args.min_train_weeks,
        test_weeks=args.test_weeks,
        embargo_weeks=args.embargo_weeks,
    )
    gbm_cfg = GBMConfig()

    # --- 1. Naive baseline ---------------------------------------------------
    logger.info("model: naive_si_rank (no fit; pure cross-sectional rank)")
    naive_score = naive_si_rank(df).reindex(df.index)

    # --- 2. EW composite -----------------------------------------------------
    # The baselines module's DEFAULT_EW_COLS expects coarse theme ranks
    # (short_rk/momentum_rk/...) which don't exist in our feature panel - we
    # rank individual factors, not themes. Hand-pick one representative rank
    # column per family. Missing columns are tolerated by ew_composite.
    _EW_COLS = [
        c for c in (
            "short_pct_ff_rk",        # short interest as % of free float
            "ShortPct_rk",            # raw short pct (fallback)
            "si_z_52w_rk",            # SI z-score
            "mom_12w_rk",             # 12-week price momentum
            "vol_4w_rk",              # short-term realised vol
            "log_mktcap_rk",          # size
            "pe_rk", "fcf_yield_rk",  # valuation (expensive = positive)
            "roe_rk", "roic_rk",      # quality
            "debt_equity_rk",         # leverage
            "revenue_growth_yoy_rk",  # growth
        ) if c in df.columns
    ]
    logger.info(
        f"model: ew_composite (no fit; equal-weight blend over {len(_EW_COLS)} *_rk cols)"
    )
    ew_score = ew_composite(df, cols=_EW_COLS).reindex(df.index)

    # --- 3. Logistic regression (walk-forward) -------------------------------
    logger.info("model: logit (walk-forward)")

    def _fit_predict_logit(X_tr, y_tr, X_te, _train_idx):
        # fit_logit_baseline wants the panel + a target Series aligned on its index.
        # X_tr already carries the (positional) panel index; passing it as
        # ``features`` and y_tr as ``target`` works directly.
        model, used_cols = fit_logit_baseline(
            X_tr, y_tr.astype(int), feature_cols=list(X_tr.columns)
        )
        return predict_logit_baseline(model, X_te, used_cols).to_numpy()

    logit_score = _walkforward_predict(
        df, feat_cols, y_bin, _fit_predict_logit,
        model_name="logit", **wf_common,
    )

    # --- 4. GBM classifier (walk-forward) ------------------------------------
    logger.info("model: gbm_cls (walk-forward)")

    def _fit_predict_gbm_cls(X_tr, y_tr, X_te, _train_idx):
        model = fit_gbm_classifier(
            X_tr, y_tr.astype(int),
            feature_cols=list(X_tr.columns),
            cfg=gbm_cfg,
        )
        return predict_score(model, X_te, list(X_te.columns)).to_numpy()

    gbm_cls_score = _walkforward_predict(
        df, feat_cols, y_bin, _fit_predict_gbm_cls,
        model_name="gbm_cls", **wf_common,
    )

    # --- 5. GBM ranker (walk-forward) ----------------------------------------
    logger.info("model: gbm_rank (walk-forward)")

    def _fit_predict_gbm_rank(X_tr, y_tr, X_te, train_idx):
        # The ranker needs group_dates aligned with the (filtered) training rows.
        # train_idx is the positional row index into ``df`` for the kept rows.
        group_dates = df.iloc[train_idx][_DATE_COL].reset_index(drop=True)
        # Sort training rows by date so groups are contiguous (LightGBM requirement).
        order = np.argsort(group_dates.to_numpy(), kind="mergesort")
        X_tr_sorted = X_tr.iloc[order].reset_index(drop=True)
        y_tr_sorted = y_tr.iloc[order].reset_index(drop=True)
        gd_sorted = group_dates.iloc[order].reset_index(drop=True)
        model = fit_gbm_ranker(
            X_tr_sorted,
            y_tr_sorted.astype(int),
            gd_sorted,
            feature_cols=list(X_tr_sorted.columns),
            cfg=gbm_cfg,
        )
        return predict_score(model, X_te, list(X_te.columns)).to_numpy()

    gbm_rank_score = _walkforward_predict(
        df, feat_cols, y_rank, _fit_predict_gbm_rank,
        model_name="gbm_rank", **wf_common,
    )

    # --- Stack OOF predictions long ------------------------------------------
    def _stamp(name: str, score: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                _DATE_COL: df[_DATE_COL].values,
                _TICKER_COL: df[_TICKER_COL].values,
                "model": name,
                "score": score.reindex(df.index).to_numpy(),
                _FWD_RET_COL: df[fwd_col].values,
            }
        )

    oof = pd.concat(
        [
            _stamp("naive", naive_score),
            _stamp("ew", ew_score),
            _stamp("logit", logit_score),
            _stamp("gbm_cls", gbm_cls_score),
            _stamp("gbm_rank", gbm_rank_score),
        ],
        ignore_index=True,
    )
    write_parquet(oof, settings.reports_dir / oof_filename)
    logger.info(f"oof_predictions: {len(oof):,} rows across {oof['model'].nunique()} models")

    # --- Per-model metrics ---------------------------------------------------
    rows: list[dict] = []
    for name, grp in oof.groupby("model"):
        ic_series = rank_ic(grp["score"], grp[_FWD_RET_COL], by=grp[_DATE_COL])
        ic_stats = ic_summary(ic_series if isinstance(ic_series, pd.Series) else pd.Series([ic_series]))
        spread_series = decile_spread(
            grp["score"], grp[_FWD_RET_COL], by=grp[_DATE_COL], n=10,
        )
        rows.append(
            {
                "model": name,
                "ic_mean": float(ic_stats.get("mean", np.nan)),
                "ic_t": float(ic_stats.get("t_stat", np.nan)),
                "ic_hit_rate": float(ic_stats.get("hit_rate", np.nan)),
                "ic_n_periods": int(ic_stats.get("n", 0)),
                "decile_spread_mean": float(spread_series.mean()) if len(spread_series) else float("nan"),
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("ic_mean", ascending=False)
    metrics_df.to_csv(settings.reports_dir / metrics_filename, index=False)
    logger.info(f"model_metrics:\n{metrics_df.to_string(index=False)}")

    # --- Calibration table for the GBM classifier ----------------------------
    gbm_cls_oof = oof[oof["model"] == "gbm_cls"].dropna(subset=["score", _FWD_RET_COL])
    if len(gbm_cls_oof):
        labels = (gbm_cls_oof[_FWD_RET_COL] < 0).astype(int)
        cal = calibration_table(gbm_cls_oof["score"], labels, n_bins=10)
        cal.to_csv(settings.reports_dir / "calibration_gbm_cls.csv", index=False)
        logger.info(f"calibration_gbm_cls: {len(cal)} bins")
    else:
        logger.warning("gbm_cls produced no usable OOF scores; skipping calibration table")

    # --- Interpretability: full-sample GBM for SHAP / gain importance --------
    logger.info("interpret: fitting full-sample GBM classifier for SHAP / gain importance")
    # Cast to plain float64 (same nullable-Float64 reason as the walk-forward loop).
    _X_all = df[feat_cols].astype("float64", copy=False)
    train_mask = y_bin.notna() & np.isfinite(_X_all.to_numpy()).all(axis=1)
    if int(train_mask.sum()) == 0:
        logger.warning("no usable rows for full-sample GBM; skipping interpret artefacts")
    else:
        X_full = _X_all.loc[train_mask]
        y_full = y_bin.loc[train_mask].astype(int)
        full_gbm = fit_gbm_classifier(X_full, y_full, feature_cols=feat_cols, cfg=gbm_cfg)

        gi = gain_importance(full_gbm, feat_cols)
        gi.to_csv(settings.reports_dir / "gain_importance.csv", index=False)
        logger.info(f"gain_importance: top features = {gi['feature'].head(5).tolist()}")

        try:
            shap_df = shap_values_sampled(
                full_gbm, X_full, feat_cols, n_sample=args.shap_n_sample,
            )
            mas = mean_abs_shap(shap_df)
            mas.to_csv(settings.reports_dir / "mean_abs_shap.csv", index=False)
            plot_shap_summary(shap_df, X_full, settings.charts_dir / "shap_summary.png")
        except Exception as exc:  # noqa: BLE001 - shap is optional + heavy
            logger.warning(f"SHAP step failed (continuing without): {exc}")

    t1 = dt.datetime.now()
    logger.info(
        f"05_train_and_validate: wrote oof_predictions.parquet, model_metrics.csv "
        f"+ interpret artefacts | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
