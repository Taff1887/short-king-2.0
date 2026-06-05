"""Walk-forward train + validate the five short-signal models on the feature panel.

Models (all walk-forward, no look-ahead):

* ``naive``       - rank by raw short-interest percentile (industry baseline).
* ``ew``          - polarity-aware equal-weight composite of ``*_rk`` columns.
* ``logit``       - L2 logistic regression on the rank features (sklearn).
* ``gbm_cls``     - LightGBM binary classifier on ``Pr(monthly forward return < 0)``.
* ``gbm_rank``    - LightGBM cross-sectional LambdaRank on inverted decile target.

The OOF predictions land in ``reports/oof_predictions.parquet`` (+ the
monthly variant) and the per-model IC + decile-spread summary in
``reports/model_metrics.csv``. SHAP and gain-based feature-importance
artefacts for the GBM classifier are saved under ``reports/`` and
``charts/``.

NOTE: this project has been refocused on **short-signal quality**
(success rate + magnitude of identified shorts), not on portfolio
returns. The downstream pipeline analyses the OOF predictions
directly via ``_short_signal_analysis.py``; there is no portfolio
backtest engine, no L/S construction, no stop-loss logic, and no
trading-cost model. Every position is treated equally and the only
thing measured is "did the stock fall, and by how much?".
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

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
_FWD_RET_COL = "fwd_ret_1m"


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
    p.add_argument(
        "--holdout-months", type=int, default=36,
        help="Reserve the LAST N months as a pure out-of-sample holdout. "
             "Walk-forward CV runs only on the in-sample period (everything "
             "before holdout_start). After CV, a single final model is fit "
             "on the full IS data and scores the holdout - this is the honest "
             "OOS performance estimate (CV folds are technically OOF too but "
             "feature selection / hyperparameters may have implicitly seen "
             "the later years). Default 36 = 3 years. Set to 0 to disable.",
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
        # and tree-model native arrays. Impute NaN to 0.5 (the neutral
        # cross-sectional rank) so linear models (logit) don't blow up;
        # tree models (LightGBM) handle NaN natively but the imputation
        # leaves them just as expressive (a tree can still split on 0.5).
        # Without this, requiring every feature finite would drop ~98 % of
        # training rows because at least one *_rk column is NaN for most
        # name-dates on a 573-column panel.
        X_tr = df.iloc[sp.train_idx][feat_cols].astype("float64", copy=False).fillna(0.5)
        X_te = df.iloc[sp.test_idx][feat_cols].astype("float64", copy=False).fillna(0.5)
        y_tr = y.iloc[sp.train_idx]

        # Drop NaN labels from training; X is already imputed above.
        tr_mask = y_tr.notna()
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
        # `walk_forward_splits` measures windows in CALENDAR weeks. For a
        # monthly panel we want 36 months train / 6 months test / 1 month
        # embargo, which is ~ 156 / 26 / 4 calendar weeks. (Same numbers as
        # the weekly default for train, which is intentional - 3 years.)
        # Override only when the user didn't pass an explicit value.
        if args.min_train_weeks == 156:
            args.min_train_weeks = 156   # 36 months of training data
        if args.test_weeks == 4:
            args.test_weeks = 26         # 6-month test windows (~6 monthly dates)
        if args.embargo_weeks == 4:
            args.embargo_weeks = 4       # 1-month embargo (>= label horizon)
        logger.info(
            "05: --monthly enabled | "
            f"min_train={args.min_train_weeks}w (~36m) "
            f"test={args.test_weeks}w (~6m) "
            f"embargo={args.embargo_weeks}w (~1m)"
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

    # ---- In-sample / Out-of-sample split ------------------------------------
    # Reserve the LAST `holdout_months` calendar months (or weeks, for the
    # weekly run) as a pure OOS holdout. Walk-forward CV operates on the IS
    # portion only. A final model is fit on the full IS data and scores the
    # holdout - that holdout Sharpe is the unbiased OOS estimate.
    dates_unique = pd.to_datetime(df[_DATE_COL].unique())
    holdout_n = int(args.holdout_months)
    # Need at least the holdout period + a token training window (12 obs).
    # holdout_n is in months (or rows of the monthly panel - same thing).
    if holdout_n > 0 and len(dates_unique) > holdout_n + 12:
        holdout_start = pd.to_datetime(sorted(dates_unique)[-holdout_n])
        is_mask = df[_DATE_COL] < holdout_start
        oos_mask = ~is_mask
        logger.info(
            f"IS/OOS split: holdout_start={holdout_start.date()} | "
            f"IS rows={int(is_mask.sum()):,} ({pd.to_datetime(df.loc[is_mask, _DATE_COL]).min().date()} "
            f"-> {pd.to_datetime(df.loc[is_mask, _DATE_COL]).max().date()}) | "
            f"OOS rows={int(oos_mask.sum()):,} ({pd.to_datetime(df.loc[oos_mask, _DATE_COL]).min().date()} "
            f"-> {pd.to_datetime(df.loc[oos_mask, _DATE_COL]).max().date()})"
        )
    else:
        holdout_start = None
        is_mask = pd.Series(True, index=df.index)
        oos_mask = pd.Series(False, index=df.index)
        if holdout_n > 0:
            logger.warning("holdout disabled - not enough periods")

    feat_cols = _feature_cols(df)
    rk_cols = _rk_cols(df)
    if not feat_cols:
        logger.error("feature panel has no '*_rk' or 'sec_*' columns; nothing to train on")
        return 2
    logger.info(f"model inputs: {len(rk_cols)} *_rk + {len(feat_cols) - len(rk_cols)} sec_* = {len(feat_cols)} cols")

    # Binary + decile targets, per-date.
    y_bin = (df[fwd_col] < 0).astype("float64").where(df[fwd_col].notna())
    # decile_target returns 0..9 where 9 == highest forward return. For a SHORT
    # signal we want the ranker to pull "worst forward return" (decile 0) to
    # the top, so we invert: y_rank = 9 - decile  =>  decile 0 becomes
    # relevance 9.
    decile = decile_target(df[fwd_col], by=df[_DATE_COL])
    y_rank = (9 - decile.astype("float64")).where(decile.notna())

    wf_common = dict(
        min_train_weeks=args.min_train_weeks,
        test_weeks=args.test_weeks,
        embargo_weeks=args.embargo_weeks,
    )
    gbm_cfg = GBMConfig()

    # IS-only frame for walk-forward CV.
    df_is = df.loc[is_mask].reset_index(drop=True)
    y_bin_is = y_bin.loc[is_mask].reset_index(drop=True)
    y_rank_is = y_rank.loc[is_mask].reset_index(drop=True)

    def _is_oos_score(
        cv_score_is: pd.Series,
        fit_predict: callable,
        y_full: pd.Series,
    ) -> pd.Series:
        """Combine walk-forward IS scores with a final-model OOS prediction.

        ``cv_score_is`` is positional-indexed against ``df_is``; ``fit_predict``
        is the same callable used in walk-forward (X_tr, y_tr, X_te, train_idx).
        We fit one final model on the entire IS panel and score the OOS rows.
        The output is a Series of length ``len(df)``, IS rows from cv_score_is
        and OOS rows from the final-model prediction (NaN where holdout is off).
        """
        full = pd.Series(np.nan, index=df.index, name="score", dtype="float64")
        full.loc[is_mask] = cv_score_is.to_numpy()
        if holdout_start is not None and int(oos_mask.sum()) > 0:
            # Fit final model on the FULL IS panel. Impute NaN features to
            # 0.5 (neutral rank) - same convention as the per-fold driver.
            X_full_is = df_is[feat_cols].astype("float64", copy=False).fillna(0.5)
            keep = y_full.loc[is_mask].notna()
            train_idx_is = np.where(keep.to_numpy())[0]
            try:
                X_te_oos = df.loc[oos_mask, feat_cols].astype("float64", copy=False).fillna(0.5)
                oos_score = fit_predict(
                    X_full_is.loc[keep].reset_index(drop=True),
                    y_full.loc[is_mask].loc[keep].reset_index(drop=True),
                    X_te_oos,
                    train_idx_is,
                )
                full.loc[oos_mask] = np.asarray(oos_score, dtype="float64")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"OOS final-fit failed: {exc}")
        return full

    # --- 1. Naive baseline ---------------------------------------------------
    logger.info("model: naive_si_rank (no fit; pure cross-sectional rank)")
    # Naive scores the entire panel; both IS and OOS rows score naturally.
    naive_score = naive_si_rank(df).reindex(df.index)

    # --- 2. EW composite -----------------------------------------------------
    # Every input rank must be SHORT-aligned (high rank = MORE shortable) for
    # the equal-weight mean to be a coherent short-attractiveness score.
    # Cross-sectional rank columns built upstream are oriented to the raw
    # factor direction, so bullish-quality / bullish-growth factors need to
    # be FLIPPED before averaging. Polarity table:
    #
    #   +1  high rank already means more shortable -> keep
    #   -1  high rank means a "good" company       -> invert via (1 - rank)
    #
    # Earlier versions of this script averaged the raw ranks directly, which
    # caused EW to score WINNERS (positive IC ~ +5 %) - the bullish-quality
    # factors dominated. The polarity-aware version below produces a proper
    # SHORT composite with negative IC.
    _EW_SPEC: tuple[tuple[str, int], ...] = (
        ("short_pct_ff_rk",        +1),  # short interest / free float
        ("ShortPct_rk",            +1),  # raw short pct
        ("si_z_12m_rk",            +1),  # SI z-score vs 52w history
        ("mom_3m_rk",             -1),  # high momentum = bullish, invert
        ("vol_1m_rk",              +1),  # high vol = low-vol anomaly says shortable
        ("log_mktcap_rk",          -1),  # mega-caps less shortable, invert
        ("pe_rk",                  +1),  # high P/E = expensive = shortable
        ("fcf_yield_rk",           -1),  # cash-rich = bullish, invert
        ("roe_rk",                 -1),  # high ROE = quality = bullish, invert
        ("roic_rk",                -1),  # high ROIC = quality = bullish, invert
        ("debt_equity_rk",         +1),  # high leverage = shortable
        ("revenue_growth_yoy_rk",  -1),  # high growth = bullish, invert
    )
    _EW_COLS = [c for c, _ in _EW_SPEC if c in df.columns]
    _EW_INVERT = [c for c, sign in _EW_SPEC if sign == -1 and c in df.columns]

    # Build a SHORT-oriented copy of just the EW columns. Bullish factors get
    # flipped via (1 - rank). The underlying cross-sectional ranks live in
    # [0, 1] so the flip stays in [0, 1].
    df_ew = df.copy()
    for c in _EW_INVERT:
        df_ew[c] = 1.0 - df_ew[c]

    logger.info(
        f"model: ew_composite (polarity-aware; {len(_EW_COLS)} cols, "
        f"{len(_EW_INVERT)} flipped for short-alignment: {_EW_INVERT})"
    )
    ew_score = ew_composite(df_ew, cols=_EW_COLS).reindex(df.index)

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

    logit_score_is = _walkforward_predict(
        df_is, feat_cols, y_bin_is, _fit_predict_logit,
        model_name="logit (IS CV)", **wf_common,
    )
    logit_score = _is_oos_score(logit_score_is, _fit_predict_logit, y_bin)

    # --- 4. GBM classifier (walk-forward) ------------------------------------
    logger.info("model: gbm_cls (walk-forward)")

    def _fit_predict_gbm_cls(X_tr, y_tr, X_te, _train_idx):
        model = fit_gbm_classifier(
            X_tr, y_tr.astype(int),
            feature_cols=list(X_tr.columns),
            cfg=gbm_cfg,
        )
        return predict_score(model, X_te, list(X_te.columns)).to_numpy()

    gbm_cls_score_is = _walkforward_predict(
        df_is, feat_cols, y_bin_is, _fit_predict_gbm_cls,
        model_name="gbm_cls (IS CV)", **wf_common,
    )
    gbm_cls_score = _is_oos_score(gbm_cls_score_is, _fit_predict_gbm_cls, y_bin)

    # --- 5. GBM ranker (walk-forward) ----------------------------------------
    logger.info("model: gbm_rank (walk-forward)")

    def _fit_predict_gbm_rank(X_tr, y_tr, X_te, train_idx):
        # The ranker needs group_dates aligned with the (filtered) training
        # rows. train_idx is the positional row index into df_is during CV
        # (and during final-fit) - use it to look up the matching Date.
        source = df_is
        group_dates = source.iloc[train_idx][_DATE_COL].reset_index(drop=True)
        # Sort by date so groups are contiguous (LightGBM ranker requirement).
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

    gbm_rank_score_is = _walkforward_predict(
        df_is, feat_cols, y_rank_is, _fit_predict_gbm_rank,
        model_name="gbm_rank (IS CV)", **wf_common,
    )
    gbm_rank_score = _is_oos_score(gbm_rank_score_is, _fit_predict_gbm_rank, y_rank)

    # --- Stack OOF predictions long ------------------------------------------
    period_labels = np.where(is_mask.to_numpy(), "IS", "OOS")

    def _stamp(name: str, score: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                _DATE_COL: df[_DATE_COL].values,
                _TICKER_COL: df[_TICKER_COL].values,
                "model": name,
                "score": score.reindex(df.index).to_numpy(),
                _FWD_RET_COL: df[fwd_col].values,
                "period": period_labels,
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

    # --- Per-model metrics (computed separately for IS-CV vs OOS-holdout) ----
    rows: list[dict] = []
    for name, grp in oof.groupby("model"):
        for period_label in ("ALL", "IS", "OOS"):
            sub = grp if period_label == "ALL" else grp[grp["period"] == period_label]
            if sub.empty or sub["score"].isna().all():
                continue
            ic_series = rank_ic(sub["score"], sub[_FWD_RET_COL], by=sub[_DATE_COL])
            ic_stats = ic_summary(
                ic_series if isinstance(ic_series, pd.Series) else pd.Series([ic_series])
            )
            spread_series = decile_spread(
                sub["score"], sub[_FWD_RET_COL], by=sub[_DATE_COL], n=10,
            )
            rows.append(
                {
                    "model": name,
                    "period": period_label,
                    "ic_mean": float(ic_stats.get("mean", np.nan)),
                    "ic_t": float(ic_stats.get("t_stat", np.nan)),
                    "ic_hit_rate": float(ic_stats.get("hit_rate", np.nan)),
                    "ic_n_periods": int(ic_stats.get("n", 0)),
                    "decile_spread_mean": float(spread_series.mean()) if len(spread_series) else float("nan"),
                }
            )
    metrics_df = pd.DataFrame(rows).sort_values(["model", "period"])
    metrics_df.to_csv(settings.reports_dir / metrics_filename, index=False)
    logger.info(f"model_metrics:\n{metrics_df.to_string(index=False)}")

    # --- Calibration table for the GBM classifier ----------------------------
    gbm_cls_oof = oof[oof["model"] == "gbm_cls"].dropna(subset=["score", _FWD_RET_COL])
    if len(gbm_cls_oof):
        labels = (gbm_cls_oof[_FWD_RET_COL] < 0).astype(int)
        cal = calibration_table(gbm_cls_oof["score"], labels, n_bins=10)
        cal.to_csv(settings.reports_dir / "calibration_gbm_cls.csv", index=False)
        logger.info(f"calibration_gbm_cls: {len(cal)} bins")

    # --- Interpretability: full-sample GBM for SHAP / gain importance --------
    logger.info("interpret: fitting full-sample GBM classifier for SHAP / gain importance")
    _X_all = df[feat_cols].astype("float64", copy=False).fillna(0.5)
    train_mask = y_bin.notna()
    if int(train_mask.sum()) > 0:
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
