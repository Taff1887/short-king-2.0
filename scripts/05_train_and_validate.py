"""Walk-forward train + validate every model on the feature panel.

Models compared (all walk-forward, no look-ahead):

* ``naive``       - rank by raw short-interest percentile (industry baseline).
* ``ew``          - equal-weight composite of the *_rk feature columns.
* ``logit``       - logistic regression on the rank features (sklearn).
* ``gbm_cls``     - LightGBM binary classifier on fwd_ret_4w < 0.
* ``gbm_rank``    - LightGBM cross-sectional ranker on the decile target.

Out-of-fold predictions land in ``reports/oof_predictions.parquet`` and the
per-model metrics table in ``reports/model_metrics.csv``. SHAP and gain-based
feature-importance charts for the GBM classifier are saved under ``charts/``.
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
from short_king.models.walk_forward import fit_predict_walkforward
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger

_TICKER_COL = "Ticker"
_DATE_COL = "Date"
_FWD_RET_COL = "ret_fwd_4w"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon-weeks", type=int, default=4)
    p.add_argument("--train-window-weeks", type=int, default=156, help="~3y training window")
    p.add_argument("--step-weeks", type=int, default=4)
    p.add_argument("--embargo-weeks", type=int, default=4)
    p.add_argument("--n-deciles", type=int, default=10)
    return p.parse_args()


def _resolve_fwd_ret(features: pd.DataFrame, horizon_weeks: int) -> str:
    """Pick the forward-return column to use as the label source."""
    candidates = [
        _FWD_RET_COL,
        f"ret_fwd_{horizon_weeks}w",
        "ret_fwd",
        "y_fwd_ret",
    ]
    for c in candidates:
        if c in features.columns:
            return c
    raise KeyError(
        f"no forward-return column in features (looked for {candidates}). "
        "Build features with a label column before training."
    )


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    feat_path = settings.processed_dir / "features.parquet"
    if not feat_path.exists():
        logger.error(f"{feat_path} not found — must run 04_build_features.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"05_train_and_validate: start {t0.isoformat(timespec='seconds')}")

    features = read_parquet(feat_path)
    fwd_col = _resolve_fwd_ret(features, args.horizon_weeks)
    logger.info(f"using forward-return column: {fwd_col}")

    # Binary + decile targets, per-date.
    df = features.copy()
    df["y_bin"] = (df[fwd_col] < 0).astype("Int8")
    df["y_dec"] = decile_target(df, label_col=fwd_col, date_col=_DATE_COL, n=args.n_deciles)

    # Common walk-forward config used by every model.
    wf_kwargs = dict(
        date_col=_DATE_COL,
        train_window=args.train_window_weeks,
        step=args.step_weeks,
        embargo=args.embargo_weeks,
    )

    # Naive + EW composites: no fitting, but we still walk-forward so the
    # downstream metrics share a sample.
    logger.info("model: naive_si_rank")
    naive_pred = naive_si_rank(df, date_col=_DATE_COL)
    logger.info("model: ew_composite")
    ew_pred = ew_composite(df, date_col=_DATE_COL)

    logger.info("model: logit (walk-forward)")
    logit_pred = fit_predict_walkforward(
        df,
        fit_fn=fit_logit_baseline,
        predict_fn=predict_logit_baseline,
        label_col="y_bin",
        **wf_kwargs,
    )

    logger.info("model: gbm_cls (walk-forward)")
    gbm_cfg = GBMConfig()
    gbm_cls_pred = fit_predict_walkforward(
        df,
        fit_fn=lambda X, y: fit_gbm_classifier(X, y, cfg=gbm_cfg),
        predict_fn=predict_score,
        label_col="y_bin",
        **wf_kwargs,
    )

    logger.info("model: gbm_rank (walk-forward)")
    gbm_rank_pred = fit_predict_walkforward(
        df,
        fit_fn=lambda X, y: fit_gbm_ranker(X, y, cfg=gbm_cfg),
        predict_fn=predict_score,
        label_col="y_dec",
        **wf_kwargs,
    )

    # Stack OOF predictions long for downstream backtest / metrics.
    def _stamp(name: str, pred: pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(pred, pd.Series):
            out = pred.rename("score").to_frame()
        else:
            out = pred.copy()
        if _TICKER_COL not in out.columns or _DATE_COL not in out.columns:
            ids = df.loc[out.index, [_TICKER_COL, _DATE_COL, fwd_col]].reset_index(drop=True)
            out = out.reset_index(drop=True)
            out[_TICKER_COL] = ids[_TICKER_COL]
            out[_DATE_COL] = ids[_DATE_COL]
            out[fwd_col] = ids[fwd_col]
        out["model"] = name
        return out[[_DATE_COL, _TICKER_COL, "model", "score", fwd_col]]

    oof = pd.concat(
        [
            _stamp("naive", naive_pred),
            _stamp("ew", ew_pred),
            _stamp("logit", logit_pred),
            _stamp("gbm_cls", gbm_cls_pred),
            _stamp("gbm_rank", gbm_rank_pred),
        ],
        ignore_index=True,
    )
    write_parquet(oof, settings.reports_dir / "oof_predictions.parquet")

    # Per-model metrics: IC summary + decile spread + calibration sample.
    rows: list[dict] = []
    for name, grp in oof.groupby("model"):
        ic = rank_ic(grp, score_col="score", label_col=fwd_col, date_col=_DATE_COL)
        ic_stats = ic_summary(ic)
        spread = decile_spread(grp, score_col="score", label_col=fwd_col, date_col=_DATE_COL)
        rows.append(
            {
                "model": name,
                "ic_mean": float(ic_stats.get("mean", np.nan)),
                "ic_t": float(ic_stats.get("t_stat", np.nan)),
                "ic_hit_rate": float(ic_stats.get("hit_rate", np.nan)),
                "decile_spread_mean": float(spread.get("mean", np.nan)) if hasattr(spread, "get") else np.nan,
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("ic_mean", ascending=False)
    metrics_df.to_csv(settings.reports_dir / "model_metrics.csv", index=False)
    logger.info(f"model_metrics:\n{metrics_df}")

    # Calibration table for the GBM classifier (probability-output model).
    cal = calibration_table(
        oof[oof["model"] == "gbm_cls"],
        score_col="score",
        label_col=fwd_col,
        n_bins=10,
    )
    cal.to_csv(settings.reports_dir / "calibration_gbm_cls.csv", index=False)

    # Interpretability: fit a single GBM on the full feature panel for SHAP.
    logger.info("interpret: fitting full-sample GBM for SHAP / gain importance")
    y_full = df["y_bin"]
    X_full = df.drop(columns=[c for c in df.columns if c.startswith("y_") or c == fwd_col],
                     errors="ignore")
    full_gbm = fit_gbm_classifier(X_full, y_full, cfg=gbm_cfg)

    gi = gain_importance(full_gbm)
    gi.to_csv(settings.reports_dir / "gain_importance.csv", index=False)

    shap_vals = shap_values_sampled(full_gbm, X_full, n_samples=2000)
    mas = mean_abs_shap(shap_vals)
    mas.to_csv(settings.reports_dir / "mean_abs_shap.csv", index=False)
    plot_shap_summary(shap_vals, X_full, out_path=settings.charts_dir / "shap_summary.png")

    t1 = dt.datetime.now()
    logger.info(
        f"05_train_and_validate: wrote oof_predictions.parquet, model_metrics.csv, "
        f"shap_summary.png | took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
