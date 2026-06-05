"""Model interpretability: gain importance + SHAP value attribution.

A tree-based factor model is only useful if we can explain *why* it ranks a
name as a likely short. Vanilla LightGBM gain importance answers "which
features did the booster split on most usefully" but it is dataset-level and
biased toward high-cardinality features; SHAP values give per-row Shapley
attributions that are consistent across models and additive (sum to the
margin prediction). This module exposes both: cheap gain/split counts for
quick dashboards, plus a sampled SHAP pass (full-panel SHAP on a multi-year
panel is prohibitively expensive) and two paper-quality matplotlib charts -
a global mean-|SHAP| bar and per-feature dependence scatters - that get
embedded in the methodology report.

The sampling is deterministic by ``random_state`` so re-runs of the report
pick up exactly the same rows; SHAP TreeExplainer is the LightGBM-aware
exact path (no background dataset needed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from short_king.utils.io import ensure_dir  # noqa: E402
from short_king.utils.logging import logger  # noqa: E402

# House style: a single navy bar/scatter color keeps the SHAP charts visually
# consistent with the rest of the report deck (see qfr.utils.viz.PALETTE).
_PRIMARY = "#1b3a5b"
_ACCENT = "#c1121f"


# --- Gain / split importance --------------------------------------------------
def gain_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Return per-feature gain + split counts as a tidy ranked frame.

    LightGBM exposes two complementary importance signals on the underlying
    booster: ``importance_type='gain'`` (total information gain attributed to
    splits on the feature - what ``model.feature_importances_`` returns for
    sklearn-style wrappers when configured with the default
    ``importance_type='split'``... so we go through the booster directly to
    avoid that footgun) and ``importance_type='split'`` (raw count of times
    the feature was used as a split variable). Both are dataset-level
    summaries; SHAP below gives the per-row decomposition.
    """
    if not feature_cols:
        raise ValueError("gain_importance requires a non-empty feature_cols list")

    booster = getattr(model, "booster_", None)
    if booster is None:
        # Plain ``lgb.Booster`` (not a sklearn wrapper). The Booster object
        # itself exposes ``feature_importance`` and ``feature_name``.
        booster = model

    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)

    # ``feature_cols`` is the canonical training order; sanity-check the
    # booster's own feature_name matches so we never silently mis-label.
    bf_names = list(booster.feature_name())
    if len(bf_names) != len(feature_cols):
        raise ValueError(
            f"feature_cols length {len(feature_cols)} != booster feature count {len(bf_names)}"
        )
    if bf_names != list(feature_cols):
        logger.warning(
            "gain_importance: booster feature order differs from feature_cols; "
            "trusting booster order"
        )
        feature_cols = bf_names

    out = pd.DataFrame({"feature": feature_cols, "gain": gain, "split": split})
    out = out.sort_values("gain", ascending=False).reset_index(drop=True)
    return out


# --- SHAP value computation ---------------------------------------------------
def shap_values_sampled(
    model,
    features: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_sample: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute SHAP values on a deterministic sub-sample of ``features``.

    TreeExplainer on LightGBM is exact and O(TLD^2) per row - cheap per row,
    but a multi-year cross-section is hundreds of thousands of rows so we
    sample. ``n_sample`` rows are drawn uniformly without replacement (seeded
    by ``random_state``); the returned frame has one column per feature plus
    ``idx`` carrying the original ``features.index`` value so callers can
    join SHAP back to the panel (date / ticker / sector).
    """
    if not feature_cols:
        raise ValueError("shap_values_sampled requires a non-empty feature_cols list")
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise KeyError(f"features is missing model columns: {missing[:5]}...")

    # Local import: shap is an optional-heavy dep and we don't want every
    # downstream module that imports ``short_king.models`` to pay the import
    # cost (numba, scipy, etc.).
    import shap

    n_avail = len(features)
    if n_avail == 0:
        raise ValueError("shap_values_sampled received an empty features frame")
    n_take = int(min(n_sample, n_avail))

    if n_take < n_avail:
        sampled = features.sample(n=n_take, random_state=random_state)
    else:
        sampled = features
    X = sampled[feature_cols]
    logger.info(
        f"shap_values_sampled: sampled {len(sampled):,} of {n_avail:,} rows "
        f"({len(feature_cols)} features)"
    )

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # LightGBM binary classifiers return a list[ndarray] (one per class);
    # take the positive-class contributions. Rankers and recent SHAP versions
    # return a single ndarray of shape (n, p) directly.
    if isinstance(raw, list):
        # Two-class case: prefer class 1 (the "is-a-short" positive). For a
        # multi-class hand-back, also pick the last (highest) class.
        arr = np.asarray(raw[-1])
    else:
        arr = np.asarray(raw)
        # Some recent shap versions return (n, p, n_classes) for classifiers.
        if arr.ndim == 3:
            arr = arr[..., -1]

    if arr.shape != (len(sampled), len(feature_cols)):
        raise ValueError(
            f"unexpected SHAP shape {arr.shape}; "
            f"expected ({len(sampled)}, {len(feature_cols)})"
        )

    shap_df = pd.DataFrame(arr, columns=list(feature_cols), index=sampled.index)
    shap_df.insert(0, "idx", sampled.index.to_numpy())
    shap_df = shap_df.reset_index(drop=True)
    return shap_df


def mean_abs_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-row SHAP values to a global feature ranking.

    Mean-|SHAP| is the standard global-importance summary: unlike gain it is
    on the model-output scale (margin units) so per-feature contributions
    are directly comparable across features and across walk-forward folds.
    """
    if shap_df is None or len(shap_df) == 0:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    feat_cols = [c for c in shap_df.columns if c != "idx"]
    if not feat_cols:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    vals = shap_df[feat_cols].abs().mean(axis=0)
    out = (
        pd.DataFrame({"feature": vals.index, "mean_abs_shap": vals.to_numpy()})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return out


# --- Charts -------------------------------------------------------------------
def plot_shap_summary(
    shap_df: pd.DataFrame,
    features: pd.DataFrame,  # noqa: ARG001 - kept for API symmetry / future overlays
    path: Path,
    *,
    top_n: int = 15,
) -> Path:
    """Save a horizontal bar chart of the top ``top_n`` mean-|SHAP| features.

    Deliberately not using ``shap.summary_plot``: that helper opens a window
    in interactive backends, depends on ``shap``'s plotting stack and produces
    a beeswarm that doesn't compose well into a multi-chart methodology PDF.
    A clean horizontal bar with the values annotated is more report-friendly.
    """
    path = Path(path)
    ensure_dir(path.parent)

    summary = mean_abs_shap(shap_df).head(int(top_n))
    if len(summary) == 0:
        # Still write a placeholder so the report build doesn't break on an
        # empty SHAP run (e.g. n_sample=0 or empty panel during smoke tests).
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no SHAP data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    # Reverse so the largest bar is on top when plotted horizontally.
    summary = summary.iloc[::-1].reset_index(drop=True)

    height = max(3.0, 0.35 * len(summary) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(summary["feature"], summary["mean_abs_shap"], color=_PRIMARY, edgecolor="none")
    ax.set_xlabel("mean(|SHAP|)  -  average impact on model output")
    ax.set_title(f"SHAP feature importance (top {len(summary)})")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate each bar with its value (kept terse - 3 sig-figs).
    xmax = float(summary["mean_abs_shap"].max())
    pad = xmax * 0.01 if xmax > 0 else 0.0
    for y, v in enumerate(summary["mean_abs_shap"].to_numpy()):
        ax.text(v + pad, y, f"{v:.3g}", va="center", fontsize=9, color="#333333")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info(f"plot_shap_summary: wrote {path}")
    return path


def plot_top_feature_dependence(
    shap_df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    top_n: int = 4,
    outdir: Path | None = None,
) -> list[Path]:
    """Per-feature dependence scatters for the ``top_n`` features by mean-|SHAP|.

    Each chart plots feature value (x) vs SHAP contribution (y), colored by
    the feature value itself. This is the canonical SHAP "dependence plot"
    minus the auto-picked interaction overlay - we keep it deliberately
    simple so the chart reads at-a-glance in a research deck.

    PNGs are written under ``outdir`` (default: cwd) as
    ``shap_dependence_<feature>.png`` and the list of paths is returned.
    """
    if shap_df is None or len(shap_df) == 0:
        logger.warning("plot_top_feature_dependence: empty shap_df, nothing to plot")
        return []

    out_dir = Path(outdir) if outdir is not None else Path.cwd()
    ensure_dir(out_dir)

    ranking = mean_abs_shap(shap_df).head(int(top_n))
    top_features = [f for f in ranking["feature"].tolist() if f in features.columns]
    if not top_features:
        logger.warning("plot_top_feature_dependence: no top features intersect features columns")
        return []

    # ``shap_df.idx`` (if present) holds the original ``features.index`` values
    # for the sampled rows; align both sides on it so feature value and SHAP
    # value belong to the same row even if either frame was re-indexed.
    if "idx" in shap_df.columns:
        idx = shap_df["idx"].to_numpy()
        try:
            x_frame = features.loc[idx, top_features]
        except KeyError:
            # Fall back to positional alignment if the index doesn't match
            # (e.g. caller reset_index'd features after training).
            logger.warning(
                "plot_top_feature_dependence: shap_df.idx not found in features.index, "
                "falling back to positional alignment"
            )
            x_frame = features[top_features].iloc[: len(shap_df)].reset_index(drop=True)
            shap_df = shap_df.reset_index(drop=True)
    else:
        x_frame = features[top_features].iloc[: len(shap_df)].reset_index(drop=True)
        shap_df = shap_df.reset_index(drop=True)

    x_frame = x_frame.reset_index(drop=True)
    shap_vals = shap_df.reset_index(drop=True)

    out_paths: list[Path] = []
    for feat in top_features:
        x = pd.to_numeric(x_frame[feat], errors="coerce").to_numpy()
        y = pd.to_numeric(shap_vals[feat], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() == 0:
            logger.warning(f"plot_top_feature_dependence: '{feat}' has no finite points, skipping")
            continue
        xs, ys = x[mask], y[mask]

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            xs,
            ys,
            c=xs,
            cmap="viridis",
            alpha=0.65,
            s=14,
            edgecolor="none",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(feat)

        ax.axhline(0.0, color=_ACCENT, linewidth=0.8, alpha=0.6)
        ax.set_xlabel(feat)
        ax.set_ylabel(f"SHAP value for {feat}")
        ax.set_title(f"SHAP dependence: {feat}")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # File-safe slug (feature names like 'short_int_pct/adv_rk' do appear).
        slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in feat)
        path = out_dir / f"shap_dependence_{slug}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        out_paths.append(path)
        logger.info(f"plot_top_feature_dependence: wrote {path}")

    return out_paths


__all__ = [
    "gain_importance",
    "shap_values_sampled",
    "mean_abs_shap",
    "plot_shap_summary",
    "plot_top_feature_dependence",
]
