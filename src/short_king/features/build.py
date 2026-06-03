"""Feature-panel orchestrator.

Single entry point for the modelling / backtest stage. Reads the clean PIT panel
produced by ``short_king.data`` and stitches the per-family feature modules
(short interest, price/momentum, liquidity, valuation, quality, leverage-growth)
into one wide feature matrix, then turns every numeric raw feature into a
within-date cross-sectional percentile rank (``*_rk``) - these ranks are the
actual model inputs because the raw factors are extremely fat-tailed and a
handful of names would otherwise dominate a z-score model. One-hot sector
dummies (``sec_*``) are appended last so a tree model can absorb sector
specific base rates without us having to demean by sector by hand.

The companion ``write_feature_panel`` persists to ``data/processed/features.parquet``
and ``feature_summary`` produces a one-page sanity table (coverage + top-pair
correlations) for the methodology notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import write_parquet
from short_king.utils.logging import logger

# Order matters: short-interest first (it's the headline signal family for this
# project), then technicals, then liquidity, then the fundamentals stack. Each
# entry maps to a module ``short_king.features.<name>`` that must expose a
# ``<name>_panel(panel) -> DataFrame`` function returning the original panel
# augmented with that family's new columns (same row identity, no row drops).
FEATURE_FAMILIES: list[str] = [
    "short",
    "price",
    "liquidity",
    "valuation",
    "quality",
    "leverage_growth",
]

# Columns that are panel identifiers / labels / passthroughs - never ranked,
# never sector-dummied, never reported as features.
_ID_COLS: frozenset[str] = frozenset(
    {
        "Ticker",
        "ticker",
        "symbol",
        "Date",
        "date",
        "sector",
        "industry",
        "exchange",
        "name",
        "company",
        "investable",
        "period_end",
    }
)

# Forward-return / label columns produced by the data layer. We must never rank
# these (would leak the label into the feature matrix) or sector-dummy them.
_LABEL_PREFIXES: tuple[str, ...] = ("ret_fwd", "y_", "label_", "target")


def _family_panel_fn(family: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Resolve ``<family>_panel`` from ``short_king.features.<family>`` lazily.

    Import is deferred so a missing sibling module only blows up when that
    family is actually requested, not at orchestrator import time.
    """
    import importlib

    module = importlib.import_module(f"short_king.features.{family}")
    fn_name = f"{family}_panel"
    if not hasattr(module, fn_name):
        raise AttributeError(
            f"short_king.features.{family} does not expose {fn_name}(panel)"
        )
    return getattr(module, fn_name)


def _ticker_date_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return the (ticker, date) column names actually present on ``df``."""
    tcol = "Ticker" if "Ticker" in df.columns else "ticker" if "ticker" in df.columns else "symbol"
    dcol = "Date" if "Date" in df.columns else "date"
    if tcol not in df.columns or dcol not in df.columns:
        raise KeyError(
            "panel must contain a ticker column (Ticker/ticker/symbol) "
            "and a date column (Date/date)"
        )
    return tcol, dcol


def _is_label(col: str) -> bool:
    return any(col.startswith(p) for p in _LABEL_PREFIXES)


def _rankable_columns(df: pd.DataFrame) -> list[str]:
    """Numeric raw-feature columns (excludes ids, labels, existing ranks, dummies)."""
    out: list[str] = []
    for c in df.columns:
        if c in _ID_COLS or _is_label(c):
            continue
        if c.endswith("_rk") or c.startswith("sec_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            out.append(c)
    return out


def cross_sectional_rank(
    df: pd.DataFrame,
    cols: list[str],
    *,
    by: str = "Date",
    suffix: str = "_rk",
) -> pd.DataFrame:
    """Append within-date percentile ranks of ``cols`` as ``<col><suffix>``.

    Ranks are pct-scaled to [0, 1], NaNs stay NaN. Single-stock dates (where
    rank is undefined) and degenerate columns (all-NaN within a date) are
    handled by pandas' rank semantics; we replace +/-inf inputs first so a
    div-by-zero raw factor doesn't poison the ranking.
    """
    if not cols:
        return df
    work = df[cols].replace([np.inf, -np.inf], np.nan)
    ranks = work.groupby(df[by], sort=False).rank(pct=True)
    ranks.columns = [f"{c}{suffix}" for c in cols]
    return pd.concat([df, ranks], axis=1)


def _sector_dummies(df: pd.DataFrame, prefix: str = "sec_") -> pd.DataFrame:
    """One-hot encode ``sector`` with ``prefix``. No-op if column missing."""
    if "sector" not in df.columns:
        logger.warning("add_sector_dummies requested but 'sector' column missing - skipping")
        return df
    dummies = pd.get_dummies(df["sector"], prefix=prefix.rstrip("_"), prefix_sep="_", dummy_na=False)
    # pd.get_dummies returns bool in pandas 2.x; cast to int8 for model libs that don't accept bool.
    dummies = dummies.astype("int8")
    return pd.concat([df, dummies], axis=1)


def build_feature_panel(
    panel: pd.DataFrame,
    *,
    families: list[str] | None = None,
    cross_sectional: bool = True,
    add_sector_dummies: bool = True,
) -> pd.DataFrame:
    """Stitch the per-family feature modules onto the clean PIT panel.

    Parameters
    ----------
    panel:
        The clean point-in-time panel from ``short_king.data`` (one row per
        ticker-date, must carry ``Ticker``/``ticker``/``symbol`` and
        ``Date``/``date``, plus ``sector`` if dummies are requested).
    families:
        Subset of ``FEATURE_FAMILIES`` to run, in the listed order. ``None``
        means all of them.
    cross_sectional:
        If ``True`` (the model default) append a within-date percentile rank
        for every numeric raw feature as ``<col>_rk``. The raw column is
        retained for diagnostics; ranks are the modelling inputs.
    add_sector_dummies:
        If ``True``, append one-hot ``sec_<sector>`` columns.
    """
    if panel is None or len(panel) == 0:
        raise ValueError("build_feature_panel received an empty panel")

    fams = list(families) if families is not None else list(FEATURE_FAMILIES)
    unknown = [f for f in fams if f not in FEATURE_FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown feature families: {unknown}. Valid: {FEATURE_FAMILIES}"
        )

    tcol, dcol = _ticker_date_cols(panel)
    out = panel.sort_values([tcol, dcol]).reset_index(drop=True).copy()
    n0_rows, n0_cols = out.shape
    logger.info(
        f"build_feature_panel: start | rows={n0_rows:,} cols={n0_cols} "
        f"families={fams} cross_sectional={cross_sectional} "
        f"add_sector_dummies={add_sector_dummies}"
    )

    for fam in fams:
        before = out.shape[1]
        try:
            fn = _family_panel_fn(fam)
            out = fn(out)
        except Exception as exc:
            # Surface the family that failed but don't swallow - upstream
            # pipeline needs to know a feature family is broken.
            logger.error(f"feature family '{fam}' failed: {exc}")
            raise
        added = out.shape[1] - before
        logger.info(f"  + {fam}_panel: +{added} cols -> {out.shape[1]} total")

    if cross_sectional:
        # Re-sort by date so the groupby is contiguous (small perf win, no
        # correctness impact). The raw columns to rank are whatever the family
        # modules produced and that are numeric, not identifiers/labels/dummies.
        out = out.sort_values([dcol, tcol]).reset_index(drop=True)
        raw_cols = _rankable_columns(out)
        out = cross_sectional_rank(out, raw_cols, by=dcol, suffix="_rk")
        logger.info(f"  + cross_sectional_rank: +{len(raw_cols)} *_rk cols")

    if add_sector_dummies:
        before = out.shape[1]
        out = _sector_dummies(out)
        logger.info(f"  + sector dummies: +{out.shape[1] - before} sec_* cols")

    out = out.sort_values([tcol, dcol]).reset_index(drop=True)
    logger.info(
        f"build_feature_panel: done | rows={len(out):,} cols={out.shape[1]} "
        f"(delta cols=+{out.shape[1] - n0_cols})"
    )
    return out


def write_feature_panel(features: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist the feature matrix to parquet. Default: ``data/processed/features.parquet``."""
    target = path if path is not None else settings.processed_dir / "features.parquet"
    target = Path(target)
    write_parquet(features, target)
    logger.info(f"wrote {target} | rows={len(features):,} cols={features.shape[1]}")
    return target


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """One-page sanity table: row count, feature count, %-NaN, top correlated pairs.

    The returned long-format frame has columns
    ``[metric, name, value]`` so it can be rendered straight into the
    methodology notebook without further reshaping. ``top_corr_pair`` rows
    carry the absolute correlation in ``value`` and the pair label
    ``"a | b"`` in ``name``.
    """
    if features is None or len(features) == 0:
        return pd.DataFrame(columns=["metric", "name", "value"])

    # Treat everything that is not an id / label as a candidate feature column.
    feat_cols = [
        c
        for c in features.columns
        if c not in _ID_COLS and not _is_label(c)
    ]
    n_rows = len(features)
    n_features = len(feat_cols)

    rows: list[dict[str, object]] = [
        {"metric": "n_rows", "name": "n_rows", "value": float(n_rows)},
        {"metric": "n_features", "name": "n_features", "value": float(n_features)},
    ]

    # Per-feature NaN share. Bool/int dummies are included (their NaN share is
    # ~0 by construction) - keeps the table self-describing.
    nan_share = features[feat_cols].isna().mean().sort_values(ascending=False)
    for col, frac in nan_share.items():
        rows.append({"metric": "pct_nan", "name": str(col), "value": float(frac)})

    # Top-5 highest |corr| pairs among the *_rk columns (the model inputs).
    # Falling back to numeric raw features if no ranks were produced.
    rk_cols = [c for c in feat_cols if c.endswith("_rk")]
    candidates = rk_cols if rk_cols else [
        c for c in feat_cols if pd.api.types.is_numeric_dtype(features[c])
        and not c.startswith("sec_")
    ]
    if len(candidates) >= 2:
        corr = features[candidates].corr(numeric_only=True).abs()
        # Mask the upper triangle + diagonal so each pair appears once.
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=0)
        pairs = (
            corr.mask(mask).stack().sort_values(ascending=False).head(5)
        )
        for (a, b), v in pairs.items():
            rows.append(
                {"metric": "top_corr_pair", "name": f"{a} | {b}", "value": float(v)}
            )

    return pd.DataFrame(rows, columns=["metric", "name", "value"])
