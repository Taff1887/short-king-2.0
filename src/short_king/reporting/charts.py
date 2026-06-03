"""Publication-quality charts for the short-selling research report.

Pure matplotlib (no seaborn font/style hijacking) so the figures render identically
on any machine. Every chart function:

* Accepts the data it needs plus an explicit output ``Path``.
* Writes a 200 DPI PNG with tight layout.
* Returns the output path so callers can chain into reports.
* Closes its figure to avoid leaking memory across the full chart suite.

The colour contract is fixed at module level so the long/short/benchmark mapping is
consistent across the deck:

* long       -> ``#2C5F8E`` (blue)
* short      -> ``#C44545`` (red)
* benchmark  -> ``#444444`` (dark grey)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Headless backend so this works in CI / scripts without a display server. Must
# happen before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from short_king.utils.io import ensure_dir  # noqa: E402
from short_king.utils.logging import logger  # noqa: E402

# --- Style contract ------------------------------------------------------
# A single source of truth for colour identity. Anything that touches "long",
# "short" or "benchmark" semantically should pull from here, not hard-code.
COLOR_LONG: str = "#2C5F8E"
COLOR_SHORT: str = "#C44545"
COLOR_BENCH: str = "#444444"
COLOR_NEUTRAL: str = "#888888"
COLOR_GRID: str = "#DDDDDD"

# Diverging palette for monthly heatmap / correlation matrices — built once so we
# don't pay the LinearSegmentedColormap construction cost on every chart call.
_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "short_king_div",
    [COLOR_SHORT, "#FFFFFF", COLOR_LONG],
    N=256,
)

# Default figure DPI for both rendering and saving. 200 DPI hits the sweet spot
# between PowerPoint-clarity and reasonable PNG file sizes.
_DPI: int = 200

# Reproducibility for any chart that does sampling/jitter.
_RNG_SEED: int = 0


def _new_fig(figsize: tuple[float, float] = (10.0, 6.0)) -> tuple[plt.Figure, plt.Axes]:
    """Return a fresh (fig, ax) with the module's house style applied."""
    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
    ax.set_axisbelow(True)
    ax.grid(True, color=COLOR_GRID, linewidth=0.6, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#333333", labelsize=9)
    return fig, ax


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write ``fig`` to ``path`` with tight layout, close it, return path."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug("Wrote chart: {}", path)
    return path


def _coerce_dates(s: pd.Series) -> pd.Series:
    """Idempotent conversion of a date-like column to ``Timestamp``."""
    return pd.to_datetime(s, errors="coerce")


# --- Coverage / data-quality charts --------------------------------------
def chart_universe_coverage(panel: pd.DataFrame, path: Path) -> Path:
    """Line chart of investable-universe size per as-of date.

    ``panel`` is the wide short-king PIT panel produced by ``short_king.data``;
    we count distinct ``Ticker`` values per ``Date`` after dropping rows where
    a name is flagged not-investable (if such a flag exists).
    """
    if "Date" not in panel.columns:
        raise ValueError("chart_universe_coverage: panel must contain a 'Date' column")

    df = panel.copy()
    df["Date"] = _coerce_dates(df["Date"])
    if "investable" in df.columns:
        df = df.loc[df["investable"].astype(bool)]

    counts = (
        df.dropna(subset=["Date"])
        .groupby("Date", as_index=True)["Ticker"]
        .nunique()
        .sort_index()
    )

    fig, ax = _new_fig(figsize=(11.0, 5.5))
    ax.plot(counts.index, counts.values, color=COLOR_LONG, linewidth=1.6)
    ax.fill_between(counts.index, counts.values, alpha=0.15, color=COLOR_LONG)
    ax.set_title("Investable universe coverage", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Distinct tickers")
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate()
    return _save(fig, path)


def chart_si_distribution(
    panel: pd.DataFrame,
    path: Path,
    *,
    latest_only: bool = True,
) -> Path:
    """Histogram of short-interest percentages.

    ``latest_only`` -> the most recent as-of date in the panel only;
    otherwise pool every observation (useful for the methodology section).
    """
    if "ShortPct" not in panel.columns:
        raise ValueError("chart_si_distribution: panel must contain a 'ShortPct' column")

    df = panel.copy()
    df["Date"] = _coerce_dates(df["Date"])
    if latest_only:
        latest = df["Date"].max()
        df = df.loc[df["Date"] == latest]
        subtitle = f"as of {pd.Timestamp(latest).date()}"
    else:
        subtitle = "pooled across all dates"

    values = df["ShortPct"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("chart_si_distribution: no non-null ShortPct observations")

    # ShortPct in ASIC PDFs is already a percent (e.g. 8.34); use np.histogram for
    # deterministic bin edges across reruns.
    upper = float(np.nanpercentile(values, 99.5))
    bins = np.histogram_bin_edges(values[values <= upper], bins=40, range=(0.0, max(upper, 1.0)))
    counts, edges = np.histogram(values, bins=bins)

    fig, ax = _new_fig(figsize=(10.0, 5.5))
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=COLOR_SHORT,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    median = float(np.nanmedian(values))
    ax.axvline(median, color=COLOR_BENCH, linestyle="--", linewidth=1.0, label=f"median = {median:.2f}%")
    ax.set_title(f"Short interest distribution ({subtitle})", fontsize=13, fontweight="bold")
    ax.set_xlabel("ShortPct (%)")
    ax.set_ylabel("Number of names")
    ax.legend(loc="upper right")
    return _save(fig, path)


# --- Feature diagnostics -------------------------------------------------
def chart_feature_distributions(
    features: pd.DataFrame,
    feature_cols: list[str],
    path: Path,
) -> Path:
    """Small-multiples histograms for the model's input features.

    Layout is 4 columns wide, rows grown as needed. Each sub-axis is a clean
    histogram (no per-axis grid noise) so the eye can scan distribution shape
    quickly. Useful for spotting features that need a winsor/log transform.
    """
    cols = [c for c in feature_cols if c in features.columns]
    if not cols:
        raise ValueError("chart_feature_distributions: no overlap between features and feature_cols")

    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3.0 * ncols, 2.4 * nrows), dpi=_DPI
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, cols, strict=False):
        values = features[col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            ax.set_axis_off()
            continue
        # Trim extreme tails for visibility — the underlying data isn't modified.
        lo, hi = np.nanpercentile(values, [0.5, 99.5])
        if lo == hi:
            lo, hi = float(values.min()), float(values.max()) + 1e-9
        clipped = values[(values >= lo) & (values <= hi)]
        bins = np.histogram_bin_edges(clipped, bins=30, range=(lo, hi))
        counts, edges = np.histogram(clipped, bins=bins)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=COLOR_LONG,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Hide trailing empty axes when the grid isn't a perfect rectangle.
    for ax in axes_flat[n:]:
        ax.set_axis_off()

    fig.suptitle("Feature distributions", fontsize=13, fontweight="bold", y=1.00)
    return _save(fig, path)


def chart_correlation_matrix(
    features: pd.DataFrame,
    feature_cols: list[str],
    path: Path,
) -> Path:
    """Annotated cross-feature correlation heatmap (matplotlib imshow).

    Pearson correlation on pairwise non-null observations. Diverging cmap
    centered at zero so signed structure pops without needing a colorbar squint.
    """
    cols = [c for c in feature_cols if c in features.columns]
    if len(cols) < 2:
        raise ValueError("chart_correlation_matrix: need at least 2 overlapping feature columns")

    corr = features[cols].corr(method="pearson")
    arr = corr.to_numpy()
    n = arr.shape[0]

    fig, ax = _new_fig(figsize=(max(7.0, 0.45 * n + 4.0), max(6.0, 0.45 * n + 3.0)))
    # Strip the grid we set in _new_fig — imshow background owns this chart.
    ax.grid(False)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(arr, cmap=_DIVERGING_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    # Annotate each cell with the correlation value; suppress text on the
    # diagonal where it's always 1.00 and just clutters.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = arr[i, j]
            text_color = "white" if abs(val) > 0.55 else "#222222"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=7, color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson r", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Feature correlation matrix", fontsize=13, fontweight="bold")
    return _save(fig, path)


def chart_feature_importance(
    importance: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 20,
) -> Path:
    """Horizontal bar chart of the top-``top_n`` features by importance.

    Accepts either a 'gain' (tree models) or 'mean_abs_shap' (SHAP) column;
    detection is automatic so the same chart helper covers both the LightGBM
    booster and the SHAP-explained variant.
    """
    if "feature" not in importance.columns:
        raise ValueError("chart_feature_importance: importance must have a 'feature' column")

    value_col = next(
        (c for c in ("mean_abs_shap", "gain", "importance") if c in importance.columns),
        None,
    )
    if value_col is None:
        raise ValueError(
            "chart_feature_importance: importance must have one of "
            "'mean_abs_shap', 'gain', or 'importance'"
        )

    df = (
        importance[["feature", value_col]]
        .dropna()
        .sort_values(value_col, ascending=False)
        .head(top_n)
        .iloc[::-1]  # reverse so largest bar sits at the top of the chart
        .reset_index(drop=True)
    )

    height = max(4.0, 0.32 * len(df) + 1.5)
    fig, ax = _new_fig(figsize=(9.0, height))
    ax.barh(
        df["feature"],
        df[value_col],
        color=COLOR_LONG,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(
        f"Top {len(df)} features by {value_col.replace('_', ' ')}",
        fontsize=13,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)
    return _save(fig, path)


# --- Backtest / performance charts ---------------------------------------
def _growth_of_one(r: pd.Series) -> pd.Series:
    """Cumulative growth of $1 from a periodic return series (NaNs -> 0)."""
    return (1.0 + r.fillna(0.0)).cumprod()


def chart_cumulative_returns(
    returns: pd.DataFrame,
    path: Path,
    *,
    bench: pd.Series | None = None,
) -> Path:
    """Multi-line growth-of-$1 chart for one or more strategies plus a benchmark.

    ``returns`` columns are interpreted by name so the palette is consistent
    across the report: any column literally named 'long' or 'short' (case-
    insensitive) gets the brand colour; everything else cycles through a muted
    secondary palette.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("chart_cumulative_returns: returns must be a non-empty DataFrame")

    fig, ax = _new_fig(figsize=(11.0, 6.0))
    secondary = ["#5B8EBF", "#888888", "#B58A3E", "#5D8C5A"]
    sec_idx = 0
    for col in returns.columns:
        growth = _growth_of_one(returns[col])
        name = str(col).strip().lower()
        if name in ("long", "long_only"):
            color, lw = COLOR_LONG, 2.0
        elif name in ("short", "short_only"):
            color, lw = COLOR_SHORT, 2.0
        elif name in ("ls", "long_short", "long-short"):
            color, lw = "#1F1F1F", 2.2
        else:
            color, lw = secondary[sec_idx % len(secondary)], 1.4
            sec_idx += 1
        ax.plot(growth.index, growth.values, label=str(col), color=color, linewidth=lw)

    if bench is not None and len(bench) > 0:
        bench_growth = _growth_of_one(bench)
        ax.plot(
            bench_growth.index, bench_growth.values,
            label=str(bench.name or "benchmark"),
            color=COLOR_BENCH, linewidth=1.5, linestyle="--",
        )

    ax.axhline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_title("Cumulative growth of $1", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.autofmt_xdate()
    return _save(fig, path)


def chart_drawdowns(returns: pd.Series, path: Path) -> Path:
    """Underwater (drawdown-from-peak) plot of a single return series."""
    if returns is None or len(returns) == 0:
        raise ValueError("chart_drawdowns: returns is empty")

    growth = _growth_of_one(returns)
    peak = growth.cummax()
    dd = growth / peak - 1.0

    fig, ax = _new_fig(figsize=(11.0, 4.5))
    ax.fill_between(dd.index, dd.values, 0.0, color=COLOR_SHORT, alpha=0.45, linewidth=0)
    ax.plot(dd.index, dd.values, color=COLOR_SHORT, linewidth=1.0)
    ax.axhline(0.0, color="#666666", linewidth=0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.set_title("Drawdown from peak", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    worst = float(dd.min())
    ax.annotate(
        f"max DD: {worst * 100:.1f}%",
        xy=(dd.idxmin(), worst),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=9,
        color="#222222",
    )
    fig.autofmt_xdate()
    return _save(fig, path)


def chart_monthly_heatmap(returns: pd.Series, path: Path) -> Path:
    """Year x Month coloured heatmap of compounded monthly returns."""
    if returns is None or len(returns) == 0:
        raise ValueError("chart_monthly_heatmap: returns is empty")

    s = returns.copy()
    s.index = pd.to_datetime(s.index)
    # Compound any sub-monthly observations into a single month-end return.
    monthly = (1.0 + s.fillna(0.0)).resample("ME").prod() - 1.0
    df = pd.DataFrame({"ret": monthly.values}, index=monthly.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    grid = df.pivot(index="year", columns="month", values="ret")
    grid = grid.reindex(columns=range(1, 13))

    arr = grid.to_numpy(dtype=float)
    bound = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.01
    bound = max(bound, 1e-4)

    fig, ax = _new_fig(figsize=(11.0, max(3.5, 0.35 * grid.shape[0] + 2.0)))
    ax.grid(False)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im = ax.imshow(arr, cmap=_DIVERGING_CMAP, norm=norm, aspect="auto")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, fontsize=9)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([str(y) for y in grid.index], fontsize=9)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            text_color = "white" if abs(v) > bound * 0.55 else "#222222"
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Monthly return", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.set_title("Monthly returns heatmap", fontsize=13, fontweight="bold")
    return _save(fig, path)


# --- Candidate / decile / calibration ------------------------------------
def chart_top_candidates(
    scores: pd.DataFrame,
    path: Path,
    *,
    as_of: pd.Timestamp | None = None,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of the top-``top_n`` short candidates.

    ``scores`` must contain ``Ticker``, ``score`` and (optionally) ``Date`` and
    ``Company`` columns. Score convention: *higher = more attractive short*.
    """
    if "Ticker" not in scores.columns or "score" not in scores.columns:
        raise ValueError("chart_top_candidates: scores must contain 'Ticker' and 'score'")

    df = scores.copy()
    if "Date" in df.columns:
        df["Date"] = _coerce_dates(df["Date"])
        snapshot = pd.Timestamp(as_of).normalize() if as_of is not None else df["Date"].max()
        df = df.loc[df["Date"] == snapshot]
        as_of_label = pd.Timestamp(snapshot).date().isoformat()
    else:
        as_of_label = "latest"

    df = df.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n)
    if df.empty:
        raise ValueError("chart_top_candidates: no rows after as-of filter")
    df = df.iloc[::-1].reset_index(drop=True)  # largest at top

    if "Company" in df.columns:
        labels = [f"{t}  -  {(c or '')[:34]}" for t, c in zip(df["Ticker"], df["Company"], strict=False)]
    else:
        labels = list(df["Ticker"].astype(str))

    fig, ax = _new_fig(figsize=(10.0, max(4.5, 0.36 * len(df) + 1.5)))
    ax.barh(
        labels, df["score"].astype(float).to_numpy(),
        color=COLOR_SHORT, alpha=0.9, edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Short score (higher = more attractive short)")
    ax.set_title(f"Top {len(df)} short candidates  -  {as_of_label}",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    return _save(fig, path)


def chart_decile_returns(returns_by_decile: pd.DataFrame, path: Path) -> Path:
    """Bar chart of mean periodic return per score decile.

    Expected shape: ten rows (deciles 1..10) with a numeric return column
    ('mean_return' / 'mean_ret' / 'ret') and an integer decile column
    ('decile'). Decile 10 is conventionally the most-shorted bucket; a healthy
    signal monotonically descends from 1 -> 10.
    """
    df = returns_by_decile.copy()
    decile_col = next((c for c in ("decile", "Decile", "bucket") if c in df.columns), None)
    ret_col = next(
        (c for c in ("mean_return", "mean_ret", "ret", "return", "avg_return")
         if c in df.columns),
        None,
    )
    if decile_col is None or ret_col is None:
        raise ValueError(
            "chart_decile_returns: dataframe must have a decile column and a "
            "mean-return column"
        )

    df = df[[decile_col, ret_col]].dropna().sort_values(decile_col)
    deciles = df[decile_col].astype(int).to_numpy()
    rets = df[ret_col].astype(float).to_numpy()

    # Colour by sign so the eye can confirm monotonicity at a glance.
    colors = [COLOR_LONG if r >= 0 else COLOR_SHORT for r in rets]

    fig, ax = _new_fig(figsize=(10.0, 5.5))
    ax.bar(deciles, rets, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color="#444444", linewidth=0.6)
    ax.set_xticks(deciles)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.2f}%"))
    ax.set_xlabel("Score decile (1 = most attractive long, 10 = most attractive short)")
    ax.set_ylabel("Mean weekly return")
    ax.set_title("Mean weekly return by score decile (gross of costs)",
                 fontsize=13, fontweight="bold")
    return _save(fig, path)


def chart_calibration(calibration_df: pd.DataFrame, path: Path) -> Path:
    """Reliability diagram for the classifier.

    ``calibration_df`` must contain a ``mean_pred`` (bin mean predicted prob)
    column and an ``observed_rate`` column. A perfectly calibrated model lies
    on the identity diagonal; the marker size (if ``count`` is present)
    reflects the number of observations in each bin.
    """
    if "mean_pred" not in calibration_df.columns or "observed_rate" not in calibration_df.columns:
        raise ValueError(
            "chart_calibration: dataframe must contain 'mean_pred' and "
            "'observed_rate' columns"
        )

    df = calibration_df.dropna(subset=["mean_pred", "observed_rate"]).sort_values("mean_pred")
    x = df["mean_pred"].astype(float).to_numpy()
    y = df["observed_rate"].astype(float).to_numpy()

    fig, ax = _new_fig(figsize=(7.0, 6.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], color=COLOR_BENCH, linestyle="--",
            linewidth=1.0, label="perfect calibration")
    if "count" in df.columns and df["count"].notna().any():
        sizes = df["count"].astype(float).to_numpy()
        scaled = 40.0 + 200.0 * (sizes / max(sizes.max(), 1.0))
        ax.scatter(x, y, s=scaled, color=COLOR_LONG, alpha=0.85,
                   edgecolor="white", linewidth=0.6, label="model")
    else:
        ax.scatter(x, y, s=80, color=COLOR_LONG, alpha=0.85,
                   edgecolor="white", linewidth=0.6, label="model")
    ax.plot(x, y, color=COLOR_LONG, linewidth=1.0, alpha=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Classifier calibration", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, path)


__all__ = [
    "COLOR_LONG",
    "COLOR_SHORT",
    "COLOR_BENCH",
    "chart_universe_coverage",
    "chart_si_distribution",
    "chart_feature_distributions",
    "chart_correlation_matrix",
    "chart_feature_importance",
    "chart_cumulative_returns",
    "chart_drawdowns",
    "chart_monthly_heatmap",
    "chart_top_candidates",
    "chart_decile_returns",
    "chart_calibration",
]
