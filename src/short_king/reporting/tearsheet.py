"""One-page tear sheet — cumulative growth, drawdowns, summary stats, monthly heatmap.

The tear sheet is the headline visual for any backtest run. A single PNG carries
the entire story so it embeds cleanly in the README and in PRs. The text-summary
counterpart returns the same numbers as plain Markdown for inline use.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never pop a window

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from short_king.utils.io import ensure_dir  # noqa: E402
from short_king.utils.logging import logger  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from short_king.portfolio.backtest import BacktestResult

# Annualisation factor — short-king is anchored to ASIC weekly (Friday) releases.
# Override per call via ``periods_per_year`` if a different cadence is used.
ANN_WEEKLY = 52
ANN_MONTHLY = 12

PALETTE = {
    "primary": "#1b3a5b",   # deep navy — strategy
    "accent":  "#c1121f",   # red — drawdown
    "bench":   "#444444",   # dark grey — benchmark
    "muted":   "#6c757d",   # axes / annotation
    "green":   "#2a9d8f",
    "gold":    "#e9c46a",
}


# --------------------------------------------------------------------------
# Result-shape helpers — keep this module loosely coupled to BacktestResult.
# We expect ``result.returns`` to exist as a periodic pd.Series indexed by date.
# Other fields are looked up best-effort.
# --------------------------------------------------------------------------
def _extract_returns(result: Any) -> pd.Series:
    """Pull the periodic-return series off a BacktestResult-like object."""
    for attr in ("returns", "net_returns", "ret", "r"):
        if hasattr(result, attr):
            s = getattr(result, attr)
            if isinstance(s, pd.Series):
                return s.dropna().astype(float).sort_index()
    raise AttributeError(
        "BacktestResult must expose a `.returns` pd.Series of periodic returns."
    )


def _extract_turnover(result: Any) -> pd.Series | None:
    for attr in ("turnover", "total_traded", "traded"):
        if hasattr(result, attr):
            s = getattr(result, attr)
            if isinstance(s, pd.Series):
                return s.astype(float)
    return None


def _infer_periods_per_year(idx: pd.DatetimeIndex) -> int:
    """Guess the annualisation factor from the index frequency."""
    if len(idx) < 3:
        return ANN_WEEKLY
    median_days = float(pd.Series(idx).diff().dt.days.dropna().median() or 7.0)
    if median_days <= 1.5:
        return 252
    if median_days <= 9.0:
        return ANN_WEEKLY
    if median_days <= 45.0:
        return ANN_MONTHLY
    return 4  # quarterly


# --------------------------------------------------------------------------
# Performance arithmetic
# --------------------------------------------------------------------------
def _cumulative_growth(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def _drawdown(r: pd.Series) -> pd.Series:
    cum = _cumulative_growth(r)
    return cum / cum.cummax() - 1.0


def _perf_stats(r: pd.Series, *, periods_per_year: int) -> dict[str, float]:
    """Headline performance stats. All returns expressed as decimals."""
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ("CAGR", "ann_vol", "Sharpe", "max_drawdown", "Calmar",
                 "hit_rate", "total_return", "n_periods")}
    cum_final = float((1.0 + r).prod())
    years = n / periods_per_year
    cagr = cum_final ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    sd = float(r.std())
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    max_dd = float(_drawdown(r).min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    return {
        "CAGR": float(cagr),
        "ann_vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "Calmar": calmar,
        "hit_rate": float((r > 0).mean()),
        "total_return": float(cum_final - 1.0),
        "n_periods": float(n),
    }


def _avg_ann_turnover(turnover: pd.Series | None, periods_per_year: int) -> float:
    """One-way annualised turnover as a fraction (e.g. 1.8 == 180%/yr)."""
    if turnover is None or turnover.empty:
        return float("nan")
    return float(turnover.dropna().mean() * 0.5 * periods_per_year)


def _monthly_returns(r: pd.Series) -> pd.Series:
    """Compound the periodic series up to month-end returns."""
    if not isinstance(r.index, pd.DatetimeIndex):
        r = r.copy()
        r.index = pd.to_datetime(r.index)
    return (1.0 + r.fillna(0.0)).resample("ME").prod() - 1.0


def _monthly_heatmap_frame(r: pd.Series) -> pd.DataFrame:
    """Year x Month matrix of returns in percent (NaN for missing cells)."""
    m = _monthly_returns(r)
    if m.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "year": m.index.year,
        "month": m.index.month,
        "ret": m.values * 100.0,
    })
    table = df.pivot(index="year", columns="month", values="ret")
    # Always show 12 month columns in order, even if some are missing.
    table = table.reindex(columns=range(1, 13))
    return table.sort_index()


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "grid.alpha": 0.30,
        "legend.frameon": False,
    })


# --------------------------------------------------------------------------
# Sub-panels
# --------------------------------------------------------------------------
def _plot_growth(ax, r: pd.Series, bench: pd.Series | None) -> None:
    cum = _cumulative_growth(r)
    ax.plot(cum.index, cum.values, lw=1.8, color=PALETTE["primary"], label="Strategy")
    if bench is not None and not bench.empty:
        b = bench.reindex(r.index).fillna(0.0)
        cum_b = _cumulative_growth(b)
        ax.plot(cum_b.index, cum_b.values, lw=1.6, color=PALETTE["bench"],
                ls="--", label=bench.name or "Benchmark")
    ax.axhline(1.0, color="#888", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_title("Cumulative growth of $1", loc="left")
    ax.legend(loc="upper left", fontsize=9)


def _plot_drawdown(ax, r: pd.Series) -> None:
    dd = _drawdown(r) * 100.0
    ax.fill_between(dd.index, dd.values, 0.0, color=PALETTE["accent"], alpha=0.30)
    ax.plot(dd.index, dd.values, lw=1.1, color=PALETTE["accent"])
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown", loc="left")


def _plot_summary_table(ax, stats: dict[str, float], turnover_ann: float,
                        bench_stats: dict[str, float] | None) -> None:
    ax.axis("off")
    rows = [
        ("CAGR",          f"{stats['CAGR'] * 100:.2f}%"),
        ("Ann. vol",      f"{stats['ann_vol'] * 100:.2f}%"),
        ("Sharpe",        f"{stats['Sharpe']:.2f}"),
        ("Max drawdown",  f"{stats['max_drawdown'] * 100:.2f}%"),
        ("Calmar",        f"{stats['Calmar']:.2f}" if np.isfinite(stats['Calmar']) else "—"),
        ("Hit rate",      f"{stats['hit_rate'] * 100:.1f}%"),
        ("Ann. turnover", f"{turnover_ann * 100:.1f}%" if np.isfinite(turnover_ann) else "—"),
        ("# periods",     f"{int(stats['n_periods'])}"),
    ]
    headers = ["Metric", "Strategy"]
    if bench_stats is not None:
        headers.append("Benchmark")
        bench_vals = [
            f"{bench_stats['CAGR'] * 100:.2f}%",
            f"{bench_stats['ann_vol'] * 100:.2f}%",
            f"{bench_stats['Sharpe']:.2f}",
            f"{bench_stats['max_drawdown'] * 100:.2f}%",
            f"{bench_stats['Calmar']:.2f}" if np.isfinite(bench_stats['Calmar']) else "—",
            f"{bench_stats['hit_rate'] * 100:.1f}%",
            "—",
            f"{int(bench_stats['n_periods'])}",
        ]
        cells = [[k, v, bv] for (k, v), bv in zip(rows, bench_vals)]
        col_widths = [0.45, 0.275, 0.275]
    else:
        cells = [[k, v] for k, v in rows]
        col_widths = [0.55, 0.45]

    tbl = ax.table(cellText=cells, colLabels=headers, loc="center",
                   cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.45)
    for j in range(len(headers)):
        h = tbl[0, j]
        h.set_facecolor(PALETTE["primary"])
        h.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cells) + 1):
        tbl[i, 0].set_text_props(fontweight="bold", ha="left")
        tbl[i, 0].set_facecolor("#eef1f6")
    ax.set_title("Summary statistics", loc="left", pad=8)


def _plot_monthly_heatmap(ax, r: pd.Series) -> None:
    table = _monthly_heatmap_frame(r)
    if table.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient data for monthly heatmap",
                ha="center", va="center", color=PALETTE["muted"])
        return
    data = table.to_numpy()
    finite = data[np.isfinite(data)]
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index.astype(int), fontsize=8)
    ax.set_title("Monthly returns (%)", loc="left")

    # Annotate each cell only when the heatmap is small enough to stay readable.
    if data.shape[0] <= 20:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    color = "white" if abs(v) > 0.6 * vmax else "#222"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Return (%)", fontsize=8)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def render_tearsheet(
    result: BacktestResult,
    *,
    title: str,
    bench: pd.Series | None = None,
    path: Path,
) -> Path:
    """Render a one-page PNG tear sheet for ``result`` and save it to ``path``.

    The PNG packs four panels using matplotlib gridspec:

    * (a) cumulative growth of $1 with optional benchmark overlay
    * (b) drawdown panel
    * (c) summary statistics table (Sharpe, max DD, hit rate, turnover, CAGR, Calmar)
    * (d) monthly-return heatmap (year x month)

    Why one PNG: a single artefact links from the README and from PR descriptions
    without juggling multiple files, which keeps the surface area for review
    small. Stats are derived from ``result.returns`` so any change to the
    backtest engine flows through automatically.
    """
    _apply_style()
    r = _extract_returns(result)
    if r.empty:
        raise ValueError("BacktestResult.returns is empty — cannot render tear sheet.")

    periods_per_year = _infer_periods_per_year(pd.DatetimeIndex(r.index))
    stats = _perf_stats(r, periods_per_year=periods_per_year)
    bench_stats: dict[str, float] | None = None
    if bench is not None and not bench.empty:
        b = bench.reindex(r.index).dropna()
        if not b.empty:
            bench_stats = _perf_stats(b, periods_per_year=periods_per_year)

    turnover_ann = _avg_ann_turnover(_extract_turnover(result), periods_per_year)

    fig: Figure = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995, x=0.02, ha="left")

    # Layout: row 0 = growth + summary table, row 1 = drawdown + heatmap.
    gs = GridSpec(
        nrows=2, ncols=2,
        figure=fig,
        width_ratios=[2.1, 1.0],
        height_ratios=[1.0, 0.9],
        hspace=0.36, wspace=0.18,
        top=0.93, bottom=0.06, left=0.06, right=0.98,
    )
    ax_growth = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax_growth)
    ax_heat = fig.add_subplot(gs[1, 1])

    _plot_growth(ax_growth, r, bench)
    _plot_summary_table(ax_table, stats, turnover_ann, bench_stats)
    _plot_drawdown(ax_dd, r)
    _plot_monthly_heatmap(ax_heat, r)

    # Footer: date range + cadence (gives a quick sense of coverage).
    start, end = r.index.min(), r.index.max()
    cadence = {252: "daily", 52: "weekly", 12: "monthly", 4: "quarterly"}.get(
        periods_per_year, f"{periods_per_year}x/yr")
    fig.text(
        0.02, 0.015,
        f"Sample: {pd.Timestamp(start):%Y-%m-%d} → {pd.Timestamp(end):%Y-%m-%d}  "
        f"|  cadence: {cadence}  |  n = {len(r)}",
        fontsize=8.5, style="italic", color=PALETTE["muted"],
    )

    out = Path(path)
    ensure_dir(out.parent)
    fig.savefig(out)
    plt.close(fig)
    logger.info(f"tearsheet saved -> {out}")
    return out


def render_text_summary(
    result: BacktestResult,
    *,
    title: str,
    bench: pd.Series | None = None,
) -> str:
    """Plain-text Markdown summary of a backtest run.

    Pair the PNG tear sheet with this text block when embedding into the README
    or a PR description — it keeps the numbers grep-able and diff-friendly.
    """
    r = _extract_returns(result)
    if r.empty:
        return f"### {title}\n\n_No returns to summarise._\n"

    periods_per_year = _infer_periods_per_year(pd.DatetimeIndex(r.index))
    stats = _perf_stats(r, periods_per_year=periods_per_year)
    turnover_ann = _avg_ann_turnover(_extract_turnover(result), periods_per_year)
    start, end = r.index.min(), r.index.max()
    cadence = {252: "daily", 52: "weekly", 12: "monthly", 4: "quarterly"}.get(
        periods_per_year, f"{periods_per_year}x/yr")

    def _pct(x: float) -> str:
        return f"{x * 100:.2f}%" if np.isfinite(x) else "—"

    def _num(x: float, fmt: str = "{:.2f}") -> str:
        return fmt.format(x) if np.isfinite(x) else "—"

    bench_block = ""
    if bench is not None and not bench.empty:
        b = bench.reindex(r.index).dropna()
        if not b.empty:
            bs = _perf_stats(b, periods_per_year=periods_per_year)
            bench_name = bench.name or "Benchmark"
            bench_block = (
                f"\n**Benchmark — {bench_name}**\n\n"
                f"| Metric | Value |\n"
                f"|---|---:|\n"
                f"| CAGR | {_pct(bs['CAGR'])} |\n"
                f"| Ann. vol | {_pct(bs['ann_vol'])} |\n"
                f"| Sharpe | {_num(bs['Sharpe'])} |\n"
                f"| Max drawdown | {_pct(bs['max_drawdown'])} |\n"
            )

    lines = [
        f"### {title}",
        "",
        f"_Sample:_ `{pd.Timestamp(start):%Y-%m-%d}` → "
        f"`{pd.Timestamp(end):%Y-%m-%d}`  ·  _cadence:_ `{cadence}`  ·  "
        f"_n =_ `{len(r)}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| CAGR | {_pct(stats['CAGR'])} |",
        f"| Ann. vol | {_pct(stats['ann_vol'])} |",
        f"| Sharpe | {_num(stats['Sharpe'])} |",
        f"| Max drawdown | {_pct(stats['max_drawdown'])} |",
        f"| Calmar | {_num(stats['Calmar'])} |",
        f"| Hit rate | {_pct(stats['hit_rate'])} |",
        f"| Ann. turnover (one-way) | {_pct(turnover_ann)} |",
        f"| Total return | {_pct(stats['total_return'])} |",
        bench_block,
    ]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_tearsheet", "render_text_summary"]
