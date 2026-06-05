"""Short-signal quality analysis -- the central artefact of this project.

For each (model, bucket-size, period) combination, take the top-N% most
shortable names by the model's score on each rebalance and ask:

  1. Of those shorts, what fraction actually fell? (success rate)
  2. When right, how right? (mean / median winning trade)
  3. When wrong, how wrong? (mean / median losing trade, worst tail)
  4. What's the asymmetry? (win-magnitude / loss-magnitude ratio)
  5. What does the return distribution look like? (percentile bands)

We deliberately do NOT model commission, borrow, slippage, stop loss,
or any portfolio construction. Each position is a single (Date, Ticker,
short_return) triple. The only thing being measured is signal quality:
"how often does the model correctly identify a stock that will fall,
and by how much in either direction?"

Buckets tested:
  decile     = top 10 %  (smallest, highest-conviction)
  quintile   = top 20 %  (standard)
  tercile    = top 33 %  (broadest)
  top-5      = a fixed 5-name basket per month (smallest possible)
  top-10     = a fixed 10-name basket per month

For each model x bucket x period (ALL, IS, OOS) we report success rate,
mean / median trade return, worst-decile and best-decile averages, and
the win/loss magnitude asymmetry.

Outputs:
  reports/short_signal_summary.csv          -- all numbers, long form
  reports/short_signal_summary.md           -- markdown leaderboard
  reports/short_signal_per_position.csv     -- every individual short
                                                position with its
                                                realised forward return
  charts/short_return_distribution.png      -- per-model histograms
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

MODELS = ("naive", "ew", "logit", "gbm_cls", "gbm_rank")

# Bucket spec: name -> (kind, value).
# 'frac' = top fraction by score; 'topn' = top fixed-N by score.
BUCKETS: dict[str, tuple[str, float]] = {
    "decile":   ("frac", 0.10),
    "quintile": ("frac", 0.20),
    "tercile":  ("frac", 0.33),
    "top-5":    ("topn", 5),
    "top-10":   ("topn", 10),
}


def _select_top(df: pd.DataFrame, kind: str, value: float) -> pd.DataFrame:
    """Within each Date group, keep the top-fraction or top-N rows by score.
    Returns a flat DataFrame (no groupby index)."""
    # Rank per-date (descending by score): rank 1 = highest score = most shortable.
    ranks = df.groupby("Date", sort=False)["score"].rank(method="first", ascending=False)
    if kind == "frac":
        # Top fraction = rank <= round(group_size * value).
        group_size = df.groupby("Date", sort=False)["score"].transform("size")
        cutoff = (group_size * float(value)).round().clip(lower=1)
        mask = ranks <= cutoff
    elif kind == "topn":
        mask = ranks <= int(value)
    else:
        raise ValueError(f"unknown bucket kind: {kind}")
    return df.loc[mask].reset_index(drop=True)

# Loss/win bins for the distribution tables (in % units; we'll convert).
WIN_BINS = (0.10, 0.25, 0.50, 1.00)   # +10%, +25%, +50%, +100% wins (stock fell that much)
LOSS_BINS = (0.10, 0.25, 0.50, 1.00)  # -10%, -25%, -50%, -100%+ losses (stock rallied)


def _short_returns_table(oof: pd.DataFrame, fwd_col: str = "fwd_ret_1m") -> pd.DataFrame:
    """Take the OOF predictions (Date, Ticker, model, score, fwd_ret_1m,
    period) and build one row per (Date, Ticker, model) with the
    *short return* = -fwd_ret_1m and convenience columns.
    """
    df = oof.dropna(subset=["score", fwd_col]).copy()
    df["stock_return"] = df[fwd_col].astype(float)
    df["short_return"] = -df["stock_return"]
    return df


def _bucket_summary(positions: pd.DataFrame) -> dict:
    """Per-(model, bucket, period) summary stats."""
    n = len(positions)
    if n == 0:
        return {}
    sr = positions["short_return"].astype(float)
    wins = sr[sr > 0]
    losses = sr[sr <= 0]
    win_rate = float((sr > 0).mean())
    summary = {
        "n_positions": int(n),
        "win_rate_pct": round(100 * win_rate, 2),
        "mean_trade_pct": round(100 * float(sr.mean()), 3),
        "median_trade_pct": round(100 * float(sr.median()), 3),
        "stdev_pct": round(100 * float(sr.std(ddof=1)), 3),
        "p05_pct": round(100 * float(sr.quantile(0.05)), 2),
        "p25_pct": round(100 * float(sr.quantile(0.25)), 2),
        "p75_pct": round(100 * float(sr.quantile(0.75)), 2),
        "p95_pct": round(100 * float(sr.quantile(0.95)), 2),
        "worst_trade_pct": round(100 * float(sr.min()), 2),
        "best_trade_pct": round(100 * float(sr.max()), 2),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "mean_win_pct": round(100 * float(wins.mean()), 3) if len(wins) else float("nan"),
        "mean_loss_pct": round(100 * float(losses.mean()), 3) if len(losses) else float("nan"),
        "median_win_pct": round(100 * float(wins.median()), 3) if len(wins) else float("nan"),
        "median_loss_pct": round(100 * float(losses.median()), 3) if len(losses) else float("nan"),
        # Asymmetry: mean(win) / |mean(loss)|. > 1.0 = average win is bigger
        # than average loss (good); < 1.0 = losses bigger than wins (bad).
        "win_loss_magnitude_ratio": (
            round(float(wins.mean()) / abs(float(losses.mean())), 3)
            if len(wins) and len(losses) and losses.mean() != 0 else float("nan")
        ),
        # Expected value per position (in %).
        "expected_value_pct": round(100 * float(sr.mean()), 3),
    }
    # Distribution: count of trades that fell more than +10%, +25%, etc.
    for thr in WIN_BINS:
        summary[f"pct_win_gt_{int(thr*100)}pct"] = round(100 * float((sr > thr).mean()), 2)
    for thr in LOSS_BINS:
        summary[f"pct_loss_gt_{int(thr*100)}pct"] = round(100 * float((sr < -thr).mean()), 2)
    return summary


def _build_positions(oof_returns: pd.DataFrame) -> pd.DataFrame:
    """For each (model, bucket, period), build the short positions and return a
    long DataFrame keyed by (model, bucket, period, Date, Ticker, score,
    stock_return, short_return).
    """
    rows: list[pd.DataFrame] = []
    for model in MODELS:
        m = oof_returns[oof_returns["model"] == model]
        if m.empty:
            continue
        for bucket, (kind, value) in BUCKETS.items():
            picks = _select_top(m, kind, value).copy()
            picks["bucket"] = bucket
            picks["model"] = model
            rows.append(picks)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    if "fwd_ret_1m" not in oof.columns:
        # Fall back to the older 4w name if rename hasn't propagated.
        col = "fwd_ret_4w" if "fwd_ret_4w" in oof.columns else None
        if col is None:
            logger.error("no forward-return column in OOF parquet")
            return 1
        oof = oof.rename(columns={col: "fwd_ret_1m"})

    oof_returns = _short_returns_table(oof)
    positions = _build_positions(oof_returns)
    if positions.empty:
        logger.error("no positions built")
        return 2

    # Per-position artefact (helps the README's deep-dive section).
    pos_csv = settings.reports_dir / "short_signal_per_position.csv"
    keep_cols = ["Date", "Ticker", "model", "bucket", "period", "score",
                 "stock_return", "short_return"]
    positions[keep_cols].to_csv(pos_csv, index=False)
    logger.info(f"wrote {pos_csv} ({len(positions):,} rows)")

    # Summary table.
    rows: list[dict] = []
    for (model, bucket, period), grp in positions.groupby(["model", "bucket", "period"]):
        s = _bucket_summary(grp)
        if not s:
            continue
        s.update({"model": model, "bucket": bucket, "period": period})
        rows.append(s)
    # Also ALL-period rollup per (model, bucket).
    for (model, bucket), grp in positions.groupby(["model", "bucket"]):
        s = _bucket_summary(grp)
        if not s:
            continue
        s.update({"model": model, "bucket": bucket, "period": "ALL"})
        rows.append(s)

    summary = pd.DataFrame(rows)
    cols_first = ["model", "bucket", "period", "n_positions", "win_rate_pct",
                  "mean_trade_pct", "median_trade_pct",
                  "mean_win_pct", "mean_loss_pct", "win_loss_magnitude_ratio",
                  "expected_value_pct"]
    cols_dist = [c for c in summary.columns if c not in cols_first]
    summary = summary[cols_first + cols_dist]
    csv_path = settings.reports_dir / "short_signal_summary.csv"
    summary.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path} ({len(summary)} rows)")

    # Markdown view focused on the headline picture: OOS results per
    # (model, bucket), ranked by win-rate then by expected value.
    def _md_table(period: str) -> str:
        sub = summary[summary["period"] == period].copy()
        sub = sub.sort_values(["bucket", "win_rate_pct"], ascending=[True, False])
        cols = ["model", "bucket", "n_positions", "win_rate_pct",
                "median_trade_pct", "mean_trade_pct",
                "mean_win_pct", "mean_loss_pct",
                "win_loss_magnitude_ratio", "worst_trade_pct"]
        body = sub[cols].copy()
        body.columns = ["model", "bucket", "n", "win %", "median",
                        "mean", "mean win", "mean loss",
                        "win/loss ratio", "worst"]
        hdr = "| " + " | ".join(body.columns) + " |"
        sep = "|" + "|".join(["---"] * len(body.columns)) + "|"
        lines = [hdr, sep]
        for _, r in body.iterrows():
            vals = []
            for c in body.columns:
                v = r[c]
                if isinstance(v, str):
                    vals.append(v)
                elif c in ("n",):
                    vals.append(f"{int(v):,}" if pd.notna(v) else "")
                elif c == "win/loss ratio":
                    vals.append(f"{v:.2f}" if pd.notna(v) else "")
                else:
                    vals.append(f"{v:+.2f}%" if pd.notna(v) else "")
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md = ["# Short-signal quality summary",
          "",
          "Per-model, per-bucket short-position outcomes. Every position is "
          "a single (Date, Ticker) short with realised one-month forward "
          "return; no costs, no portfolio construction.",
          "",
          "**Columns**:",
          "* `win %` = share of positions where the stock fell.",
          "* `median` / `mean` = central tendency of the per-position short "
          "return (positive = stock fell, the short won).",
          "* `mean win` / `mean loss` = average outcome conditional on winning "
          "/ losing.",
          "* `win/loss ratio` = `mean win / |mean loss|`. >1 = average win "
          "is bigger than average loss; <1 = losses dominate.",
          "* `worst` = single worst position (the big squeeze).",
          ""]
    for period in ("OOS", "IS", "ALL"):
        md += [f"\n## {period}\n", _md_table(period)]
    md_path = settings.reports_dir / "short_signal_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"wrote {md_path}")

    # Distribution chart: per-model histogram of short returns for the
    # decile bucket OOS. Visualises the asymmetry the user cares about.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 2.5 * len(MODELS)),
                                  sharex=True, dpi=160)
        for ax, model in zip(axes, MODELS):
            sub = positions[(positions.model == model) &
                            (positions.bucket == "decile") &
                            (positions.period == "OOS")]
            if sub.empty:
                ax.set_visible(False); continue
            sr = sub["short_return"].clip(-3.0, 3.0)  # cap for plot only
            ax.hist(sr, bins=60, color="#0072B2" if sr.mean() > 0 else "#D55E00",
                    alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(float(sr.mean()), color="red", linewidth=1.5, linestyle="--",
                        label=f"mean = {sr.mean()*100:+.2f} %")
            ax.axvline(float(sr.median()), color="green", linewidth=1.5, linestyle=":",
                        label=f"median = {sr.median()*100:+.2f} %")
            wr = float((sub["short_return"] > 0).mean()) * 100
            ax.set_title(f"{model} -- decile OOS (n={len(sub):,}, win-rate = {wr:.1f} %)",
                          fontsize=10, fontweight="bold", loc="left")
            ax.set_ylabel("frequency", fontsize=9)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.grid(axis="y", linestyle=":", alpha=0.3)
        axes[-1].set_xlabel("Short return per position "
                             "(positive = stock fell, short won; capped at ±300 % for visibility)",
                             fontsize=9)
        fig.suptitle("Short-return distribution per model (top-decile OOS)",
                      fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        chart_path = settings.charts_dir / "short_return_distribution.png"
        fig.savefig(chart_path)
        plt.close(fig)
        logger.info(f"wrote {chart_path}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"distribution chart failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
