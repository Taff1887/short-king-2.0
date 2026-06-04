"""OOS short-book analysis with the stop LOSS DISABLED.

Reconstructs every OOS short position for each of the 5 models, computes
the raw per-position monthly return (NO -16 % floor), and produces:

1. Per-model summary: per-position win-rate, mean / median return, biggest
   loss, # positions losing > 25 % / > 50 %.
2. Top-20 winning and losing positions globally (no model filter).
3. Per-model "PLS-style stories" - how many positions a single name had,
   what its average per-position return was, and was the model directionally
   right or just unlucky.

Outputs:
    reports/no_stop_per_model.csv
    reports/no_stop_per_position.csv
    reports/no_stop.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.construct import long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


MODELS = ("naive", "ew", "logit", "gbm_cls", "gbm_rank")


def _build_position_pnl(model: str, oof, prices_panel, features) -> pd.DataFrame:
    sub = oof[(oof["model"] == model) & (oof["period"] == "OOS")].dropna(subset=["score"])
    if sub.empty:
        return pd.DataFrame()
    joined = sub.merge(
        prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
        on=["Date", "Ticker"], how="left",
    )
    w = long_short_decile(joined, n_deciles=5)
    shorts = w[w["weight"] < 0].copy()
    if shorts.empty:
        return pd.DataFrame()
    # forward 1m return is already in features.
    fwd = features[["Date", "Ticker", "fwd_ret_4w", "ShortPct", "mktCap"]].drop_duplicates(["Date", "Ticker"])
    df = shorts.merge(fwd, on=["Date", "Ticker"], how="left")
    df = df.dropna(subset=["fwd_ret_4w"])
    # trade_return: positive = stock fell = short won.
    df["trade_return"] = df["fwd_ret_4w"] * np.sign(df["weight"])
    df["position_pnl"] = df["weight"] * df["fwd_ret_4w"]
    df["model"] = model
    return df


def main() -> None:
    settings.ensure_dirs()

    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    features = read_parquet(settings.processed_dir / "features_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")

    prices_panel = clean[["Date", "Ticker", "adjClose", "investable"]].copy()
    if "adv_aud" in features.columns:
        f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")

    per_model_rows: list[dict] = []
    all_positions: list[pd.DataFrame] = []
    for model in MODELS:
        df = _build_position_pnl(model, oof, prices_panel, features)
        if df.empty:
            continue
        all_positions.append(df)
        # ---- summary ----
        win_rate = float((df["trade_return"] > 0).mean())
        mean_ret = float(df["trade_return"].mean())
        median_ret = float(df["trade_return"].median())
        worst = float(df["trade_return"].min())
        best = float(df["trade_return"].max())
        std = float(df["trade_return"].std())
        # Big-loss buckets.
        n_loss_25 = int((df["trade_return"] < -0.25).sum())
        n_loss_50 = int((df["trade_return"] < -0.50).sum())
        n_win_25 = int((df["trade_return"] > 0.25).sum())
        n_total = len(df)
        # Aggregate book-level P&L (no stop applied).
        per_month = df.groupby("Date")["position_pnl"].sum()
        total_pnl_book = float(per_month.sum())
        # Approximate cost: 2 * 25bps * |weight|.sum() per month + borrow.
        cost_per_month = df.groupby("Date").apply(
            lambda s: 2 * 25 / 1e4 * s["weight"].abs().sum() + s["weight"].abs().sum() * (1.5 / 100) / 12,
            include_groups=False,
        )
        net_per_month = per_month.sub(cost_per_month, fill_value=0).sort_index()
        # Sharpe and MaxDD on the monthly book series.
        if len(net_per_month) > 1:
            sharpe = net_per_month.mean() / net_per_month.std() * np.sqrt(12)
            eq = (1.0 + net_per_month).cumprod()
            mdd = float((eq / eq.cummax() - 1.0).min())
        else:
            sharpe = float("nan")
            mdd = float("nan")
        per_model_rows.append({
            "model": model,
            "n_positions": n_total,
            "win_rate_%": round(100 * win_rate, 1),
            "mean_trade_%": round(100 * mean_ret, 2),
            "median_trade_%": round(100 * median_ret, 2),
            "std_trade_%": round(100 * std, 2),
            "best_position_%": round(100 * best, 1),
            "worst_position_%": round(100 * worst, 1),
            "n_loss_>25%": n_loss_25,
            "n_loss_>50%": n_loss_50,
            "n_win_>25%": n_win_25,
            "short_book_Sharpe_OOS": round(float(sharpe), 2),
            "short_book_MaxDD_%": round(100 * mdd, 1),
            "total_pnl_book_%": round(100 * total_pnl_book, 1),
        })

    summary = pd.DataFrame(per_model_rows).sort_values("win_rate_%", ascending=False)
    summary_csv = settings.reports_dir / "no_stop_per_model.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info(f"wrote {summary_csv}")

    if all_positions:
        cat = pd.concat(all_positions, ignore_index=True)
        cat = cat.sort_values("trade_return", ascending=False)
        cat.to_csv(settings.reports_dir / "no_stop_per_position.csv", index=False)

        # Most consistent winners + worst losers per model (use logit, the canonical).
        logit_df = cat[cat["model"] == "logit"]
        agg = logit_df.groupby("Ticker").agg(
            Company=("trade_return", lambda s: ""),  # placeholder
            n_positions=("trade_return", "size"),
            avg_trade=("trade_return", "mean"),
            total_pnl=("position_pnl", "sum"),
            worst_trade=("trade_return", "min"),
            best_trade=("trade_return", "max"),
            win_rate=("trade_return", lambda s: float((s > 0).mean())),
            avg_si=("ShortPct", "mean"),
        ).reset_index()
        # Pull Company from features.
        company_map = (
            features.dropna(subset=["Company"])
                    .groupby("Ticker")["Company"].first().to_dict()
        )
        agg["Company"] = agg["Ticker"].map(company_map).fillna("")
        agg["Company"] = (
            agg["Company"].astype(str).str.replace(" ORDINARY", "", regex=False).str.title()
        )
        agg = agg.sort_values("total_pnl", ascending=False)
        agg_csv = settings.reports_dir / "no_stop_per_ticker_logit.csv"
        agg.to_csv(agg_csv, index=False)

    # ---- markdown ----
    md = "# Stop-DISABLED OOS short-book analysis\n\n"
    md += "All 5 models re-evaluated on the OOS holdout (2023-06 → 2026-05) "
    md += "with the 15 % per-position stop turned OFF. Per-position "
    md += "returns are the raw monthly forward returns — no floor. The "
    md += "naive baseline still uses raw ShortPct; the trained models "
    md += "(logit / gbm_cls / gbm_rank) use the same final-fit predictions "
    md += "as the headline run.\n\n"
    md += "**The 50 % win-rate question**: what share of monthly short "
    md += "positions ended in profit (= stock fell)?\n\n"
    cols = [
        "model", "n_positions", "win_rate_%", "mean_trade_%", "median_trade_%",
        "std_trade_%", "worst_position_%", "n_loss_>25%", "n_loss_>50%",
        "n_win_>25%", "short_book_Sharpe_OOS", "short_book_MaxDD_%",
    ]
    try:
        md += summary[cols].to_markdown(index=False)
    except (ImportError, ValueError):
        md += summary[cols].to_string(index=False)
    md += "\n\n### Read this carefully\n\n"
    md += "- **`win_rate_%`** is the per-position monthly win-rate — how "
    md += "  often the stock fell during the month we were short it. The "
    md += "  goal is > 50 %.\n"
    md += "- **`worst_position_%`** is the single worst monthly outcome. "
    md += "  Without the stop, this is uncapped — a stock that rallied "
    md += "  40 % in a month gives a 40 % per-position loss.\n"
    md += "- **`n_loss_>25%`** and **`n_loss_>50%`** count the catastrophic "
    md += "  single-month positions (un-stopped). These are the squeezes / "
    md += "  takeover bumps / commodity rallies that the 15 % stop normally "
    md += "  caps at −16 %.\n\n"
    if all_positions:
        md += "## Worst 20 single-position outcomes (no stop)\n\n"
        worst20 = cat.nsmallest(20, "trade_return")[
            ["model", "Date", "Ticker", "ShortPct", "trade_return", "fwd_ret_4w"]
        ].copy()
        worst20["trade_return"] = (worst20["trade_return"] * 100).round(1)
        worst20["fwd_ret_4w"] = (worst20["fwd_ret_4w"] * 100).round(1)
        worst20["ShortPct"] = worst20["ShortPct"].round(2)
        worst20["Date"] = pd.to_datetime(worst20["Date"]).dt.strftime("%Y-%m")
        worst20 = worst20.rename(columns={
            "trade_return": "trade_return_%",
            "fwd_ret_4w": "stock_ret_%",
            "ShortPct": "SI_%",
        })
        try:
            md += worst20.to_markdown(index=False)
        except (ImportError, ValueError):
            md += worst20.to_string(index=False)
        md += "\n"

    md_path = settings.reports_dir / "no_stop.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()
