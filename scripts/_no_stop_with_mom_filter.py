"""No-stop OOS short book WITH a momentum filter.

Idea: do not short any name in the top 30 % of cross-sectional 12-week
momentum. Avoids "fighting the tape" - the squeezes that killed our
naked-short results (APX +218 %, 4DX +217 %, BRN +177 %) were all
already rallying when we shorted them.

Filter is applied AFTER the model picks the top-quintile (by score)
shorts. If a candidate has mom_12w_rk > momentum_cutoff, drop it.
Bottom-quintile longs are unchanged.

Sweep momentum_cutoff ∈ {1.0 (off), 0.9, 0.8, 0.7, 0.6, 0.5} for
each model. Report per-position win-rate + the short-leg Sharpe with
the stop OFF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.construct import long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


MODELS = ("naive", "ew", "logit", "gbm_cls", "gbm_rank")
MOM_CUTOFFS = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50)


def main() -> None:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    features = read_parquet(settings.processed_dir / "features_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")

    prices_panel = clean[["Date", "Ticker", "adjClose", "investable"]].copy()
    if "adv_aud" in features.columns:
        f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")

    # Pre-join the per-row 12-week momentum rank for filtering.
    mom = features[["Date", "Ticker", "mom_12w_rk", "ShortPct", "fwd_ret_4w"]].drop_duplicates(["Date", "Ticker"])
    mom["mom_12w_rk"] = pd.to_numeric(mom["mom_12w_rk"], errors="coerce").fillna(0.5)

    rows: list[dict] = []
    for model in MODELS:
        sub = oof[(oof["model"] == model) & (oof["period"] == "OOS")].dropna(subset=["score"])
        joined = sub.merge(
            prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
            on=["Date", "Ticker"], how="left",
        )
        w = long_short_decile(joined, n_deciles=5)
        shorts = w[w["weight"] < 0].copy()
        if shorts.empty:
            continue
        shorts = shorts.merge(mom, on=["Date", "Ticker"], how="left").dropna(subset=["fwd_ret_4w"])

        for cutoff in MOM_CUTOFFS:
            filt = shorts[shorts["mom_12w_rk"] < cutoff].copy()
            if filt.empty:
                continue
            # Re-equal-weight the survivors so each month's short book sums to -1.
            n_per_month = filt.groupby("Date").size().rename("n_kept")
            filt = filt.merge(n_per_month, on="Date")
            filt["weight"] = -1.0 / filt["n_kept"]

            filt["trade_return"] = filt["fwd_ret_4w"] * np.sign(filt["weight"])
            filt["position_pnl"] = filt["weight"] * filt["fwd_ret_4w"]

            win_rate = float((filt["trade_return"] > 0).mean())
            mean_trade = float(filt["trade_return"].mean())
            median_trade = float(filt["trade_return"].median())
            worst = float(filt["trade_return"].min())
            n_loss_25 = int((filt["trade_return"] < -0.25).sum())
            n_loss_50 = int((filt["trade_return"] < -0.50).sum())
            n_kept_total = len(filt)
            n_dropped = len(shorts) - n_kept_total

            per_month = filt.groupby("Date")["position_pnl"].sum()
            cost = filt.groupby("Date").apply(
                lambda s: 2 * 25 / 1e4 * s["weight"].abs().sum() + s["weight"].abs().sum() * (1.5 / 100) / 12,
                include_groups=False,
            )
            net_pm = per_month.sub(cost, fill_value=0).sort_index()
            if len(net_pm) > 2:
                sharpe = net_pm.mean() / net_pm.std() * np.sqrt(12)
                eq = (1.0 + net_pm).cumprod()
                mdd = float((eq / eq.cummax() - 1.0).min())
            else:
                sharpe = float("nan")
                mdd = float("nan")

            rows.append({
                "model": model,
                "mom_cutoff": cutoff,
                "n_kept": n_kept_total,
                "n_dropped": n_dropped,
                "win_rate_%": round(100 * win_rate, 1),
                "mean_trade_%": round(100 * mean_trade, 2),
                "median_trade_%": round(100 * median_trade, 2),
                "worst_position_%": round(100 * worst, 1),
                "n_loss_>25%": n_loss_25,
                "n_loss_>50%": n_loss_50,
                "short_Sharpe_OOS": round(float(sharpe), 2),
                "short_MaxDD_%": round(100 * mdd, 1),
            })

    df = pd.DataFrame(rows)
    csv_path = settings.reports_dir / "no_stop_mom_filter.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    md = "# No-stop short book with momentum filter\n\n"
    md += "Drop any name in the top X % of 12-week cross-sectional momentum "
    md += "BEFORE shorting. Re-equal-weight survivors. No stop loss applied. "
    md += "`mom_cutoff=1.0` = no filter (every top-quintile name shorted), "
    md += "`mom_cutoff=0.50` = only short below-median-momentum names.\n\n"
    md += "**Key columns**: `win_rate_%` = share of monthly short positions "
    md += "that ended in profit (target > 50 %). `worst_position_%` = single "
    md += "worst monthly outcome (unstopped, so uncapped). `short_Sharpe_OOS` "
    md += "= annualised Sharpe of the short-leg-only monthly book.\n\n"
    # Pivot Sharpe by (model, mom_cutoff).
    pv_sharpe = df.pivot_table(index="mom_cutoff", columns="model",
                                values="short_Sharpe_OOS")
    pv_win = df.pivot_table(index="mom_cutoff", columns="model", values="win_rate_%")
    pv_worst = df.pivot_table(index="mom_cutoff", columns="model", values="worst_position_%")
    pv_n50 = df.pivot_table(index="mom_cutoff", columns="model", values="n_loss_>50%")

    def _md(df_, title):
        try:
            body = df_.to_markdown()
        except (ImportError, ValueError):
            body = df_.to_string()
        return f"\n### {title}\n\n{body}\n"

    md += _md(pv_sharpe, "Short-leg OOS Sharpe (no stop, with momentum filter)")
    md += _md(pv_win, "Per-position win-rate %")
    md += _md(pv_worst, "Worst single-position % (uncapped)")
    md += _md(pv_n50, "Count of positions losing > 50 %")

    md_path = settings.reports_dir / "no_stop_mom_filter.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()
