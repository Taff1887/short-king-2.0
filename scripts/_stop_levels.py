"""Stop-level sweep: re-run the production backtest at stop_loss_pct ∈
{1.0, 0.20, 0.15, 0.12, 0.10, 0.08}. Use the SAME engine + costs as the
headline so the comparison is apples-to-apples — only the stop floor moves.
Long-short quintile (L/S net dollar-neutral) only.

Output: ``reports/stop_levels.csv`` + a markdown summary.
"""

from __future__ import annotations

import pandas as pd

from short_king.portfolio.backtest import CostConfig, _summarise_returns, backtest_weekly
from short_king.portfolio.construct import long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

STOP_LEVELS = (None, 0.20, 0.15, 0.12, 0.10, 0.08)


def main() -> None:
    settings.ensure_dirs()

    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    features = read_parquet(settings.processed_dir / "features_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")

    prices_panel = clean[["Date", "Ticker", "adjClose", "investable"]].copy()
    if "adv_aud" in features.columns:
        f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")

    rows: list[dict] = []
    for stop in STOP_LEVELS:
        for model in ("naive", "logit", "gbm_rank"):
            sub = oof[oof["model"] == model].dropna(subset=["score"])
            joined = sub.merge(
                prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
                on=["Date", "Ticker"], how="left",
            )
            weights = long_short_decile(joined, n_deciles=5)
            cfg = CostConfig(
                bps_round_trip=25.0, slippage_bps=5.0, annual_borrow_pct=1.5,
                stop_loss_pct=stop,
                stop_slippage_pct=0.01,
                periods_per_year=12,
            )
            res = backtest_weekly(weights, prices_panel, cost_config=cfg)

            # IS / OOS split via merge with the oof period column.
            ret = res.returns.merge(
                sub[["Date", "period"]].drop_duplicates(),
                on="Date", how="left",
            )
            for label in ("IS", "OOS"):
                segment = ret[ret["period"] == label]
                if len(segment) < 6:
                    continue
                s = _summarise_returns(
                    net=segment.set_index("Date")["ret_net"],
                    gross=segment.set_index("Date")["ret_gross"],
                    turnover=segment.set_index("Date")["turnover"],
                    periods_per_year=12,
                )
                rows.append({
                    "model": model,
                    "period": label,
                    "stop_pct": "off" if stop is None else f"{stop:.2f}",
                    "Sharpe": round(float(s["Sharpe"]), 2),
                    "CAGR_%": round(100 * float(s["CAGR"]), 1),
                    "MaxDD_%": round(100 * float(s["max_drawdown"]), 1),
                    "hit_rate_%": round(100 * float(s["hit_rate"]), 1),
                })

    df = pd.DataFrame(rows)
    csv_path = settings.reports_dir / "stop_levels.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Markdown pivot: rows = stop_pct, cols = model, value = OOS Sharpe.
    oos = df[df["period"] == "OOS"]
    sharpe_pivot = oos.pivot_table(index="stop_pct", columns="model", values="Sharpe")
    cagr_pivot = oos.pivot_table(index="stop_pct", columns="model", values="CAGR_%")
    mdd_pivot = oos.pivot_table(index="stop_pct", columns="model", values="MaxDD_%")
    # Order index by descending stop tightness (off → 0.08).
    order = ["off", "0.20", "0.15", "0.12", "0.10", "0.08"]
    sharpe_pivot = sharpe_pivot.reindex([i for i in order if i in sharpe_pivot.index])
    cagr_pivot = cagr_pivot.reindex(sharpe_pivot.index)
    mdd_pivot = mdd_pivot.reindex(sharpe_pivot.index)

    def _md(df_: pd.DataFrame, title: str) -> str:
        try:
            body = df_.to_markdown()
        except (ImportError, ValueError):
            body = df_.to_string()
        return f"\n### {title}\n\n{body}\n"

    md = "# Stop-level sweep — long-short quintile OOS\n\n"
    md += "Same engine, same costs, same Friday rebalance, same 100 bps stop-fill "
    md += "slippage. Only the cumulative stop floor moves.\n"
    md += _md(sharpe_pivot, "OOS Sharpe")
    md += _md(cagr_pivot, "OOS CAGR (%)")
    md += _md(mdd_pivot, "OOS MaxDD (%)")
    md_path = settings.reports_dir / "stop_levels.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()
