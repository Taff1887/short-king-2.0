"""Sweep ``stop_slippage_pct`` across {0, 1, 2 %} for the headline strategies.

Re-runs the backtest engine in-memory (no parquet writes) and reports the
Sharpe / CAGR / MaxDD for the best two models at each slippage level.
Output saved to ``reports/stop_sensitivity.csv`` and printed for the README.
"""

from __future__ import annotations

import pandas as pd

from short_king.portfolio.backtest import CostConfig, backtest_weekly
from short_king.portfolio.construct import decile_short, long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

SLIPPAGES = (0.0, 0.005, 0.01, 0.015, 0.02)
N_BUCKETS = 5

oof = read_parquet(settings.reports_dir / "oof_predictions.parquet")
clean = read_parquet(settings.processed_dir / "master_clean.parquet")
features = read_parquet(settings.processed_dir / "features.parquet")

# Build prices_panel the same way 06_backtest does.
keep = [c for c in ("Date", "Ticker", "adjClose", "investable") if c in clean.columns]
prices_panel = clean[keep].copy()
if "adv_aud" in features.columns:
    f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
    prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")

rows: list[dict] = []
for slip in SLIPPAGES:
    cost = CostConfig(
        bps_round_trip=25.0, slippage_bps=5.0, annual_borrow_pct=1.5,
        stop_loss_pct=0.15, stop_slippage_pct=slip,
    )
    for model in ("naive", "logit", "gbm_rank", "gbm_cls", "ew"):
        sub = oof[oof["model"] == model]
        if sub.empty:
            continue
        joined = sub.merge(
            prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
            on=["Date", "Ticker"], how="left",
        )
        ls = long_short_decile(joined, n_deciles=N_BUCKETS)
        if ls.empty:
            continue
        res = backtest_weekly(ls, prices_panel, cost_config=cost)
        rows.append({
            "stop_slippage_pct": f"{slip:.3f}",
            "model": model,
            "strategy": "long_short_quintile",
            "CAGR": float(res.summary["CAGR"]),
            "Sharpe": float(res.summary["Sharpe"]),
            "MaxDD": float(res.summary["max_drawdown"]),
            "n_stops": int(res.summary["n_stops_total"]),
            "stop_slippage_drag": float(res.summary["stop_slippage_drag_total"]),
        })

df = pd.DataFrame(rows)
df["CAGR"] = df["CAGR"].round(4)
df["Sharpe"] = df["Sharpe"].round(3)
df["MaxDD"] = df["MaxDD"].round(4)
df["stop_slippage_drag"] = df["stop_slippage_drag"].round(4)

out_csv = settings.reports_dir / "stop_sensitivity.csv"
df.to_csv(out_csv, index=False)
logger.info(f"wrote {out_csv}")

# Pretty pivot for the README - one row per model, columns = slippage levels.
pivot = (
    df.pivot_table(index="model", columns="stop_slippage_pct", values="Sharpe")
      .reindex(["naive", "logit", "gbm_rank", "gbm_cls", "ew"])
      .round(2)
)
print("\nSharpe sensitivity (long-short quintile, by stop_slippage_pct):")
print(pivot.to_string())
