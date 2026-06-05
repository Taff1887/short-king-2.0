"""Build the headline cumulative-growth chart and the headline summary table
using the STOP-LOSS-APPLIED backtest parquets, plus the ASX 200 buy & hold
benchmark with its own Sharpe / CAGR / MaxDD over the same windows.

Inputs (already produced by _apply_stop_loss_full.py + 06_backtest.py):
  * reports/backtest_monthly_<model>_<strategy>.parquet     (stopped returns)
  * reports/backtest_summary_monthly.csv                    (stopped summary)
  * reports/oof_predictions_monthly.parquet                 (for IS/OOS dates)

Outputs:
  * charts/cumulative_returns_monthly.png  -- 6 strategy curves + ASX 200
  * reports/headline_table.csv             -- 6 strategies + ASX 200 row
  * reports/headline_table.md              -- markdown for README
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.backtest import _summarise_returns
from short_king.reporting import charts as rc
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

MODELS = ("naive", "ew", "logit")
STRATEGIES = ("quintile_short", "long_short_quintile")
PERIODS_PER_YEAR = 12


def _asx200_returns(dates: pd.DatetimeIndex) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        start = (dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        raw = yf.download("^AXJO", start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"ASX200 fetch failed: {exc}")
        return None
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    close = raw["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    snap = close.reindex(dates, method="ffill")
    rets = snap.pct_change().dropna()
    rets.name = "ASX 200 (buy & hold)"
    return rets


def _load_returns(suffix: str = "") -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for model in MODELS:
        for strategy in STRATEGIES:
            p = settings.reports_dir / f"backtest_monthly_{model}_{strategy}{suffix}.parquet"
            if not p.exists():
                logger.warning(f"missing {p}, skipping")
                continue
            df = read_parquet(p)
            s = pd.Series(
                df["ret_net"].to_numpy(),
                index=pd.to_datetime(df["Date"]).dt.normalize(),
                name=f"{model}/{strategy}",
            ).sort_index()
            series[s.name] = s
    return series


def main() -> int:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")

    # Load both the STOPPED parquets (headline) and the NO-STOP copies if
    # available. The no-stop set lives alongside as *_nostop.parquet so the
    # README can show both charts side-by-side.
    series_map = _load_returns()  # stopped (headline)
    series_map_nostop = _load_returns("_nostop")

    if not series_map:
        logger.error("no backtest parquets found")
        return 1

    # Build the returns DataFrame for the chart (one column per strategy).
    returns_df = pd.DataFrame(series_map).sort_index()
    returns_df_nostop = (
        pd.DataFrame(series_map_nostop).sort_index() if series_map_nostop else None
    )

    # ASX 200 returns on the same monthly grid.
    bench = _asx200_returns(pd.DatetimeIndex(returns_df.index))
    if bench is None:
        logger.warning("ASX 200 benchmark unavailable, chart will skip it")

    # ---- Chart (stopped headline) -------------------------------------------
    chart_path = settings.charts_dir / "cumulative_returns_monthly.png"
    rc.chart_cumulative_returns(returns_df, chart_path, bench=bench)
    logger.info(f"wrote {chart_path}")

    # ---- Chart (no-stop comparison) -----------------------------------------
    if returns_df_nostop is not None:
        bench_nostop = _asx200_returns(pd.DatetimeIndex(returns_df_nostop.index))
        nostop_path = settings.charts_dir / "cumulative_returns_monthly_nostop.png"
        rc.chart_cumulative_returns(returns_df_nostop, nostop_path, bench=bench_nostop)
        logger.info(f"wrote {nostop_path}")

    # ---- Headline table with ASX 200 row -------------------------------------
    # IS / OOS periods per the OOF predictions.
    oof_dates = oof[["Date", "period"]].drop_duplicates("Date")
    oof_dates["Date"] = pd.to_datetime(oof_dates["Date"]).dt.normalize()

    def _row(name: str, rets: pd.Series, period_label: str, dates_in: pd.DatetimeIndex) -> dict:
        r = rets.reindex(dates_in).dropna()
        if len(r) < 3:
            return None
        s = _summarise_returns(
            net=r, gross=r, turnover=pd.Series(0.0, index=r.index),
            periods_per_year=PERIODS_PER_YEAR,
        )
        return {
            "strategy": name, "period": period_label,
            "n_months": int(s.get("n_weeks", 0)),
            "Sharpe": float(s.get("Sharpe", float("nan"))),
            "CAGR": float(s.get("CAGR", float("nan"))),
            "ann_vol": float(s.get("ann_vol", float("nan"))),
            "MaxDD": float(s.get("max_drawdown", float("nan"))),
            "hit_rate": float(s.get("hit_rate", float("nan"))),
        }

    out_rows: list[dict] = []
    for name, rets in series_map.items():
        for period_label in ("ALL", "IS", "OOS"):
            if period_label == "ALL":
                d = pd.DatetimeIndex(rets.index)
            else:
                d = pd.DatetimeIndex(oof_dates.loc[oof_dates.period == period_label, "Date"])
            row = _row(name, rets, period_label, d)
            if row:
                out_rows.append(row)
    if bench is not None:
        for period_label in ("ALL", "IS", "OOS"):
            if period_label == "ALL":
                d = pd.DatetimeIndex(bench.index)
            else:
                d = pd.DatetimeIndex(oof_dates.loc[oof_dates.period == period_label, "Date"])
            row = _row("ASX 200 (buy & hold)", bench, period_label, d)
            if row:
                out_rows.append(row)

    table_df = pd.DataFrame(out_rows)
    csv_path = settings.reports_dir / "headline_table.csv"
    table_df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Markdown view per period.
    def _md(df: pd.DataFrame) -> str:
        cols = ["strategy", "n_months", "Sharpe", "CAGR", "ann_vol", "MaxDD", "hit_rate"]
        d = df[cols].copy()
        d["Sharpe"] = d["Sharpe"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "")
        d["CAGR"] = d["CAGR"].map(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "")
        d["ann_vol"] = d["ann_vol"].map(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "")
        d["MaxDD"] = d["MaxDD"].map(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "")
        d["hit_rate"] = d["hit_rate"].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "")
        hdr = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        rows = ["| " + " | ".join(str(v) for v in r) + " |"
                for r in d.itertuples(index=False, name=None)]
        return "\n".join([hdr, sep, *rows])

    md_parts: list[str] = ["# Headline summary (stop-loss applied)\n"]
    for period_label in ("OOS", "ALL", "IS"):
        sub = table_df[table_df.period == period_label].sort_values("Sharpe", ascending=False)
        md_parts.append(f"\n## {period_label}\n")
        md_parts.append(_md(sub))
    md_path = settings.reports_dir / "headline_table.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
