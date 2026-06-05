"""Derive the LONG-ONLY component of each model's L/S quintile and chart it
alongside the short-only and L/S full versions, plus the ASX 200
benchmark.

Math:
  L/S quintile net return = long_leg_ret + short_leg_ret (both costed)
  quintile_short net return = short_leg_ret (the SAME short_leg, just
    on its own with the same short-side costs)
  Therefore:
    long_only_ret = ls_ret - short_only_ret

This works for the NET returns because the long leg only carries
commission (no borrow), and the L/S file has the long-leg commission
included while the short-only file does not -- subtracting cancels out
the shared short-side costs and leaves the long-only net.

Uses the NO-STOP parquets (long leg doesn't have stops anyway, and the
stop only affects the short leg, so subtraction is cleaner on the
no-stop versions).

Outputs:
  charts/long_only_components.png
  reports/long_only_summary.csv  -- Sharpe / CAGR / MaxDD per
                                    (model, leg_type, period)
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


def main() -> int:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    oof_dates = oof[["Date", "period"]].drop_duplicates("Date")
    oof_dates["Date"] = pd.to_datetime(oof_dates["Date"]).dt.normalize()

    series_map: dict[str, pd.Series] = {}
    summary_rows: list[dict] = []
    for model in MODELS:
        ls_p = settings.reports_dir / f"backtest_monthly_{model}_long_short_quintile_nostop.parquet"
        so_p = settings.reports_dir / f"backtest_monthly_{model}_quintile_short_nostop.parquet"
        if not ls_p.exists() or not so_p.exists():
            logger.warning(f"{model}: missing nostop parquets, skipping")
            continue
        ls = read_parquet(ls_p)
        so = read_parquet(so_p)
        ls["Date"] = pd.to_datetime(ls["Date"]).dt.normalize()
        so["Date"] = pd.to_datetime(so["Date"]).dt.normalize()
        ls_ret = ls.set_index("Date")["ret_net"]
        so_ret = so.set_index("Date")["ret_net"]
        # Long-only = L/S - short-only (the short-side costs cancel out;
        # long-leg commission stays in L/S and not in short-only).
        long_only = (ls_ret - so_ret).dropna()
        long_only.name = f"{model}/long_only"
        series_map[long_only.name] = long_only

        # Per-period summary stats.
        for period_label in ("ALL", "IS", "OOS"):
            for leg_label, series in (("long_only", long_only),
                                       ("long_short_quintile", ls_ret),
                                       ("quintile_short", so_ret)):
                if period_label == "ALL":
                    d = pd.DatetimeIndex(series.index)
                else:
                    d = pd.DatetimeIndex(oof_dates.loc[oof_dates.period == period_label, "Date"])
                r = series.reindex(d).dropna()
                if len(r) < 6:
                    continue
                s = _summarise_returns(r, r, pd.Series(0.0, index=r.index),
                                        periods_per_year=PERIODS_PER_YEAR)
                summary_rows.append({
                    "model": model, "leg": leg_label, "period": period_label,
                    "n_months": int(s.get("n_weeks", 0)),
                    "Sharpe": round(float(s.get("Sharpe", np.nan)), 3),
                    "CAGR_%": round(float(s.get("CAGR", np.nan)) * 100, 2),
                    "ann_vol_%": round(float(s.get("ann_vol", np.nan)) * 100, 2),
                    "MaxDD_%": round(float(s.get("max_drawdown", np.nan)) * 100, 2),
                    "hit_rate_%": round(float(s.get("hit_rate", np.nan)) * 100, 1),
                })

    if not series_map:
        logger.error("no series built")
        return 1

    # Build a returns frame with all three long-only curves.
    returns_df = pd.DataFrame(series_map).sort_index()
    bench = _asx200_returns(pd.DatetimeIndex(returns_df.index))

    # ASX 200 row in the summary too (full window + IS / OOS slices).
    if bench is not None:
        for period_label in ("ALL", "IS", "OOS"):
            if period_label == "ALL":
                d = pd.DatetimeIndex(bench.index)
            else:
                d = pd.DatetimeIndex(oof_dates.loc[oof_dates.period == period_label, "Date"])
            r = bench.reindex(d).dropna()
            if len(r) < 6:
                continue
            s = _summarise_returns(r, r, pd.Series(0.0, index=r.index),
                                    periods_per_year=PERIODS_PER_YEAR)
            summary_rows.append({
                "model": "ASX200", "leg": "buy_and_hold", "period": period_label,
                "n_months": int(s.get("n_weeks", 0)),
                "Sharpe": round(float(s.get("Sharpe", np.nan)), 3),
                "CAGR_%": round(float(s.get("CAGR", np.nan)) * 100, 2),
                "ann_vol_%": round(float(s.get("ann_vol", np.nan)) * 100, 2),
                "MaxDD_%": round(float(s.get("max_drawdown", np.nan)) * 100, 2),
                "hit_rate_%": round(float(s.get("hit_rate", np.nan)) * 100, 1),
            })

    csv_path = settings.reports_dir / "long_only_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    chart_path = settings.charts_dir / "long_only_components.png"
    rc.chart_cumulative_returns(returns_df, chart_path, bench=bench)
    logger.info(f"wrote {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
