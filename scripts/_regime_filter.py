"""Test a market-regime filter that skips the short leg during sharp ASX 200
recovery rallies (when the index has just bounced hard from a drawdown,
junk-rally squeezes destroy systematic short books).

The hypothesis: "don't short into a rally" - use a *trailing*, no-look-
ahead signal. Two variants tested:

  T-3m: skip short leg when ASX 200 trailing 3-month return > +X %
  T-6m: skip short leg when ASX 200 trailing 6-month return > +X %

Skip mechanic per month:
  * long_short_quintile: keep long leg, drop short leg -> half-book
    (long-only on the strategy's bottom quintile), no short notional.
  * quintile_short: skip entirely, return 0 % for the month.

Tested thresholds: 5 %, 10 %, 15 %, 20 % (trailing-3m gating).
No look-ahead: ASX 200 return computed strictly BEFORE each rebalance date.

Outputs:
  reports/regime_filter.csv  -- per (model, strategy, threshold) Sharpe/CAGR/etc
  reports/regime_filter.md   -- markdown summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.backtest import _summarise_returns
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

MODELS = ("naive", "ew", "logit")
STRATEGIES = ("quintile_short", "long_short_quintile")
TRAIL_MONTHS = (3, 6)
THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
PERIODS_PER_YEAR = 12


def _asx200_monthly_returns(rebalance_dates: pd.DatetimeIndex) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance required for ASX200 fetch")
    start = (rebalance_dates.min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (rebalance_dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    raw = yf.download("^AXJO", start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    close = raw["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    snap = close.reindex(rebalance_dates, method="ffill")
    return snap


def _trailing_return(px: pd.Series, months: int) -> pd.Series:
    """Trailing months-month return, ending at each rebalance date.
    Strict no-look-ahead: uses price at (date - months) periods ago.
    """
    shifted = px.shift(months)
    return px / shifted - 1.0


def _split_legs_from_returns(short_only: pd.Series, ls_quintile: pd.Series) -> pd.Series:
    """Recover the LONG-leg return from (ls = long + short, short_only = short).
    short_only is the short leg's contribution to the L/S book; in the L/S
    quintile construction with equal weights summing to -1 on each leg, the
    L/S net return = mean(long_stock_ret) + mean(short_return) where the
    short-only series is the latter. So long = ls - short_only.
    """
    return ls_quintile - short_only


def main() -> int:
    settings.ensure_dirs()
    rows: list[dict] = []
    px = None

    for model in MODELS:
        # Use the NO-STOP versions so we isolate the regime-filter effect from
        # the stop-loss effect. The user's question is about the underlying
        # signal, not the stop.
        try:
            df_so = read_parquet(
                settings.reports_dir / f"backtest_monthly_{model}_quintile_short_nostop.parquet")
            df_ls = read_parquet(
                settings.reports_dir / f"backtest_monthly_{model}_long_short_quintile_nostop.parquet")
        except FileNotFoundError as exc:
            logger.warning(f"{model}: missing nostop parquets -- skipping ({exc})")
            continue
        df_so["Date"] = pd.to_datetime(df_so["Date"]).dt.normalize()
        df_ls["Date"] = pd.to_datetime(df_ls["Date"]).dt.normalize()
        so = df_so.set_index("Date")["ret_net"]
        ls = df_ls.set_index("Date")["ret_net"]
        # Long leg recovered from ls = long + short_contrib. The short leg's
        # contribution to the L/S book is the short-only return (both sized
        # to a 1.0 notional leg).
        long_leg = ls - so

        if px is None:
            px = _asx200_monthly_returns(pd.DatetimeIndex(sorted(so.index)))

        # OOF predictions for IS/OOS labels.
        oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
        period_map = (oof[oof["model"] == model][["Date", "period"]]
                      .drop_duplicates("Date")
                      .assign(Date=lambda d: pd.to_datetime(d["Date"]).dt.normalize())
                      .set_index("Date")["period"])

        # Baseline (no filter) for each strategy.
        for strategy, baseline in (("quintile_short", so), ("long_short_quintile", ls)):
            for period_label in ("ALL", "IS", "OOS"):
                if period_label == "ALL":
                    sub = baseline
                else:
                    keep = period_map.reindex(baseline.index) == period_label
                    sub = baseline.loc[keep.fillna(False)]
                if len(sub) < 6:
                    continue
                s = _summarise_returns(sub, sub, pd.Series(0.0, index=sub.index),
                                        periods_per_year=PERIODS_PER_YEAR)
                rows.append({
                    "model": model, "strategy": strategy,
                    "filter": "NONE", "trail_months": 0, "threshold_pct": 0,
                    "period": period_label,
                    "n_months": int(s.get("n_weeks", 0)),
                    "skipped_short_months": 0,
                    "Sharpe": round(float(s.get("Sharpe", np.nan)), 3),
                    "CAGR_%": round(float(s.get("CAGR", np.nan)) * 100, 2),
                    "MaxDD_%": round(float(s.get("max_drawdown", np.nan)) * 100, 2),
                    "hit_rate_%": round(float(s.get("hit_rate", np.nan)) * 100, 1),
                })

        # Filtered variants.
        for tm in TRAIL_MONTHS:
            tr = _trailing_return(px, tm)
            for thr in THRESHOLDS:
                # rally_mask = ASX 200 trailing-tm return > threshold AT THE
                # REBALANCE DATE. tr is shift(tm), so already uses prior data
                # only (no look-ahead).
                rally = (tr > thr).reindex(so.index).fillna(False)

                # Short-only filtered: 0 % in rally months.
                so_filt = so.where(~rally, 0.0)
                # L/S filtered: keep long leg only in rally months.
                ls_filt = ls.where(~rally, long_leg.where(~rally, 0.0))
                # ^ the second .where keeps long_leg in rally months, ls
                # in normal months.

                for strategy, filt_series in (("quintile_short", so_filt),
                                              ("long_short_quintile", ls_filt)):
                    skipped = int(rally.sum())
                    for period_label in ("ALL", "IS", "OOS"):
                        if period_label == "ALL":
                            sub = filt_series
                            sub_rally = rally
                        else:
                            keep = period_map.reindex(filt_series.index) == period_label
                            sub = filt_series.loc[keep.fillna(False)]
                            sub_rally = rally.loc[keep.fillna(False)]
                        if len(sub) < 6:
                            continue
                        s = _summarise_returns(sub, sub, pd.Series(0.0, index=sub.index),
                                                periods_per_year=PERIODS_PER_YEAR)
                        rows.append({
                            "model": model, "strategy": strategy,
                            "filter": f"trail{tm}m>{int(thr*100)}%",
                            "trail_months": tm,
                            "threshold_pct": int(thr * 100),
                            "period": period_label,
                            "n_months": int(s.get("n_weeks", 0)),
                            "skipped_short_months": int(sub_rally.sum()),
                            "Sharpe": round(float(s.get("Sharpe", np.nan)), 3),
                            "CAGR_%": round(float(s.get("CAGR", np.nan)) * 100, 2),
                            "MaxDD_%": round(float(s.get("max_drawdown", np.nan)) * 100, 2),
                            "hit_rate_%": round(float(s.get("hit_rate", np.nan)) * 100, 1),
                        })

    df = pd.DataFrame(rows)
    csv_path = settings.reports_dir / "regime_filter.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Markdown summary: focus on full-period (ALL) and OOS for the L/S
    # quintile (the user's headline strategy). Show NONE + each
    # threshold side by side.
    def _summary(strategy: str, period: str) -> str:
        sub = df[(df.strategy == strategy) & (df.period == period)].copy()
        sub["filter_short"] = sub["filter"]
        pv = sub.pivot_table(index="model", columns="filter", values="Sharpe").reindex(MODELS)
        cols_order = ["NONE"] + [f"trail3m>{int(t*100)}%" for t in THRESHOLDS] + [f"trail6m>{int(t*100)}%" for t in THRESHOLDS]
        cols_order = [c for c in cols_order if c in pv.columns]
        pv = pv[cols_order]
        hdr = "| Model | " + " | ".join(cols_order) + " |"
        sep = "|" + "|".join(["---"] * (len(cols_order) + 1)) + "|"
        body = []
        for m in MODELS:
            cells = [f"{pv.loc[m, c]:+.3f}" if pd.notna(pv.loc[m, c]) else "" for c in cols_order]
            body.append(f"| {m} | " + " | ".join(cells) + " |")
        return "\n".join([hdr, sep, *body])

    md = ["# Market-regime filter sweep (skip shorts during ASX 200 rally regimes)",
          "",
          "Rule: in any rebalance month where ASX 200 trailing-N-month return "
          "exceeded the threshold AT THE REBALANCE DATE, skip the short leg "
          "(short-only -> 0 %; L/S quintile -> long-only that month). No look-"
          "ahead: trailing return uses only prior closes. **All numbers use the "
          "NO-STOP backtests** so the filter effect is isolated from stop-loss "
          "noise.",
          ""]
    for strategy in STRATEGIES:
        for period in ("ALL", "OOS"):
            md += [f"\n## {strategy} -- {period} Sharpe\n", _summary(strategy, period)]
    # Plus: how many rebalances each filter skipped.
    md += ["\n## Months skipped (by filter)\n"]
    sk = df[df.period == "ALL"].pivot_table(
        index="filter", values="skipped_short_months", aggfunc="max"
    ).reindex([f"trail3m>{int(t*100)}%" for t in THRESHOLDS]
              + [f"trail6m>{int(t*100)}%" for t in THRESHOLDS])
    md += ["| Filter | Months skipped (of ~191 total) |",
           "|---|---:|"]
    for idx, row in sk.iterrows():
        md.append(f"| {idx} | {int(row['skipped_short_months'])} |")
    md.append("")

    md_path = settings.reports_dir / "regime_filter.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
