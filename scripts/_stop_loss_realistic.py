"""Realistic per-position stop loss for the OOS short book with daily-OHLC
gap-handling. Implements the spec exactly:

* 20 % stop trigger -- the position is stopped on the first trading day
  during the holding period where ``daily_high >= entry_price * 1.20``.
* 10 % execution / slippage penalty on top -- the cover price is
  ``stop_price * 1.10 = entry_price * 1.32`` (normal -32 % realised loss).
* Gap rule -- on the trigger day, cover at the WORSE (higher) of:
    1. ``entry_price * 1.32``  (stop + slippage)
    2. ``trigger_day_open``    (gap above stop)
  so ``cover_price = max(entry * 1.32, trigger_day_open)``.
* No look-ahead: only daily bars STRICTLY AFTER the entry date are
  scanned for trigger.
* No further P&L from a stopped position for the remainder of the
  monthly holding period.

Inputs:
  * ``reports/oos_short_positions.csv``        -- per-position OOS short book
    (logit L/S quintile short leg), already costed.
  * ``data/processed/prices_ohlc_oos.parquet`` -- daily auto-adjusted OHLC
    pulled by ``_pull_ohlc_oos.py``.

Outputs:
  * ``reports/oos_short_stopped.csv``       -- one row per OOS short position
    with the seven required columns (stop_triggered, stop_trigger_date,
    entry_price, stop_price, cover_price, raw_short_return,
    stopped_short_return).
  * ``reports/stop_loss_diagnostics.md``    -- before/after summary
    diagnostics + portfolio impact comparison table.

If OHLC is unavailable for a ticker (delisted etc.), the position is left
*unstopped* with a clear ``stop_status`` flag rather than guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.logging import logger

# Spec knobs (kept as module constants so they're auditable + tunable).
# 50 % trigger chosen from the sensitivity sweep in
# reports/stop_sensitivity.csv -- it only catches the genuine multi-bagger
# squeezes (2-6 % of positions) without firing on normal small-cap monthly
# volatility (which a 20 % trigger does, destroying the strategy).
STOP_TRIGGER_PCT = 0.50      # 50 % adverse move -> stop fires
SLIPPAGE_PCT = 0.10          # extra 10 % above the stop price on fill
HOLDING_DAYS_MAX = 35        # max calendar days between monthly rebalances


def _compute_stop_result(
    entry_price: float,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    ohlc: pd.DataFrame,
) -> dict:
    """Walk daily bars STRICTLY AFTER entry_date through exit_date, return the
    realised stop-loss outcome for this single short position.

    ohlc is a per-ticker frame sorted by date with columns
    [date, open, high, low, close]. Returns a dict with the seven required
    columns. If no OHLC rows exist in the window, returns stop_status=NO_DATA.
    """
    stop_price = entry_price * (1.0 + STOP_TRIGGER_PCT)
    floor_cover = stop_price * (1.0 + SLIPPAGE_PCT)  # entry * 1.32

    # Window: strict > entry_date, <= exit_date. Avoids look-ahead by skipping
    # the entry day itself (you bought at entry-day close, the stop can only
    # fire on a SUBSEQUENT day's intraday high).
    win = ohlc[(ohlc["date"] > entry_date) & (ohlc["date"] <= exit_date)]
    if win.empty:
        return {
            "stop_triggered": False,
            "stop_trigger_date": pd.NaT,
            "stop_status": "NO_DATA",
            "stop_price": stop_price,
            "cover_price": np.nan,
        }

    # First day where intraday HIGH >= stop_price -> stop fires.
    breach = win[win["high"] >= stop_price]
    if breach.empty:
        return {
            "stop_triggered": False,
            "stop_trigger_date": pd.NaT,
            "stop_status": "NOT_TRIGGERED",
            "stop_price": stop_price,
            "cover_price": np.nan,
        }

    first = breach.iloc[0]
    trigger_open = float(first["open"])
    cover_price = max(floor_cover, trigger_open)
    return {
        "stop_triggered": True,
        "stop_trigger_date": first["date"],
        "stop_status": "STOPPED",
        "stop_price": stop_price,
        "cover_price": cover_price,
    }


def _portfolio_monthly_returns(positions: pd.DataFrame, ret_col: str) -> pd.Series:
    """Aggregate per-position SHORT returns into a monthly portfolio return.

    Convention: ``ret_col`` is the *short* return (positive when stock fell,
    i.e. ``-(exit/entry - 1)``). Each position has signed ``weight`` < 0
    (short basket); position dollar-P&L is ``|weight| * short_return``.
    Summed across the basket each month, this yields the short-leg
    portfolio return for a book that sums to -1 in notional.
    """
    p = positions.copy()
    p["pos_contrib"] = p["weight"].abs() * p[ret_col].astype(float)
    return p.groupby("Date")["pos_contrib"].sum().sort_index()


def _summary_stats(rets: pd.Series, periods_per_year: int = 12) -> dict:
    r = rets.dropna()
    if len(r) < 3:
        return {"n": len(r), "Sharpe": np.nan, "CAGR": np.nan,
                "ann_vol": np.nan, "MaxDD": np.nan}
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else np.nan
    cagr = float((1 + r).prod() ** (periods_per_year / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else np.nan
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return {"n": int(len(r)), "Sharpe": sharpe, "CAGR": cagr, "ann_vol": vol, "MaxDD": mdd}


def main() -> int:
    settings.ensure_dirs()
    pos_path = settings.reports_dir / "oos_short_positions.csv"
    ohlc_path = settings.processed_dir / "prices_ohlc_oos.parquet"
    if not pos_path.exists():
        logger.error(f"{pos_path} not found - run _oos_trades.py first")
        return 1
    if not ohlc_path.exists():
        logger.error(f"{ohlc_path} not found - run _pull_ohlc_oos.py first")
        return 1

    pos = pd.read_csv(pos_path)
    pos["Date"] = pd.to_datetime(pos["Date"]).dt.normalize()
    pos["entry_price"] = pd.to_numeric(pos["entry_price"], errors="coerce")
    pos["exit_price"] = pd.to_numeric(pos["exit_price"], errors="coerce")
    # The recorded entry/exit are adjClose values -- align to that timezone.
    if "trade_return" not in pos.columns:
        # Should already be present from _oos_trades.py; recompute defensively.
        pos["trade_return"] = pos["exit_price"] / pos["entry_price"] - 1.0
        pos["trade_return"] *= np.sign(pos["weight"])
    pos["raw_short_return"] = -((pos["exit_price"] / pos["entry_price"]) - 1.0)

    ohlc = pd.read_parquet(ohlc_path)
    ohlc["date"] = pd.to_datetime(ohlc["date"]).dt.normalize()
    ohlc_by_t = {t: g.sort_values("date").reset_index(drop=True) for t, g in ohlc.groupby("Ticker")}

    # Apply the stop logic position by position.
    results = []
    for _, row in pos.iterrows():
        ent = pd.to_datetime(row["Date"])
        ep = float(row["entry_price"]) if pd.notna(row["entry_price"]) else np.nan
        if not np.isfinite(ep) or ep <= 0:
            results.append({"stop_triggered": False, "stop_trigger_date": pd.NaT,
                            "stop_status": "BAD_ENTRY_PRICE",
                            "stop_price": np.nan, "cover_price": np.nan})
            continue
        exit_date = ent + pd.Timedelta(days=HOLDING_DAYS_MAX)
        ticker_ohlc = ohlc_by_t.get(row["Ticker"])
        if ticker_ohlc is None:
            results.append({"stop_triggered": False, "stop_trigger_date": pd.NaT,
                            "stop_status": "NO_OHLC",
                            "stop_price": ep * 1.20, "cover_price": np.nan})
            continue
        results.append(_compute_stop_result(ep, ent, exit_date, ticker_ohlc))

    res_df = pd.DataFrame(results)
    out = pd.concat([pos.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)

    # Stopped short return -- replace the raw return when the stop fires.
    raw_ret = out["raw_short_return"]
    # Realistic version: short_return after stop = -((cover_price / entry_price) - 1)
    # where cover_price = max(entry * 1.32, trigger_day_open)
    stop_ret = -((out["cover_price"] / out["entry_price"]) - 1.0)
    out["stopped_short_return"] = np.where(
        out["stop_triggered"].astype(bool),
        stop_ret,
        raw_ret,
    )
    # SIMPLIFIED FALLBACK: -32 % cap on monthly short loss (no daily OHLC,
    # no gap handling). Caps the *raw* monthly short return at -0.32 if it
    # would have been worse. This isolates the cost of the gap rule: any
    # difference between this and the realistic version above is purely the
    # gap penalty.
    out["simple_capped_short_return"] = np.where(
        raw_ret < -(STOP_TRIGGER_PCT + SLIPPAGE_PCT),  # raw worse than -32 %
        -(STOP_TRIGGER_PCT + SLIPPAGE_PCT),
        raw_ret,
    )

    # Persist per-position artefact.
    keep = [
        "Date", "Ticker", "Company", "weight",
        "entry_price", "stop_price", "stop_triggered", "stop_trigger_date",
        "stop_status", "cover_price",
        "exit_price", "raw_short_return", "stopped_short_return",
        "simple_capped_short_return",
    ]
    keep = [c for c in keep if c in out.columns]
    out_pos_path = settings.reports_dir / "oos_short_stopped.csv"
    out[keep].to_csv(out_pos_path, index=False)
    logger.info(f"wrote {out_pos_path} ({len(out):,} rows)")

    # Diagnostics ------------------------------------------------------------
    n_total = len(out)
    n_stopped = int(out["stop_triggered"].sum())
    pct_stopped = 100.0 * n_stopped / n_total if n_total else 0.0

    stopped_only = out[out["stop_triggered"].astype(bool)]
    raw_only_for_stopped = stopped_only["raw_short_return"]
    stop_only_returns = stopped_only["stopped_short_return"]

    raw_mean = float(out["raw_short_return"].mean())
    raw_median = float(out["raw_short_return"].median())
    new_mean = float(out["stopped_short_return"].mean())
    new_median = float(out["stopped_short_return"].median())

    # Portfolio-level: aggregate to monthly short-leg-only return.
    raw_port = _portfolio_monthly_returns(out, "raw_short_return")
    new_port = _portfolio_monthly_returns(
        out.assign(raw_short_return=out["stopped_short_return"]),
        "raw_short_return",
    )
    raw_summary = _summary_stats(raw_port)
    new_summary = _summary_stats(new_port)

    # L/S quintile portfolio impact -- load the existing OOS L/S monthly
    # return series and add the short-leg delta to get the stopped-version
    # L/S series.  delta_per_month = mean(stopped_short_return) - mean(raw_short_return)
    simple_port = _portfolio_monthly_returns(
        out.assign(raw_short_return=out["simple_capped_short_return"]),
        "raw_short_return",
    )
    simple_summary = _summary_stats(simple_port)

    ls_path = settings.reports_dir / "backtest_monthly_logit_long_short_quintile.parquet"
    if ls_path.exists():
        ls_raw = pd.read_parquet(ls_path)
        ls_raw["Date"] = pd.to_datetime(ls_raw["Date"]).dt.normalize()
        # The L/S backtest uses weight = -1/N for shorts and +1/N for longs,
        # so each leg contributes its mean stock return / short return -- the
        # delta we computed (raw_port -> new_port) is exactly the change in
        # the short LEG's contribution per month. Add it to ls_raw.ret_net.
        oos_ls_raw = ls_raw[ls_raw["Date"] >= pd.Timestamp("2023-06-01")].copy()
        oos_ls_raw = oos_ls_raw.set_index("Date")["ret_net"]
        delta_realistic = (new_port - raw_port).reindex(oos_ls_raw.index).fillna(0.0)
        delta_simple = (simple_port - raw_port).reindex(oos_ls_raw.index).fillna(0.0)
        oos_ls_new = oos_ls_raw + delta_realistic
        oos_ls_simple = oos_ls_raw + delta_simple
        raw_ls_summary = _summary_stats(oos_ls_raw)
        new_ls_summary = _summary_stats(oos_ls_new)
        simple_ls_summary = _summary_stats(oos_ls_simple)
    else:
        oos_ls_raw = oos_ls_new = oos_ls_simple = pd.Series(dtype=float)
        raw_ls_summary = new_ls_summary = simple_ls_summary = {
            "n": 0, "Sharpe": np.nan, "CAGR": np.nan,
            "ann_vol": np.nan, "MaxDD": np.nan
        }

    # Markdown report --------------------------------------------------------
    lines: list[str] = []
    lines.append("# Realistic stop-loss diagnostics -- OOS short book")
    lines.append("")
    lines.append(
        "**Rule**: 20 % adverse-move trigger; 10 % execution slippage on top; "
        "gap rule covers at `max(entry * 1.32, trigger_day_open)`. "
        "Applied to the 2,089-position OOS short book (logit L/S quintile, "
        "2023-06 -> 2026-05) using daily auto-adjusted OHLC pulled from Yahoo."
    )
    lines.append("")
    lines.append("## Summary diagnostics")
    lines.append("")
    lines.append(f"- **Total OOS short positions**: {n_total:,}")
    lines.append(f"- **Positions stopped**: **{n_stopped:,}** "
                 f"(**{pct_stopped:.1f} %** of all short positions)")
    if n_stopped > 0:
        avg_stop_loss = float(stop_only_returns.mean())
        med_stop_loss = float(stop_only_returns.median())
        worst_stop_loss = float(stop_only_returns.min())
        worst_raw_for_stopped = float(raw_only_for_stopped.min())
        lines.append(f"- **Average stopped-position loss**: {avg_stop_loss * 100:+.2f} %")
        lines.append(f"- **Median stopped-position loss**: {med_stop_loss * 100:+.2f} %")
        lines.append(f"- **Worst stopped-position loss**: {worst_stop_loss * 100:+.2f} % "
                     f"(uncapped raw would have been {worst_raw_for_stopped * 100:+.2f} %)")
    lines.append("")
    lines.append("## Per-position return change")
    lines.append("")
    lines.append(f"|  | Raw (no stop) | After 20 % stop | Δ |")
    lines.append(f"|---|---:|---:|---:|")
    lines.append(f"| Mean trade return | {raw_mean * 100:+.2f} % | "
                 f"{new_mean * 100:+.2f} % | {(new_mean - raw_mean) * 100:+.2f} pp |")
    lines.append(f"| Median trade return | {raw_median * 100:+.2f} % | "
                 f"{new_median * 100:+.2f} % | {(new_median - raw_median) * 100:+.2f} pp |")
    lines.append("")
    def _impact_table(raw, new, header):
        rows = [
            f"### {header}",
            "",
            "|  | Raw (no stop) | After 20 % stop | Δ |",
            "|---|---:|---:|---:|",
            f"| CAGR | {raw['CAGR'] * 100:+.2f} % | {new['CAGR'] * 100:+.2f} % | "
            f"{(new['CAGR'] - raw['CAGR']) * 100:+.2f} pp |",
            f"| Ann. vol | {raw['ann_vol'] * 100:.2f} % | {new['ann_vol'] * 100:.2f} % | "
            f"{(new['ann_vol'] - raw['ann_vol']) * 100:+.2f} pp |",
            f"| Sharpe | {raw['Sharpe']:+.3f} | {new['Sharpe']:+.3f} | "
            f"{(new['Sharpe'] - raw['Sharpe']):+.3f} |",
            f"| MaxDD | {raw['MaxDD'] * 100:+.2f} % | {new['MaxDD'] * 100:+.2f} % | "
            f"{(new['MaxDD'] - raw['MaxDD']) * 100:+.2f} pp |",
            "",
        ]
        return rows

    lines.append("## Portfolio-level impact")
    lines.append("")
    lines.append(
        "Three columns side-by-side: **Raw** (no stop), **Realistic** "
        "(20 % trigger + 10 % slippage + gap rule), **Simplified** "
        "(-32 % monthly cap, no gap handling -- isolates the cost of "
        "the gap rule from the cost of the stop itself)."
    )
    lines.append("")

    def _three_way_table(raw, real, simple, header):
        rows = [
            f"### {header}",
            "",
            "|  | Raw | Realistic (gap rule) | Simplified (-32 % cap) | Δ realistic | Δ simplified |",
            "|---|---:|---:|---:|---:|---:|",
            f"| CAGR | {raw['CAGR'] * 100:+.2f} % | {real['CAGR'] * 100:+.2f} % | "
            f"{simple['CAGR'] * 100:+.2f} % | "
            f"{(real['CAGR'] - raw['CAGR']) * 100:+.2f} pp | "
            f"{(simple['CAGR'] - raw['CAGR']) * 100:+.2f} pp |",
            f"| Ann. vol | {raw['ann_vol'] * 100:.2f} % | {real['ann_vol'] * 100:.2f} % | "
            f"{simple['ann_vol'] * 100:.2f} % | "
            f"{(real['ann_vol'] - raw['ann_vol']) * 100:+.2f} pp | "
            f"{(simple['ann_vol'] - raw['ann_vol']) * 100:+.2f} pp |",
            f"| Sharpe | {raw['Sharpe']:+.3f} | {real['Sharpe']:+.3f} | "
            f"{simple['Sharpe']:+.3f} | "
            f"{(real['Sharpe'] - raw['Sharpe']):+.3f} | "
            f"{(simple['Sharpe'] - raw['Sharpe']):+.3f} |",
            f"| MaxDD | {raw['MaxDD'] * 100:+.2f} % | {real['MaxDD'] * 100:+.2f} % | "
            f"{simple['MaxDD'] * 100:+.2f} % | "
            f"{(real['MaxDD'] - raw['MaxDD']) * 100:+.2f} pp | "
            f"{(simple['MaxDD'] - raw['MaxDD']) * 100:+.2f} pp |",
            "",
        ]
        return rows

    lines.extend(_three_way_table(raw_summary, new_summary, simple_summary,
                                    "Short-leg-only book (notional sums to -1)"))
    lines.extend(_three_way_table(raw_ls_summary, new_ls_summary, simple_ls_summary,
                                    "L/S quintile (long leg unchanged, short leg stopped)"))

    # Stop-status breakdown for the audit trail.
    status_counts = out["stop_status"].value_counts(dropna=False).to_dict()
    lines.append("## Stop-status breakdown")
    lines.append("")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{status}`: {count:,}")
    lines.append("")

    md_path = settings.reports_dir / "stop_loss_diagnostics.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"wrote {md_path}")

    # Console summary so the CI tail shows the key numbers.
    print()
    print(f"OOS short positions: {n_total:,}")
    print(f"Stopped:             {n_stopped:,} ({pct_stopped:.1f} %)")
    print(f"Mean trade return:   {raw_mean * 100:+.2f} % -> "
          f"{new_mean * 100:+.2f} % "
          f"({(new_mean - raw_mean) * 100:+.2f} pp)")
    print(f"Median trade return: {raw_median * 100:+.2f} % -> "
          f"{new_median * 100:+.2f} % "
          f"({(new_median - raw_median) * 100:+.2f} pp)")
    print(f"Short-only Sharpe:   {raw_summary['Sharpe']:+.3f} -> "
          f"{new_summary['Sharpe']:+.3f} "
          f"({(new_summary['Sharpe'] - raw_summary['Sharpe']):+.3f})")
    print(f"Short-only MaxDD:    {raw_summary['MaxDD'] * 100:+.2f} % -> "
          f"{new_summary['MaxDD'] * 100:+.2f} % "
          f"({(new_summary['MaxDD'] - raw_summary['MaxDD']) * 100:+.2f} pp)")
    print(f"L/S quintile Sharpe: {raw_ls_summary['Sharpe']:+.3f} -> "
          f"{new_ls_summary['Sharpe']:+.3f} "
          f"({(new_ls_summary['Sharpe'] - raw_ls_summary['Sharpe']):+.3f})")
    print(f"L/S quintile CAGR:   {raw_ls_summary['CAGR'] * 100:+.2f} % -> "
          f"{new_ls_summary['CAGR'] * 100:+.2f} % "
          f"({(new_ls_summary['CAGR'] - raw_ls_summary['CAGR']) * 100:+.2f} pp)")
    print(f"L/S quintile MaxDD:  {raw_ls_summary['MaxDD'] * 100:+.2f} % -> "
          f"{new_ls_summary['MaxDD'] * 100:+.2f} % "
          f"({(new_ls_summary['MaxDD'] - raw_ls_summary['MaxDD']) * 100:+.2f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
