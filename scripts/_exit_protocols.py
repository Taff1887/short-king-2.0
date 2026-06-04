"""Compare exit protocols on the OOS short book.

Reconstructs every OOS short position (logit model, L/S quintile,
2023-06 → 2026-05) and replays the holding period DAILY rather than only
checking at month-end. Each protocol can exit early on any trading day
during the hold; whichever triggers first wins.

Protocols:
    A. monthly_stop_15: existing baseline. End-of-month cumulative cap at
       -15 % per position (-16 % with 100 bps fill slippage).
    B. daily_breakout_10: on any single trading day during the hold, if the
       stock rises >10 % vs the prior close, exit at that day's close.
       Captures squeezes early.
    C. cumulative_trail_10: trailing cumulative-from-entry. If the stock
       has rallied >10 % from the entry close at any day during the hold,
       exit at that day's close.
    D. daily_breakout_10 + cumulative_trail_10: both armed; first-to-fire
       wins.

For each protocol, aggregate per-position contributions to a monthly book
return and compute Sharpe / MaxDD / win-rate / stop-fire rate on the OOS
holdout. Output ``reports/exit_protocols_oos.csv`` and a markdown summary
to ``reports/exit_protocols.md``.

Costs match the headline backtest: 25 bps round-trip per side commission +
5 bps slippage on weight changes + 1.5 % p.a. borrow + 100 bps fill
slippage on stop exits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from short_king.portfolio.backtest import CostConfig
from short_king.portfolio.construct import long_short_decile
from short_king.portfolio.metrics import cagr, max_drawdown, sharpe, sortino
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


@dataclass(frozen=True)
class ExitProtocol:
    name: str
    daily_breakout_pct: float | None    # exit if single-day return > this
    cumulative_pct: float | None        # exit if cum return from entry > this
    label: str


PROTOCOLS: tuple[ExitProtocol, ...] = (
    ExitProtocol("A_monthly_only", None,  0.15, "Monthly EOM stop at +15 % cumulative (current default)"),
    ExitProtocol("B_daily_10",     0.10,  None, "Daily intraday: exit if any single day rises >10 %"),
    ExitProtocol("C_trail_10",     None,  0.10, "Cumulative trailing: exit if stock rises >10 % from entry"),
    ExitProtocol("D_both_10",      0.10,  0.10, "Both: daily 10 % AND cumulative 10 % (first-to-fire)"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="logit",
                   choices=("naive", "ew", "logit", "gbm_cls", "gbm_rank"))
    p.add_argument("--stop-slippage-pct", type=float, default=0.01,
                   help="Slippage on stop fill (default 100 bps; same as headline).")
    return p.parse_args()


def _simulate_position(
    weight: float,
    entry_date: pd.Timestamp,
    exit_target_date: pd.Timestamp,
    daily_px: pd.DataFrame,
    proto: ExitProtocol,
    slip_pct: float,
) -> dict:
    """Day-by-day replay of one short position.

    Returns a dict with the realised stock return at exit, the realised
    book contribution (signed weight x stock return), the exit reason and
    the exit date.
    """
    series = daily_px[
        (daily_px["date"] >= entry_date) & (daily_px["date"] <= exit_target_date)
    ][["date", "adjClose"]].sort_values("date").reset_index(drop=True)
    if len(series) < 2:
        return dict(
            stock_return=np.nan,
            position_pnl=0.0,
            exit_reason="no_price_data",
            exit_date=entry_date,
            days_held=0,
            slip_extra=0.0,
        )
    entry_px = float(series.iloc[0]["adjClose"])
    if entry_px <= 0 or not np.isfinite(entry_px):
        return dict(
            stock_return=np.nan,
            position_pnl=0.0,
            exit_reason="bad_entry_price",
            exit_date=entry_date,
            days_held=0,
            slip_extra=0.0,
        )

    daily_thresh = proto.daily_breakout_pct
    cum_thresh = proto.cumulative_pct
    is_short = weight < 0
    sign = -1.0 if is_short else 1.0  # for a short, "bad" = stock up

    for i in range(1, len(series)):
        px_today = float(series.iloc[i]["adjClose"])
        px_yest = float(series.iloc[i - 1]["adjClose"])
        if not np.isfinite(px_today) or not np.isfinite(px_yest) or px_yest <= 0:
            continue
        daily_ret = px_today / px_yest - 1.0
        cum_ret = px_today / entry_px - 1.0
        # For shorts: "against us" = positive moves. For longs: negative.
        adverse_daily = daily_ret * sign
        adverse_cum = cum_ret * sign

        if daily_thresh is not None and adverse_daily > daily_thresh:
            # Stop fires today. Realised stock return is cum_ret with slippage.
            slip_extra = slip_pct
            realised_stock = cum_ret + (slip_extra * sign)  # worsen by slip
            return dict(
                stock_return=realised_stock,
                position_pnl=weight * realised_stock,
                exit_reason="daily_stop",
                exit_date=series.iloc[i]["date"],
                days_held=i,
                slip_extra=slip_extra,
            )
        if cum_thresh is not None and adverse_cum > cum_thresh:
            slip_extra = slip_pct
            realised_stock = cum_ret + (slip_extra * sign)
            return dict(
                stock_return=realised_stock,
                position_pnl=weight * realised_stock,
                exit_reason="cum_stop",
                exit_date=series.iloc[i]["date"],
                days_held=i,
                slip_extra=slip_extra,
            )

    # No stop fired: held to the exit target (next rebalance close).
    last_px = float(series.iloc[-1]["adjClose"])
    cum_ret = last_px / entry_px - 1.0
    return dict(
        stock_return=cum_ret,
        position_pnl=weight * cum_ret,
        exit_reason="rebalance",
        exit_date=series.iloc[-1]["date"],
        days_held=len(series) - 1,
        slip_extra=0.0,
    )


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    features = read_parquet(settings.processed_dir / "features_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")
    prices = read_parquet(settings.processed_dir / "prices_long.parquet")
    cost = CostConfig()
    slip_pct = float(args.stop_slippage_pct)

    # OOS L/S quintile shorts only.
    sub = oof[(oof["model"] == args.model) & (oof["period"] == "OOS")].dropna(subset=["score"])
    if sub.empty:
        logger.error(f"no OOS scores for model={args.model}")
        return 1

    prices_panel_cols = [c for c in ("Date", "Ticker", "adjClose", "investable") if c in clean.columns]
    prices_panel = clean[prices_panel_cols].copy()
    if "adv_aud" in features.columns:
        f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")
    joined = sub.merge(
        prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
        on=["Date", "Ticker"], how="left",
    )
    weights = long_short_decile(joined, n_deciles=5)
    shorts = weights[weights["weight"] < 0].copy()
    shorts["Date"] = pd.to_datetime(shorts["Date"]).dt.normalize()
    shorts["Symbol"] = shorts["Ticker"].astype(str) + ".AX"

    # Determine each position's exit_target = next rebalance Date in the OOS window.
    oos_dates = sorted(shorts["Date"].unique())
    next_date_map = {d: oos_dates[i + 1] if i + 1 < len(oos_dates) else d
                     for i, d in enumerate(oos_dates)}
    shorts["exit_target"] = shorts["Date"].map(next_date_map)
    # Drop the last cohort (no forward data to simulate).
    shorts = shorts[shorts["Date"] != oos_dates[-1]].copy()

    logger.info(
        f"exit-protocol replay: {len(shorts):,} short positions over {len(oos_dates) - 1} "
        f"OOS rebalance pairs ({oos_dates[0].date()} -> {oos_dates[-1].date()})"
    )

    # Pre-index daily prices for fast slicing.
    px = prices[["symbol", "date", "adjClose"]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px_by_sym = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol")}

    all_results: dict[str, pd.DataFrame] = {}
    for proto in PROTOCOLS:
        records: list[dict] = []
        for _, row in shorts.iterrows():
            sym = row["Symbol"]
            sub_px = px_by_sym.get(sym, pd.DataFrame(columns=["symbol", "date", "adjClose"]))
            r = _simulate_position(
                weight=float(row["weight"]),
                entry_date=row["Date"],
                exit_target_date=row["exit_target"],
                daily_px=sub_px,
                proto=proto,
                slip_pct=slip_pct,
            )
            records.append({
                "Date": row["Date"],
                "Ticker": row["Ticker"],
                "weight": float(row["weight"]),
                "stock_return": r["stock_return"],
                "position_pnl": r["position_pnl"],
                "exit_reason": r["exit_reason"],
                "exit_date": r["exit_date"],
                "days_held": r["days_held"],
            })
        df = pd.DataFrame(records)
        all_results[proto.name] = df

    # Aggregate to monthly book returns for each protocol.
    summary_rows: list[dict] = []
    for proto in PROTOCOLS:
        df = all_results[proto.name]
        # Per-month book return from this protocol's short positions only.
        per_month = df.groupby("Date")["position_pnl"].sum()
        # Apply per-month commission + borrow (approximate; identical across
        # protocols for the rebalance trade itself - the stop-out adds extra).
        # Commission on the initial entry: |weight| * 2 * bps_round_trip.
        per_month_commission = (
            df.groupby("Date")["weight"].apply(lambda s: 2 * cost.bps_round_trip / 1e4 * s.abs().sum())
        )
        per_month_borrow = (
            df.groupby("Date")["weight"].apply(
                lambda s: s.abs().sum() * (cost.annual_borrow_pct / 100) / cost.periods_per_year
            )
        )
        # Extra exit-trade commission when a stop fires (any non-rebalance exit).
        df_stopped = df[~df["exit_reason"].isin(["rebalance", "no_price_data", "bad_entry_price"])]
        per_month_extra = (
            df_stopped.groupby("Date")["weight"].apply(
                lambda s: 2 * cost.bps_round_trip / 1e4 * s.abs().sum()
            )
        )
        per_month_net = (
            per_month
            - per_month_commission.reindex(per_month.index, fill_value=0)
            - per_month_borrow.reindex(per_month.index, fill_value=0)
            - per_month_extra.reindex(per_month.index, fill_value=0)
        )
        per_month_net = per_month_net.sort_index()

        n = len(per_month_net)
        s_cagr = cagr(per_month_net, periods_per_year=12)
        s_sharpe = sharpe(per_month_net, periods_per_year=12)
        s_sortino = sortino(per_month_net, periods_per_year=12)
        s_mdd = max_drawdown(per_month_net)
        win_pos = float((df["position_pnl"] > 0).mean())

        n_daily_stops = int((df["exit_reason"] == "daily_stop").sum())
        n_cum_stops = int((df["exit_reason"] == "cum_stop").sum())
        n_rebal = int((df["exit_reason"] == "rebalance").sum())
        n_total = len(df)
        avg_days = float(df["days_held"].mean())

        summary_rows.append({
            "protocol": proto.name,
            "label": proto.label,
            "n_positions": n_total,
            "avg_days_held": round(avg_days, 1),
            "daily_stops": n_daily_stops,
            "cum_stops": n_cum_stops,
            "held_to_rebal": n_rebal,
            "stop_fire_rate_%": round(100 * (n_daily_stops + n_cum_stops) / max(n_total, 1), 1),
            "win_rate_%": round(100 * win_pos, 1),
            "OOS_CAGR_%": round(100 * s_cagr, 2),
            "OOS_Sharpe": round(s_sharpe, 2),
            "OOS_Sortino": round(s_sortino, 2),
            "OOS_MaxDD_%": round(100 * s_mdd, 2),
            "n_months": n,
        })

    out = pd.DataFrame(summary_rows)
    csv_path = settings.reports_dir / "exit_protocols_oos.csv"
    out.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    md = "# Exit-protocol comparison — OOS short book\n\n"
    md += f"Replay of every OOS short position (model = {args.model}, "
    md += "L/S quintile, 36-month holdout) with day-by-day price monitoring. "
    md += "Each protocol can exit early during the monthly hold; "
    md += "whichever trigger fires first wins. Same costs as headline backtest.\n\n"
    md += "_For dollar-neutral L/S impact see the README — the long leg is "
    md += "identical across protocols; only the short-leg exit logic changes._\n\n"
    cols = ["protocol", "label", "OOS_Sharpe", "OOS_CAGR_%", "OOS_MaxDD_%",
            "win_rate_%", "stop_fire_rate_%", "daily_stops", "cum_stops",
            "held_to_rebal", "avg_days_held"]
    disp = out[cols].copy()
    try:
        md += disp.to_markdown(index=False)
    except (ImportError, ValueError):
        hdr = "| " + " | ".join(disp.columns) + " |\n"
        sep = "|" + "|".join(["---"] * len(disp.columns)) + "|\n"
        body = "\n".join("| " + " | ".join(str(v) for v in r.values) + " |" for _, r in disp.iterrows())
        md += hdr + sep + body
    md_path = settings.reports_dir / "exit_protocols.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
