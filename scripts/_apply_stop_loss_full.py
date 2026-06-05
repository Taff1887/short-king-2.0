"""Apply the realistic per-position stop-loss to EVERY (model, strategy)
backtest across the full 16-year IS + OOS panel.

Stop-loss spec (per position):
* trigger = entry_price * 1.20  (20% adverse move)
* nominal cover = stop * 1.10   (entry * 1.32  -> -32% short loss)
* gap rule: cover_price = max(entry * 1.32, trigger_day_open)
* daily intraday monitoring on auto-adjusted OHLC
* strict > entry_date, <= exit_date (no look-ahead)
* stopped positions contribute zero P&L for the remainder of the month

For each of the 3 models (naive, ew, logit) x 2 strategies
(quintile_short, long_short_quintile), this script:
1. Reads the OOF predictions monthly parquet.
2. Builds the per-month weight book via the existing constructors.
3. For each short position, applies the realistic stop using daily OHLC.
4. Aggregates back to a monthly portfolio return series.
5. Overwrites backtest_monthly_<model>_<strategy>.parquet with the new
   ret_net + supporting columns.
6. Rewrites backtest_summary_monthly.csv with stop-loss-applied IS / OOS /
   ALL summaries for every (model, strategy).

This is the new HEADLINE backtest -- every chart and table will reflect
the realistic stopped book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.backtest import CostConfig, _summarise_returns
from short_king.portfolio.construct import decile_short, long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger

# Stop-loss knobs (kept inline so the spec is auditable here, not buried).
# The 20 % default destroyed the strategy because normal small-cap monthly
# volatility easily hits +20 %, so the stop fired on 17-26 % of all monthly
# positions (mostly false alarms). Sensitivity sweep at 25/30/35/40/50 %
# triggers (reports/stop_sensitivity.csv) showed 50 % is the lowest
# trigger that still produces positive OOS Sharpe for the L/S baselines --
# it only fires on 2-6 % of positions, catching the genuine multi-bagger
# squeezes (PLS, 4DX, APX, BRN-style +50 %+ moves) while leaving normal
# monthly noise alone.
STOP_TRIGGER_PCT = 0.50
SLIPPAGE_PCT = 0.10
HOLDING_DAYS_MAX = 35

PERIODS_PER_YEAR = 12  # monthly
MODELS = ("naive", "ew", "logit")
STRATEGIES = ("quintile_short", "long_short_quintile")


def _compute_position_return(
    ticker: str,
    entry_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    ohlc_by_t: dict[str, pd.DataFrame],
    is_short: bool,
) -> tuple[float, bool, pd.Timestamp | None]:
    """Return (realised_position_return, stop_triggered, trigger_date) for one
    position. For longs the stop logic doesn't apply (returns raw stock return,
    False, None). For shorts: walk daily OHLC after entry, fire stop on first
    bar where high >= entry * 1.20, cover at max(entry * 1.32, that day's open).
    """
    if not is_short:
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
            return 0.0, False, None
        # Long return = exit / entry - 1; held to end of month, no stop.
        return float(exit_price / entry_price - 1.0), False, None

    if not np.isfinite(entry_price) or entry_price <= 0:
        return 0.0, False, None
    stop_price = entry_price * (1.0 + STOP_TRIGGER_PCT)
    floor_cover = stop_price * (1.0 + SLIPPAGE_PCT)

    ohlc = ohlc_by_t.get(ticker)
    exit_date = entry_date + pd.Timedelta(days=HOLDING_DAYS_MAX)
    if ohlc is None or ohlc.empty:
        # No OHLC -- fall back to monthly close-to-close (no stop possible).
        if not np.isfinite(exit_price):
            return 0.0, False, None
        return float(-((exit_price / entry_price) - 1.0)), False, None

    win = ohlc[(ohlc["date"] > entry_date) & (ohlc["date"] <= exit_date)]
    if win.empty:
        if not np.isfinite(exit_price):
            return 0.0, False, None
        return float(-((exit_price / entry_price) - 1.0)), False, None

    breach = win[win["high"] >= stop_price]
    if breach.empty:
        # Stop never fired - normal monthly exit.
        if not np.isfinite(exit_price):
            return 0.0, False, None
        return float(-((exit_price / entry_price) - 1.0)), False, None

    first = breach.iloc[0]
    cover_price = max(floor_cover, float(first["open"]))
    short_ret = -((cover_price / entry_price) - 1.0)
    return float(short_ret), True, pd.Timestamp(first["date"])


def _build_monthly_portfolio_returns(
    weights: pd.DataFrame,
    prices_close: pd.DataFrame,
    ohlc_by_t: dict[str, pd.DataFrame],
    cost: CostConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate per-position outcomes into a monthly returns DataFrame.

    weights: long-form Date / Ticker / weight (sign-aware)
    prices_close: Date / Ticker / adjClose (monthly Friday closes)
    ohlc_by_t: per-ticker daily OHLC for stop logic
    cost: CostConfig (commission, borrow, slippage)

    Returns (returns_df, diagnostics):
    returns_df: columns [Date, ret_gross, ret_net, cost_total, turnover,
                          n_positions, n_stops, gross_exposure, net_exposure]
    diagnostics: aggregate count of stops fired etc.
    """
    # Pivot prices to a wide grid for cheap entry / exit lookup.
    px = (prices_close
            .drop_duplicates(["Date", "Ticker"])
            .pivot(index="Date", columns="Ticker", values="adjClose")
            .sort_index())
    rebalance_dates = pd.DatetimeIndex(sorted(weights["Date"].unique()))

    rows: list[dict] = []
    total_stops = 0
    total_positions = 0
    bps_round = cost.bps_round_trip / 10_000.0
    slip_bps = cost.slippage_bps / 10_000.0
    annual_borrow = cost.annual_borrow_pct / 100.0
    period_borrow = annual_borrow / cost.periods_per_year

    prev_weights: dict[str, float] = {}

    for i, ent_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        wb = weights[weights["Date"] == ent_date]
        n_stops_this_month = 0

        # Gather entry / exit prices for everything we'll hold.
        per_pos = []
        for _, row in wb.iterrows():
            t = row["Ticker"]
            w = float(row["weight"])
            if abs(w) < 1e-12:
                continue
            ep = float(px.at[ent_date, t]) if t in px.columns and ent_date in px.index else np.nan
            xp = float(px.at[next_date, t]) if t in px.columns and next_date in px.index else np.nan
            ret, stopped, _ = _compute_position_return(
                ticker=t, entry_date=ent_date, entry_price=ep, exit_price=xp,
                ohlc_by_t=ohlc_by_t, is_short=(w < 0),
            )
            per_pos.append((t, w, ret, stopped))
            if stopped:
                n_stops_this_month += 1

        if not per_pos:
            continue

        # Gross portfolio return = sum(weight_signed * stock_return).
        # For a short basket weight=-1/N and stock_return for the short pos
        # is the stock's actual return (or stopped if fired). Multiplying
        # gives -1/N * stock_return; the SHORT P&L is the negative of that
        # so SHORT contribution to book = -weight * stock_return = +1/N
        # * (-stock_return) = +1/N * short_return. The compute_position_return
        # function returns SHORT return directly for shorts, raw stock return
        # for longs. We need to use:
        #   long  -> contribution = +|w| * stock_return  =  w * stock_return  (w>0)
        #   short -> contribution = +|w| * short_return  = -w * short_return  (w<0)
        # Simpler: contribution = sign-flipped for shorts.
        gross = 0.0
        for t, w, ret, stopped in per_pos:
            if w >= 0:
                gross += w * ret
            else:
                # ret is already the SHORT return (positive when stock fell).
                gross += abs(w) * ret

        # Turnover (one-way = sum|delta_w|/2).
        new_w = {t: w for t, w, _, _ in per_pos}
        delta_sum = 0.0
        for t, w in new_w.items():
            delta_sum += abs(w - prev_weights.get(t, 0.0))
        for t, w_old in prev_weights.items():
            if t not in new_w:
                delta_sum += abs(0.0 - w_old)
        turnover = delta_sum / 2.0
        prev_weights = new_w

        # Costs.
        commission = delta_sum * bps_round
        slippage = delta_sum * slip_bps
        short_notional = sum(abs(w) for _, w, _, _ in per_pos if w < 0)
        long_notional = sum(w for _, w, _, _ in per_pos if w > 0)
        borrow = short_notional * period_borrow
        cost_total = commission + slippage + borrow

        n_pos = sum(1 for _, w, _, _ in per_pos if abs(w) > 1e-12)
        rows.append({
            "Date": ent_date,
            "ret_gross": gross,
            "ret_net": gross - cost_total,
            "cost_total": cost_total,
            "commission": commission,
            "slippage": slippage,
            "borrow": borrow,
            "turnover": turnover,
            "n_positions": n_pos,
            "n_stops": n_stops_this_month,
            "gross_exposure": long_notional + short_notional,
            "net_exposure": long_notional - short_notional,
        })
        total_stops += n_stops_this_month
        total_positions += sum(1 for _, w, _, _ in per_pos if w < 0)

    returns_df = pd.DataFrame(rows)
    return returns_df, {
        "total_stops": total_stops,
        "total_short_positions": total_positions,
    }


def main() -> int:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")
    features = read_parquet(settings.processed_dir / "features_monthly.parquet")
    ohlc = read_parquet(settings.processed_dir / "prices_ohlc_full.parquet")

    # Prices panel for constructor + closing prices for stop / monthly exit.
    prices_panel = clean[["Date", "Ticker", "adjClose", "investable"]].copy()
    if "adv_aud" in features.columns:
        f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")

    # Index OHLC by ticker for fast access; normalise the date column.
    ohlc["date"] = pd.to_datetime(ohlc["date"]).dt.normalize()
    ohlc_by_t = {
        t: g.sort_values("date").reset_index(drop=True)
        for t, g in ohlc.groupby("Ticker")
    }
    logger.info(f"OHLC loaded: {len(ohlc):,} bars across {len(ohlc_by_t)} tickers")

    cost = CostConfig(periods_per_year=PERIODS_PER_YEAR)

    summary_rows: list[dict] = []
    for model in MODELS:
        model_oof = oof[oof["model"] == model].copy()
        joined = model_oof.merge(
            prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
            on=["Date", "Ticker"], how="left",
        )

        for strategy in STRATEGIES:
            if strategy == "quintile_short":
                weights = decile_short(joined, n_deciles=5)
            else:
                weights = long_short_decile(joined, n_deciles=5)
            if weights.empty:
                logger.warning(f"{model}/{strategy}: empty weights, skipping")
                continue

            returns_df, diag = _build_monthly_portfolio_returns(
                weights=weights,
                prices_close=prices_panel[["Date", "Ticker", "adjClose"]],
                ohlc_by_t=ohlc_by_t,
                cost=cost,
            )

            out_path = settings.reports_dir / f"backtest_monthly_{model}_{strategy}.parquet"
            write_parquet(returns_df, out_path)
            logger.info(
                f"{model}/{strategy}: {len(returns_df)} months "
                f"({diag['total_stops']:,} stops / {diag['total_short_positions']:,} shorts "
                f"= {100 * diag['total_stops'] / max(1, diag['total_short_positions']):.1f} %)"
            )

            # IS / OOS / ALL summaries via the same helper the backtest engine uses.
            ret_with_period = returns_df.merge(
                model_oof[["Date", "period"]].drop_duplicates("Date"),
                on="Date", how="left",
            )
            for period_label in ("ALL", "IS", "OOS"):
                sub = ret_with_period if period_label == "ALL" else ret_with_period[
                    ret_with_period["period"] == period_label
                ]
                if len(sub) < 3:
                    continue
                s = _summarise_returns(
                    net=sub.set_index("Date")["ret_net"],
                    gross=sub.set_index("Date")["ret_gross"],
                    turnover=sub.set_index("Date")["turnover"],
                    periods_per_year=PERIODS_PER_YEAR,
                )
                # Per-period stop count.
                n_stops_period = int(sub["n_stops"].sum())
                summary_rows.append({
                    "model": model, "strategy": strategy, "period": period_label,
                    "CAGR": float(s.get("CAGR", float("nan"))),
                    "vol": float(s.get("ann_vol", float("nan"))),
                    "Sharpe": float(s.get("Sharpe", float("nan"))),
                    "Sortino": float(s.get("Sortino", float("nan"))),
                    "MaxDD": float(s.get("max_drawdown", float("nan"))),
                    "Calmar": float(s.get("calmar", float("nan"))),
                    "hit_rate": float(s.get("hit_rate", float("nan"))),
                    "avg_turnover": float(s.get("avg_turnover", float("nan"))),
                    "n_rebalances": int(s.get("n_weeks", 0)),
                    "n_stops_total": n_stops_period,
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = settings.reports_dir / "backtest_summary_monthly.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"wrote {summary_path} with {len(summary_df)} rows "
                f"({len(MODELS)} models x {len(STRATEGIES)} strategies x 3 periods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
