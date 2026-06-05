"""Sensitivity sweep over stop-loss trigger level.

For each candidate trigger in {0.20, 0.25, 0.30, 0.35, 0.40, 0.50}, re-run
the realistic stop-loss logic (gap rule + 10 % slippage) across every
(model, strategy) backtest and report the resulting Sharpe / CAGR / MaxDD
/ stop-rate.

Goal: find the trigger level that avoids being tripped by normal monthly
volatility (which is destroying the strategy at 20 %) while still catching
the genuine multi-bagger squeezes.

Output: reports/stop_sensitivity.csv + reports/stop_sensitivity.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.backtest import CostConfig, _summarise_returns
from short_king.portfolio.construct import decile_short, long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

SLIPPAGE_PCT = 0.10  # 10 % execution penalty (kept fixed across the sweep)
HOLDING_DAYS_MAX = 35
PERIODS_PER_YEAR = 12
MODELS = ("naive", "ew", "logit")
STRATEGIES = ("quintile_short", "long_short_quintile")
TRIGGERS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def _compute_position_return(
    ticker: str,
    entry_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    ohlc_by_t: dict[str, pd.DataFrame],
    is_short: bool,
    trigger_pct: float,
) -> tuple[float, bool]:
    """Returns (realised position return for the leg, stop_triggered).
    For longs the stop logic doesn't apply.
    """
    if not is_short:
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
            return 0.0, False
        return float(exit_price / entry_price - 1.0), False

    if not np.isfinite(entry_price) or entry_price <= 0:
        return 0.0, False
    stop_price = entry_price * (1.0 + trigger_pct)
    floor_cover = stop_price * (1.0 + SLIPPAGE_PCT)

    ohlc = ohlc_by_t.get(ticker)
    exit_date = entry_date + pd.Timedelta(days=HOLDING_DAYS_MAX)
    if ohlc is None or ohlc.empty:
        if not np.isfinite(exit_price):
            return 0.0, False
        return float(-((exit_price / entry_price) - 1.0)), False

    win = ohlc[(ohlc["date"] > entry_date) & (ohlc["date"] <= exit_date)]
    if win.empty:
        if not np.isfinite(exit_price):
            return 0.0, False
        return float(-((exit_price / entry_price) - 1.0)), False

    breach = win[win["high"] >= stop_price]
    if breach.empty:
        if not np.isfinite(exit_price):
            return 0.0, False
        return float(-((exit_price / entry_price) - 1.0)), False

    first = breach.iloc[0]
    cover_price = max(floor_cover, float(first["open"]))
    return float(-((cover_price / entry_price) - 1.0)), True


def _build_monthly_returns(
    weights: pd.DataFrame,
    px_wide: pd.DataFrame,
    ohlc_by_t: dict[str, pd.DataFrame],
    cost: CostConfig,
    trigger_pct: float,
) -> tuple[pd.DataFrame, int, int]:
    rebalance_dates = pd.DatetimeIndex(sorted(weights["Date"].unique()))
    bps_round = cost.bps_round_trip / 10_000.0
    slip_bps = cost.slippage_bps / 10_000.0
    annual_borrow = cost.annual_borrow_pct / 100.0
    period_borrow = annual_borrow / cost.periods_per_year

    rows: list[dict] = []
    total_stops = 0
    total_shorts = 0
    prev_w: dict[str, float] = {}

    for i, ent_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        wb = weights[weights["Date"] == ent_date]
        gross = 0.0
        n_stops = 0
        new_w: dict[str, float] = {}
        n_shorts = 0

        for _, row in wb.iterrows():
            t = row["Ticker"]; w = float(row["weight"])
            if abs(w) < 1e-12: continue
            ep = float(px_wide.at[ent_date, t]) if t in px_wide.columns and ent_date in px_wide.index else np.nan
            xp = float(px_wide.at[next_date, t]) if t in px_wide.columns and next_date in px_wide.index else np.nan
            ret, stopped = _compute_position_return(t, ent_date, ep, xp, ohlc_by_t, w < 0, trigger_pct)
            new_w[t] = w
            if w >= 0:
                gross += w * ret
            else:
                gross += abs(w) * ret
                n_shorts += 1
                if stopped:
                    n_stops += 1

        if not new_w:
            continue
        delta = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in set(new_w) | set(prev_w))
        turnover = delta / 2.0
        commission = delta * bps_round
        slippage = delta * slip_bps
        short_notional = sum(abs(w) for w in new_w.values() if w < 0)
        borrow = short_notional * period_borrow
        cost_total = commission + slippage + borrow
        rows.append({
            "Date": ent_date, "ret_gross": gross,
            "ret_net": gross - cost_total, "turnover": turnover,
            "n_stops": n_stops,
        })
        total_stops += n_stops
        total_shorts += n_shorts
        prev_w = new_w

    return pd.DataFrame(rows), total_stops, total_shorts


def main() -> int:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")
    feat = read_parquet(settings.processed_dir / "features_monthly.parquet")
    ohlc = read_parquet(settings.processed_dir / "prices_ohlc_full.parquet")

    prices_panel = clean[["Date", "Ticker", "adjClose", "investable"]].copy()
    if "adv_aud" in feat.columns:
        f = feat[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")
    ohlc["date"] = pd.to_datetime(ohlc["date"]).dt.normalize()
    ohlc_by_t = {t: g.sort_values("date").reset_index(drop=True) for t, g in ohlc.groupby("Ticker")}
    px_wide = (prices_panel[["Date", "Ticker", "adjClose"]]
                .drop_duplicates(["Date", "Ticker"])
                .pivot(index="Date", columns="Ticker", values="adjClose").sort_index())

    cost = CostConfig(periods_per_year=PERIODS_PER_YEAR)
    oof_dates = oof[["Date", "period"]].drop_duplicates("Date")
    oof_dates["Date"] = pd.to_datetime(oof_dates["Date"]).dt.normalize()

    rows: list[dict] = []
    for model in MODELS:
        model_oof = oof[oof["model"] == model].copy()
        joined = model_oof.merge(
            prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
            on=["Date", "Ticker"], how="left",
        )
        for strategy in STRATEGIES:
            weights = (decile_short(joined, n_deciles=5) if strategy == "quintile_short"
                       else long_short_decile(joined, n_deciles=5))
            if weights.empty: continue

            for tp in TRIGGERS:
                rdf, stops, shorts = _build_monthly_returns(weights, px_wide, ohlc_by_t, cost, tp)
                if rdf.empty: continue
                rdf["Date"] = pd.to_datetime(rdf["Date"]).dt.normalize()
                merged = rdf.merge(model_oof[["Date", "period"]].drop_duplicates("Date"),
                                    on="Date", how="left")
                for period in ("ALL", "IS", "OOS"):
                    sub = merged if period == "ALL" else merged[merged["period"] == period]
                    if len(sub) < 6: continue
                    s = _summarise_returns(
                        net=sub.set_index("Date")["ret_net"],
                        gross=sub.set_index("Date")["ret_gross"],
                        turnover=sub.set_index("Date")["turnover"],
                        periods_per_year=PERIODS_PER_YEAR,
                    )
                    rows.append({
                        "model": model, "strategy": strategy,
                        "trigger_pct": tp, "period": period,
                        "n_months": int(s.get("n_weeks", 0)),
                        "stop_rate_%": round(100 * stops / max(1, shorts), 1),
                        "Sharpe": round(float(s.get("Sharpe", np.nan)), 3),
                        "CAGR": round(float(s.get("CAGR", np.nan)) * 100, 2),
                        "MaxDD": round(float(s.get("max_drawdown", np.nan)) * 100, 2),
                        "hit_rate_%": round(float(s.get("hit_rate", np.nan)) * 100, 1),
                    })
                logger.info(f"{model}/{strategy} trigger={tp:.0%}: "
                            f"{stops:,} stops / {shorts:,} shorts = {100*stops/max(1,shorts):.1f}%")

    df = pd.DataFrame(rows)
    csv_path = settings.reports_dir / "stop_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Compact markdown view: L/S quintile OOS Sharpe across triggers.
    def _md_pivot(period: str, strategy: str, metric: str, fmt: str = "{:+.3f}") -> str:
        sub = df[(df.period == period) & (df.strategy == strategy)]
        pv = sub.pivot_table(index="model", columns="trigger_pct", values=metric).reindex(MODELS)
        cols = sorted(pv.columns)
        hdr = "| Model | " + " | ".join(f"{int(c*100)}%" for c in cols) + " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        body = []
        for m in MODELS:
            cells = [fmt.format(pv.loc[m, c]) if pd.notna(pv.loc[m, c]) else "" for c in cols]
            body.append(f"| {m} | " + " | ".join(cells) + " |")
        return "\n".join([hdr, sep] + body)

    md_lines: list[str] = ["# Stop-loss trigger sensitivity sweep\n"]
    md_lines.append(
        "Same realistic stop logic (10 % slippage + gap rule on daily OHLC) "
        "across triggers from 20 % through 50 %. 'No stop' = uncapped.\n"
    )
    for strategy in STRATEGIES:
        md_lines.append(f"\n## {strategy}\n")
        for period in ("OOS", "ALL", "IS"):
            md_lines.append(f"\n### {period} Sharpe\n")
            md_lines.append(_md_pivot(period, strategy, "Sharpe"))
            md_lines.append(f"\n### {period} CAGR (%)\n")
            md_lines.append(_md_pivot(period, strategy, "CAGR", "{:+.2f}"))
            md_lines.append(f"\n### {period} MaxDD (%)\n")
            md_lines.append(_md_pivot(period, strategy, "MaxDD", "{:+.2f}"))
    md_lines.append("\n## Stop-fire rate (% of all short positions, full panel)\n")
    rate = df[df.period == "ALL"].pivot_table(
        index="model", columns="trigger_pct", values="stop_rate_%"
    ).reindex(MODELS)
    cols = sorted(rate.columns)
    md_lines.append("| Model | " + " | ".join(f"{int(c*100)}%" for c in cols) + " |")
    md_lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for m in MODELS:
        cells = [f"{rate.loc[m, c]:.1f}%" if pd.notna(rate.loc[m, c]) else "" for c in cols]
        md_lines.append(f"| {m} | " + " | ".join(cells) + " |")

    md_path = settings.reports_dir / "stop_sensitivity.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
