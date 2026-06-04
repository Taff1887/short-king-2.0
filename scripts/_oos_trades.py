"""Per-trade OOS analysis: which names were shorted, P&L per name, why.

Reconstructs the L/S quintile SHORT positions from the out-of-sample holdout
(2023-06 to 2026-05). For each (Date, Ticker) in the short basket:

1. Compute the realised one-month forward return (uncapped — no stop loss).
2. Apply commission + slippage + borrow costs (matched to the headline
   ``CostConfig``).
3. Compute the *contribution to book* = signed_weight * realised return.

Then aggregate by Ticker over the entire OOS window:

* ``n_months_shorted`` - how many monthly rebalances this name was in the short basket
* ``total_contribution`` - cumulative book-level P&L from this name
* ``best_month`` / ``worst_month`` - extreme single-month outcomes
* ``avg_short_return`` - average per-position return when shorted
* ``hit_rate`` - share of months where the short was profitable
* signal context at first entry: SI %, market cap, momentum rank, valuation rank,
  quality rank, growth rank - the "why" behind the short

Outputs ``reports/oos_trades.csv`` (full ranked table) and
``reports/oos_trades.md`` (markdown extract with best + worst N).
"""

from __future__ import annotations

import argparse

import numpy as np  # noqa: F401  # kept for downstream readers
import pandas as pd

from short_king.portfolio.backtest import CostConfig
from short_king.portfolio.construct import long_short_decile
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


# Header columns we'll surface as "why short?" - one rank-feature per family,
# plus the raw ShortPct level. Higher rank = more shortable (for these specific
# columns - they're SHORT-aligned by construction).
WHY_COLS: tuple[tuple[str, str], ...] = (
    ("ShortPct", "Short %"),
    ("ShortPct_rk", "SI rank"),
    ("mom_12w_rk", "12w-mom rank"),
    ("vol_4w_rk", "vol rank"),
    ("log_mktcap_rk", "size rank"),
    ("pe_rk", "P/E rank"),
    ("fcf_yield_rk", "FCF-yld rank"),
    ("roe_rk", "ROE rank"),
    ("debt_equity_rk", "lev rank"),
    ("revenue_growth_yoy_rk", "growth rank"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="logit",
                   choices=("naive", "ew", "logit", "gbm_cls", "gbm_rank"),
                   help="Which model's OOF scores to backtest at trade level (default logit "
                        "- best OOS Sharpe among the trained models).")
    p.add_argument("--n-buckets", type=int, default=5,
                   help="Quintiles by default (5); 10 for deciles.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Show top-N winners and top-N losers in the markdown extract.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    features = read_parquet(settings.processed_dir / "features_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")
    prices = read_parquet(settings.processed_dir / "prices_long.parquet")

    cost = CostConfig()  # default = same as headline backtest

    # Restrict to the chosen model + OOS rows.
    sub = oof[(oof["model"] == args.model) & (oof["period"] == "OOS")].copy()
    sub = sub.dropna(subset=["score"])
    if sub.empty:
        logger.error(f"no OOS scores for model={args.model}")
        return 1
    logger.info(f"oos_trades: model={args.model} | "
                f"oos rows={len(sub):,} | dates={sub['Date'].nunique()}")

    # Merge investable + adv_aud to feed the constructor.
    prices_panel_cols = [c for c in ("Date", "Ticker", "adjClose", "investable") if c in clean.columns]
    prices_panel = clean[prices_panel_cols].copy()
    if "adv_aud" in features.columns:
        f = features[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        prices_panel = prices_panel.merge(f, on=["Date", "Ticker"], how="left")
    joined = sub.merge(
        prices_panel[["Date", "Ticker", "investable", "adv_aud"]],
        on=["Date", "Ticker"], how="left",
    )

    # Build the L/S quintile target weights for the OOS panel.
    weights = long_short_decile(joined, n_deciles=args.n_buckets)
    # SHORT leg only (the user asked specifically about shorted names).
    shorts = weights[weights["weight"] < 0].copy()
    logger.info(f"OOS short positions across all months: {len(shorts):,} "
                f"(unique tickers: {shorts['Ticker'].nunique()})")

    # One-month forward return per (Date, Ticker), reusing the same merge_asof
    # logic as the backtest engine.
    px = prices.rename(columns={"symbol": "Symbol", "date": "px_date"})[
        ["Symbol", "px_date", "adjClose"]
    ].dropna(subset=["adjClose"])
    px["px_date"] = pd.to_datetime(px["px_date"]).dt.normalize().astype("datetime64[ns]")
    px = px.sort_values(["Symbol", "px_date"])
    # Entry price = adjClose at the rebalance date.
    shorts["Symbol"] = shorts["Ticker"].astype(str) + ".AX"
    shorts["Date"] = pd.to_datetime(shorts["Date"]).dt.normalize().astype("datetime64[ns]")
    entry = pd.merge_asof(
        shorts.sort_values("Date").reset_index(drop=True),
        px.rename(columns={"px_date": "Date", "adjClose": "entry_price"}).sort_values("Date"),
        on="Date", by="Symbol", direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    # Exit price: ~30 days forward (next monthly rebalance).
    entry["exit_target"] = entry["Date"] + pd.Timedelta(days=35)
    px_exit = px.rename(columns={"px_date": "exit_target", "adjClose": "exit_price"})
    entry = pd.merge_asof(
        entry.sort_values("exit_target").reset_index(drop=True),
        px_exit.sort_values("exit_target"),
        on="exit_target", by="Symbol", direction="backward",
        tolerance=pd.Timedelta(days=14),
    )

    # Stock return = exit/entry - 1; position return for a short = -weight * stock_return.
    entry["stock_ret"] = entry["exit_price"] / entry["entry_price"] - 1.0
    entry = entry.dropna(subset=["stock_ret"])
    # Position contribution = signed_weight * stock_return (no stop applied).
    entry["pos_contrib"] = entry["weight"] * entry["stock_ret"]

    # Per-position cost: commission + slippage on |weight| (entry + exit) +
    # borrow ~ short_weight * 1.5%/12.
    bps_round = cost.bps_round_trip / 10_000.0
    slip_bps = cost.slippage_bps / 10_000.0
    borrow_m = (cost.annual_borrow_pct / 100.0) / cost.periods_per_year
    entry["cost"] = (
        entry["weight"].abs() * 2 * bps_round
        + entry["weight"].abs() * 2 * slip_bps
        + entry["weight"].abs() * borrow_m
    )
    entry["pos_pnl_net"] = entry["pos_contrib"] - entry["cost"]

    # Per-position return PER A$ NOTIONAL. For a SHORT (weight < 0) we want
    # trade_return positive when the stock falls: trade_return = stock_ret * sign(weight).
    # E.g. weight=-0.005, stock_ret=-0.10 -> trade_return = -0.10 * -1 = +0.10 (winning short).
    entry["trade_return"] = entry["stock_ret"] * np.sign(entry["weight"])

    # Attach the "why" features as-of the entry date.
    why_keys = [c for c, _ in WHY_COLS if c in features.columns]
    ctx = features[["Date", "Ticker", "Company", *why_keys]].drop_duplicates(["Date", "Ticker"])
    ctx["Date"] = pd.to_datetime(ctx["Date"]).dt.normalize().astype("datetime64[ns]")
    entry = entry.merge(ctx, on=["Date", "Ticker"], how="left")

    # Write the full per-position table.
    entry["mktCap_AUDm"] = (clean.set_index(["Date", "Ticker"])
                                 .reindex(pd.MultiIndex.from_arrays(
                                     [entry["Date"], entry["Ticker"]]))["mktCap"].to_numpy() / 1e6)
    out_cols = [
        "Date", "Ticker", "Company", "weight", "entry_price", "exit_price",
        "trade_return", "pos_contrib", "cost", "pos_pnl_net",
        "mktCap_AUDm", *why_keys,
    ]
    pos_csv = settings.reports_dir / "oos_short_positions.csv"
    entry[out_cols].to_csv(pos_csv, index=False)
    logger.info(f"wrote {pos_csv} ({len(entry):,} rows)")

    # Aggregate by ticker.
    agg = entry.groupby("Ticker").agg(
        Company=("Company", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
        n_months_shorted=("Ticker", "size"),
        total_pnl_book=("pos_pnl_net", "sum"),
        avg_trade_return=("trade_return", "mean"),
        best_month=("trade_return", "max"),
        worst_month=("trade_return", "min"),
        hit_rate=("trade_return", lambda s: float((s > 0).mean())),
        avg_weight=("weight", "mean"),
        avg_mktCap_AUDm=("mktCap_AUDm", "mean"),
        avg_short_pct=("ShortPct", "mean"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
    ).reset_index()
    agg = agg.sort_values("total_pnl_book", ascending=False).reset_index(drop=True)
    agg_csv = settings.reports_dir / "oos_trades.csv"
    agg.to_csv(agg_csv, index=False)
    logger.info(f"wrote {agg_csv} ({len(agg):,} tickers)")

    # Markdown extract: top N winners + top N losers.
    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Company"] = (
            out["Company"].astype(str).str.replace(" ORDINARY", "", regex=False).str.title()
        )
        out["total_pnl_book_%"] = (out["total_pnl_book"] * 100).round(3)
        out["avg_trade_%"] = (out["avg_trade_return"] * 100).round(2)
        out["best_%"] = (out["best_month"] * 100).round(2)
        out["worst_%"] = (out["worst_month"] * 100).round(2)
        out["hit_%"] = (out["hit_rate"] * 100).round(1)
        out["mktCap_$m"] = out["avg_mktCap_AUDm"].round(0)
        out["avg_SI_%"] = out["avg_short_pct"].round(2)
        out["first"] = pd.to_datetime(out["first_date"]).dt.strftime("%Y-%m")
        out["last"] = pd.to_datetime(out["last_date"]).dt.strftime("%Y-%m")
        return out[[
            "Ticker", "Company", "n_months_shorted", "total_pnl_book_%",
            "avg_trade_%", "best_%", "worst_%", "hit_%",
            "avg_SI_%", "mktCap_$m", "first", "last",
        ]]

    md_path = settings.reports_dir / "oos_trades.md"
    lines = [
        f"# OOS short trades — model = {args.model}",
        "",
        f"_Reconstructed from {len(entry):,} OOS short positions across "
        f"{entry['Ticker'].nunique()} unique tickers in the 36-month "
        f"holdout (2023-06 → 2026-05). Per-position P&L is uncapped "
        "(no stop loss) — commission + borrow + slippage apply per the "
        "headline backtest._",
        "",
        "**Columns:** "
        "`n_months_shorted` = number of monthly rebalances the ticker was in "
        "the SHORT quintile. "
        "`total_pnl_book_%` = cumulative contribution to book over all those "
        "months (positive = the strategy made money shorting this name). "
        "`avg_trade_%` = average per-position return (positive = the stock "
        "fell, the short made money). "
        "`best_%` / `worst_%` = best / worst single month per-position return. "
        "`hit_%` = share of monthly shorts that were profitable. "
        "`avg_SI_%` = average reported short interest at the time of shorting. "
        "`mktCap_$m` = average market cap when shorted (AUD m).",
        "",
        f"## Top {args.top_n} winners (most profitable shorts in OOS)",
        "",
    ]
    winners = _fmt(agg.head(args.top_n))
    losers = _fmt(agg.sort_values("total_pnl_book").head(args.top_n))

    def _to_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except (ImportError, ValueError):
            hdr = "| " + " | ".join(df.columns) + " |"
            sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
            rows = ["| " + " | ".join(str(v) for v in r.values) + " |" for _, r in df.iterrows()]
            return "\n".join([hdr, sep, *rows])

    lines.append(_to_md(winners))
    lines += ["", f"## Top {args.top_n} losers (worst shorts in OOS)", ""]
    lines.append(_to_md(losers))

    # Aggregate stats.
    short_only_pnl = float(entry.loc[entry["weight"] < 0, "pos_pnl_net"].sum())
    median_trade = float(entry["trade_return"].median())
    win_rate = float((entry["trade_return"] > 0).mean())
    lines += [
        "",
        "## Aggregate OOS stats (short leg only)",
        "",
        f"- **Total short-leg P&L**: {short_only_pnl * 100:+.1f} % of book "
        f"(summed over {len(entry):,} monthly positions)",
        f"- **Median per-position return**: {median_trade * 100:+.2f} %",
        f"- **Win rate** (per-position): {win_rate * 100:.1f} %",
        f"- **Best single month**: "
        f"`{agg.iloc[0]['Ticker']}` ({_fmt(agg.head(1)).iloc[0]['best_%']:+.2f} %)",
        f"- **Worst single month**: "
        f"`{agg.iloc[-1]['Ticker']}` ({_fmt(agg.tail(1)).iloc[0]['worst_%']:+.2f} %)",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
