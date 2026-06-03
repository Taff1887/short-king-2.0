"""Backtest each model's OOF scores across two portfolio constructions.

For every model in ``reports/oof_predictions.parquet`` we build two short
portfolios:

* ``decile_short``      - short the worst decile (top-N% by score), equal-weight.
* ``long_short_decile`` - dollar-neutral decile spread (short worst - long best).

Each is run through ``backtest_weekly`` with the same ``CostConfig`` (commission
+ borrow + slippage), producing per-week return frames. Cumulative-return,
drawdown and monthly-heatmap PNGs are written via ``reporting.charts``, and an
aggregated ``backtest_summary.csv`` is saved to ``reports/``.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from short_king.portfolio.backtest import CostConfig, backtest_weekly
from short_king.portfolio.construct import decile_short, long_short_decile
from short_king.reporting import charts as rc
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger

_DATE_COL = "Date"
_TICKER_COL = "Ticker"
_FWD_RET_COL = "fwd_ret_4w"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-deciles", type=int, default=10, help="Number of deciles.")
    p.add_argument("--bps-round-trip", type=float, default=25.0,
                   help="Per-side commission + half-spread, bps.")
    p.add_argument("--annual-borrow-pct", type=float, default=1.5,
                   help="Annualised borrow fee on short positions, %% p.a.")
    p.add_argument("--slippage-bps", type=float, default=5.0,
                   help="One-sided slippage on weight changes, bps.")
    return p.parse_args()


def _build_prices_panel(clean: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Long [Date, Ticker, adjClose] panel plus investable / adv_aud for the
    constructors. Ticker is resolved from 'Ticker' or 'Symbol'."""
    def _with_ticker(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if _TICKER_COL not in out.columns and "Symbol" in out.columns:
            out[_TICKER_COL] = out["Symbol"]
        return out

    c = _with_ticker(clean)
    keep = [k for k in (_DATE_COL, _TICKER_COL, "adjClose", "investable") if k in c.columns]
    panel = c[keep].copy()

    if "adv_aud" in features.columns:
        f = _with_ticker(features)[[_DATE_COL, _TICKER_COL, "adv_aud"]]
        f = f.drop_duplicates([_DATE_COL, _TICKER_COL])
        panel = panel.merge(f, on=[_DATE_COL, _TICKER_COL], how="left")
    return panel


def _summary_row(model: str, strategy: str, result) -> dict:
    """BacktestResult.summary -> the task's flat row schema."""
    s = result.summary
    return {
        "model": model, "strategy": strategy,
        "CAGR": float(s.get("CAGR", float("nan"))),
        "vol": float(s.get("ann_vol", float("nan"))),
        "Sharpe": float(s.get("Sharpe", float("nan"))),
        "Sortino": float(s.get("Sortino", float("nan"))),
        "MaxDD": float(s.get("max_drawdown", float("nan"))),
        "Calmar": float(s.get("calmar", float("nan"))),
        "hit_rate": float(s.get("hit_rate", float("nan"))),
        "avg_turnover": float(s.get("avg_turnover", float("nan"))),
        "n_rebalances": int(s.get("n_weeks", 0)),
    }


def _returns_series(returns_df: pd.DataFrame, label: str) -> pd.Series:
    """Net-of-cost weekly return series, DatetimeIndex-keyed."""
    return pd.Series(
        returns_df["ret_net"].to_numpy(),
        index=pd.to_datetime(returns_df[_DATE_COL]),
        name=label,
    ).sort_index()


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    oof_path = settings.reports_dir / "oof_predictions.parquet"
    clean_path = settings.processed_dir / "master_clean.parquet"
    feat_path = settings.processed_dir / "features.parquet"
    if not oof_path.exists():
        logger.error(f"{oof_path} not found - must run 05_train_and_validate.py first")
        return 1
    if not clean_path.exists():
        logger.error(f"{clean_path} not found - must run 04_build_features.py first")
        return 1
    if not feat_path.exists():
        logger.error(f"{feat_path} not found - must run 04_build_features.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"06_backtest: start {t0.isoformat(timespec='seconds')}")

    oof = read_parquet(oof_path)
    clean = read_parquet(clean_path)
    features = read_parquet(feat_path)
    prices_panel = _build_prices_panel(clean, features)
    logger.info(
        f"loaded: oof rows={len(oof):,} models={sorted(oof['model'].unique())} | "
        f"prices_panel rows={len(prices_panel):,}"
    )

    cost = CostConfig(
        bps_round_trip=args.bps_round_trip,
        annual_borrow_pct=args.annual_borrow_pct,
        slippage_bps=args.slippage_bps,
    )

    strategies = {
        "decile_short": lambda g: decile_short(g, n_deciles=args.n_deciles),
        "long_short_decile": lambda g: long_short_decile(g, n_deciles=args.n_deciles),
    }

    summary_rows: list[dict] = []
    returns_panel: dict[str, pd.Series] = {}

    for model_name, model_oof in oof.groupby("model"):
        # The constructors need `investable` from master_clean and `adv_aud`
        # from features alongside Date / Ticker / score. Merge once per model.
        joined = model_oof.merge(
            prices_panel[[_DATE_COL, _TICKER_COL] +
                         [c for c in ("investable", "adv_aud") if c in prices_panel.columns]],
            on=[_DATE_COL, _TICKER_COL],
            how="left",
        )

        for strategy_name, build_fn in strategies.items():
            weights = build_fn(joined)
            if weights.empty:
                logger.warning(f"{model_name}/{strategy_name}: empty weights - skipping")
                continue

            result = backtest_weekly(
                target_weights=weights,
                prices_panel=prices_panel,
                cost_config=cost,
            )

            out_path = settings.reports_dir / f"backtest_{model_name}_{strategy_name}.parquet"
            write_parquet(result.returns, out_path)

            summary_rows.append(_summary_row(model_name, strategy_name, result))
            returns_panel[f"{model_name}/{strategy_name}"] = _returns_series(
                result.returns, f"{model_name}/{strategy_name}"
            )

            logger.info(
                f"backtest {model_name}/{strategy_name}: weeks={int(result.summary.get('n_weeks', 0))} "
                f"-> {out_path.name}"
            )

    if not summary_rows:
        logger.error("no backtests produced - aborting summary/chart writes.")
        return 2

    # Aggregate summary across (model, strategy).
    summary_df = pd.DataFrame(summary_rows)
    summary_path = settings.reports_dir / "backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"wrote {summary_path.name} with {len(summary_df)} (model, strategy) rows")

    # Charts: cumulative across all (model, strategy); drawdown + monthly
    # heatmap for the best Sharpe row.
    returns_frame = pd.DataFrame(returns_panel).sort_index()
    best_row = summary_df.sort_values("Sharpe", ascending=False).iloc[0]
    best_key = f"{best_row['model']}/{best_row['strategy']}"
    best_series = returns_panel.get(best_key)

    chart_jobs = [
        ("cumulative_returns.png",
         lambda: rc.chart_cumulative_returns(returns_frame, settings.charts_dir / "cumulative_returns.png")),
    ]
    if best_series is not None and len(best_series) > 0:
        chart_jobs += [
            ("drawdowns.png",
             lambda: rc.chart_drawdowns(best_series, settings.charts_dir / "drawdowns.png")),
            ("monthly_heatmap.png",
             lambda: rc.chart_monthly_heatmap(best_series, settings.charts_dir / "monthly_heatmap.png")),
        ]
    for name, fn in chart_jobs:
        try:
            fn()
        except Exception as exc:
            logger.warning(f"chart {name} failed: {exc}")
    logger.info(f"best by Sharpe: {best_key} (Sharpe={best_row['Sharpe']:.2f})")

    t1 = dt.datetime.now()
    logger.info(
        f"06_backtest: done | {len(summary_df)} strategies | "
        f"took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
