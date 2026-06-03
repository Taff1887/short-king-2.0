"""Backtest each model's OOF scores across three portfolio constructions.

For every model in ``reports/oof_predictions.parquet`` we build three short
portfolios:

* ``top_k_short``     - short the K names with the highest predicted-decline
  score (sized equal-weight).
* ``decile_short``    - short the worst decile (top-N% by score).
* ``long_short_decile`` - decile spread (short worst - long best); a hedged
  read on the ranking.

Each portfolio is run through ``backtest_weekly`` with the same ``CostConfig``
(borrow + commission), producing per-week return series. Cumulative return,
drawdown and per-month heatmap PNGs are written via ``reporting.charts`` and
the headline ``summary_table`` is saved as ``reports/summary_table.csv``.
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from short_king.portfolio.backtest import CostConfig, backtest_weekly
from short_king.portfolio.construct import (
    decile_short,
    long_short_decile,
    top_k_short,
)
from short_king.portfolio.metrics import summary_table
from short_king.reporting import charts as rc
from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=20, help="Names to short in top_k variant.")
    p.add_argument("--decile", type=int, default=10, help="Number of deciles (default 10).")
    p.add_argument("--borrow-bps", type=float, default=300.0, help="Borrow cost in bps p.a.")
    p.add_argument("--commission-bps", type=float, default=5.0, help="Per-side commission bps.")
    return p.parse_args()


_FWD_RET_CANDIDATES = ("ret_fwd_4w", "ret_fwd", "y_fwd_ret")


def _fwd_ret_col(df: pd.DataFrame) -> str:
    for c in _FWD_RET_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"oof predictions missing a forward-return col; tried {_FWD_RET_CANDIDATES}")


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    oof_path = settings.reports_dir / "oof_predictions.parquet"
    if not oof_path.exists():
        logger.error(f"{oof_path} not found — must run 05_train_and_validate.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"06_backtest: start {t0.isoformat(timespec='seconds')}")

    oof = read_parquet(oof_path)
    fwd_col = _fwd_ret_col(oof)
    cost = CostConfig(borrow_bps=args.borrow_bps, commission_bps=args.commission_bps)

    variants = {
        "top_k": lambda g: top_k_short(g, score_col="score", k=args.top_k),
        "decile": lambda g: decile_short(g, score_col="score", n_deciles=args.decile),
        "ls_decile": lambda g: long_short_decile(g, score_col="score", n_deciles=args.decile),
    }

    all_summaries: list[pd.DataFrame] = []
    for model_name, model_oof in oof.groupby("model"):
        for variant_name, build_fn in variants.items():
            weights = build_fn(model_oof)
            bt = backtest_weekly(weights, returns=model_oof[fwd_col], cost=cost)
            out_path = (
                settings.reports_dir / f"backtest_{model_name}_{variant_name}.parquet"
            )
            write_parquet(bt, out_path)

            summ = summary_table(bt)
            if isinstance(summ, pd.Series):
                summ = summ.to_frame().T
            summ = summ.copy()
            summ["model"] = model_name
            summ["variant"] = variant_name
            all_summaries.append(summ)

            # Per-strategy chart trio.
            stem = settings.charts_dir / f"{model_name}_{variant_name}"
            try:
                rc.plot_cumulative(bt, out_path=stem.with_name(stem.name + "_cum.png"))
                rc.plot_drawdown(bt, out_path=stem.with_name(stem.name + "_dd.png"))
                rc.plot_monthly_heatmap(
                    bt, out_path=stem.with_name(stem.name + "_heatmap.png")
                )
            except Exception as exc:
                logger.warning(f"charts {model_name}/{variant_name}: {exc}")

            logger.info(f"backtest {model_name}/{variant_name}: wrote {out_path.name}")

    if not all_summaries:
        logger.error("no backtests produced — aborting summary write.")
        return 2

    summary = pd.concat(all_summaries, ignore_index=True)
    cols = ["model", "variant"] + [c for c in summary.columns if c not in ("model", "variant")]
    summary = summary[cols]
    summary.to_csv(settings.reports_dir / "summary_table.csv", index=False)

    t1 = dt.datetime.now()
    logger.info(
        f"06_backtest: wrote summary_table.csv with {len(summary)} strategies | "
        f"took {(t1 - t0).total_seconds():.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
