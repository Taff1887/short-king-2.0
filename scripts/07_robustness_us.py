"""US robustness check: re-run the methodology on the S&P 500 (stub).

Intent
------
Demonstrate the methodology travels by repeating the ASX exercise on a
survivorship-bias-free S&P 500 panel:

1. :func:`short_king.data.universe.build_sp500_pit` — point-in-time S&P 500
   membership over the same study window (Wikipedia/CRSP-style PIT, no
   look-ahead from current index composition).
2. :func:`short_king.data.fundamentals.fetch_many` + cached FMP fundamentals
   for each in-universe symbol (same FMP endpoints as the ASX path).
3. :func:`short_king.data.prices.fetch_many_adjusted` for total-return prices.
4. ``assemble_pit_panel`` / ``conservative_clean`` / ``build_feature_panel``
   reused **without the short-interest feature family**. The FMP stable API
   does not expose a weekly US short-interest analogue; the US FINRA / NYSE
   short-interest series is bi-monthly / quarterly, an order of magnitude
   coarser than the weekly ASIC panel that drives the ASX signal. Mixing
   cadences would either force weekly forward-fills that fabricate
   information or force the whole US panel down to quarterly, which destroys
   the cross-sectional ranking exercise. We therefore drop the short family
   entirely on US and document this as a known limitation in
   ``docs/methodology.md`` rather than fake a comparable signal.
5. Re-target the model. On ASX the headline label is the
   regression-style ``fwd_ret_4w``; on US we run the
   :class:`short_king.models.baselines.GBM_cls` baseline against a *binary*
   ``fwd_ret_4w`` target (sign of the 4-week forward return), so the
   robustness exercise tests the cross-sectional ranking machinery rather
   than parity of point predictions across markets.
6. Compare factor loadings cross-market: line up the per-feature SHAP /
   gain importances from the ASX GBM with the US GBM_cls and report the
   rank correlation. A positive correlation is the headline robustness
   finding — same features matter on a different market — and a negative or
   zero correlation flags ASX-specific overfit. ``backtest_weekly`` is run
   on the US predictions to sanity-check that the ranking actually trades.

This script is presently a stub: the end-to-end wiring lives in the modules
above but the integration test is deferred to the next iteration so the
primary ASX results stay the headline deliverable. The file exists so the
intent is discoverable in the repo layout.
"""

from __future__ import annotations

import argparse
import datetime as dt

from short_king.utils.config import settings
from short_king.utils.logging import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--start", type=str, default="2018-01-01", help="Study window start (YYYY-MM-DD)."
    )
    p.add_argument(
        "--end",
        type=str,
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="Study window end (YYYY-MM-DD).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()
    t0 = dt.datetime.now()
    logger.info(
        f"07_robustness_us: start {t0.isoformat(timespec='seconds')} | "
        f"window {args.start} -> {args.end}"
    )

    # TODO: end-to-end wiring. Pieces (already implemented elsewhere):
    #
    #   from short_king.data.universe     import build_sp500_pit
    #   from short_king.data.fundamentals import fetch_many, write_raw_tables
    #   from short_king.data.prices       import fetch_many_adjusted
    #   from short_king.data.assemble     import assemble_pit_panel, write_panel
    #   from short_king.data.clean        import conservative_clean
    #   from short_king.features.build    import build_feature_panel
    #   from short_king.models.walk_forward import fit_predict_walkforward
    #   from short_king.portfolio.backtest  import backtest_weekly, CostConfig
    #
    # Rationale for the stub: the US short-interest signal (FINRA bi-monthly)
    # has a fundamentally different cadence from the weekly ASIC panel, and
    # gluing it in cleanly would extend this milestone. The robustness section
    # of the README will reference this file and ``docs/methodology.md``.

    logger.warning(
        "US robustness path: not yet wired end-to-end — see docs/methodology.md"
    )
    t1 = dt.datetime.now()
    logger.info(f"07_robustness_us: done | took {(t1 - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
