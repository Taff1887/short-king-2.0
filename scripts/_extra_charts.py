"""Generate the rest of the diagnostic charts the headline report skipped.

Runs after 04 (features), 05 (oof + interpret artefacts) and 06 (backtest).
Adds universe coverage, SI distribution, feature correlation, top short
candidates, decile-return spread and calibration to ``charts/``.
"""

from __future__ import annotations

import pandas as pd

from short_king.reporting import charts as rc
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


def main() -> None:
    settings.ensure_dirs()
    out = settings.charts_dir

    panel = read_parquet(settings.processed_dir / "master_clean.parquet")
    features = read_parquet(settings.processed_dir / "features.parquet")
    oof = read_parquet(settings.reports_dir / "oof_predictions.parquet")

    # 1. Universe coverage by Friday
    rc.chart_universe_coverage(panel, out / "universe_coverage.png")
    logger.info("wrote universe_coverage.png")

    # 2. SI distribution (latest Friday)
    rc.chart_si_distribution(panel, out / "si_distribution.png", latest_only=True)
    logger.info("wrote si_distribution.png")

    # 3. Feature correlation matrix on a representative subset of *_rk cols
    chosen_features = [
        c for c in (
            "short_pct_ff_rk", "ShortPct_rk", "si_z_12m_rk", "days_to_cover_rk",
            "mom_1w_rk", "mom_1m_rk", "mom_3m_rk", "mom_6m_rk",
            "vol_1m_rk", "vol_3m_rk",
            "log_mktcap_rk", "adv_aud_rk", "turnover_pct_rk", "amihud_rk",
            "pe_rk", "pb_rk", "ev_ebitda_rk",
            "fcf_yield_rk", "earnings_yield_rk",
            "roe_rk", "roic_rk", "gross_margin_rk", "operating_margin_rk",
            "debt_equity_rk", "interest_coverage_rk",
            "revenue_growth_yoy_rk", "eps_growth_yoy_rk", "asset_growth_yoy_rk",
        ) if c in features.columns
    ]
    rc.chart_correlation_matrix(features, chosen_features, out / "feature_correlation.png")
    logger.info(f"wrote feature_correlation.png ({len(chosen_features)} features)")

    # 4. Feature distributions for the same chosen subset (small multiples)
    try:
        rc.chart_feature_distributions(features, chosen_features[:12], out / "feature_distributions.png")
        logger.info("wrote feature_distributions.png")
    except Exception as exc:
        logger.warning(f"chart_feature_distributions failed: {exc}")

    # 5. Top short candidates as of the latest Friday — use the best model's
    #    OOF (or fall back to ShortPct rank if no model has scores there).
    latest = pd.to_datetime(oof["Date"]).max()
    latest_oof = oof[oof["Date"] == latest].copy()
    if not latest_oof.empty:
        best_model = (
            oof.groupby("model").apply(lambda g: pd.Series([g["score"].dropna().mean()]))
            .iloc[:, 0]
            .idxmax()
        )
        latest_oof = latest_oof[latest_oof["model"] == best_model]
        latest_oof = latest_oof.rename(columns={"score": "score"})
        try:
            rc.chart_top_candidates(
                latest_oof, out / "top_short_candidates.png",
                as_of=latest, top_n=15,
            )
            logger.info(f"wrote top_short_candidates.png (model={best_model}, asof={latest.date()})")
        except Exception as exc:
            logger.warning(f"chart_top_candidates failed: {exc}")

    # 6. Decile-return spread per model (mean fwd_ret by score decile)
    decile_records: list[dict] = []
    for model, g in oof.groupby("model"):
        g = g.dropna(subset=["score", "fwd_ret_1m"])
        if len(g) < 100:
            continue
        deciles = pd.qcut(g["score"], 10, labels=False, duplicates="drop")
        means = g.groupby(deciles)["fwd_ret_1m"].mean()
        for d, r in means.items():
            decile_records.append({"model": model, "decile": int(d), "mean_fwd_ret_1m": float(r)})
    decile_df = pd.DataFrame(decile_records)
    if not decile_df.empty:
        try:
            rc.chart_decile_returns(decile_df, out / "decile_returns.png")
            logger.info("wrote decile_returns.png")
        except Exception as exc:
            # Fall back to a basic matplotlib bar group; the spec for
            # chart_decile_returns may not accept this exact shape.
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
            wide = decile_df.pivot_table(index="decile", columns="model", values="mean_fwd_ret_1m")
            wide.plot(kind="bar", ax=ax)
            ax.set_xlabel("Score decile (0 = lowest predicted shortability, 9 = highest)")
            ax.set_ylabel("Mean 1-month forward return")
            ax.set_title("Forward-return spread by score decile (out-of-fold)")
            ax.axhline(0, color="black", linewidth=0.7)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(out / "decile_returns.png")
            plt.close(fig)
            logger.info(f"wrote decile_returns.png (manual fallback, reason: {exc})")

    # (Previously: GBM calibration plot. GBM models were removed from the
    # project.)


if __name__ == "__main__":
    main()
