"""Produce the "as-of latest Friday" short-candidate table.

Joins the latest available out-of-fold predictions with the cleaned PIT panel
to surface the top-N short candidates by *consensus* trained-model rank. Used
as a worked example in the README ("what does the model recommend today?").

Outputs:
    reports/current_positions.csv         - full ranked table (top 30)
    reports/current_positions.md          - markdown extract (top 15) for README
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

TRAINED_MODELS = ("logit", "gbm_cls", "gbm_rank")
MIN_MKT_CAP_AUD = 200_000_000
MIN_FULL_COVERAGE_FRAC = 0.7   # logit must score >= 70 % of the day's universe
TOP_N_CSV = 30
TOP_N_MD = 15


def _latest_full_date(oof: pd.DataFrame) -> pd.Timestamp:
    """Latest Friday on which every trained model has scored at least
    ``MIN_FULL_COVERAGE_FRAC`` of the day's universe. Using a coverage gate
    (rather than just `>= 1 non-null score`) ensures the consensus rank is
    apples-to-apples - logit has fewer walk-forward folds than the GBMs so
    naively grabbing the latest date can leave logit blank for most names."""
    work = oof[oof["model"].isin(TRAINED_MODELS)].copy()
    daily_total = work.groupby("Date")["Ticker"].nunique().rename("universe")
    scored = (
        work.dropna(subset=["score"])
            .groupby(["Date", "model"])["Ticker"]
            .nunique()
            .unstack("model")
            .reindex(columns=list(TRAINED_MODELS))
    )
    frac = scored.div(daily_total, axis=0)
    full = frac[(frac >= MIN_FULL_COVERAGE_FRAC).all(axis=1)].index
    if full.empty:
        # Fall back to "at least one non-null score per model"; tighter gate
        # didn't fit so log it and pick the latest date with any coverage.
        logger.warning(
            f"no Friday meets the {MIN_FULL_COVERAGE_FRAC:.0%} coverage gate; "
            f"falling back to >=1 score per model"
        )
        full = scored.dropna(how="any").index
    return pd.Timestamp(full.max())


def _consensus_rank(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Average within-date percentile rank of the given columns. High = short."""
    pres = [c for c in cols if c in panel.columns]
    if not pres:
        raise KeyError(f"none of {cols} present in panel")
    ranks = panel[pres].rank(pct=True)
    return ranks.mean(axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monthly", action="store_true",
                   help="Read oof_predictions_monthly.parquet and write "
                        "current_positions_monthly.{csv,md}.")
    args = p.parse_args()
    settings.ensure_dirs()

    oof_name = "oof_predictions_monthly.parquet" if args.monthly else "oof_predictions.parquet"
    out_csv_name = "current_positions_monthly.csv" if args.monthly else "current_positions.csv"
    out_md_name = "current_positions_monthly.md" if args.monthly else "current_positions.md"
    oof = read_parquet(settings.reports_dir / oof_name)
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")

    as_of = _latest_full_date(oof)
    logger.info(f"current_positions: as-of = {as_of.date()}")

    # Wide score panel for the as-of date.
    snap = oof[oof["Date"] == as_of]
    wide = snap.pivot_table(
        index="Ticker", columns="model", values="score", aggfunc="mean"
    )
    wide = wide.add_prefix("score_")

    # Consensus = average rank across the three trained models. High = strong
    # short conviction (all three want this name lower).
    trained_cols = [f"score_{m}" for m in TRAINED_MODELS if f"score_{m}" in wide.columns]
    wide["consensus_rank"] = _consensus_rank(wide, trained_cols)

    # Context from the cleaned panel for the same Friday.
    ctx_cols = [
        "Ticker", "Company", "sector", "industry", "mktCap",
        "ShortPct", "adjClose", "investable",
    ]
    ctx = clean.loc[clean["Date"] == as_of, [c for c in ctx_cols if c in clean.columns]]
    out = wide.merge(ctx, left_index=True, right_on="Ticker", how="left")

    # Require all three trained models to have scored the ticker on this date
    # (drops names where logit's walk-forward didn't cover the row).
    full_score_mask = wide[trained_cols].notna().all(axis=1)
    full_idx = wide.index[full_score_mask]
    out = out[out["Ticker"].isin(full_idx)]

    # Investable gate + explicit min market cap (the upstream `investable`
    # column doesn't re-apply the size threshold after `conservative_clean`).
    if "investable" in out.columns:
        out = out[out["investable"].astype(bool)]
    if "mktCap" in out.columns:
        out = out[out["mktCap"].fillna(0) >= MIN_MKT_CAP_AUD]

    out = out.sort_values(
        ["consensus_rank", "ShortPct"], ascending=[False, False],
    ).reset_index(drop=True)
    out["short_rank"] = np.arange(1, len(out) + 1)

    # Pretty columns for the markdown extract.
    md_cols = [
        "short_rank", "Ticker", "Company", "sector",
        "mktCap_AUDm", "ShortPct",
        "score_logit", "score_gbm_cls", "score_gbm_rank", "consensus_rank",
    ]
    out["mktCap_AUDm"] = (out["mktCap"] / 1e6).round(0)

    csv_cols = [
        "short_rank", "Ticker", "Company", "sector", "industry",
        "mktCap", "mktCap_AUDm", "ShortPct", "adjClose",
        "score_logit", "score_gbm_cls", "score_gbm_rank", "consensus_rank",
    ]
    csv_cols = [c for c in csv_cols if c in out.columns]
    out_csv = out[csv_cols].head(TOP_N_CSV)
    csv_path = settings.reports_dir / out_csv_name
    out_csv.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path} ({len(out_csv)} rows)")

    # Markdown extract for the README.
    md_cols = [c for c in md_cols if c in out.columns]
    md_df = out[md_cols].head(TOP_N_MD).copy()
    # Tidy numeric formatting for the markdown.
    fmt_map = {
        "ShortPct": "{:.2f}",
        "score_logit": "{:.3f}",
        "score_gbm_cls": "{:.3f}",
        "score_gbm_rank": "{:.3f}",
        "consensus_rank": "{:.3f}",
        "mktCap_AUDm": "{:,.0f}",
    }
    for c, fmt in fmt_map.items():
        if c in md_df.columns:
            md_df[c] = md_df[c].map(lambda v, _f=fmt: _f.format(v) if pd.notna(v) else "")
    md_df["Company"] = md_df["Company"].astype(str).str.replace(" ORDINARY", "", regex=False).str.title()
    # Sector isn't populated in the current pipeline (we don't fetch FMP profile);
    # show a clean "-" rather than literal "nan" in the markdown.
    if "sector" in md_df.columns:
        md_df["sector"] = md_df["sector"].fillna("-").replace({"nan": "-", "None": "-"})

    md_path = settings.reports_dir / out_md_name
    header = (
        f"# Top {TOP_N_MD} short candidates — as of {as_of.date()}\n\n"
        f"Ranked by **consensus rank** of three trained models (logit + GBM "
        f"classifier + GBM LambdaRank). Higher rank = stronger short conviction "
        f"(all three models agree the name is in the bearish tail).\n\n"
        f"_Investable gate: A$200m mkt cap, fundamentals present, not flagged "
        f"as corrupted. Universe size at as-of: {len(out)} names._\n\n"
    )
    try:
        body = md_df.to_markdown(index=False)
    except (ImportError, ValueError):
        body = "| " + " | ".join(md_df.columns) + " |\n"
        body += "|" + "|".join(["---"] * len(md_df.columns)) + "|\n"
        for _, r in md_df.iterrows():
            body += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    md_path.write_text(header + body + "\n", encoding="utf-8")
    logger.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()
