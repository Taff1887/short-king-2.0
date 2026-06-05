"""'What is each model shorting right now?' -- the most recent rebalance.

Produces a richer view than `_current_positions.py`:

  1. **Top 10 by consensus** -- the names every model agrees are
     shortable (the safest picks).
  2. **Top 5 per individual model** -- so the reader can see what
     each model alone is flagging (the disagreements are often
     the most informative).
  3. **Factor breakdown** for the top consensus picks -- which of
     the EW polarity-aware ranks lit up for each name.

Output: reports/current_short_picks.md (designed to slot into the
README as-is).
"""

from __future__ import annotations

import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

MODELS = ("naive", "ew", "logit", "gbm_cls", "gbm_rank")

# Factor breakdown columns (raw rank, friendly label, invert?).
FACTORS: list[tuple[str, str, bool]] = [
    ("ShortPct_rk", "SI", False),
    ("si_z_12m_rk", "SI z", False),
    ("mom_3m_rk", "mom", True),
    ("vol_1m_rk", "vol", False),
    ("pe_rk", "P/E", False),
    ("fcf_yield_rk", "FCF-y", True),
    ("roe_rk", "ROE", True),
    ("debt_equity_rk", "D/E", False),
    ("revenue_growth_yoy_rk", "growth", True),
]

TOP_CONS = 10
TOP_PER_MODEL = 5


def _company_clean(name: object) -> str:
    if pd.isna(name):
        return ""
    return str(name).replace(" ORDINARY", "").title()


def _md_table(rows: list[list], headers: list[str]) -> str:
    hdr = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([hdr, sep, *body])


def main() -> int:
    settings.ensure_dirs()
    feat = read_parquet(settings.processed_dir / "features_monthly.parquet")
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")

    as_of = feat["Date"].max()
    latest = feat[feat["Date"] == as_of].copy()
    # Investability gate matches the backtest constructor.
    latest = latest[latest["investable"].fillna(False).astype(bool)]
    latest = latest[latest["mktCap"].fillna(0) >= 100_000_000]
    n_universe = len(latest)

    # Pull per-model scores for the latest rebalance.
    scores = (
        oof[oof["Date"] == as_of]
        .pivot_table(index="Ticker", columns="model", values="score")
        .reset_index()
    )
    score_cols = [f"score_{m}" for m in MODELS]
    scores = scores.rename(columns={m: f"score_{m}" for m in MODELS})
    latest = latest.merge(scores, on="Ticker", how="left")

    # Add polarity-aware factor columns (high = more shortable).
    for raw_col, label, invert in FACTORS:
        if raw_col not in latest.columns:
            latest[label] = float("nan")
            continue
        val = pd.to_numeric(latest[raw_col], errors="coerce").fillna(0.5)
        latest[label] = (1.0 - val) if invert else val

    # Cross-sectional rank of each model's score on this date so we can
    # build a consensus across models on different output scales (gbm_rank
    # outputs are roughly z-scored, others are 0-1; ranking normalises).
    rank_cols = []
    for col in score_cols:
        rk_col = f"rk_{col}"
        latest[rk_col] = pd.to_numeric(latest[col], errors="coerce").rank(
            pct=True, ascending=True,
        )
        rank_cols.append(rk_col)
    latest["consensus_rk"] = latest[rank_cols].mean(axis=1)

    md_lines: list[str] = [
        f"# Current short picks — as of {as_of.date()}",
        "",
        f"_Latest monthly rebalance: **{as_of.date()}**. "
        f"Investable universe size: **{n_universe} names** "
        "(≥ A$100 m market cap, has fresh fundamentals, has a valid "
        "adjusted close)._",
        "",
        "**Score columns** (one per model, all 0-1 except gbm_rank which is "
        "z-scored):",
        "* `score_naive` — rank of `ShortPct` across the cross-section",
        "* `score_ew` — polarity-aware equal-weight composite of 12 ranks",
        "* `score_logit` — L2 logistic-regression Pr(monthly return < 0)",
        "* `score_gbm_cls` — LightGBM binary classifier probability",
        "* `score_gbm_rank` — LightGBM LambdaRank raw output "
        "(higher = ranked closer to the top of the bearish tail)",
        "",
        "Higher = more shortable on every model. `consensus_rk` is the "
        "**average of each model's cross-sectional percentile rank** on this "
        "date — robust to the different output scales.",
        "",
    ]

    # --- 1. Top by consensus --------------------------------------------------
    top_cons = latest.sort_values("consensus_rk", ascending=False).head(TOP_CONS).copy()
    top_cons["Company"] = top_cons["Company"].map(_company_clean)
    rows = []
    for i, (_, r) in enumerate(top_cons.iterrows(), 1):
        rows.append([
            i, r["Ticker"], r["Company"],
            f"{r['mktCap'] / 1e6:,.0f}" if pd.notna(r["mktCap"]) else "",
            f"{r['ShortPct']:.2f}" if pd.notna(r["ShortPct"]) else "",
            *[f"{r[c]:.3f}" if pd.notna(r[c]) else "" for c in score_cols],
            f"{r['consensus_rk']:.3f}",
        ])
    md_lines += [
        f"## Top {TOP_CONS} by consensus across all 5 models",
        "",
        "The names every model agrees are in the bearish tail (the safest "
        "picks — if all 5 models cluster on the same names, the signal is "
        "broad-based rather than driven by any single factor).",
        "",
        _md_table(
            rows,
            ["#", "Ticker", "Company", "Mkt Cap (A$m)", "Short %",
             "score_naive", "score_ew", "score_logit",
             "score_gbm_cls", "score_gbm_rank", "consensus_rk"],
        ),
        "",
    ]

    # --- 2. Top 5 per individual model ----------------------------------------
    md_lines += [
        f"## Top {TOP_PER_MODEL} per individual model",
        "",
        "What each model alone says are its highest-conviction shorts. "
        "Cross-reference with the consensus list above — names that appear in "
        "multiple individual top-5s are broad agreement signals; names that "
        "appear in only one model are 'this model alone thinks this' (often "
        "the most informative disagreements).",
        "",
    ]
    for model in MODELS:
        col = f"score_{model}"
        sub = latest.dropna(subset=[col]).sort_values(col, ascending=False).head(TOP_PER_MODEL)
        rows = []
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            rows.append([
                i, r["Ticker"], _company_clean(r["Company"]),
                f"{r['mktCap'] / 1e6:,.0f}" if pd.notna(r["mktCap"]) else "",
                f"{r['ShortPct']:.2f}" if pd.notna(r["ShortPct"]) else "",
                f"{r[col]:.3f}",
            ])
        md_lines += [
            f"### Top {TOP_PER_MODEL} per `{model}`",
            "",
            _md_table(
                rows,
                ["#", "Ticker", "Company", "Mkt Cap (A$m)", "Short %",
                 f"score_{model}"],
            ),
            "",
        ]

    # --- 3. Factor breakdown for the consensus top -----------------------------
    md_lines += [
        f"## Why these names? Factor breakdown for the top {TOP_CONS} consensus picks",
        "",
        "Every cell is 0-1; higher = more shortable on that dimension. "
        "`(inv)` columns are naturally-bullish ranks flipped via `1 − rank` so "
        "the polarity is consistent across the table.",
        "",
    ]
    factor_labels = [lbl for _, lbl, inv in FACTORS]
    factor_cols_md = [
        "SI", "SI z",
        "mom (inv)", "vol", "P/E",
        "FCF-y (inv)", "ROE (inv)", "D/E", "growth (inv)",
    ]
    rows = []
    for i, (_, r) in enumerate(top_cons.iterrows(), 1):
        cells = [i, r["Ticker"]]
        for lbl in factor_labels:
            cells.append(f"{r[lbl]:.2f}" if pd.notna(r.get(lbl)) else "")
        # Mean of all factor cells = quick read of how multi-factor bearish
        # the name is.
        avg = pd.Series([r[l] for l in factor_labels]).dropna().mean()
        cells.append(f"{avg:.2f}" if pd.notna(avg) else "")
        rows.append(cells)
    md_lines += [
        _md_table(
            rows,
            ["#", "Ticker", *factor_cols_md, "EW factor avg"],
        ),
        "",
        "**How to read it:**",
        "* Rows where most cells are near 1.0 = multi-factor shorts (crowded "
        "SI + falling momentum + low quality + high leverage all at once). "
        "These are the safest setups.",
        "* Rows where only the SI columns are high but fundamentals are "
        "neutral / bullish = pure crowded-short plays. Higher squeeze risk "
        "because the signal rests on one dimension only.",
        "* `EW factor avg` is the simple mean across the 9 columns — gives a "
        "one-number read of multi-factor agreement.",
        "",
    ]

    md_path = settings.reports_dir / "current_short_picks.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    csv_cols = ["Ticker", "Company", "mktCap", "ShortPct",
                *score_cols, "consensus_rk", *factor_labels]
    csv_cols = [c for c in csv_cols if c in latest.columns]
    out = latest.sort_values("consensus_rk", ascending=False).head(30)[csv_cols]
    csv_path = settings.reports_dir / "current_short_picks.csv"
    out.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
