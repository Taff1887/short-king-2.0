"""Build a 'what is currently being shorted, and why' view of the top
short basket on the most recent rebalance.

The 'most successful strategy' (excluding the ASX 200 benchmark) is
**naive L/S quintile** at +0.39 OOS Sharpe. naive is trivial -- sort by
ShortPct, take the top 20 % -- so the 'why' for naive is just 'high SI'.

To give the reader a richer 'why', this report shows the top short
candidates and breaks out the polarity-aware factor ranks that the EW
composite uses (the most interpretable signal stack on the project):

  ShortPct_rk      : crowded short -- high = more shortable
  si_z_12m_rk      : SI z-score vs 12m history -- high = SI building
  mom_3m_rk_inv    : 1 - 3-month momentum rank -- high = falling
  vol_1m_rk        : 1-month vol -- high = skittish
  pe_rk            : P/E rank -- high = expensive
  fcf_yield_rk_inv : 1 - FCF yield rank -- high = cash-poor
  roe_rk_inv       : 1 - ROE rank -- high = low quality
  debt_equity_rk   : leverage -- high = levered
  revenue_growth_yoy_rk_inv : 1 - growth rank -- high = shrinking

Each column is 0-1 (0 = least shortable, 1 = most). A row average gives a
quick read of how many factors point bearish for that name. A name with
many cells near 1.0 is a 'multi-factor short' (the EW thesis); a name
with only the SI-related cells lit is a 'naive crowded-short' play.

Output: reports/current_short_basket.csv + .md (the markdown view is
designed to slot directly into the README).
"""

from __future__ import annotations

import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger

# (raw_rank_column, friendly_label, invert?)
# If invert is True, we report (1 - rank) so that "high column value = more
# shortable" across the board.
FACTORS: list[tuple[str, str, bool]] = [
    ("ShortPct_rk", "SI %", False),
    ("si_z_12m_rk", "SI z", False),
    ("mom_3m_rk", "3m-mom (inv)", True),
    ("vol_1m_rk", "vol", False),
    ("pe_rk", "P/E", False),
    ("fcf_yield_rk", "FCF-yld (inv)", True),
    ("roe_rk", "ROE (inv)", True),
    ("debt_equity_rk", "D/E", False),
    ("revenue_growth_yoy_rk", "rev-gth (inv)", True),
]

# How many names to show in the markdown.
TOP_N = 15


def main() -> int:
    settings.ensure_dirs()
    feat = read_parquet(settings.processed_dir / "features_monthly.parquet")
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")

    as_of = feat["Date"].max()
    latest_feat = feat[feat["Date"] == as_of].copy()
    latest_oof = oof[oof["Date"] == as_of].copy()
    # features_monthly already carries investable + mktCap from assemble, so
    # we can apply the gate directly without re-merging master_clean.

    # Investability gate -- match the backtest constructor.
    inv = latest_feat[latest_feat["investable"].fillna(False).astype(bool)].copy()
    inv = inv[inv["mktCap"].fillna(0) >= 100_000_000]
    logger.info(f"current_short_basket: as-of={as_of.date()}, "
                f"investable universe size = {len(inv)}")

    # Pivot OOF scores wide (one column per model).
    score_w = latest_oof.pivot_table(
        index="Ticker", columns="model", values="score",
    ).reset_index().rename(columns={
        "naive": "score_naive", "ew": "score_ew", "logit": "score_logit",
    })
    inv = inv.merge(score_w, on="Ticker", how="left")

    # Build the polarity-aware factor breakdown.
    for raw_col, label, invert in FACTORS:
        if raw_col not in inv.columns:
            inv[label] = float("nan")
            continue
        val = pd.to_numeric(inv[raw_col], errors="coerce").fillna(0.5)
        inv[label] = (1.0 - val) if invert else val

    # Sort by **naive score** (the most successful non-benchmark strategy
    # per Table 1 in the README). naive's score is rank-percentile of
    # ShortPct, so top of the naive list is the highest reported short
    # interest -- in line with "what's being shorted".
    top = inv.sort_values("score_naive", ascending=False).head(TOP_N).copy()

    # Composite EW score from the polarity-aware factors -- the bearish-
    # consensus metric. Mean across the 9 factor columns we just built.
    factor_labels = [lbl for _, lbl, _ in FACTORS]
    top["EW factor avg"] = top[factor_labels].mean(axis=1).round(3)

    # Pretty columns for the markdown.
    pretty = top[[
        "Ticker", "Company",
    ] + ([] if "Company" not in top.columns else [])].copy()  # placeholder

    keep = ["Ticker", "Company", "mktCap", "ShortPct",
            "score_naive", "score_ew", "score_logit",
            *factor_labels, "EW factor avg"]
    keep = [c for c in keep if c in top.columns]
    out = top[keep].copy()
    out["mktCap (A$m)"] = (out["mktCap"] / 1e6).round(0)

    # Drop raw mktCap, reorder. Round scores to 0.000 / factors to 0.00.
    final_cols = ["Ticker", "Company", "mktCap (A$m)", "ShortPct",
                  "score_naive", "score_ew", "score_logit",
                  *factor_labels, "EW factor avg"]
    final_cols = [c for c in final_cols if c in out.columns]
    out = out[final_cols].reset_index(drop=True)
    out.insert(0, "#", range(1, len(out) + 1))

    csv_path = settings.reports_dir / "current_short_basket.csv"
    out.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Markdown: hand-rolled (no tabulate dep).
    md_lines = [
        f"# Current short basket — what's being shorted, and why",
        "",
        f"_As of {as_of.date()} (last ASIC release of the month). "
        "Ranked by **naive score** (the highest-Sharpe non-benchmark "
        "strategy: rank by reported short interest). Per-factor "
        "polarity-aware ranks alongside show **why** each name is "
        "shortable across the EW composite's 12 signals -- every cell "
        "is 0-1 where 1 = most shortable on that factor._",
        "",
        "**Score columns:**",
        "* `score_naive` -- rank of `ShortPct` across the cross-section "
        "(higher = more crowded short).",
        "* `score_ew` -- polarity-aware equal-weight composite of 12 ranks "
        "(higher = bearish across many dimensions).",
        "* `score_logit` -- L2 logistic regression Pr(monthly return < 0).",
        "",
        "**Factor columns (all 0-1, higher = more shortable; `(inv)` = "
        "naturally bullish raw rank flipped via `1 - rank`):**",
        "* `SI %` -- raw short-interest %.",
        "* `SI z` -- short interest z-score vs 12-month history.",
        "* `3m-mom (inv)` -- low 3-month momentum.",
        "* `vol` -- 1-month realised volatility.",
        "* `P/E` -- expensive valuation.",
        "* `FCF-yld (inv)` -- low free cash flow yield.",
        "* `ROE (inv)` -- low return on equity.",
        "* `D/E` -- high leverage.",
        "* `rev-gth (inv)` -- low / negative revenue growth.",
        "",
    ]
    # Header + rows. Format scores 0.000, factors 0.00, mktCap 0.
    score_cols = {"score_naive", "score_ew", "score_logit", "EW factor avg"}
    factor_cols = set(factor_labels)
    cols = list(out.columns)
    hdr = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    md_lines.append(hdr)
    md_lines.append(sep)
    for _, row in out.iterrows():
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if c == "#":
                cells.append(str(int(v)))
            elif c == "Ticker":
                cells.append(str(v))
            elif c == "Company":
                comp = "" if pd.isna(v) else str(v).replace(" ORDINARY", "").title()
                cells.append(comp)
            elif c == "mktCap (A$m)":
                cells.append("" if pd.isna(v) else f"{v:,.0f}")
            elif c == "ShortPct":
                cells.append("" if pd.isna(v) else f"{v:.2f}")
            elif c in score_cols:
                cells.append("" if pd.isna(v) else f"{v:.3f}")
            elif c in factor_cols:
                cells.append("" if pd.isna(v) else f"{v:.2f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        md_lines.append("| " + " | ".join(cells) + " |")

    md_lines += [
        "",
        "### How to read this table",
        "",
        "* **A row of mostly dark cells (cells near 1.0)** = the name is "
        "shortable across multiple dimensions. These are the kind of "
        "multi-factor shorts the EW composite is built to find -- "
        "crowded SI + falling momentum + low quality + high leverage.",
        "* **A row with high SI columns but low fundamentals columns** = "
        "a 'pure crowded-short' play. Naive will rank it highly because "
        "of `score_naive` (just SI rank), but the EW composite will "
        "weight it less if its quality / valuation / momentum aren't "
        "also bearish.",
        "* **A row with low `EW factor avg` despite a top-15 spot** "
        "means naive is the only model pushing the name onto the list. "
        "These are the names most exposed to squeeze risk -- where the "
        "broad market doesn't share the consensus short view.",
        "",
    ]

    md_path = settings.reports_dir / "current_short_basket.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
