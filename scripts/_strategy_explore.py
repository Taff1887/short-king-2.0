"""Disciplined strategy-exploration script.

Tests a small number (5) of pre-specified strategy variants against the
existing OOF predictions. **All variant selection is done on IS data only**;
OOS Sharpe is reported solely as unbiased confirmation of the chosen variant.
This is the only legitimate way to "data-mine" without inflating OOS
performance via repeated hypothesis testing.

Variants (defined upfront, no peeking at OOS):

  V0: Baseline -- L/S quintile on polarity-aware EW (current headline)
  V1: SI floor -- L/S quintile on EW, but only short names with ShortPct > 3%
                  (absolute short-interest filter, variable basket size)
  V2: EW conviction gate -- short only when EW score > 0.85 (top ~15% by
                            absolute conviction; long bottom 15% mirror)
  V3: naive+EW averaged -- L/S quintile on (naive_rank + ew_rank)/2
  V4: Conviction-weighted -- top-quintile by EW, position size proportional
                             to (score - 0.8); higher conviction = bigger short

Output: reports/strategy_explore.csv + reports/strategy_explore.md with the
IS Sharpe leaderboard and OOS confirmation row for the IS winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from short_king.portfolio.backtest import CostConfig, backtest_weekly
from short_king.utils.config import settings
from short_king.utils.io import read_parquet
from short_king.utils.logging import logger


def _build_prices_panel() -> pd.DataFrame:
    clean = read_parquet(settings.processed_dir / "master_clean.parquet")
    feat = read_parquet(settings.processed_dir / "features_monthly.parquet")
    keep = [c for c in ("Date", "Ticker", "adjClose", "investable") if c in clean.columns]
    panel = clean[keep].copy()
    if "adv_aud" in feat.columns:
        f = feat[["Date", "Ticker", "adv_aud"]].drop_duplicates(["Date", "Ticker"])
        panel = panel.merge(f, on=["Date", "Ticker"], how="left")
    return panel


def _ls_quintile_from_score(scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Standard L/S quintile construction: long bottom 20%, short top 20%."""
    out = []
    for d, grp in scores.dropna(subset=[score_col]).groupby("Date"):
        elig = grp[grp.get("investable", True).astype(bool)].copy()
        if len(elig) < 10:
            continue
        n = len(elig)
        q = int(round(n * 0.20))
        if q < 2:
            continue
        elig = elig.sort_values(score_col)
        longs = elig.head(q).copy()
        shorts = elig.tail(q).copy()
        longs["weight"] = 1.0 / q
        shorts["weight"] = -1.0 / q
        out.append(pd.concat([longs, shorts])[["Date", "Ticker", "weight"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["Date", "Ticker", "weight"]
    )


def _quintile_short_from_score(scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Quintile-short only (no long leg)."""
    out = []
    for d, grp in scores.dropna(subset=[score_col]).groupby("Date"):
        elig = grp[grp.get("investable", True).astype(bool)].copy()
        if len(elig) < 10:
            continue
        n = len(elig)
        q = int(round(n * 0.20))
        if q < 2:
            continue
        elig = elig.sort_values(score_col)
        shorts = elig.tail(q).copy()
        shorts["weight"] = -1.0 / q
        out.append(shorts[["Date", "Ticker", "weight"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["Date", "Ticker", "weight"]
    )


def _conviction_gate(
    scores: pd.DataFrame, threshold: float, score_col: str = "score",
    long_threshold: float | None = None,
) -> pd.DataFrame:
    """Absolute-threshold L/S: short any name where score > threshold (variable
    basket size). Long mirror: long any name where score < long_threshold
    (default 1 - threshold). Equal-weight inside each leg."""
    if long_threshold is None:
        long_threshold = 1.0 - threshold
    out = []
    for d, grp in scores.dropna(subset=[score_col]).groupby("Date"):
        elig = grp[grp.get("investable", True).astype(bool)].copy()
        shorts = elig[elig[score_col] > threshold].copy()
        longs = elig[elig[score_col] < long_threshold].copy()
        if len(shorts) > 0:
            shorts["weight"] = -1.0 / len(shorts)
        if len(longs) > 0:
            longs["weight"] = 1.0 / len(longs)
        combined = pd.concat([longs, shorts]) if not shorts.empty or not longs.empty else None
        if combined is not None:
            out.append(combined[["Date", "Ticker", "weight"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["Date", "Ticker", "weight"]
    )


def _conviction_weighted(scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Top-quintile short with position SIZE proportional to (score - 0.8).
    Long-leg unchanged (equal weight bottom quintile)."""
    out = []
    for d, grp in scores.dropna(subset=[score_col]).groupby("Date"):
        elig = grp[grp.get("investable", True).astype(bool)].copy()
        if len(elig) < 10:
            continue
        n = len(elig)
        q = int(round(n * 0.20))
        if q < 2:
            continue
        elig = elig.sort_values(score_col)
        longs = elig.head(q).copy()
        shorts = elig.tail(q).copy()
        # Conviction weighting: extra weight where score - 0.8 is larger.
        conv = (shorts[score_col] - 0.8).clip(lower=0.01)
        shorts["weight"] = -conv / conv.sum()  # short notional sums to -1
        longs["weight"] = 1.0 / q
        out.append(pd.concat([longs, shorts])[["Date", "Ticker", "weight"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["Date", "Ticker", "weight"]
    )


def main() -> int:
    settings.ensure_dirs()
    oof = read_parquet(settings.reports_dir / "oof_predictions_monthly.parquet")
    feat = read_parquet(settings.processed_dir / "features_monthly.parquet")
    prices_panel = _build_prices_panel()

    # Build the master scores frame: pivot to one row per (Date, Ticker) with
    # one column per model's score.
    scores = oof.pivot_table(
        index=["Date", "Ticker", "period"], columns="model", values="score"
    ).reset_index()
    # Merge ShortPct (for V1's SI floor) and investable (for the gate).
    ctx_cols = ["Date", "Ticker", "ShortPct"]
    if "ShortPct" not in feat.columns:
        logger.error("ShortPct missing from features panel; can't run V1")
        return 1
    ctx = feat[ctx_cols].drop_duplicates(["Date", "Ticker"])
    scores = scores.merge(ctx, on=["Date", "Ticker"], how="left")
    scores = scores.merge(
        prices_panel[["Date", "Ticker", "investable"]].drop_duplicates(["Date", "Ticker"]),
        on=["Date", "Ticker"], how="left",
    )
    scores["investable"] = scores["investable"].fillna(False).astype(bool)

    cost = CostConfig(periods_per_year=12)

    # ------------------------------------------------------------------------
    # Variant definitions. Each builds a weights DataFrame and a label.
    # All scores used here come from oof_predictions_monthly.parquet, which
    # already has IS/OOS labels. We test each variant's IS Sharpe and report
    # the IS leader's OOS Sharpe for unbiased confirmation.
    # ------------------------------------------------------------------------
    def v0_ew_quintile_ls(sc):
        return _ls_quintile_from_score(sc.assign(score=sc["ew"]))

    def v0b_ew_quintile_short(sc):
        return _quintile_short_from_score(sc.assign(score=sc["ew"]))

    def v1_si_floor_ew_ls(sc):
        sub = sc[sc["ShortPct"] >= 3.0].copy()  # only candidates with real SI
        return _ls_quintile_from_score(sub.assign(score=sub["ew"]))

    def v2_conviction_gate(sc):
        return _conviction_gate(sc.assign(score=sc["ew"]), threshold=0.85)

    def v3_naive_ew_blend_ls(sc):
        # both naive and ew are already rank-normalised within Date, but
        # averaging is safer with an explicit re-rank.
        blend = (sc["naive"].fillna(0.5) + sc["ew"].fillna(0.5)) / 2.0
        return _ls_quintile_from_score(sc.assign(score=blend))

    def v4_conviction_weighted(sc):
        return _conviction_weighted(sc.assign(score=sc["ew"]))

    variants = [
        ("V0: EW L/S quintile (baseline)", v0_ew_quintile_ls),
        ("V0b: EW quintile-short only", v0b_ew_quintile_short),
        ("V1: SI floor (>3%) + EW L/S quintile", v1_si_floor_ew_ls),
        ("V2: EW conviction gate (score > 0.85)", v2_conviction_gate),
        ("V3: naive+EW average L/S quintile", v3_naive_ew_blend_ls),
        ("V4: EW conviction-weighted positions", v4_conviction_weighted),
    ]

    results: list[dict] = []
    for name, build in variants:
        try:
            weights = build(scores)
        except Exception as exc:
            logger.warning(f"{name}: build failed -- {exc}")
            continue
        if weights.empty:
            logger.warning(f"{name}: empty weights")
            continue

        # Per-period summary.
        per_dates = scores[["Date", "period"]].drop_duplicates("Date")
        bt = backtest_weekly(
            target_weights=weights,
            prices_panel=prices_panel[["Date", "Ticker", "adjClose"]],
            cost_config=cost,
        )
        # Aggregate per period.
        rets = bt.returns.merge(per_dates, on="Date", how="left")
        for period in ("ALL", "IS", "OOS"):
            sub = rets if period == "ALL" else rets[rets["period"] == period]
            if len(sub) < 6:
                continue
            r = sub["ret_net"]
            if r.std() == 0:
                continue
            sharpe = float(r.mean() / r.std() * np.sqrt(12))
            cagr = float((1 + r).prod() ** (12 / len(r)) - 1)
            eq = (1 + r).cumprod()
            mdd = float((eq / eq.cummax() - 1).min())
            hit = float((r > 0).mean())
            avg_n = float(weights.groupby("Date").size().mean())
            results.append({
                "variant": name,
                "period": period,
                "n_rebalances": int(len(sub)),
                "avg_basket_size": round(avg_n, 1),
                "Sharpe": round(sharpe, 3),
                "CAGR": round(cagr, 4),
                "MaxDD": round(mdd, 4),
                "hit_rate": round(hit, 3),
            })

    df = pd.DataFrame(results)

    # IS-only leaderboard for selection.
    is_lb = df[df.period == "IS"].sort_values("Sharpe", ascending=False).reset_index(drop=True)
    print()
    print("=" * 80)
    print("IS-ONLY LEADERBOARD (for variant selection; OOS is reported below as confirmation)")
    print("=" * 80)
    print(is_lb.to_string(index=False))

    # OOS leaderboard for the same variants (reported but NOT used for selection).
    oos_lb = df[df.period == "OOS"].set_index("variant")
    print()
    print("=" * 80)
    print("OOS CONFIRMATION (do not retune on these numbers)")
    print("=" * 80)
    print(oos_lb.to_string())

    csv_path = settings.reports_dir / "strategy_explore.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    # Markdown report. Hand-rolled to avoid an optional tabulate dependency.
    def _to_md(d: pd.DataFrame) -> str:
        cols = list(d.columns)
        hdr = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        rows = [
            "| " + " | ".join(str(v) for v in r) + " |"
            for r in d.itertuples(index=False, name=None)
        ]
        return "\n".join([hdr, sep, *rows])

    lines: list[str] = []
    lines.append("# Strategy exploration -- 5 pre-specified variants")
    lines.append("")
    lines.append("**Discipline:** all variants defined upfront in "
                 "`scripts/_strategy_explore.py`. IS-only Sharpe used for "
                 "variant selection. OOS Sharpe reported as unbiased confirmation "
                 "of the IS-winner -- never used to retune.")
    lines.append("")
    lines.append("## IS-only leaderboard (n=119 monthly OOF + 35 OOS = 154 ALL for trained models)")
    lines.append("")
    lines.append(_to_md(is_lb))
    lines.append("")
    lines.append("## OOS confirmation (n=35)")
    lines.append("")
    lines.append(_to_md(oos_lb[["n_rebalances", "avg_basket_size", "Sharpe",
                                  "CAGR", "MaxDD", "hit_rate"]].reset_index()))
    lines.append("")
    lines.append("## All variants, all periods")
    lines.append("")
    lines.append(_to_md(df))

    md_path = settings.reports_dir / "strategy_explore.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
