"""Panel validation and conservative cleaning.

Once the weekly point-in-time panel is assembled (``master_pit.parquet``), we
run a battery of sanity checks against it before any feature is built on top.
The philosophy mirrors the sister ``qfr`` project: keep the panel layout
intact, and only *null out* the specific cells that are demonstrably wrong
rather than dropping whole rows or whole symbols. That way the membership
mask stays honest and downstream coverage metrics keep meaning.

Two checks run today:

* **No look-ahead** — every fundamental row must carry a non-negative
  ``filing_lag_days``. A negative value means we joined a filing whose
  ``acceptedDate`` post-dates the rebalance week, which is a contamination
  bug in the as-of merge.
* **Corrupted weekly returns** — single (symbol, week) cells whose one-week
  return exceeds a generous threshold (150% default) are almost always
  unadjusted splits, halted-stock typos or vendor glitches, not real moves.
  Those cells get their prices/returns nulled and the row is flagged.

Outputs ``master_clean.parquet`` — the canonical analytics panel for the
feature, model and portfolio layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from short_king.utils.config import settings
from short_king.utils.io import write_parquet
from short_king.utils.logging import logger

# --- Column conventions ---------------------------------------------------
# Price + return columns we treat as "tainted" when a cell is flagged. Any
# column matching one of these names (or the ret_fwd_*w / ret_*w family) is
# nulled in ``conservative_clean``.
_PRICE_COL = "adjClose"
_RETURN_PREFIXES: tuple[str, ...] = ("ret_1w", "ret_fwd_", "ret_")


def _return_columns(panel: pd.DataFrame) -> list[str]:
    """Return columns currently present on the panel that look like returns."""
    cols: list[str] = []
    for c in panel.columns:
        if c == "ret_corr" or c == "ret_corruption":
            # belt-and-braces: skip diagnostic columns that happen to start with 'ret_'
            continue
        if any(c == p or c.startswith(p) for p in _RETURN_PREFIXES):
            cols.append(c)
    return cols


# --- Look-ahead audit -----------------------------------------------------
def check_no_lookahead(panel: pd.DataFrame) -> dict:
    """Assert ``filing_lag_days >= 0`` for every row that carries a filing.

    A negative lag means the as-of merge attached a filing accepted *after*
    the rebalance date — i.e. point-in-time has been violated. Rows without a
    filing (``has_fundamentals == False``) are skipped, since their lag is
    undefined.
    """
    if "filing_lag_days" not in panel.columns:
        logger.warning("check_no_lookahead: panel is missing 'filing_lag_days' column")
        return {"n_rows": int(len(panel)), "n_violations": 0, "max_violation_days": 0.0}

    lag = pd.to_numeric(panel["filing_lag_days"], errors="coerce")
    has_lag = lag.notna()
    violations = has_lag & (lag < 0)
    n_violations = int(violations.sum())
    max_violation = float(-lag[violations].min()) if n_violations else 0.0

    if n_violations:
        sample = panel.loc[violations, ["date", "symbol", "filing_lag_days"]].head(5)
        logger.warning(
            f"check_no_lookahead: {n_violations:,} rows with filing_lag_days < 0 "
            f"(max violation = {max_violation:.0f} days). Sample:\n{sample}"
        )
    else:
        logger.info(f"check_no_lookahead: clean ({int(has_lag.sum()):,} dated rows, no violations)")

    return {
        "n_rows": int(len(panel)),
        "n_violations": n_violations,
        "max_violation_days": max_violation,
    }


# --- Corrupted weekly return detection ------------------------------------
def detect_corrupted_series(
    panel: pd.DataFrame,
    *,
    max_abs_weekly_return: float = 1.5,
) -> pd.DataFrame:
    """Flag ``(symbol, week)`` cells whose one-week return looks impossible.

    A 150% one-week move is almost never a real return — it is usually an
    unadjusted split, a halted-then-reopened tick, or a vendor data typo. We
    compute ``ret_1w`` on the fly from ``adjClose`` (rather than trusting any
    pre-existing column) so the detector remains independent of upstream
    return-engineering choices.

    Returns a long DataFrame ``[date, symbol, adjClose, ret_1w, reason]`` of
    the suspect rows; never mutates the input.
    """
    required = {"date", "symbol", _PRICE_COL}
    missing = required - set(panel.columns)
    if missing:
        logger.warning(f"detect_corrupted_series: panel missing columns {sorted(missing)}; nothing flagged")
        return pd.DataFrame(columns=["date", "symbol", _PRICE_COL, "ret_1w", "reason"])

    px = panel[["date", "symbol", _PRICE_COL]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)

    prev = px.groupby("symbol")[_PRICE_COL].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = px[_PRICE_COL] / prev - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    px["ret_1w"] = ret

    mask = ret.abs() > float(max_abs_weekly_return)
    suspects = px.loc[mask].copy()
    suspects["reason"] = f"abs_weekly_return>{max_abs_weekly_return:g}"
    suspects = suspects.reset_index(drop=True)

    if not suspects.empty:
        logger.warning(
            f"detect_corrupted_series: {len(suspects):,} suspect (symbol, week) cells "
            f"(threshold = {max_abs_weekly_return:.2f}); {suspects['symbol'].nunique()} symbols affected"
        )
    else:
        logger.info(
            f"detect_corrupted_series: no cells exceed |ret_1w| > {max_abs_weekly_return:.2f}"
        )

    return suspects


# --- Conservative cleaning ------------------------------------------------
def conservative_clean(panel: pd.DataFrame, suspects: pd.DataFrame) -> pd.DataFrame:
    """Null prices/returns in flagged cells; keep the rest of the row intact.

    We do *not* drop the rows — the membership mask is what makes the panel
    point-in-time honest, so removing rows would silently distort universe
    coverage. Instead we null the price and any return column for the cell,
    add ``suspect_week`` and ``excluded_corrupted`` boolean flags, and rebuild
    ``has_price`` and ``investable`` so downstream filters do the right thing.
    """
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])

    # Always present, even when no suspects -> downstream code can rely on them.
    out["suspect_week"] = False
    out["excluded_corrupted"] = False

    if suspects is not None and not suspects.empty:
        keys = pd.MultiIndex.from_arrays(
            [pd.to_datetime(suspects["date"]).to_numpy(), suspects["symbol"].to_numpy()]
        )
        idx = pd.MultiIndex.from_arrays([out["date"].to_numpy(), out["symbol"].to_numpy()])
        bad = idx.isin(keys)
        out.loc[bad, "suspect_week"] = True
        out.loc[bad, "excluded_corrupted"] = True

        # Null the price and every return column for the flagged cells. Keep
        # the membership, sector and fundamental fields — those are still valid.
        null_cols = [c for c in (_PRICE_COL, *_return_columns(out)) if c in out.columns]
        for col in null_cols:
            out.loc[bad, col] = np.nan

        n_bad = int(bad.sum())
        logger.info(
            f"conservative_clean: nulled {len(null_cols)} columns across {n_bad:,} cells "
            f"({suspects['symbol'].nunique()} symbols)"
        )
    else:
        logger.info("conservative_clean: no suspect cells to null")

    # Refresh derived flags. ``has_price`` is the canonical price-coverage flag;
    # ``investable`` mirrors the rule from the assembly step.
    if _PRICE_COL in out.columns:
        out["has_price"] = out[_PRICE_COL].notna()

    if {"has_price", "has_fundamentals", "fresh_filing"} <= set(out.columns):
        out["investable"] = (
            out["has_price"]
            & out["has_fundamentals"].astype(bool)
            & out["fresh_filing"].astype(bool)
            & ~out["excluded_corrupted"]
        )
    elif "has_price" in out.columns:
        # Minimal fallback when the panel doesn't carry fundamentals flags.
        out["investable"] = out["has_price"] & ~out["excluded_corrupted"]

    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


# --- Persistence ----------------------------------------------------------
def write_clean(panel: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the cleaned panel to ``data/processed/master_clean.parquet``."""
    path = path or (settings.processed_dir / "master_clean.parquet")
    write_parquet(panel, path)
    logger.info(f"write_clean: wrote {path} ({len(panel):,} rows, {panel.shape[1]} cols)")
    return path


# --- Quality report -------------------------------------------------------
def quality_report(panel_raw: pd.DataFrame, panel_clean: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side coverage summary of the raw vs cleaned panel.

    Useful as a one-glance sanity check after a re-run: the row count must be
    identical (we never drop rows), price coverage should drop only by the
    suspect-cell count, and the investable count should drop by the same
    amount minus rows that were already excluded for other reasons.
    """

    def _row(p: pd.DataFrame) -> dict[str, float | int]:
        n = int(len(p))
        n_symbols = int(p["symbol"].nunique()) if "symbol" in p.columns else 0
        n_weeks = int(p["date"].nunique()) if "date" in p.columns else 0
        has_price = int(p["has_price"].sum()) if "has_price" in p.columns else 0
        has_fund = int(p["has_fundamentals"].sum()) if "has_fundamentals" in p.columns else 0
        fresh = int(p["fresh_filing"].sum()) if "fresh_filing" in p.columns else 0
        investable = int(p["investable"].sum()) if "investable" in p.columns else 0
        suspect = int(p["suspect_week"].sum()) if "suspect_week" in p.columns else 0
        excluded = int(p["excluded_corrupted"].sum()) if "excluded_corrupted" in p.columns else 0
        return {
            "n_rows": n,
            "n_symbols": n_symbols,
            "n_weeks": n_weeks,
            "n_has_price": has_price,
            "n_has_fundamentals": has_fund,
            "n_fresh_filing": fresh,
            "n_investable": investable,
            "n_suspect_week": suspect,
            "n_excluded_corrupted": excluded,
            "pct_investable": (investable / n) if n else 0.0,
        }

    raw_row = _row(panel_raw)
    clean_row = _row(panel_clean)
    delta_row = {
        k: (clean_row[k] - raw_row[k]) if isinstance(raw_row[k], (int, float)) else None
        for k in raw_row
    }
    report = pd.DataFrame([raw_row, clean_row, delta_row], index=["raw", "clean", "delta"])

    logger.info(
        f"quality_report: investable {raw_row['n_investable']:,} -> {clean_row['n_investable']:,} "
        f"(Δ {delta_row['n_investable']:+,}); "
        f"flagged {clean_row['n_suspect_week']:,} suspect cells"
    )
    return report


__all__ = [
    "check_no_lookahead",
    "detect_corrupted_series",
    "conservative_clean",
    "write_clean",
    "quality_report",
]
