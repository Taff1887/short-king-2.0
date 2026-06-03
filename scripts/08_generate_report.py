"""Assemble the final results writeup.

Reads ``reports/model_metrics.csv`` (from 05) and ``reports/backtest_summary.csv``
(from 06; falls back to the canonical ``summary_table.csv`` the backtest
script actually emits), collects every PNG under ``charts/`` and writes:

* ``reports/RESULTS.md`` — markdown summary with the metrics tables and an
  inline image link per chart.
* an in-place edit of the top-level ``README.md`` "Results" section: the
  placeholder line is swapped for the top-decile-short row of the best model
  plus the cumulative-returns chart link.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from short_king.utils.config import PROJECT_ROOT, settings
from short_king.utils.logging import logger

_README_PLACEHOLDER = "> Charts and tables appear here once `scripts/08_generate_report.py` has run."


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--no-readme-patch",
        action="store_true",
        help="Skip the in-place README.md placeholder replacement.",
    )
    return p.parse_args()


def _md_table(df: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table from a small DataFrame.

    ``DataFrame.to_markdown`` requires the optional ``tabulate`` dependency;
    fall back to a hand-rolled pipe table if it is not installed.
    """
    if df.empty:
        return "_(no rows)_"
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except (ImportError, ModuleNotFoundError, ValueError):
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [
            "| " + " | ".join(_fmt_cell(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *rows])


def _fmt_cell(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _rel(path: Path) -> str:
    """Project-root-relative POSIX path for markdown links."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_backtest_summary() -> Path | None:
    """The task spec names this ``backtest_summary.csv``; 06 currently
    writes ``summary_table.csv``. Accept either."""
    for name in ("backtest_summary.csv", "summary_table.csv"):
        path = settings.reports_dir / name
        if path.exists():
            return path
    return None


def _best_decile_row(bt: pd.DataFrame) -> pd.DataFrame:
    """Pick the top-decile-short row for the best model.

    "Best" = highest Sharpe (column name varies across backtest outputs).
    Returns a 1-row DataFrame, or empty if no decile rows are present.
    """
    if "variant" in bt.columns:
        decile_mask = bt["variant"].astype(str).str.lower().str.contains("decile")
        decile_mask &= ~bt["variant"].astype(str).str.lower().str.contains("ls_")
        candidates = bt.loc[decile_mask]
    else:
        candidates = bt
    if candidates.empty:
        # Fall back to the whole table so we still surface *something*.
        candidates = bt
    sort_col = next(
        (c for c in ("Sharpe", "sharpe", "ann_sharpe", "annual_sharpe") if c in candidates.columns),
        None,
    )
    if sort_col:
        candidates = candidates.sort_values(sort_col, ascending=False)
    return candidates.head(1).reset_index(drop=True)


def _patch_readme(replacement_block: str) -> bool:
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        logger.warning("README.md not found at project root - skipping patch.")
        return False
    text = readme.read_text(encoding="utf-8")
    if _README_PLACEHOLDER not in text:
        logger.info("README placeholder already replaced - leaving as-is.")
        return False
    # Only the placeholder + the immediately-following blank line are touched.
    target = _README_PLACEHOLDER + "\n\n"
    if target in text:
        new_text = text.replace(target, replacement_block.rstrip() + "\n\n", 1)
    else:
        new_text = text.replace(_README_PLACEHOLDER, replacement_block.rstrip(), 1)
    readme.write_text(new_text, encoding="utf-8")
    logger.info(f"patched README placeholder ({len(replacement_block)} chars)")
    return True


def _pick_cum_chart(chart_files: list[Path]) -> Path | None:
    """Prefer a *_cum.png chart (cumulative-returns) for the README link."""
    for p in chart_files:
        if p.stem.endswith("_cum"):
            return p
    return chart_files[0] if chart_files else None


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    metrics_path = settings.reports_dir / "model_metrics.csv"
    if not metrics_path.exists():
        logger.error(f"{metrics_path} not found - must run 05_train_and_validate.py first")
        return 1
    bt_path = _resolve_backtest_summary()
    if bt_path is None:
        logger.error(
            f"neither backtest_summary.csv nor summary_table.csv found under "
            f"{settings.reports_dir} - must run 06_backtest.py first"
        )
        return 1

    t0 = dt.datetime.now()
    logger.info(f"08_generate_report: start {t0.isoformat(timespec='seconds')}")

    metrics_df = pd.read_csv(metrics_path)
    backtest_df = pd.read_csv(bt_path)
    logger.info(
        f"loaded {len(metrics_df)} model rows, {len(backtest_df)} backtest rows "
        f"({bt_path.name})"
    )

    chart_files = sorted(settings.charts_dir.glob("*.png"))
    logger.info(f"found {len(chart_files)} chart PNGs")

    # ---- RESULTS.md ----------------------------------------------------
    lines: list[str] = []
    lines.append("# short-king-2.0 - Results")
    lines.append("")
    lines.append(f"_Generated {dt.datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Model metrics (out-of-fold)")
    lines.append("")
    lines.append(_md_table(metrics_df))
    lines.append("")
    lines.append("## Backtest summary")
    lines.append("")
    lines.append(_md_table(backtest_df))
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    if chart_files:
        for p in chart_files:
            lines.append(f"![{p.stem}]({_rel(p)})")
    else:
        lines.append("_(no charts found under `charts/`)_")
    lines.append("")

    results_md = settings.reports_dir / "RESULTS.md"
    results_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"wrote {results_md}")

    # ---- README in-place patch ----------------------------------------
    if not args.no_readme_patch:
        headline = _best_decile_row(backtest_df)
        block_parts: list[str] = []
        if not headline.empty:
            block_parts.append(_md_table(headline))
        else:
            block_parts.append("_(no backtest rows available)_")
        cum_chart = _pick_cum_chart(chart_files)
        if cum_chart is not None:
            block_parts.append("")
            block_parts.append(f"![{cum_chart.stem}]({_rel(cum_chart)})")
        block_parts.append("")
        block_parts.append(f"_Full writeup: [`reports/RESULTS.md`]({_rel(results_md)})._")
        _patch_readme("\n".join(block_parts))

    t1 = dt.datetime.now()
    logger.info(f"08_generate_report: done | took {(t1 - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
