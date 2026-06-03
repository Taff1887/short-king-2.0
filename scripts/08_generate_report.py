"""Assemble the final results writeup: tearsheets, RESULTS.md, and README patch.

Reads everything under ``reports/`` and ``charts/`` produced by steps 05-06,
re-renders a full tearsheet for each strategy via
:func:`short_king.reporting.tearsheet.render_tearsheet`, and emits two
artefacts:

* ``reports/RESULTS.md`` — markdown writeup with the headline metrics table
  and inline links to every generated chart (cumulative / drawdown / heatmap /
  SHAP).
* ``README.md`` — the project root README's "Results" placeholder line is
  replaced in-place with a condensed block (headline table + key chart links).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from short_king.reporting.tearsheet import render_tearsheet
from short_king.utils.config import PROJECT_ROOT, settings
from short_king.utils.io import read_parquet
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
    """GitHub-flavoured markdown table from a small DataFrame."""
    if df.empty:
        return "_(no rows)_"
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except Exception:
        # to_markdown requires tabulate; fall back to a hand-rolled pipe table.
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [
            "| " + " | ".join(f"{v}" for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *rows])


def _rel(path: Path) -> str:
    """Path relative to project root, POSIX-style for markdown."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _patch_readme(headline: str) -> bool:
    """Replace the README placeholder with ``headline``. Returns True if patched."""
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        logger.warning("README.md not found at project root — skipping patch.")
        return False
    text = readme.read_text(encoding="utf-8")
    if _README_PLACEHOLDER not in text:
        logger.info("README placeholder already replaced — leaving as-is.")
        return False
    new_text = text.replace(_README_PLACEHOLDER, headline)
    readme.write_text(new_text, encoding="utf-8")
    logger.info(f"patched README placeholder ({len(headline)} chars)")
    return True


def main() -> int:
    args = _parse_args()
    settings.ensure_dirs()

    summary_path = settings.reports_dir / "summary_table.csv"
    if not summary_path.exists():
        logger.error(f"{summary_path} not found — must run 06_backtest.py first")
        return 1

    t0 = dt.datetime.now()
    logger.info(f"08_generate_report: start {t0.isoformat(timespec='seconds')}")

    summary = pd.read_csv(summary_path)
    headline = summary.copy()
    # Order strategies by best Sharpe (column name varies; pick what's there).
    sort_col = next(
        (c for c in ("sharpe", "Sharpe", "ann_sharpe", "annual_sharpe") if c in headline.columns),
        None,
    )
    if sort_col:
        headline = headline.sort_values(sort_col, ascending=False)
    logger.info(f"summary_table loaded: {len(summary)} strategies")

    # Per-strategy tearsheets.
    tearsheet_paths: list[Path] = []
    for bt_path in sorted(settings.reports_dir.glob("backtest_*.parquet")):
        try:
            bt = read_parquet(bt_path)
        except Exception as exc:
            logger.warning(f"could not read {bt_path.name}: {exc}")
            continue
        stem = bt_path.stem.replace("backtest_", "")
        out = settings.reports_dir / f"tearsheet_{stem}.html"
        try:
            render_tearsheet(bt, out_path=out, title=stem)
            tearsheet_paths.append(out)
            logger.info(f"tearsheet -> {out.name}")
        except Exception as exc:
            logger.warning(f"render_tearsheet {stem}: {exc}")

    # RESULTS.md
    chart_files = sorted(settings.charts_dir.glob("*.png"))
    lines: list[str] = []
    lines.append("# Short King 2.0 — Results")
    lines.append("")
    lines.append(f"_Generated {dt.datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Headline strategies")
    lines.append("")
    lines.append(_md_table(headline))
    lines.append("")
    lines.append("## Tearsheets")
    lines.append("")
    for p in tearsheet_paths:
        lines.append(f"- [{p.stem}]({_rel(p)})")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for p in chart_files:
        lines.append(f"![{p.stem}]({_rel(p)})")
    lines.append("")

    results_md = settings.reports_dir / "RESULTS.md"
    results_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"wrote {results_md}")

    # README in-place patch with a condensed block.
    if not args.no_readme_patch:
        # Keep only the top 5 by Sharpe for the README to stay compact.
        compact = headline.head(5)
        readme_block_lines = [
            _md_table(compact),
            "",
            f"_Full writeup: [`reports/RESULTS.md`]({_rel(results_md)})._",
        ]
        # Include up to 3 representative charts.
        for p in chart_files[:3]:
            readme_block_lines.append("")
            readme_block_lines.append(f"![{p.stem}]({_rel(p)})")
        _patch_readme("\n".join(readme_block_lines))

    t1 = dt.datetime.now()
    logger.info(f"08_generate_report: done | took {(t1 - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
