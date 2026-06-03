# Changelog — Short King v1.0 → v2.0

This document tracks the migration from the original
[Taff1887/short-king](https://github.com/Taff1887/short-king) one-notebook
prototype to a structured research project.

## v2.0 (this repo)

### Repository & engineering
- Migrated single 430 KB `Short King.ipynb` to an `src/short_king/` package
  with separate `data/`, `features/`, `models/`, `portfolio/`, `reporting/`,
  `utils/` submodules.
- Added a numbered `scripts/` pipeline (01–08), `tests/` (incl. an explicit
  no-look-ahead test), `docs/` (methodology + data dictionary), `charts/` and
  `reports/` outputs.
- Added `pyproject.toml` (uv-managed), `.env.example`, `.gitignore`,
  `.gitattributes`, MIT `LICENSE`, this `CHANGELOG.md`, and a structured
  `README.md`.
- API keys are loaded from a gitignored `.env` via `pydantic-settings`.
- The original notebook is preserved unchanged in `legacy/` so reviewers can
  compare before/after.

### Data
- **ASIC scraper:** rewritten with on-disk caching, retry, dual parser
  (`tabula-py` + `pdfplumber` fallback), explicit validation of every
  release, and structured logging.
- **Fundamentals + prices:** Yahoo replaced as the primary source with
  Financial Modeling Prep (Premium) — point-in-time lagged to filing date.
- **Cross-validation:** Yahoo retained as an independent cross-check (same
  pattern used in the sister `quant-factor-ranking` project).
- **Universe:** explicit ASX equity universe (ORDINARY filter + minimum
  market cap + liquidity floor) with a point-in-time membership snapshot per
  rebalance date. S&P 500 PIT universe added for robustness.

### Features
- Expanded from 5 price/SI-only signals to ~25 features across short,
  price, liquidity, valuation, quality, leverage, growth and sector groups.
- All cross-sectional ranks computed within each rebalance date.

### Modelling
- Three baselines: naive SI-percentile rank, equal-weight composite
  (z-score), and a reproduction of the v1.0 logistic regression.
- Advanced model: LightGBM cross-sectional ranker (LambdaRank) on
  fortnight-forward decile target.
- Walk-forward expanding-window CV with purge+embargo to avoid overlap-
  driven look-ahead.
- SHAP and gain-based feature importance committed to `charts/`.

### Portfolio & backtest
- Weekly rebalance, top-decile short basket as the primary book; a
  dollar-neutral long–short variant as a diagnostic.
- Costs modelled: 25 bps round-trip commission + 150 bps annualised
  borrow (configurable).
- Liquidity gate at A$200m market cap and minimum ADV.
- Sharpe, Sortino, max drawdown, hit rate, turnover, monthly heatmap.

### Reporting
- Universe coverage, SI distribution, feature correlation, model
  interpretability, cumulative-return curves with drawdowns, monthly
  returns heatmap, and a one-page tearsheet — all committed PNGs and CSVs.

## v1.0 (preserved in `legacy/`)

Original one-notebook prototype:
- 796 weekly ASIC PDFs scraped with `tabula`.
- 5 features (`mom_12w`, `si_percentile_52w`, `si_up_weeks`,
  `mom_si_interact`, `vol_4w`).
- LogisticRegression, single fit on first 400 weeks.
- Top-K weekly shorts, $1 per trade, 10% hard stop, 4-week hold.
- Reported: 21 trades, return on peak capital ≈ −3 %, total $ PnL ≈ −$0.18.
