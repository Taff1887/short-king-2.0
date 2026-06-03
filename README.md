# Short King 2.0 — ASX Short-Selling Research

> Cross-sectional short-selling model on the Australian Securities Exchange.
> Uses ASIC weekly short-position disclosures plus fundamentals/price/liquidity
> data to rank short candidates, with a costed weekly-rebalance backtest and a
> US S&P 500 robustness check.

**Status:** under construction (full results, charts and tables populate on the
first end-to-end pipeline run — see [How to reproduce](#how-to-reproduce)).

---

## What is this?

This is a from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)). The original
was a single 430KB Jupyter notebook with five hand-built signals and a logistic
regression that produced 21 trades and a negative return on peak capital.

**Version 2.0** is a proper research project:

| Dimension | v1.0 (original) | v2.0 (this repo) |
|---|---|---|
| Repo structure | one notebook | src-layout package + scripts + tests + docs |
| Data sources | ASIC PDFs + Yahoo Finance only | ASIC + Financial Modeling Prep (Premium) + Yahoo cross-check |
| Universe | implicit (all ASX shorts) | explicit point-in-time ASX equities + S&P 500 robustness |
| Features | 5 (all price/SI derived) | ~25 across short, price, liquidity, valuation, quality, leverage, growth |
| Model | one logit fit | baselines (naive, EW composite, logit) **and** LightGBM ranker, walk-forward CV with purge+embargo, SHAP |
| Look-ahead audit | none | explicit lag of fundamentals to filing date + test suite |
| Backtest | 21 trades, no costs, no Sharpe/MaxDD/turnover | weekly rebalance, top-decile short basket, borrow + commission costs, full performance attribution |
| Reporting | inline plots | publication-quality charts, monthly returns table, methodology doc, data dictionary |

See [`CHANGELOG.md`](CHANGELOG.md) for the full delta.

---

## Repo layout

```
short-king-2.0/
├── src/short_king/
│   ├── data/        ASIC scraper, FMP client, universe, PIT assembly
│   ├── features/    short signals, price, liquidity, valuation, quality, ...
│   ├── models/      baselines, LightGBM ranker, walk-forward CV, SHAP
│   ├── portfolio/   construction + backtest engine
│   ├── reporting/   charts + tearsheet
│   └── utils/       config, logging, IO
├── scripts/         pipeline entry points (numbered 01..08)
├── notebooks/       consolidated research narrative
├── tests/           pytest suite incl. no-lookahead checks
├── data/            raw + processed (gitignored)
├── charts/          committed PNGs that appear in this README
├── reports/         CSV summary tables (factor screens, backtest stats)
├── docs/            methodology.md, data_dictionary.md
└── legacy/          the original v1.0 notebook (preserved for comparison)
```

---

## How to reproduce

```bash
# 1. Clone and set up the environment
git clone https://github.com/Taff1887/short-king-2.0.git
cd short-king-2.0
cp .env.example .env       # then put your FMP_API_KEY in .env

# 2. Install (uv recommended)
uv sync --extra dev        # creates .venv/ and installs everything

# 3. Run the pipeline (each script is idempotent and caches)
uv run python scripts/01_pull_asic.py             # weekly ASIC short-position PDFs
uv run python scripts/02_pull_fmp_fundamentals.py # quarterly fundamentals
uv run python scripts/03_pull_fmp_prices.py       # adjusted-close history
uv run python scripts/04_build_features.py        # PIT panel + feature matrix
uv run python scripts/05_train_and_validate.py    # baselines + LightGBM + CV + SHAP
uv run python scripts/06_backtest.py              # portfolio + weekly backtest
uv run python scripts/07_robustness_us.py         # US S&P 500 robustness check
uv run python scripts/08_generate_report.py       # writes charts/ + reports/
```

Java (any version, for `tabula-py`) needs to be on `PATH` so ASIC PDFs can be
parsed; `pdfplumber` is included as a Java-free fallback.

---

## Methodology — one paragraph

For every Friday in the backtest window we observe ASIC's end-of-week
short-position report (lagged four business days from the as-of date — a
real-world tradability constraint) and combine it with point-in-time
fundamentals (lagged to the SEC/ASX filing date), price history and liquidity.
Stocks are scored cross-sectionally; the top-decile shorts form a basket. The
basket is rebalanced weekly, charged 25 bps round-trip plus an annualised
borrow cost, and benchmarked against an equal-weight short of the investable
universe and a long-S&P/ASX-200 reference. We compare baselines (naive
SI-percentile, equal-weight composite, logistic regression) against a LightGBM
cross-sectional ranker and report Sharpe, max drawdown, hit rate, turnover and
monthly returns. Full detail in [`docs/methodology.md`](docs/methodology.md).

---

## Results

> Charts and tables appear here once `scripts/08_generate_report.py` has run.

---

## Honest limitations

* ASIC short-position data is reported with a four-business-day lag — the
  backtest enters on the release date, not the as-of date, to reflect that.
* FMP fundamentals coverage is materially better for large-caps than small/
  mid-caps; the universe is gated on `min market cap = A$200m` to avoid the
  thinnest names where coverage gaps would dominate.
* Short borrow costs vary by name and time; we use a flat 150 bps annualised
  default and document sensitivity in the methodology doc.
* The backtest excludes capital-raise and short-squeeze dynamics that can
  dominate realised PnL for individual names.

---

## License

MIT — see [`LICENSE`](LICENSE).
