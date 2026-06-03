# Short King 2.0 — ASX Short-Selling Research

> A cross-sectional short-selling model on the Australian Securities Exchange.
> 500-stock universe over 260 weekly rebalances (~5 years), ASIC weekly
> short-position disclosures + FMP fundamentals, walk-forward CV with
> purge + embargo, costed weekly backtest of five models across two
> portfolio constructions.

This is a from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)). The original
was a single 430 KB Jupyter notebook with five hand-built signals, one
LogisticRegression fit, and 21 total trades with a −3 % return on peak
capital. **Version 2.0** is a proper research project — see the comparison
table and headline results below.

---

## Headline result

The best-Sharpe strategy on the 2021-06 → 2026-05 window:

| model | strategy | CAGR | Vol | **Sharpe** | Sortino | MaxDD | Calmar | Hit-rate | Turnover (1-way) | n_rebalances |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | long_short_decile | **+8.1 %** | 20.1 % | **0.49** | 0.72 | −18.3 % | 0.44 | 55.2 % | 25.1 % | 212 |
| logit | long_short_decile | +3.6 % | 15.1 % | 0.31 | 0.47 | −22.0 % | 0.16 | 52.7 % | 37.8 % | 91 |
| gbm_cls | long_short_decile | −1.8 % | 19.2 % | 0.00 | 0.00 | −39.3 % | −0.05 | 51.5 % | 92.6 % | 99 |
| ew | long_short_decile | −5.9 % | 18.7 % | −0.23 | −0.33 | −40.7 % | −0.14 | 42.9 % | 37.5 % | 212 |
| gbm_rank | long_short_decile | −22.7 % | 28.4 % | −0.76 | −1.02 | −56.8 % | −0.40 | 46.5 % | 70.4 % | 99 |

_Net of 25 bps round-trip commission, 1.5 % p.a. borrow on the short leg,
5 bps slippage on weight changes. Decile-short books are in
[`reports/RESULTS.md`](reports/RESULTS.md)._

![Cumulative growth of $1, dollar-neutral L/S deciles](charts/cumulative_returns.png)

**The honest read.** On this 5-year ASX window — meme-stock-and-AI-rally era,
high realised short-squeeze risk — *no model meaningfully beats the
naive-SI-rank baseline as a stand-alone short alpha*. The naive long-short
decile is the only strategy with positive Sharpe; the trained ML models
correctly *learn* the right cross-sectional sign (negative IC, see below)
but a 4-week label horizon plus a dollar-neutral construction is not enough
to overcome borrow + execution costs in this regime. That is itself a
research finding worth publishing, and §[Limitations](#limitations) lays out
which of these would change in a different period or with richer features.

---

## Information coefficients (out-of-fold)

What the model *thinks* about each stock, scored against the realised 4-week
forward total return. All five models are evaluated on the same walk-forward
folds (purge + 4-week embargo to match the label horizon):

| model | IC mean | IC t-stat | IC hit-rate | n periods | Decile-spread mean |
|---|---:|---:|---:|---:|---:|
| ew (equal-weight composite) | **+7.5 %** | **+9.12** | 78.2 % | 211 | −0.001 |
| naive (rank ShortPct) | +1.2 % | +2.13 | 54.0 % | 211 | −0.008 |
| gbm_cls (LightGBM classifier) | −3.0 % | −2.55 | 36.7 % | 98 | −0.004 |
| logit (rebuilt v1 baseline) | −4.3 % | −4.29 | 31.1 % | 90 | −0.007 |
| gbm_rank (LightGBM LambdaRank) | −4.5 % | −2.51 | 36.7 % | 98 | +0.010 |

**Reading the signs.** A positive IC means the model's "shortable" score
correlates with *higher* forward returns — exactly backwards for a short
book. The trained models (logit, gbm_cls, gbm_rank) all produce
*statistically significant **negative** ICs*, i.e. they correctly identify
underperformers. The naive and EW baselines have positive ICs because they
mechanically rank by characteristics that, in the 2021-2026 ASX regime,
turned out to predict *winners* — high short interest paid for the longs
(squeeze risk), and a blind equal-weight blend of value + quality + momentum
ranks is essentially a long signal. The fact that the *negative-IC* trained
models still don't make money on a dollar-neutral L/S construction tells us
the cross-sectional spread is real but thin, and that costs + borrow eat the
edge.

---

## v1 → v2 delta

| Dimension | v1.0 (original) | v2.0 (this repo) |
|---|---|---|
| Repo structure | one notebook | src-layout package + scripts + tests + docs |
| Data — short interest | ASIC PDFs (tabula) | ASIC PDFs (dual parser, cached, validated) |
| Data — prices / fundamentals | Yahoo Finance | **Financial Modeling Prep** (Premium); Yahoo cross-check |
| Universe | implicit, all ASX | explicit point-in-time, top-500 by frequency, A$200m mcap gate |
| Features | 5 (price + SI only) | **~25 raw → 562 cross-sectional ranks** across short, price, liquidity, valuation, quality, leverage, growth |
| Target | binary `y_down = 4w_ret < 0` | + cross-sectional decile-rank target for LambdaRank |
| Cross-validation | single 400-week train / forward test | **walk-forward expanding window, purge + 4-week embargo, 24 folds** |
| Models | one logit | naive, EW composite, logit, LightGBM classifier, LightGBM LambdaRank |
| Interpretability | none | SHAP + gain-based feature importance, calibration table |
| Look-ahead audit | none | explicit lag of fundamentals to `acceptedDate` + unit test |
| Backtest | 21 trades, no costs, ad hoc | **weekly rebalance**, 10 (model × strategy) variants, 25 bps round-trip + 1.5 % p.a. borrow + 5 bps slippage |
| Metrics | total $ PnL only | CAGR, Vol, **Sharpe, Sortino, MaxDD, Calmar**, hit-rate, turnover, monthly heatmap |
| Reporting | inline plots | **9 publication-quality PNGs** + CSV summary + `reports/RESULTS.md` + methodology + data dictionary |
| Reproducibility | none | uv lockfile, deterministic on-disk caches (ASIC + FMP), 38-test pytest suite |

Full granular delta: [`CHANGELOG.md`](CHANGELOG.md). Original notebook preserved
unchanged in [`legacy/original_notebook.ipynb`](legacy/original_notebook.ipynb).

---

## What's in this repo

```
short-king-2.0/
├── src/short_king/
│   ├── data/         FMP client, ASIC scraper, prices, fundamentals,
│   │                 yahoo cross-check, ASX/SP500 universes, PIT assembly + clean
│   ├── features/     short signals, price, liquidity, valuation, quality,
│   │                 leverage/growth + cross-sectional rank orchestrator
│   ├── models/       baselines (naive / EW / logit), LightGBM clf + LambdaRank,
│   │                 walk-forward CV (purge + embargo), SHAP, IC metrics
│   ├── portfolio/    top-K / decile / long-short construction; weekly backtest
│   │                 with costs + borrow + slippage; Sharpe/Sortino/MaxDD/Calmar
│   ├── reporting/    publication-quality charts + tearsheet
│   └── utils/        config (pydantic-settings), loguru, IO, dates
├── scripts/          01..08 pipeline + utility helpers
├── tests/            38 pytest assertions (utils, features, models, no-lookahead, backtest math)
├── docs/             methodology.md, data_dictionary.md
├── charts/           publication PNGs (committed)
├── reports/          backtest_*.parquet, model_metrics.csv, RESULTS.md, oof_predictions.parquet
└── legacy/           original v1 notebook (preserved)
```

---

## Methodology — one paragraph

For every Friday in the 2021-06 → 2026-05 window we observe ASIC's
end-of-week short-position report (lagged 4 business days to the release
date — a real-world tradability constraint), and combine it with
point-in-time fundamentals (lagged to the FMP `acceptedDate`), adjusted
prices and liquidity measures into one ~25-feature panel. Every numeric
factor is converted to a within-date percentile rank so the modelling layer
sees a uniform [0, 1] distribution per Friday. Targets are (a) the binary
`fwd_ret_4w < 0` and (b) the cross-sectional decile rank of `fwd_ret_4w`.
Models are walk-forward fit with a 156-week minimum train window, 4-week
test, 4-week embargo (matching the label horizon). The top-decile short and
dollar-neutral long-short books are rebalanced weekly and charged
commission + borrow + slippage. Full detail:
[`docs/methodology.md`](docs/methodology.md).

---

## Charts

| | |
|---|---|
| ![Universe coverage](charts/universe_coverage.png) | ![Short-interest distribution](charts/si_distribution.png) |
| **Universe coverage** by Friday. Top-500 most-shorted tickers across 5 years. | **Short-interest distribution** — pooled across the panel. |
| ![Feature correlation](charts/feature_correlation.png) | ![Feature distributions](charts/feature_distributions.png) |
| **Cross-feature correlation** (rank columns) — confirms low collinearity. | **Feature distributions** — uniform after cross-sectional ranking. |
| ![Decile returns](charts/decile_returns.png) | ![Top short candidates](charts/top_short_candidates.png) |
| **OOF mean 4-week forward return by score decile.** Negative-slope = working short signal. | **Top short candidates** on the latest panel Friday (model = logit). |
| ![Cumulative](charts/cumulative_returns.png) | ![Drawdowns](charts/drawdowns.png) |
| **Cumulative $1** across the 10 (model × strategy) backtests. | **Underwater plot** for the best (naive L/S decile). |
| ![Monthly heatmap](charts/monthly_heatmap.png) | |
| **Monthly returns** — best strategy. | |

### Model interpretability — what's the GBM picking up?

Top SHAP features for the full-sample LightGBM classifier
(higher mean-|SHAP| = stronger contribution to the score):

| Rank | Feature | Mean &#124;SHAP&#124; | Family |
|---:|---|---:|---|
| 1 | `mom_4w_rk` | 0.109 | momentum |
| 2 | `asset_growth_yoy_rk` | 0.098 | growth (textbook short signal) |
| 3 | `drawdown_52w_rk` | 0.093 | price / risk |
| 4 | `balance_sheet_preferredStock_rk` | 0.092 | leverage / dilution |
| 5 | `mom_1w_rk` | 0.089 | short-term reversal |
| 6 | `mom_12w_skip1_rk` | 0.087 | momentum (skip-1w) |
| 7 | `mom_12w_rk` | 0.078 | momentum |
| 8 | `cash_flow_accountsPayables_rk` | 0.070 | working capital |
| 9 | `mom_26w_rk` | 0.065 | medium-term momentum |
| 10 | `cash_flow_incomeTaxesPaid_rk` | 0.061 | quality |

![SHAP feature importance — LightGBM classifier](charts/shap_summary.png)

The economic read is reassuring: the tree model picks up known short-side
factors (asset growth, drawdown, preferred-stock issuance, working-capital
stress) alongside multi-horizon momentum — none of it surprising for a
quant short signal. Full ranked lists in `reports/gain_importance.csv` and
`reports/mean_abs_shap.csv`.

---

## How to reproduce

```bash
git clone https://github.com/Taff1887/short-king-2.0.git
cd short-king-2.0
cp .env.example .env       # then put your FMP_API_KEY in .env
uv sync --extra dev        # one-line install of the whole stack
```

```bash
# Pipeline (each step caches; re-runs are cheap)
uv run python scripts/01_pull_asic.py             --weeks 260
uv run python scripts/02_pull_fmp_fundamentals.py --top-tickers 500
uv run python scripts/_refilter_asic.py           # re-applies the top-500 filter
uv run python scripts/03_pull_fmp_prices.py
uv run python scripts/04_build_features.py
uv run python scripts/05_train_and_validate.py
uv run python scripts/06_backtest.py
uv run python scripts/_extra_charts.py            # generates the diagnostic chart bundle
uv run python scripts/08_generate_report.py       # writes reports/RESULTS.md + patches the README
```

End-to-end runtime (after caches are warm): **< 1 minute** for 01, ~15 min
for 02 (~3500 FMP calls), ~2 min for 03 (~500 FMP calls), ~25 s for 04,
~5 min for 05 (5 models × 24 folds), ~5 s for 06 + 08. ~25 min cold;
< 1 min hot.

Java (any version, for `tabula-py`) needs to be on `PATH` so ASIC PDFs
parse; `pdfplumber` is included as a Java-free fallback.

---

## Limitations

* **Window choice (2021–2026)** captures the meme-stock era + post-COVID
  rally + 2022 bear + 2023-2025 AI rally — a particularly unfriendly regime
  for shorts. A longer (2010+) window would dampen this; the v1 sister
  project `quant-factor-ranking` runs the same pattern for US S&P 500 on
  2010+ and finds modest +1.6 % CAPM α in the long-book direction.
* **Label horizon = 4 weeks.** Short alpha tends to accrue over longer
  horizons; we picked 4w to stay close to v1's `H=4`. A 12-week label would
  use the `fwd_ret_12w` already produced by `assemble.py`.
* **Sector dummies skipped.** The PIT panel exposes `sector`/`industry`
  columns (sourced from FMP `profile`), but our scripted run does not yet
  pull profiles, so `add_sector_dummies` no-ops. Easy follow-up: add a
  profile-fetch step to `scripts/02_pull_fmp_fundamentals.py`.
* **Borrow cost is a flat 150 bps p.a.** Real ASX borrow rates vary by
  name and by date (small-caps + hot shorts can be > 5 % p.a.); the
  `CostConfig` is fully parameterised and the backtest reruns in < 5 s.
* **Capital-raise / squeeze dynamics absent.** A short book that's
  delta-hedged by paper rights and capital-raise dilutions in reality is
  modelled here purely on adjusted-close returns — a real-world dampener
  the backtest doesn't see.
* **US robustness check (`scripts/07_robustness_us.py`) is a documented
  stub.** FMP/FINRA US short interest is bi-monthly, not weekly, so the
  same data pipeline does not transplant cleanly. The script's docstring
  explains what an end-to-end US version would do.

---

## License

MIT — see [`LICENSE`](LICENSE).
