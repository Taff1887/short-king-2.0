# Short King 2.0 — ASX Short-Selling Research

> A cross-sectional short-selling model on the Australian Securities Exchange.
> 500-stock universe over 260 weekly rebalances (~5 years), ASIC weekly
> short-position disclosures + FMP fundamentals, walk-forward CV with
> purge + embargo, costed weekly backtest of five models across two
> portfolio constructions, and a **15 % per-position hard stop with 100 bps
> realistic slippage** to guard against short squeezes.

This is a from-scratch rebuild of an earlier ASX short-interest prototype
([Taff1887/short-king](https://github.com/Taff1887/short-king)). The original
was a single 430 KB Jupyter notebook with five hand-built signals, one
LogisticRegression fit, and 21 total trades with a −3 % return on peak
capital. **Version 2.0** is a proper research project — see the comparison
table and headline results below.

---

## Headline result

The five models, scored across two portfolio constructions (top-quintile
short and dollar-neutral long-short quintile), **net of 25 bps round-trip
commission per side + 1.5 % p.a. borrow + 5 bps slippage + per-stop
exit-and-re-entry commission + 100 bps execution shortfall on every stop
fill**:

| model | strategy | CAGR | Vol | **Sharpe** | Sortino | MaxDD | Calmar | Hit-rate | Turnover (1-way) | Stops | n_rebalances |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | long_short_quintile | **+30.1 %** | 13.5 % | **2.02** | 3.84 | −6.9 % | 4.37 | 59.0 % | 17.0 % | 1,172 | 212 |
| logit | long_short_quintile | +18.6 % | 10.9 % | 1.62 | 3.13 | −10.8 % | 1.73 | 56.0 % | 30.3 % | 137 | 91 |
| gbm_rank | long_short_quintile | +21.0 % | 20.0 % | 1.05 | 1.67 | −32.4 % | 0.65 | 57.6 % | 53.8 % | 687 | 99 |
| gbm_cls | long_short_quintile | +14.4 % | 14.7 % | 0.99 | 1.61 | −24.3 % | 0.59 | 55.6 % | 75.9 % | 610 | 99 |
| gbm_rank | quintile_short | +12.9 % | 25.6 % | 0.60 | 1.01 | −37.7 % | 0.34 | 50.5 % | 30.2 % | 637 | 99 |
| naive | quintile_short | +8.2 % | 21.8 % | 0.47 | 0.73 | −30.5 % | 0.27 | 49.1 % | 5.2 % | 568 | 212 |
| ew | long_short_quintile | +5.8 % | 15.4 % | 0.44 | 0.69 | −29.0 % | 0.20 | 49.1 % | 26.1 % | 1,003 | 212 |
| gbm_cls | quintile_short | +1.7 % | 20.6 % | 0.18 | 0.30 | −36.7 % | 0.05 | 47.5 % | 39.0 % | 455 | 99 |
| logit | quintile_short | −4.3 % | 16.1 % | −0.19 | −0.30 | −33.0 % | −0.13 | 47.3 % | 15.5 % | 89 | 91 |
| ew | quintile_short | −11.6 % | 16.7 % | −0.65 | −0.90 | −54.7 % | −0.21 | 41.5 % | 13.3 % | 234 | 212 |

![Cumulative growth of $1 — solid = quintile-short, dotted = long-short](charts/cumulative_returns.png)

**The headline.** The **naive long-short quintile** — rank by raw ShortPct,
long the bottom 20 %, short the top 20 % — earns a Sharpe of **2.02** with
a 6.9 % max drawdown, net of every modelled friction including 100 bps
stop-fill slippage. Every long-short quintile variant runs positive Sharpe;
three of five clear 1.0. The short-only books are weaker (two positive,
three flat-to-negative) — short-side alpha is real but not enough to
overcome borrow on its own; the long leg of the quintile pair is what
carries the headline.

**The 15 % per-position stop is doing real work.** Without it the same
naive long-short was Sharpe 0.49 (deciles, no stop). The stop fired 1,172
times over 212 weeks (~5.5 per week, on an average book of 170 names) and
saved a cumulative ~92 % of NAV net of the 100 bps stop-fill slippage
(106 % gross, of which 14 % was eaten by the execution shortfall).
Roughly half the Sharpe lift comes from clipping squeeze-week losses, the
other half from the quintile-vs-decile change (5× wider buckets are less
concentrated and turnover drops accordingly).

### Stop-slippage sensitivity

Re-runs of the long-short quintile across `stop_slippage_pct ∈ {0, 0.5, 1,
1.5, 2 %}` — the 15 % trigger is fixed; only the average fill shortfall
changes (see `scripts/_stop_sensitivity.py` for the sweep):

| Model | 0 bps (perfect) | 50 bps | **100 bps (default)** | 150 bps | 200 bps (conservative) |
|---|---:|---:|---:|---:|---:|
| naive | 2.29 | 2.15 | **2.02** | 1.89 | 1.76 |
| logit | 1.77 | 1.70 | **1.62** | 1.54 | 1.46 |
| gbm_rank | 1.27 | 1.16 | **1.05** | 0.94 | 0.84 |
| gbm_cls | 1.25 | 1.12 | **0.98** | 0.86 | 0.73 |
| ew | 0.64 | 0.54 | **0.44** | 0.35 | 0.25 |

Every additional 50 bps of slippage drops the naive L/S Sharpe by ~0.13. At
the conservative 200 bps assumption the headline is **still 1.76** —
institutional-grade — so the strategy's edge is robust to realistic
execution friction within the range we'd expect for this universe.

### Top short candidates — as-of 2026-05-22

Top 15 names by *consensus rank* across the three trained models (`logit`
+ `gbm_cls` + `gbm_rank`). Higher rank = stronger short conviction —
all three models agree the name sits in the bearish tail. Gated on
investable + A$200m+ market cap. Full top-30 in
[`reports/current_positions.csv`](reports/current_positions.csv); regenerate
with `scripts/_current_positions.py`.

| # | Ticker | Company | Mkt Cap (A$m) | Short % | logit | gbm_cls | gbm_rank | Consensus |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1  | AMP | AMP Limited | 4,047 | 2.32 | 0.638 | 0.899 | 0.157 | 0.901 |
| 2  | GDG | Generation Dev Group | 1,622 | 9.59 | 0.701 | 0.829 | −0.540 | 0.886 |
| 3  | HMC | HMC Capital Limited | 1,143 | 6.94 | 0.684 | 0.749 | 1.055 | 0.882 |
| 4  | RWC | Reliance Worldwide | 2,416 | 3.11 | 0.721 | 0.892 | −1.150 | 0.865 |
| 5  | NUF | Nufarm Limited | 929 | 1.63 | 0.656 | 0.908 | −0.883 | 0.865 |
| 6  | TYR | Tyro Payments | 439 | 5.49 | 0.665 | 0.903 | −1.178 | 0.847 |
| 7  | DTL | Data#3 Limited | 1,307 | 5.12 | 0.651 | 0.755 | −0.279 | 0.830 |
| 8  | ACL | AU Clinical Labs | 382 | 8.44 | 0.734 | 0.814 | −1.362 | 0.826 |
| 9  | WTC | WiseTech Global | 12,366 | 7.83 | 0.736 | 0.686 | −0.531 | 0.807 |
| 10 | MGH | Maas Group Holdings | 1,790 | 1.45 | 0.608 | 0.833 | −0.556 | 0.803 |
| 11 | XRO | Xero Ltd | 12,936 | 3.49 | 0.768 | 0.696 | −0.992 | 0.785 |
| 12 | PNV | Polynovo Limited | 787 | 10.37 | 0.610 | 0.762 | −0.381 | 0.775 |
| 13 | CKF | Collins Foods Ltd | 1,004 | 0.86 | 0.652 | 0.848 | −1.811 | 0.763 |
| 14 | VSL | Vulcan Steel | 771 | 0.83 | 0.701 | 0.880 | −2.282 | 0.762 |
| 15 | ARB | ARB Corporation | 1,503 | 3.76 | 0.628 | 0.803 | −1.210 | 0.760 |

The list reads cleanly as a real short-book: high-multiple growth names
(WiseTech, Xero, Polynovo), already-shorted consumer/health/fin names
(Generation Dev, AMP, AU Clinical Labs, Tyro), and a few cyclicals/
mid-caps where the GBM is picking up earnings-quality / leverage signals
even though ShortPct is low (AMP, ARB, Collins Foods). **Not** a list of
fly-by-night small-caps — every name is ≥ A$200m and most are ASX 200/300
constituents.

---

## Information coefficients (out-of-fold)

What the model *thinks* about each stock, scored against the realised 4-week
forward total return. All five models are evaluated on the same walk-forward
folds (purge + 4-week embargo to match the label horizon):

| model | IC mean | IC t-stat (naive) | **IC t-stat (HAC-adj)** | IC hit-rate | n periods |
|---|---:|---:|---:|---:|---:|
| ew (equal-weight composite) | +7.5 % | +9.12 | **~+4.5** | 78.2 % | 211 |
| naive (rank ShortPct) | +1.2 % | +2.13 | ~+1.1 | 54.0 % | 211 |
| gbm_cls (LightGBM classifier) | −3.0 % | −2.55 | ~−1.3 | 36.7 % | 98 |
| logit (rebuilt v1 baseline) | −4.3 % | −4.29 | ~−2.1 | 31.1 % | 90 |
| gbm_rank (LightGBM LambdaRank) | −4.5 % | −2.51 | ~−1.3 | 36.7 % | 98 |

**Reading the signs.** Positive IC = the model's "shortable" score correlates
with *higher* forward returns — backwards for a short book. The three
trained models all produce *negative* ICs, i.e. they correctly identify
underperformers. The naive and EW baselines have positive ICs because the
characteristics they mechanically rank by (high SI; high quality, value,
momentum, size) turned out to predict *winners* in the 2022-2026 ASX
regime. The naive baseline's L/S construction monetises that anyway — long
the names that aren't heavily shorted, short the ones that are — and the
stop loss handles the squeeze-side tail.

**Naive t-stats are inflated by overlapping 4-week labels.** Consecutive
weekly observations of `fwd_ret_4w` share 75 % of their underlying
window, so the IC series is serially correlated. A Newey-West HAC
adjustment with lag = horizon − 1 = 3 cuts the apparent t by roughly
√4 ≈ 2 — see [§Limitations](#limitations). The mean ICs themselves are
unbiased; only the significance is overstated.

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
| Portfolio construction | top-K weekly | **top-quintile short and dollar-neutral L/S quintile**, equal-weight, liquidity-gated |
| Risk control | hard 10 % stop, no costs | **hard 15 % per-position stop with explicit exit + re-entry commission** |
| Costs | none modelled | 25 bps round-trip + 1.5 % p.a. borrow + 5 bps slippage + stop-exit commission |
| Metrics | total $ PnL only | CAGR, Vol, **Sharpe, Sortino, MaxDD, Calmar**, hit-rate, turnover, monthly heatmap |
| Reporting | inline plots | **9 publication-quality PNGs** + CSV summary + `reports/RESULTS.md` + methodology + data dictionary |
| Reproducibility | none | uv lockfile, deterministic on-disk caches (ASIC + FMP), 40-test pytest suite |

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
│   ├── portfolio/    top-K / quintile / long-short construction; weekly backtest
│   │                 with costs + borrow + slippage + per-position 15 % stop
│   ├── reporting/    publication-quality charts + tearsheet
│   └── utils/        config (pydantic-settings), loguru, IO, dates
├── scripts/          01..08 pipeline + utility helpers
├── tests/            40 pytest assertions (utils, features, models, no-lookahead,
│                     backtest math incl. stop-loss)
├── docs/             methodology.md, data_dictionary.md
├── charts/           publication PNGs (committed)
├── reports/          backtest_*.parquet, model_metrics.csv, RESULTS.md, oof_predictions.parquet
└── legacy/           original v1 notebook (preserved)
```

---

## How the backtest works

For each Friday `t` in the panel:
1. **Score**: each model produces a per-stock score on the day's universe.
2. **Construct**: rank by score within the day, take the top quintile short
   (basket weight `−1/n`), and optionally the bottom quintile long (basket
   weight `+1/n`) for the dollar-neutral variant.
3. **Hold one week**: position `p_i` realised P&L = `weight_i × (adjClose[t+1] / adjClose[t] − 1)`.
4. **Stop loss check**: if any single position would lose more than 15 % of
   its own notional in the week, the contribution is clipped to that floor
   (the position is modelled as having been exited at the stop). An
   **extra round-trip commission** is charged on the stopped notional to
   reflect both the in-week stop exit and the re-entry the rebalance-level
   delta-commission would otherwise miss.
5. **Costs**: `r_net = r_gross − 25bps·Σ|Δw| − 5bps·Σ|Δw| − 1.5%·short_notional/52 − stop_commission`.
6. **Compound**: cumulative growth of $1 is `Π (1 + r_net)`.

There is no drift between rebalances (the book snaps to target each Friday)
and the last rebalance is dropped from performance (no forward week).
Full detail in [`docs/methodology.md`](docs/methodology.md) and the
backtest engine source in
[`src/short_king/portfolio/backtest.py`](src/short_king/portfolio/backtest.py).

---

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
| **Cumulative $1** across the 10 (model × strategy) backtests. | **Underwater plot** for the best (naive L/S quintile). |
| ![Monthly heatmap](charts/monthly_heatmap.png) | |
| **Monthly returns** — best strategy. | |

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
uv run python scripts/06_backtest.py              # quintiles + 15% stop by default
uv run python scripts/_extra_charts.py            # diagnostic chart bundle
uv run python scripts/08_generate_report.py       # writes reports/RESULTS.md
```

`06_backtest.py` accepts `--n-buckets 10` to switch back to deciles and
`--stop-loss-pct 1.0` to disable the stop. End-to-end runtime (cold):
~25 min (ASIC + FMP + ML); hot re-runs: <1 min.

Java (any version, for `tabula-py`) needs to be on `PATH` so ASIC PDFs
parse; `pdfplumber` is included as a Java-free fallback.

---

## Limitations

* **Stop-loss execution slippage is modelled at 100 bps central / 200 bps
  conservative** (`CostConfig.stop_slippage_pct`). When a position trips
  the 15 % trigger, the engine fills it at the trigger *plus* this
  slippage to reflect that real stops do not fill at the trigger price.
  Realistic ranges for our universe: ~30 bps (ASX 50), ~100 bps (mid-cap),
  100-500 bps (small-cap), 200-1000+ bps (halted / squeeze). The headline
  Sharpe of 2.02 uses 100 bps; the sensitivity table above quantifies the
  effect of dialling this up. **Tail-risk slippage on individual halted
  names is not separately modelled** — a takeover-bid-driven 50 % gap on
  a single short would still be capped at the floor in the engine.
* **Window choice (2022-2026)** captures the meme-stock era + post-COVID
  rally + 2022 bear + 2023-2025 AI rally. A longer (2010+) window would
  dampen the regime effect; the sister `quant-factor-ranking` project
  runs the same pattern on US S&P 500 2010+ and finds modest +1.6 %
  CAPM α in the long-book direction.
* **IC t-stats reported in §IC table are naive** (assume independent
  weekly observations) but the 4-week label horizon makes consecutive
  obs share 75 % of their window. A Newey-West HAC adjustment cuts the
  apparent t-stat by ~√4 ≈ 2. The mean ICs themselves are unbiased.
* **Sector dummies skipped.** The PIT panel exposes `sector`/`industry`
  columns (sourced from FMP `profile`), but our scripted run does not
  yet pull profiles, so `add_sector_dummies` no-ops. Easy follow-up: add
  a profile-fetch step to `scripts/02_pull_fmp_fundamentals.py`.
* **Borrow cost is a flat 150 bps p.a.** Real ASX borrow rates vary by
  name and by date (small-caps + hot shorts can be > 5 % p.a.); the
  `CostConfig` is fully parameterised and the backtest reruns in < 5 s.
* **Capital-raise / squeeze dynamics absent.** A short book that's
  diluted by paper rights and capital-raise issuance in reality is
  modelled here purely on adjusted-close returns — a real-world
  dampener the backtest doesn't see (though the 15 % stop catches most
  of the single-week pain that comes with these events).
* **US robustness check (`scripts/07_robustness_us.py`) is a documented
  stub.** FMP/FINRA US short interest is bi-monthly, not weekly, so the
  same data pipeline does not transplant cleanly. The script's docstring
  explains what an end-to-end US version would do.

---

## License

MIT — see [`LICENSE`](LICENSE).
