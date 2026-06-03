"""Weekly rebalance backtest engine for the ASX short-king signal stack.

Drives a long/short (or long-only / short-only) book through a sequence of
weekly rebalance dates, applies realistic frictions (commission + half-spread
per side, one-sided slippage on weight *changes*, and an annualised borrow
charge on short positions) and reports portfolio analytics with no
look-ahead bias.

Conventions
-----------
* ``target_weights`` is long-form ``[Date, Ticker, weight]``. A row's
  ``weight`` is the *signed* dollar allocation at the open of ``Date``:
  positive = long, negative = short. The constructor upstream is responsible
  for ensuring the per-Date gross/net exposure is whatever the strategy
  intends (e.g. dollar-neutral has +1 long / -1 short summing to 0 net).
* ``prices_panel`` is long-form ``[Date, Ticker, adjClose]`` on the same
  weekly calendar as ``target_weights``; the next Date's adjClose is the
  realised close-to-close return for the week.
* No drift between rebalances - we assume rebalancing to target each week,
  so ``old_weight`` at week t+1 equals the *target_weights at t*. This is
  the standard textbook assumption that keeps turnover well-defined; the
  drift-correction would only matter for very long holding periods.
* Returns are compounded across weeks via :math:`\\prod (1 + r_t)`. The
  annualisation factor is :math:`52` (weekly bars).

Public API
----------
:class:`CostConfig` :class:`BacktestResult` :func:`backtest_weekly`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from short_king.utils.logging import logger

# Weekly annualisation factor. ASIC short-selling reports and the strategy
# rebalance both live on a Friday calendar -> 52 periods per year.
PERIODS_PER_YEAR: int = 52


# ---------------------------------------------------------------------------
# Configuration & result containers
# ---------------------------------------------------------------------------
@dataclass
class CostConfig:
    """Friction model for the weekly rebalance.

    Attributes
    ----------
    bps_round_trip:
        Commission plus half-spread, **per side**, charged on both legs of
        a turnover trade. Default 25 bps reflects retail ASX commissions +
        bid/ask half-spread in the sub-ASX200 tail of the universe.
    annual_borrow_pct:
        Annualised borrow fee on *short* positions, expressed in percent
        (i.e. ``1.5`` means 150 bps p.a.). ASX HTB names can be 5-15 % so
        this is a conservative baseline.
    slippage_bps:
        Extra one-sided market-impact charge on the *change* in weight at
        each rebalance, on top of ``bps_round_trip``.
    bench_borrow_for_longs:
        If True, also apply ``annual_borrow_pct`` to the long book (some
        prime-broker structures charge a leverage/financing fee on the
        long leg). Default False - shorts only.
    stop_loss_pct:
        Per-position hard stop. Any position whose realised weekly P&L is
        worse than ``-stop_loss_pct * |weight|`` is clipped to that floor
        (i.e. the position is treated as having been exited at the stop
        price). Each stop trigger also incurs an extra round-trip
        commission to model the in-week exit *and* the re-entry that the
        rebalance-level delta-commission would otherwise miss. Set to
        ``None`` (or ``>= 1.0``) to disable. Default 0.15 reflects the v1
        notebook's hard-stop convention and is the standard short-squeeze
        guard for ASX small/mid-caps.
    """

    bps_round_trip: float = 25.0
    annual_borrow_pct: float = 1.5
    slippage_bps: float = 5.0
    bench_borrow_for_longs: bool = False
    stop_loss_pct: float | None = 0.15


@dataclass
class BacktestResult:
    """Container for a weekly backtest run.

    Attributes
    ----------
    returns:
        Per-rebalance DataFrame with columns ``[Date, ret_gross, ret_net,
        turnover, cost_total, borrow, n_positions]``. Indexed by integer
        position; ``Date`` is the *entry* date of the week's holding period.
    weights:
        The input target weights, sorted and de-duplicated, retained for
        downstream attribution / reporting.
    trades:
        Long-form change log ``[Date, Ticker, old_weight, new_weight,
        delta_weight]`` containing only rows where ``delta_weight != 0``.
    summary:
        One-row :class:`pandas.Series` of portfolio statistics (see
        :func:`_summarise_returns`).
    """

    returns: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.Series
    bench_returns: pd.Series | None = field(default=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalise_weights(target_weights: pd.DataFrame) -> pd.DataFrame:
    """Validate, dedupe and sort the target-weights long frame."""
    required = {"Date", "Ticker", "weight"}
    missing = required - set(target_weights.columns)
    if missing:
        raise KeyError(f"target_weights missing columns: {sorted(missing)}")
    w = target_weights[["Date", "Ticker", "weight"]].copy()
    w["Date"] = pd.to_datetime(w["Date"]).dt.normalize()
    w["Ticker"] = w["Ticker"].astype(str)
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    # Collapse duplicate (Date, Ticker) rows by summing - the constructor
    # may emit fragmentary weights from multiple sources; summing is the
    # conservative thing because zero-sum cancellation is preserved.
    w = (
        w.groupby(["Date", "Ticker"], as_index=False)["weight"].sum()
         .sort_values(["Date", "Ticker"])
         .reset_index(drop=True)
    )
    return w


def _normalise_prices(prices_panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and pivot the long prices panel to a wide [Date, Ticker] frame."""
    required = {"Date", "Ticker", "adjClose"}
    missing = required - set(prices_panel.columns)
    if missing:
        raise KeyError(f"prices_panel missing columns: {sorted(missing)}")
    p = prices_panel[["Date", "Ticker", "adjClose"]].copy()
    p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
    p["Ticker"] = p["Ticker"].astype(str)
    p["adjClose"] = pd.to_numeric(p["adjClose"], errors="coerce")
    # Last observation wins for duplicate (Date, Ticker) cells.
    p = (
        p.drop_duplicates(subset=["Date", "Ticker"], keep="last")
         .sort_values(["Date", "Ticker"])
    )
    wide = p.pivot(index="Date", columns="Ticker", values="adjClose").sort_index()
    return wide


def _period_returns(prices_wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Close-to-close returns from ``date_t`` to ``date_{t+1}`` per ticker.

    For each consecutive pair of rebalance dates we use the closes at those
    two dates to construct the realised holding-period return. The last
    rebalance has no forward window and is dropped from the returns frame.
    Stocks missing a close on either anchor return NaN (handled at the
    weighting step - their contribution is set to zero).
    """
    snap = prices_wide.reindex(dates)
    # Forward fill so a one-week price gap doesn't kill the whole row;
    # bigger gaps stay NaN and are excluded from the weighted average.
    snap = snap.ffill(limit=1)
    rets = snap.pct_change().shift(-1)
    # Drop the trailing row (no next-week close available).
    return rets.iloc[:-1]


def _trades_long(weights_long: pd.DataFrame) -> pd.DataFrame:
    """Long-form delta-weight log; only rows where the weight actually changed."""
    # Pivot to a wide [Date, Ticker] grid filled with zeros so non-held names
    # contribute zero to old/new weight on either side of a rebalance.
    w_wide = (
        weights_long.pivot(index="Date", columns="Ticker", values="weight")
                    .fillna(0.0)
                    .sort_index()
    )
    prev = w_wide.shift(1).fillna(0.0)
    delta = w_wide - prev
    long = (
        delta.stack(future_stack=True)
             .rename("delta_weight")
             .reset_index()
    )
    long["old_weight"] = prev.stack(future_stack=True).reset_index(drop=True)
    long["new_weight"] = w_wide.stack(future_stack=True).reset_index(drop=True)
    long = long[["Date", "Ticker", "old_weight", "new_weight", "delta_weight"]]
    return long.loc[long["delta_weight"].abs() > 1e-12].reset_index(drop=True)


def _summarise_returns(
    net: pd.Series,
    gross: pd.Series,
    turnover: pd.Series,
    bench: pd.Series | None = None,
) -> pd.Series:
    """Roll the per-week net-return series into a one-line portfolio summary.

    All annualisation uses 52 weekly bars per year. Sharpe and Sortino are
    excess of zero (no risk-free) since the project benchmarks against an
    explicit ``bench_returns`` passed by the caller; treating cash as zero
    keeps the metric comparable across regimes.
    """
    r = net.dropna()
    n = len(r)
    if n == 0:
        return pd.Series(
            {
                "n_weeks": 0, "total_return": np.nan, "CAGR": np.nan,
                "ann_vol": np.nan, "Sharpe": np.nan, "Sortino": np.nan,
                "max_drawdown": np.nan, "calmar": np.nan, "hit_rate": np.nan,
                "avg_turnover": np.nan, "ann_turnover": np.nan,
                "gross_CAGR": np.nan, "cost_drag_pa": np.nan,
                "alpha_pa": np.nan, "beta": np.nan,
            }
        )

    cum = float((1.0 + r).prod())
    total_return = cum - 1.0
    years = n / PERIODS_PER_YEAR
    cagr = cum ** (1.0 / years) - 1.0 if years > 0 else np.nan

    vol = float(r.std(ddof=1))
    ann_vol = vol * np.sqrt(PERIODS_PER_YEAR) if vol > 0 else np.nan
    sharpe = (r.mean() / vol) * np.sqrt(PERIODS_PER_YEAR) if vol > 0 else np.nan

    # Sortino - downside deviation only (negative returns vs zero target).
    downside = r.clip(upper=0.0)
    dd_vol = float(np.sqrt((downside ** 2).mean()))
    sortino = (r.mean() / dd_vol) * np.sqrt(PERIODS_PER_YEAR) if dd_vol > 0 else np.nan

    # Max drawdown on the equity curve.
    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    calmar = (cagr / abs(max_dd)) if (pd.notna(cagr) and max_dd < 0) else np.nan

    hit_rate = float((r > 0).mean())

    # Turnover statistics (turnover here is the per-rebalance one-way figure
    # = sum |delta_w| / 2; annualised by 52).
    to_avg = float(turnover.mean()) if len(turnover) else np.nan
    to_ann = to_avg * PERIODS_PER_YEAR if pd.notna(to_avg) else np.nan

    # Gross vs net summary for a quick cost-drag readout.
    g = gross.dropna()
    if len(g):
        g_cum = float((1.0 + g).prod())
        g_cagr = g_cum ** (1.0 / years) - 1.0 if years > 0 else np.nan
    else:
        g_cagr = np.nan
    cost_drag_pa = (g_cagr - cagr) if (pd.notna(g_cagr) and pd.notna(cagr)) else np.nan

    # Optional CAPM-style stats vs benchmark.
    alpha_pa = np.nan
    beta = np.nan
    if bench is not None:
        joined = pd.concat([r.rename("p"), bench.rename("b")], axis=1).dropna()
        if len(joined) >= 12 and joined["b"].std(ddof=1) > 0:
            slope, intercept = np.polyfit(joined["b"].to_numpy(), joined["p"].to_numpy(), 1)
            beta = float(slope)
            alpha_pa = float(intercept) * PERIODS_PER_YEAR

    return pd.Series(
        {
            "n_weeks": int(n),
            "total_return": total_return,
            "CAGR": cagr,
            "ann_vol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": hit_rate,
            "avg_turnover": to_avg,
            "ann_turnover": to_ann,
            "gross_CAGR": g_cagr,
            "cost_drag_pa": cost_drag_pa,
            "alpha_pa": alpha_pa,
            "beta": beta,
        }
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def backtest_weekly(
    target_weights: pd.DataFrame,
    prices_panel: pd.DataFrame,
    *,
    cost_config: CostConfig | None = None,
    bench_returns: pd.Series | None = None,
) -> BacktestResult:
    """Run the weekly rebalance backtest.

    Parameters
    ----------
    target_weights:
        Long-form ``[Date, Ticker, weight]``. Signed weights: positive =
        long, negative = short. Weights live at the *entry* of each weekly
        holding period.
    prices_panel:
        Long-form ``[Date, Ticker, adjClose]``. Must cover every date in
        ``target_weights`` *and* at least one forward observation past the
        final rebalance for the last holding period.
    cost_config:
        Friction model. Defaults to :class:`CostConfig()`.
    bench_returns:
        Optional weekly benchmark return series indexed by Date - used to
        compute alpha and beta in the summary. Must be on the same Friday
        calendar as ``target_weights``.

    Returns
    -------
    :class:`BacktestResult`
    """
    cfg = cost_config or CostConfig()
    weights = _normalise_weights(target_weights)
    if weights.empty:
        raise ValueError("backtest_weekly received empty target_weights")

    prices_wide = _normalise_prices(prices_panel)
    rebalance_dates = pd.DatetimeIndex(weights["Date"].unique()).sort_values()
    if len(rebalance_dates) < 2:
        raise ValueError(
            "backtest_weekly needs at least 2 rebalance dates to realise a return "
            f"(got {len(rebalance_dates)})"
        )

    # Per-week stock returns from rebalance t -> t+1.
    stock_rets = _period_returns(prices_wide, rebalance_dates)

    # Wide weights with the *union* universe of tickers ever held across all
    # rebalance dates so positions that exit are picked up with a zero new
    # weight (and hence a non-zero delta).
    w_wide = (
        weights.pivot(index="Date", columns="Ticker", values="weight")
               .reindex(rebalance_dates)
               .fillna(0.0)
    )
    # Align stock returns to the weight grid (drops the final rebalance,
    # which has no realised forward week).
    held_dates = w_wide.index[:-1]
    w_held = w_wide.loc[held_dates]
    r_held = stock_rets.reindex(index=held_dates, columns=w_held.columns).fillna(0.0)

    # Raw per-position weekly P&L contribution: long position earns the stock
    # return, short position earns -stock_return. Encoded by signed weight.
    contrib_raw = w_held * r_held
    gross_pre_stop = contrib_raw.sum(axis=1)

    # ---- Per-position hard stop ----------------------------------------------
    # If a single position would lose more than ``stop_loss_pct`` of its own
    # notional in the week, clip it to that floor. For a short with weight
    # -0.05 and a 15 % stop, that caps the contribution at -0.0075 of book
    # (= -15 % of the position's notional).
    stop_pct = cfg.stop_loss_pct
    if stop_pct is not None and stop_pct < 1.0:
        floor = -float(stop_pct) * w_held.abs()
        stopped = contrib_raw < floor
        contrib_capped = contrib_raw.where(~stopped, floor)
    else:
        stopped = pd.DataFrame(False, index=contrib_raw.index, columns=contrib_raw.columns)
        contrib_capped = contrib_raw
    gross_per_week = contrib_capped.sum(axis=1)
    n_stops = stopped.sum(axis=1).astype(int)
    stop_loss_savings = (gross_per_week - gross_pre_stop)  # >= 0 (clipping helps)

    # Old weight = the previous *target* (no drift). For the first held week
    # the old weight is zero (book starts flat).
    old_w = w_wide.shift(1).fillna(0.0).loc[held_dates]
    new_w = w_held
    delta_w = new_w - old_w

    # Turnover (one-way) = sum |delta_w| / 2 - the standard convention so
    # that a full portfolio replacement reads 100 %, not 200 %.
    turnover = delta_w.abs().sum(axis=1) / 2.0

    # Commission + half-spread + slippage. ``bps_round_trip`` is per-side
    # but applies to both sides, so we charge it on |delta_w| once - i.e.
    # turnover_two_way = sum |delta_w| (not divided by 2). Slippage is the
    # one-sided market-impact extra on top.
    two_way_traded = delta_w.abs().sum(axis=1)
    commission = two_way_traded * cfg.bps_round_trip / 10_000.0
    slippage = two_way_traded * cfg.slippage_bps / 10_000.0

    # Stop-loss exit cost: each stop fires an in-week close trade AND the
    # rebalance delta-commission would otherwise miss the implied re-entry,
    # so we charge BOTH sides (2x ``bps_round_trip``) on the stopped notional.
    stopped_notional = (stopped.astype(float) * w_held.abs()).sum(axis=1)
    stop_commission = stopped_notional * 2.0 * cfg.bps_round_trip / 10_000.0

    # Borrow cost on *short* positions. Charged on the average weight held
    # during the period (here equal to ``new_w`` because no drift). Per-week
    # rate = annual / 52.
    short_notional = new_w.clip(upper=0.0).abs().sum(axis=1)
    long_notional = new_w.clip(lower=0.0).sum(axis=1)
    borrow_rate_wk = (cfg.annual_borrow_pct / 100.0) / PERIODS_PER_YEAR
    borrow = short_notional * borrow_rate_wk
    if cfg.bench_borrow_for_longs:
        borrow = borrow + long_notional * borrow_rate_wk

    cost_total = commission + slippage + borrow + stop_commission
    net_per_week = gross_per_week - cost_total

    n_positions = (new_w.abs() > 1e-12).sum(axis=1).astype(int)

    returns_df = pd.DataFrame(
        {
            "Date": held_dates,
            "ret_gross": gross_per_week.to_numpy(),
            "ret_gross_pre_stop": gross_pre_stop.to_numpy(),
            "stop_loss_savings": stop_loss_savings.to_numpy(),
            "ret_net": net_per_week.to_numpy(),
            "turnover": turnover.to_numpy(),
            "cost_total": cost_total.to_numpy(),
            "borrow": borrow.to_numpy(),
            "commission": commission.to_numpy(),
            "stop_commission": stop_commission.to_numpy(),
            "slippage": slippage.to_numpy(),
            "n_positions": n_positions.to_numpy(),
            "n_stops": n_stops.to_numpy(),
            "gross_exposure": (long_notional + short_notional).to_numpy(),
            "net_exposure": (long_notional - short_notional).to_numpy(),
        }
    ).reset_index(drop=True)

    # Bench return alignment for alpha/beta.
    bench_aligned: pd.Series | None = None
    if bench_returns is not None:
        b = bench_returns.copy()
        b.index = pd.to_datetime(b.index).normalize()
        bench_aligned = b.reindex(held_dates).rename("bench")

    summary = _summarise_returns(
        pd.Series(net_per_week.to_numpy(), index=held_dates, name="ret_net"),
        pd.Series(gross_per_week.to_numpy(), index=held_dates, name="ret_gross"),
        pd.Series(turnover.to_numpy(), index=held_dates, name="turnover"),
        bench=bench_aligned,
    )
    # Stop-loss diagnostics piggy-backed onto the summary Series so they
    # surface in the reports/backtest_summary.csv aggregation step.
    summary["n_stops_total"] = int(n_stops.sum())
    summary["stop_loss_savings_total"] = float(stop_loss_savings.sum())
    summary["stop_commission_total"] = float(stop_commission.sum())

    trades = _trades_long(weights)

    logger.info(
        "backtest_weekly: weeks=%d, CAGR=%.2f%%, Sharpe=%.2f, "
        "ann_turnover=%.1f%%, n_trades=%d",
        int(summary["n_weeks"]),
        100 * float(summary["CAGR"]) if pd.notna(summary["CAGR"]) else float("nan"),
        float(summary["Sharpe"]) if pd.notna(summary["Sharpe"]) else float("nan"),
        100 * float(summary["ann_turnover"]) if pd.notna(summary["ann_turnover"]) else float("nan"),
        len(trades),
    )

    return BacktestResult(
        returns=returns_df,
        weights=weights,
        trades=trades,
        summary=summary,
        bench_returns=bench_aligned,
    )


__all__ = [
    "CostConfig",
    "BacktestResult",
    "backtest_weekly",
    "PERIODS_PER_YEAR",
]
