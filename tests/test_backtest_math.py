"""Numerical sanity tests for the weekly backtest engine.

We build a deterministic 4-week, 3-ticker book where every per-week return is
hand-computed, then assert the engine reproduces those numbers to floating
point precision. Two flavours:

* zero-cost configuration so net == gross exactly,
* non-zero borrow + commission + slippage so the cost drag is the closed-form
  analytic amount.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from short_king.portfolio.backtest import CostConfig, backtest_weekly


# ---------------------------------------------------------------------------
# Synthetic, fully-deterministic inputs
# ---------------------------------------------------------------------------
REBALANCE_DATES = pd.to_datetime(
    ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"]
)
TICKERS = ["AAA", "BBB", "CCC"]

# Constant weights across all 4 rebalances:
#   AAA = +0.5 (long), BBB = -0.5 (short), CCC = 0.0 (untouched).
# Gross exposure = 1.0, net exposure = 0.0 -> a clean dollar-neutral L/S book
# with no leverage. Holding 3 weeks (D0..D2) -> the last rebalance D3 has no
# forward window and is dropped by the engine.
WEIGHTS = (
    pd.DataFrame(
        [
            *[{"Date": d, "Ticker": "AAA", "weight": 0.5} for d in REBALANCE_DATES],
            *[{"Date": d, "Ticker": "BBB", "weight": -0.5} for d in REBALANCE_DATES],
            *[{"Date": d, "Ticker": "CCC", "weight": 0.0} for d in REBALANCE_DATES],
        ]
    )
)

# Geometric ramps -> per-week stock return is identical every week per ticker:
#   AAA: +10 %/week, BBB: -5 %/week, CCC: flat.
PRICES = pd.DataFrame(
    [
        # AAA ramps 100 -> 110 -> 121 -> 133.1 -> 146.41
        *[{"Date": d, "Ticker": "AAA", "adjClose": 100.0 * (1.10**i)}
          for i, d in enumerate(REBALANCE_DATES)],
        # BBB ramps 100 -> 95 -> 90.25 -> 85.7375 -> 81.450625
        *[{"Date": d, "Ticker": "BBB", "adjClose": 100.0 * (0.95**i)}
          for i, d in enumerate(REBALANCE_DATES)],
        # CCC flat at 100.
        *[{"Date": d, "Ticker": "CCC", "adjClose": 100.0} for d in REBALANCE_DATES],
    ]
)

# Per-week gross return is the same every held week with these constant
# weights and geometric prices:
#   gross_t = 0.5 * 0.10 + (-0.5) * (-0.05) + 0.0 * 0.0 = 0.075
EXPECTED_GROSS_PER_WEEK = 0.075


# ---------------------------------------------------------------------------
# Zero-cost: net must equal gross exactly to ~machine precision
# ---------------------------------------------------------------------------
def test_zero_cost_weekly_net_returns_match_hand_computed() -> None:
    cfg = CostConfig(
        bps_round_trip=0.0,
        annual_borrow_pct=0.0,
        slippage_bps=0.0,
        bench_borrow_for_longs=False,
    )
    res = backtest_weekly(WEIGHTS, PRICES, cost_config=cfg)

    # 3 held weeks (D0, D1, D2); D3 has no forward window.
    assert len(res.returns) == 3
    assert list(res.returns["Date"]) == list(REBALANCE_DATES[:-1])

    # Gross == hand-computed value for every held week.
    np.testing.assert_allclose(
        res.returns["ret_gross"].to_numpy(),
        np.full(3, EXPECTED_GROSS_PER_WEEK),
        atol=1e-12, rtol=0,
    )

    # Costs are all zero by construction.
    assert (res.returns["cost_total"] == 0.0).all()
    assert (res.returns["borrow"] == 0.0).all()
    assert (res.returns["commission"] == 0.0).all()
    assert (res.returns["slippage"] == 0.0).all()

    # Net == gross to ~floating-point precision (1e-9 is comfortably generous
    # against the 1e-15-ish actual error from a single subtraction).
    np.testing.assert_allclose(
        res.returns["ret_net"].to_numpy(),
        res.returns["ret_gross"].to_numpy(),
        atol=1e-9, rtol=0,
    )

    # Spot-check the exposure / position-count diagnostics too.
    assert (res.returns["n_positions"] == 2).all()  # AAA and BBB only (CCC=0)
    np.testing.assert_allclose(res.returns["gross_exposure"].to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(res.returns["net_exposure"].to_numpy(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Non-zero costs: cost drag must match the closed-form analytic amount
# ---------------------------------------------------------------------------
def test_costs_reduce_returns_by_analytic_amount() -> None:
    # Pick round numbers so the arithmetic is easy to follow on paper:
    #   commission 100 bps round-trip, slippage 50 bps, borrow 5.2 %/yr.
    # Two-way traded weight is 1.0 only at D0 (initial entry, 0 -> 0.5 long
    # and 0 -> -0.5 short, total |delta| = 1.0). D1 and D2 keep the same
    # weights, so two_way_traded = 0 there. Borrow charges every week on
    # the |short notional| of 0.5.
    cfg = CostConfig(
        bps_round_trip=100.0,
        annual_borrow_pct=5.2,
        slippage_bps=50.0,
        bench_borrow_for_longs=False,
    )
    res = backtest_weekly(WEIGHTS, PRICES, cost_config=cfg)

    # 5.2 %/yr / 52 = 0.1 %/week on |short notional| = 0.5 -> 0.0005/week.
    expected_borrow_per_week = 0.5 * (0.052 / 52.0)
    # Initial-entry commission: |delta| = 1.0 at 100 bps -> 0.01.
    expected_commission_w0 = 1.0 * (100.0 / 10_000.0)
    # Initial-entry slippage: |delta| = 1.0 at 50 bps -> 0.005.
    expected_slippage_w0 = 1.0 * (50.0 / 10_000.0)

    expected_borrow = np.array([expected_borrow_per_week] * 3)
    expected_commission = np.array([expected_commission_w0, 0.0, 0.0])
    expected_slippage = np.array([expected_slippage_w0, 0.0, 0.0])
    expected_cost = expected_borrow + expected_commission + expected_slippage
    expected_gross = np.full(3, EXPECTED_GROSS_PER_WEEK)
    expected_net = expected_gross - expected_cost

    # Engine breakdown matches term-by-term.
    np.testing.assert_allclose(res.returns["borrow"].to_numpy(), expected_borrow, atol=1e-12)
    np.testing.assert_allclose(res.returns["commission"].to_numpy(), expected_commission, atol=1e-12)
    np.testing.assert_allclose(res.returns["slippage"].to_numpy(), expected_slippage, atol=1e-12)
    np.testing.assert_allclose(res.returns["cost_total"].to_numpy(), expected_cost, atol=1e-12)
    np.testing.assert_allclose(res.returns["ret_gross"].to_numpy(), expected_gross, atol=1e-12)
    np.testing.assert_allclose(res.returns["ret_net"].to_numpy(), expected_net, atol=1e-12)

    # Costs strictly reduce returns (sanity-check on the sign convention).
    assert (res.returns["ret_net"] < res.returns["ret_gross"]).all()


# ---------------------------------------------------------------------------
# Turnover & trade log: deltas only fire at the initial entry
# ---------------------------------------------------------------------------
def test_turnover_zero_after_initial_entry() -> None:
    """Constant weights -> turnover = 0.5 (one-way) at D0, then 0 forever."""
    cfg = CostConfig(bps_round_trip=0.0, annual_borrow_pct=0.0, slippage_bps=0.0)
    res = backtest_weekly(WEIGHTS, PRICES, cost_config=cfg)
    # Turnover convention in the engine: one-way = sum|delta|/2.
    # At D0: |0.5| + |-0.5| + |0| = 1.0 -> one-way turnover = 0.5.
    assert res.returns["turnover"].iloc[0] == pytest.approx(0.5)
    assert res.returns["turnover"].iloc[1] == pytest.approx(0.0)
    assert res.returns["turnover"].iloc[2] == pytest.approx(0.0)

    # Trade log carries only the two non-zero initial entries (CCC stays flat
    # at zero so its delta is zero and is not logged).
    assert len(res.trades) == 2
    assert set(res.trades["Ticker"]) == {"AAA", "BBB"}
    assert (res.trades["old_weight"] == 0.0).all()


def test_borrow_only_for_longs_flag_charges_both_legs() -> None:
    """``bench_borrow_for_longs=True`` adds borrow on the long notional too."""
    cfg_short_only = CostConfig(
        bps_round_trip=0.0, slippage_bps=0.0, annual_borrow_pct=5.2,
        bench_borrow_for_longs=False,
    )
    cfg_both_legs = CostConfig(
        bps_round_trip=0.0, slippage_bps=0.0, annual_borrow_pct=5.2,
        bench_borrow_for_longs=True,
    )
    r1 = backtest_weekly(WEIGHTS, PRICES, cost_config=cfg_short_only).returns["borrow"].to_numpy()
    r2 = backtest_weekly(WEIGHTS, PRICES, cost_config=cfg_both_legs).returns["borrow"].to_numpy()
    # Long notional = 0.5 (AAA), short notional = 0.5 (BBB) -> charging both
    # legs exactly doubles the borrow line.
    np.testing.assert_allclose(r2, 2.0 * r1, atol=1e-12)
