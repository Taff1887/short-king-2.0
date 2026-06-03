"""Unit tests for ``short_king.models.metrics``: IC, rank-IC, decile spread.

These are deterministic, closed-form examples — no stochasticity, no
network. The point is to lock the metric definitions so any future refactor
(e.g. switching from Pearson to Spearman, or changing the spread convention)
is caught immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from short_king.models.metrics import decile_spread, ic, ic_summary, rank_ic


# ---------------------------------------------------------------------------
# rank_ic / ic on deterministic series
# ---------------------------------------------------------------------------
def test_rank_ic_perfect_monotone_returns_one() -> None:
    """Strictly increasing predictions vs strictly increasing targets -> rho = 1."""
    pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    tgt = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    assert rank_ic(pred, tgt) == pytest.approx(1.0)
    # Same input through Pearson IC — also +1 for an affine relationship.
    assert ic(pred, tgt) == pytest.approx(1.0)


def test_rank_ic_anti_monotone_returns_minus_one() -> None:
    """Predictions sorted opposite to targets -> rho = -1."""
    pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    tgt = pd.Series([50.0, 40.0, 30.0, 20.0, 10.0])
    assert rank_ic(pred, tgt) == pytest.approx(-1.0)
    assert ic(pred, tgt) == pytest.approx(-1.0)


def test_ic_is_pearson_correlation() -> None:
    """Pearson IC matches numpy's reference correlation exactly."""
    rng = np.random.default_rng(42)
    p = pd.Series(rng.normal(size=200))
    # Targets = 2*pred + noise -> non-trivial but well-defined Pearson IC.
    t = 2.0 * p + pd.Series(rng.normal(scale=0.5, size=200))
    expected = np.corrcoef(p.to_numpy(), t.to_numpy())[0, 1]
    assert ic(p, t) == pytest.approx(expected, rel=1e-12)


def test_rank_ic_is_spearman_correlation() -> None:
    """Spearman / rank IC matches the pandas reference exactly."""
    rng = np.random.default_rng(7)
    p = pd.Series(rng.normal(size=100))
    # Non-linear (cubic) target -> Spearman differs from Pearson; ranks line up.
    t = p**3 + pd.Series(rng.normal(scale=0.1, size=100))
    expected = p.corr(t, method="spearman")
    assert rank_ic(p, t) == pytest.approx(expected, rel=1e-12)


def test_rank_ic_per_group_returns_series() -> None:
    """Passing ``by`` (e.g. Date) returns one IC per group."""
    dates = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    by = pd.Series(list(dates.repeat(5)))
    # Group 1 is perfectly monotone (+1), group 2 is anti-monotone (-1).
    pred = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
    tgt = pd.Series([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=float)
    s = rank_ic(pred, tgt, by=by)
    assert isinstance(s, pd.Series)
    assert len(s) == 2
    vals = s.sort_index().to_numpy()
    np.testing.assert_allclose(vals, [1.0, -1.0], atol=1e-12)


def test_ic_summary_reports_mean_std_t_hitrate_n() -> None:
    """Aggregator should report mean / std / t-stat / hit-rate / n correctly."""
    s = pd.Series([0.1, 0.2, -0.05, 0.15, 0.0])
    summary = ic_summary(s)
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(float(s.mean()))
    assert summary["std"] == pytest.approx(float(s.std(ddof=1)))
    expected_t = summary["mean"] / (summary["std"] / np.sqrt(5))
    assert summary["t_stat"] == pytest.approx(expected_t)
    # Hit-rate counts strictly positive IC periods: 3 out of 5 = 0.6
    assert summary["hit_rate"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# decile_spread on a synthetic monotone signal
# ---------------------------------------------------------------------------
def test_decile_spread_positive_on_monotone_signal() -> None:
    """When predictions and targets are perfectly co-monotone, top-bottom > 0."""
    dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    n_per_date = 50  # large enough that qcut with n=10 gives 5 names per bucket
    rows = []
    for d in dates:
        for i in range(n_per_date):
            # pred linear in i; target identical -> perfect monotone signal.
            rows.append({"Date": d, "pred": float(i), "tgt": float(i)})
    df = pd.DataFrame(rows)
    s = decile_spread(df["pred"], df["tgt"], by=df["Date"], n=10)
    assert isinstance(s, pd.Series)
    # 3 dates -> 3 spreads, every one strictly positive.
    assert len(s) == 3
    assert (s > 0).all(), f"expected all positive spreads, got {s.to_dict()}"

    # Closed-form: with i = 0..49, top decile is i in [45..49] (mean 47),
    # bottom decile is i in [0..4] (mean 2) -> spread = 45 every week.
    np.testing.assert_allclose(s.to_numpy(), 45.0, atol=1e-9)


def test_decile_spread_anti_monotone_returns_negative() -> None:
    """Inverted signal -> spread should flip sign exactly."""
    dates = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    n_per_date = 50
    rows = []
    for d in dates:
        for i in range(n_per_date):
            # pred = i, target = -i: top-pred decile coincides with bottom-tgt decile.
            rows.append({"Date": d, "pred": float(i), "tgt": -float(i)})
    df = pd.DataFrame(rows)
    s = decile_spread(df["pred"], df["tgt"], by=df["Date"], n=10)
    assert (s < 0).all()
    np.testing.assert_allclose(s.to_numpy(), -45.0, atol=1e-9)


def test_decile_spread_nan_when_group_smaller_than_n_buckets() -> None:
    """A cross-section of fewer names than ``n`` buckets yields NaN for that period."""
    dates = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    rows = []
    # First date has only 5 names -> qcut(n=10) is undefined -> NaN.
    for i in range(5):
        rows.append({"Date": dates[0], "pred": float(i), "tgt": float(i)})
    # Second date has 50 names -> well-defined spread.
    for i in range(50):
        rows.append({"Date": dates[1], "pred": float(i), "tgt": float(i)})
    df = pd.DataFrame(rows)
    s = decile_spread(df["pred"], df["tgt"], by=df["Date"], n=10)
    assert np.isnan(s.iloc[0])
    assert s.iloc[1] == pytest.approx(45.0, abs=1e-9)
