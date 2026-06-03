"""Unit tests for ``short_king.features``: transforms, price features and signals.

Synthetic-input only; no network and no on-disk data. Each test pins one
observable property (mean/std/range/exact-value), so a regression in any
single transform surfaces directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from short_king.features.price import momentum
from short_king.features.short_signals import si_persistence
from short_king.features.transforms import (
    cross_sectional_rank,
    cross_sectional_zscore,
    safe_divide,
    winsorize,
)


# ---------------------------------------------------------------------------
# transforms: cross_sectional_zscore / cross_sectional_rank / winsorize / safe_divide
# ---------------------------------------------------------------------------
def _panel(seed: int = 0, n_dates: int = 4, n_tickers: int = 30) -> pd.DataFrame:
    """Deterministic cross-section panel for transform tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-05", periods=n_dates, freq="W-FRI")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        # Heavy-tailed feature so winsor / rank / z all have something to do.
        x = rng.standard_t(df=3, size=n_tickers)
        for t, v in zip(tickers, x):
            rows.append({"Date": d, "Ticker": t, "metric": float(v)})
    return pd.DataFrame(rows)


def test_cross_sectional_zscore_is_finite_zero_mean_unit_std_per_date() -> None:
    """Within each Date the z-scored series should have mean ~0 and std ~1."""
    df = _panel()
    out = cross_sectional_zscore(df, by="Date", cols=["metric"])
    assert "metric_z" in out.columns
    # No infinities or NaNs introduced (input was finite for every row).
    z = out["metric_z"]
    assert np.isfinite(z).all(), "z-score should be finite for finite input"

    # Per-date stats.
    grp = out.groupby("Date", observed=True)["metric_z"]
    # mean ~ 0 (winsorisation can shift it a tiny amount; allow 1e-9 tolerance
    # against population-stat z-score with ddof=0).
    means = grp.mean()
    assert (means.abs() < 1e-9).all(), f"per-date means not ~0: {means.to_dict()}"
    # std (ddof=0) is exactly 1 by construction; pandas std default ddof=1, so
    # check both styles with a small tolerance.
    stds_pop = grp.std(ddof=0)
    assert np.allclose(stds_pop, 1.0, atol=1e-9), f"per-date std(ddof=0) not ~1: {stds_pop.to_dict()}"


def test_cross_sectional_rank_values_in_unit_interval() -> None:
    df = _panel()
    out = cross_sectional_rank(df, by="Date", cols=["metric"], pct=True)
    assert "metric_rk" in out.columns
    rk = out["metric_rk"].dropna()
    assert ((rk >= 0.0) & (rk <= 1.0)).all(), "ranks must be in [0, 1]"
    # Each cross-section's max rank is 1.0 (largest value gets pct=1) when
    # ascending=True (default) and no ties on the extreme.
    per_date_max = out.groupby("Date", observed=True)["metric_rk"].max()
    assert np.allclose(per_date_max, 1.0)
    per_date_min = out.groupby("Date", observed=True)["metric_rk"].min()
    # min rank = 1/N for an N-ticker cross-section.
    n = df.groupby("Date", observed=True).size().iloc[0]
    assert np.allclose(per_date_min, 1.0 / n)


def test_winsorize_clips_to_quantiles() -> None:
    # Inject obvious outliers; the 1%/99% quantiles of a 0..100 ramp are
    # 1.0 and 99.0 by default in pandas (linear interpolation).
    s = pd.Series(list(range(0, 101)) + [-1_000_000, 1_000_000], dtype=float)
    w = winsorize(s, lower=0.01, upper=0.99)
    # No values outside the clipping bounds.
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    assert (w >= lo).all() and (w <= hi).all()
    # Extreme outliers got clipped to those bounds (not just shrunk).
    assert w.iloc[-2] == lo  # -1e6 -> lo
    assert w.iloc[-1] == hi  # +1e6 -> hi
    # Interior values unchanged.
    interior = s.iloc[:101]
    np.testing.assert_array_equal(
        w.iloc[:101].clip(lo, hi).to_numpy(),
        interior.clip(lo, hi).to_numpy(),
    )


def test_safe_divide_returns_nan_on_zero_denominator() -> None:
    num = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    den = pd.Series([2.0, 0.0, 4.0, np.nan, -5.0])
    out = safe_divide(num, den)
    # Index 0: 1/2 = 0.5
    assert out.iloc[0] == pytest.approx(0.5)
    # Index 1: denom 0 -> NaN (no inf)
    assert np.isnan(out.iloc[1])
    # Index 2: 3/4 = 0.75
    assert out.iloc[2] == pytest.approx(0.75)
    # Index 3: NaN denom -> NaN
    assert np.isnan(out.iloc[3])
    # Index 4: 5/-5 = -1
    assert out.iloc[4] == pytest.approx(-1.0)
    # And critically: no infinities anywhere in the output.
    assert not np.isinf(out).any()


# ---------------------------------------------------------------------------
# price: momentum on a synthetic ramp
# ---------------------------------------------------------------------------
def test_momentum_on_ramp_returns_expected_pct_change() -> None:
    """A linear adjClose ramp [1,2,3,...,10] gives pct_change(4) = +400% at step 4."""
    dates = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    df = pd.DataFrame({
        "Ticker": ["AAA"] * 10,
        "Date": dates,
        "adjClose": np.arange(1, 11, dtype=float),
    })
    mom = momentum(df, weeks=4)
    # First four rows have no 4-week-back price -> NaN.
    assert mom.iloc[:4].isna().all()
    # Row 4: (5 - 1) / 1 = 4.0  (a 400% gain over 4 weeks)
    assert mom.iloc[4] == pytest.approx(4.0)
    # Row 5: (6 - 2) / 2 = 2.0
    assert mom.iloc[5] == pytest.approx(2.0)
    # Row 9: (10 - 6) / 6 = 0.6666...
    assert mom.iloc[9] == pytest.approx(4.0 / 6.0)


def test_momentum_groups_by_ticker() -> None:
    """Two tickers must not leak prices into each other's momentum window."""
    dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    df = pd.DataFrame({
        "Ticker": ["A"] * 6 + ["B"] * 6,
        "Date": list(dates) + list(dates),
        "adjClose": [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60],
    })
    mom = momentum(df, weeks=4)
    # Both tickers have NaN for their first 4 rows.
    assert mom.iloc[0:4].isna().all()
    assert mom.iloc[6:10].isna().all()
    # Ticker A: (5-1)/1 = 4
    assert mom.iloc[4] == pytest.approx(4.0)
    # Ticker B: (50-10)/10 = 4 -- same shape, scaled prices, same return.
    assert mom.iloc[10] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# short_signals.si_persistence — counts trailing weeks of rising ShortPct
# ---------------------------------------------------------------------------
def test_si_persistence_hand_crafted_count() -> None:
    """``si_persistence`` counts up-weeks of ``ShortPct`` in a trailing window.

    For a per-ticker series we know exactly which weeks were "up", so the
    rolling count is closed-form. We use window=5 (the module default) and
    pin the value at every index of the hand-crafted series.
    """
    # Ticker A: clean run of 5 rising weeks then 2 falling weeks.
    #   ShortPct      : [5, 6, 7, 8, 9, 10, 9, 8]
    #   diff > 0      : [NaN, T, T, T, T,  T, F, F]
    #   ups (1/0)     : [NaN, 1, 1, 1, 1,  1, 0, 0]
    #   rolling sum, window=5, min_periods=3:
    #     idx 0 (1 obs, all NaN)             -> NaN
    #     idx 1 (window=[NaN,1])             -> NaN  (1 valid <  3)
    #     idx 2 (window=[NaN,1,1])           -> NaN  (2 valid <  3)
    #     idx 3 (window=[NaN,1,1,1])         -> 3.0  (3 valid -> sum 1+1+1)
    #     idx 4 (window=[NaN,1,1,1,1])       -> 4.0
    #     idx 5 (window=[1,1,1,1,1])         -> 5.0
    #     idx 6 (window=[1,1,1,1,0])         -> 4.0
    #     idx 7 (window=[1,1,1,0,0])         -> 3.0
    # Ticker B: monotonically falling -> never an up-week -> always 0 after warm-up.
    dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    panel = pd.DataFrame({
        "Ticker": ["A"] * 8 + ["B"] * 8,
        "Date": list(dates) + list(dates),
        "ShortPct": [5, 6, 7, 8, 9, 10, 9, 8] + [10, 9, 8, 7, 6, 5, 4, 3],
    })

    result = si_persistence(panel, window=5)
    assert isinstance(result, pd.Series)
    assert result.name == "si_persistence"

    a = result.iloc[:8].to_numpy()
    b = result.iloc[8:].to_numpy()

    # Ticker A: pinned hand-computed values.
    expected_a = np.array([np.nan, np.nan, np.nan, 3.0, 4.0, 5.0, 4.0, 3.0])
    np.testing.assert_array_equal(np.isnan(a), np.isnan(expected_a))
    np.testing.assert_allclose(a[~np.isnan(a)], expected_a[~np.isnan(expected_a)], atol=1e-12)

    # Ticker B: zero up-weeks once the rolling warm-up clears.
    assert np.isnan(b[0]) and np.isnan(b[1]) and np.isnan(b[2])
    np.testing.assert_allclose(b[3:], np.zeros(5), atol=1e-12)

    # Aggregate count of "high-persistence" cells (>= 3) across the panel.
    # Ticker A contributes 5 (indices 3..7); Ticker B contributes 0.
    assert int((result >= 3).sum()) == 5
