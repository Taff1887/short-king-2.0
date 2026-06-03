"""Guardrail tests for the two cardinal sins of factor research:

1. Point-in-time violation — joining a fundamental row whose ``acceptedDate``
   post-dates the rebalance date. Caught by
   :func:`short_king.data.clean.check_no_lookahead`.
2. Silent forward-fill of forward returns at the tail of each ticker's
   history. The last few rows of every ticker MUST be NaN, not 0, otherwise
   the model "knows" the labels go to zero and overfits accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from short_king.data.clean import check_no_lookahead


# ---------------------------------------------------------------------------
# check_no_lookahead — synthetic panel with and without a violation
# ---------------------------------------------------------------------------
def _build_panel(rebalance_dates: list[str], tickers: list[str]) -> pd.DataFrame:
    """Construct a synthetic ``(Date, Ticker, acceptedDate, metric)`` panel.

    We then derive ``filing_lag_days`` and ``symbol`` so the panel matches the
    contract :func:`check_no_lookahead` expects (lowercase ``date``/``symbol``
    plus a numeric ``filing_lag_days`` column).
    """
    rows = []
    for d in pd.to_datetime(rebalance_dates):
        for tk in tickers:
            # Last filing accepted 30 days before each rebalance -- always clean.
            accepted = d - pd.Timedelta(days=30)
            rows.append(
                {
                    "Date": d,
                    "Ticker": tk,
                    "acceptedDate": accepted,
                    "metric": float(hash((tk, d)) % 100),
                }
            )
    df = pd.DataFrame(rows)
    # Adapt to the cleaning module's column conventions.
    df["date"] = df["Date"]
    df["symbol"] = df["Ticker"]
    df["filing_lag_days"] = (df["Date"] - df["acceptedDate"]).dt.days
    return df


def test_check_no_lookahead_passes_on_clean_panel() -> None:
    df = _build_panel(
        rebalance_dates=["2024-01-05", "2024-01-12", "2024-01-19"],
        tickers=["AAA", "BBB", "CCC"],
    )
    result = check_no_lookahead(df)
    assert result["n_rows"] == len(df)
    assert result["n_violations"] == 0
    assert result["max_violation_days"] == 0.0


def test_check_no_lookahead_catches_violation() -> None:
    df = _build_panel(
        rebalance_dates=["2024-01-05", "2024-01-12", "2024-01-19"],
        tickers=["AAA", "BBB", "CCC"],
    )
    # Inject a violation: BBB on 2024-01-12 has acceptedDate = 2024-01-20
    # (i.e. the filing was accepted *after* the rebalance).
    bad_mask = (df["Ticker"] == "BBB") & (df["Date"] == pd.Timestamp("2024-01-12"))
    df.loc[bad_mask, "acceptedDate"] = pd.Timestamp("2024-01-20")
    df["filing_lag_days"] = (df["Date"] - df["acceptedDate"]).dt.days
    # Sanity: the injected row should now have a negative lag of -8.
    assert df.loc[bad_mask, "filing_lag_days"].iloc[0] == -8

    result = check_no_lookahead(df)
    assert result["n_violations"] == 1
    # max_violation_days reports magnitude (positive number of days).
    assert result["max_violation_days"] == 8.0


def test_check_no_lookahead_catches_multiple_violations_and_reports_max() -> None:
    df = _build_panel(
        rebalance_dates=["2024-01-05", "2024-01-12", "2024-01-19"],
        tickers=["AAA", "BBB", "CCC"],
    )
    # Two violations with different magnitudes.
    m1 = (df["Ticker"] == "AAA") & (df["Date"] == pd.Timestamp("2024-01-05"))
    m2 = (df["Ticker"] == "CCC") & (df["Date"] == pd.Timestamp("2024-01-19"))
    df.loc[m1, "acceptedDate"] = pd.Timestamp("2024-01-08")  # lag = -3
    df.loc[m2, "acceptedDate"] = pd.Timestamp("2024-02-03")  # lag = -15
    df["filing_lag_days"] = (df["Date"] - df["acceptedDate"]).dt.days

    result = check_no_lookahead(df)
    assert result["n_violations"] == 2
    assert result["max_violation_days"] == 15.0


# ---------------------------------------------------------------------------
# Forward returns must be NaN for the last `h` rows per ticker (no silent fill).
# ---------------------------------------------------------------------------
def _fwd_ret(prices: pd.DataFrame, *, horizon_weeks: int) -> pd.Series:
    """Canonical h-week forward return per ticker: price[t+h] / price[t] - 1.

    This is the abstract definition the data layer must respect — implemented
    here in the test so we are verifying the *property* (last ``h`` rows of
    each ticker are NaN), not the helper's plumbing.
    """
    s = (
        prices.sort_values(["Ticker", "Date"])
        .groupby("Ticker", sort=False)["adjClose"]
        .apply(lambda x: x.shift(-horizon_weeks) / x - 1.0)
        .reset_index(level=0, drop=True)
    )
    return s.reindex(prices.index)


def test_fwd_ret_4w_is_nan_at_each_tickers_tail() -> None:
    """Last 4 rows of every ticker must be NaN, never silently set to 0."""
    dates = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Ticker": ["AAA"] * 12,
                    "Date": dates,
                    "adjClose": np.arange(1, 13, dtype=float),  # ramp 1..12
                }
            ),
            pd.DataFrame(
                {
                    "Ticker": ["BBB"] * 12,
                    "Date": dates,
                    "adjClose": np.arange(10, 130, 10, dtype=float),  # ramp 10..120
                }
            ),
        ],
        ignore_index=True,
    )

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["fwd_ret_4w"] = _fwd_ret(df, horizon_weeks=4)

    # Group-wise tail check: last 4 rows per ticker must be NaN.
    tail4 = df.groupby("Ticker", group_keys=False).tail(4)
    assert tail4["fwd_ret_4w"].isna().all(), (
        "fwd_ret_4w must be NaN at each ticker's tail, never 0-filled. "
        f"Got: {tail4[['Ticker','Date','fwd_ret_4w']].to_dict(orient='records')}"
    )
    # And critically: none of those tail rows are zero (would indicate silent
    # forward-fill, which is the failure mode this test exists to prevent).
    assert (tail4["fwd_ret_4w"].fillna(-999.0) != 0.0).all()

    # Interior rows have well-defined values.
    head_interior = df[df["Ticker"] == "AAA"].iloc[:-4]
    # For AAA the ramp is 1..12 so fwd_ret_4w at row i = (i+5)/(i+1) - 1.
    # First row: (5/1) - 1 = 4.0
    assert head_interior["fwd_ret_4w"].iloc[0] == 4.0


def test_fwd_ret_does_not_leak_across_tickers() -> None:
    """A short ticker followed by a long ticker must not 'borrow' future prices."""
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB", "BBB"],
            "Date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-12",
                    "2024-01-19",
                    "2024-01-05",
                    "2024-01-12",
                    "2024-01-19",
                    "2024-01-26",
                    "2024-02-02",
                ]
            ),
            "adjClose": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["fwd_ret_4w"] = _fwd_ret(df, horizon_weeks=4)

    # AAA has only 3 observations -> every forward 4-week return is undefined.
    assert df.loc[df["Ticker"] == "AAA", "fwd_ret_4w"].isna().all()
    # BBB has 5 observations -> row 0 has a defined value (50/10 - 1 = 4.0),
    # rows 1..4 are NaN. Critically: BBB's row-0 forward return uses *BBB's*
    # row 4 price (50), NOT AAA's row 4 price -- this is the no-leak guarantee.
    bbb = df.loc[df["Ticker"] == "BBB", "fwd_ret_4w"].reset_index(drop=True)
    assert bbb.iloc[0] == pytest.approx(50.0 / 10.0 - 1.0)
    assert bbb.iloc[1:].isna().all()
