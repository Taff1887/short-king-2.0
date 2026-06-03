"""Date helpers — ASIC short-selling reports are anchored to Friday end-of-week.

The ASIC file dated ``RRYYYYMMDD-001-...`` is the daily aggregate for that
release date; the as-of position is four business days earlier.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from pandas.tseries.offsets import BDay


def most_recent_friday(ref_date: dt.date | None = None) -> dt.date:
    """The last calendar Friday on/before ``ref_date`` (default = today)."""
    ref_date = ref_date or dt.date.today()
    offset = (ref_date.weekday() - 4) % 7
    return ref_date - dt.timedelta(days=offset)


def fridays_back(weeks: int, anchor: dt.date | None = None) -> list[dt.date]:
    """``weeks`` consecutive Fridays, anchored on the most recent Friday."""
    anchor = anchor or most_recent_friday()
    return [anchor - dt.timedelta(weeks=k) for k in range(weeks)]


def asof_date(release_date: pd.Timestamp | dt.date) -> pd.Timestamp:
    """ASIC release_date - 4 business days = the report's as-of date."""
    rd = pd.Timestamp(release_date).normalize()
    return (rd - BDay(4)).normalize()


def week_end_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Friday week-ends in ``[start, end]``."""
    return pd.date_range(start=start, end=end, freq="W-FRI")


def month_end_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="ME")


def quarter_end_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="QE")
