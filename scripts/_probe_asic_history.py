"""Find the EARLIEST archived ASIC daily-aggregate report.

ASIC introduced daily short-position reporting on 1 June 2010. The URL
template we use only consistently archives from mid-2010 — we probe every
Friday between mid-May 2010 and end-July 2010 to find the exact floor.
"""

from __future__ import annotations

import datetime as dt

import requests

from short_king.utils.config import settings
from short_king.utils.dates import most_recent_friday


def url_for(d: dt.date) -> str:
    return settings.asic_base_url.format(datestr=d.strftime("%Y%m%d"))


def head_ok(url: str, timeout: float = 8.0) -> tuple[int, int | None]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        cl = int(r.headers.get("content-length", 0)) or None
        return r.status_code, cl
    except Exception:
        return -1, None


def main() -> None:
    # Walk every business day from 14 May 2010 → 1 Aug 2010.
    start = dt.date(2010, 5, 14)
    end = dt.date(2010, 8, 1)
    cur = start
    print(f"{'date':12s}  {'dow':3s}  {'status':>6s}  {'bytes':>10s}")
    earliest_ok = None
    while cur <= end:
        if cur.weekday() < 5:  # Mon-Fri
            url = url_for(cur)
            code, length = head_ok(url)
            ok = "OK" if code == 200 else f"-{code}"
            length_s = f"{length:,}" if length else "-"
            print(f"{str(cur):12s}  {cur.strftime('%a'):3s}  {ok:>6s}  {length_s:>10s}")
            if code == 200 and earliest_ok is None:
                earliest_ok = cur
        cur += dt.timedelta(days=1)
    print()
    if earliest_ok:
        print(f"Earliest archived report: {earliest_ok} ({earliest_ok.strftime('%A')})")
        # Compute span to today.
        today = dt.date.today()
        days = (today - earliest_ok).days
        years = days / 365.25
        print(f"Span to today ({today}): {days} days = {years:.2f} years")
    else:
        print("No reports found in the probe window.")


if __name__ == "__main__":
    main()
