"""Test how far back ASIC publishes daily-aggregate short-position reports.

Pings the URL pattern at a handful of historical Fridays and reports which
returned a parseable PDF. Cheap (HTTP HEAD on each candidate) so we don't
spam the ASIC server.
"""

from __future__ import annotations

import datetime as dt

import requests

from short_king.utils.config import settings
from short_king.utils.dates import most_recent_friday

CANDIDATE_YEARS = [2025, 2023, 2020, 2017, 2014, 2012, 2011, 2010]


def url_for(d: dt.date) -> str:
    return settings.asic_base_url.format(datestr=d.strftime("%Y%m%d"))


def head_ok(url: str, timeout: float = 8.0) -> tuple[int, int | None]:
    """(status_code, content_length-or-None) via HEAD."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        cl = int(r.headers.get("content-length", 0)) or None
        return r.status_code, cl
    except Exception as exc:
        return -1, None


def main() -> None:
    today = dt.date.today()
    print(f"# Probing ASIC daily-aggregate short-position reports (today = {today})\n")
    print(f"{'Friday':12s}  {'status':>6s}  {'bytes':>10s}  url")
    for year in CANDIDATE_YEARS:
        # Pick a few Fridays per year (June, Sept, Dec) for coverage.
        for month, day_hint in [(6, 15), (9, 15), (12, 15)]:
            try:
                ref = dt.date(year, month, day_hint)
            except ValueError:
                continue
            if ref >= today:
                continue
            fri = most_recent_friday(ref)
            url = url_for(fri)
            code, length = head_ok(url)
            ok = "OK" if code == 200 else f"-{code}"
            length_s = f"{length:,}" if length else "-"
            print(f"{str(fri):12s}  {ok:>6s}  {length_s:>10s}  {url[-60:]}")
        print()


if __name__ == "__main__":
    main()
