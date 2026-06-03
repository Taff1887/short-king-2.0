"""Smoke tests for ``short_king.utils``.

These are intentionally minimal — they assert that the foundations (config
loading, Friday-anchored dates, deterministic hashing) work as the rest of
the codebase assumes. If any of these break, downstream tests will see
confusing failures, so we test the foundations explicitly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from short_king.utils.config import PROJECT_ROOT, Settings, get_settings, settings
from short_king.utils.dates import (
    asof_date,
    fridays_back,
    most_recent_friday,
)
from short_king.utils.io import hash_key


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_settings_loads_and_exposes_paths() -> None:
    """``settings`` should load from .env and expose the standard path tree."""
    assert isinstance(settings, Settings)
    # Singleton-ish: get_settings() reuses the same instance (lru_cache).
    assert get_settings() is settings

    # Project root is the repo root, not the cwd.
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").exists()

    # Derived paths are all under data/ or repo root, never absolute strings.
    assert isinstance(settings.data_dir, Path)
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.external_dir == settings.data_dir / "external"
    assert settings.asic_cache_dir == settings.raw_dir / "asic_cache"
    assert settings.fmp_cache_dir == settings.raw_dir / "fmp_cache"
    assert settings.charts_dir == PROJECT_ROOT / "charts"
    assert settings.reports_dir == PROJECT_ROOT / "reports"

    # Sensible non-empty defaults for the API config.
    assert settings.fmp_base_url.startswith("https://")
    assert settings.fmp_max_retries >= 1
    assert settings.fmp_calls_per_minute >= 1
    assert settings.asic_max_workers >= 1


def test_settings_ensure_dirs_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ensure_dirs`` should create the standard tree and be safe to re-run."""
    # Build a Settings subclass that points everything under tmp_path so we
    # don't pollute the real data/ tree during tests.
    class _TmpSettings(Settings):
        @property
        def data_dir(self) -> Path:  # type: ignore[override]
            return tmp_path / "data"

        @property
        def charts_dir(self) -> Path:  # type: ignore[override]
            return tmp_path / "charts"

        @property
        def reports_dir(self) -> Path:  # type: ignore[override]
            return tmp_path / "reports"

    s = _TmpSettings()
    s.ensure_dirs()
    s.ensure_dirs()  # idempotent
    for p in (
        s.raw_dir,
        s.processed_dir,
        s.external_dir,
        s.asic_cache_dir,
        s.fmp_cache_dir,
        s.charts_dir,
        s.reports_dir,
    ):
        assert p.is_dir(), f"{p} should exist after ensure_dirs"


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # A Friday returns itself.
        (dt.date(2024, 1, 5), dt.date(2024, 1, 5)),
        # Saturday rolls back to Friday.
        (dt.date(2024, 1, 6), dt.date(2024, 1, 5)),
        # Sunday rolls back to Friday.
        (dt.date(2024, 1, 7), dt.date(2024, 1, 5)),
        # Monday rolls back to the *previous* Friday.
        (dt.date(2024, 1, 8), dt.date(2024, 1, 5)),
        # Mid-week (Wednesday) rolls back to the most recent Friday.
        (dt.date(2024, 1, 10), dt.date(2024, 1, 5)),
        # Thursday — one day before Friday — still rolls back a full week.
        (dt.date(2024, 1, 11), dt.date(2024, 1, 5)),
    ],
)
def test_most_recent_friday(ref: dt.date, expected: dt.date) -> None:
    got = most_recent_friday(ref)
    assert got == expected
    # Result must always be a Friday (weekday == 4).
    assert got.weekday() == 4


def test_most_recent_friday_default_today_is_a_friday() -> None:
    got = most_recent_friday()
    assert got.weekday() == 4
    assert got <= dt.date.today()


def test_fridays_back_yields_consecutive_fridays() -> None:
    anchor = dt.date(2024, 1, 5)  # Friday
    out = fridays_back(4, anchor=anchor)
    assert len(out) == 4
    assert out[0] == anchor
    # Each step is 7 days earlier than the previous.
    for prev, cur in zip(out[:-1], out[1:]):
        assert (prev - cur).days == 7
    # All are Fridays.
    assert all(d.weekday() == 4 for d in out)


def test_asof_date_is_release_minus_four_business_days() -> None:
    # 2024-01-12 (Fri) - 4 business days = 2024-01-08 (Mon).
    rd = pd.Timestamp("2024-01-12")
    assert asof_date(rd) == pd.Timestamp("2024-01-08")
    # Across a weekend: 2024-01-09 (Tue) - 4 BDays = 2024-01-03 (Wed).
    assert asof_date(pd.Timestamp("2024-01-09")) == pd.Timestamp("2024-01-03")
    # Accepts a python date too.
    assert asof_date(dt.date(2024, 1, 12)) == pd.Timestamp("2024-01-08")


# ---------------------------------------------------------------------------
# io.hash_key
# ---------------------------------------------------------------------------
def test_hash_key_is_deterministic_and_order_sensitive() -> None:
    a = hash_key("fmp", "income_statement", "BHP.AX", 2024)
    b = hash_key("fmp", "income_statement", "BHP.AX", 2024)
    # Same inputs -> same key.
    assert a == b
    # Reordering parts changes the key (cache keys are positional).
    assert hash_key("fmp", "income_statement", "BHP.AX", 2024) != hash_key(
        "fmp", "income_statement", 2024, "BHP.AX"
    )
    # Different inputs -> different key.
    assert hash_key("fmp", "balance_sheet", "BHP.AX", 2024) != a


def test_hash_key_short_hex_string() -> None:
    """The cache-key format must stay stable: 16-char lowercase hex."""
    h = hash_key("anything", 1, None)
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
