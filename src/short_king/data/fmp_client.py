"""Financial Modeling Prep (FMP) REST client for ASX equities.

A thin, deterministic wrapper around the FMP *stable* API with three properties
that matter for reproducible short-selling research:

* **On-disk JSON caching** — every raw payload is cached under
  ``settings.fmp_cache_dir`` keyed by endpoint + sorted params. The API key is
  never part of the cache key, so caches travel with the repo. Pass
  ``force_refresh=True`` to bypass.
* **Robust retries** — transient network errors, rate-limit (429) and 5xx
  responses are retried with exponential backoff via ``tenacity``.
* **Polite throttling** — a minimum interval between calls keeps the client
  inside tier limits when looping over hundreds of ASX symbols.

ASX tickers use the ``.AX`` suffix on FMP (e.g. ``BHP.AX``, ``CBA.AX``). An
empty list response is treated as a "no data" condition, not an error — common
for thinly covered small-caps.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from short_king.utils.config import Settings
from short_king.utils.config import settings as default_settings
from short_king.utils.io import hash_key, read_json_cache, write_json_cache
from short_king.utils.logging import logger


class FMPError(RuntimeError):
    """Non-retryable FMP API error (bad key, malformed request, ...)."""


class FMPTransientError(RuntimeError):
    """Retryable rate-limit (429) or server (5xx) error."""


class FMPClient:
    """Caching, rate-limited client for the FMP REST API (stable endpoints)."""

    def __init__(self, settings: Settings | None = None, *, force_refresh: bool = False):
        self.settings = settings or default_settings
        if not self.settings.fmp_api_key:
            logger.warning("FMP_API_KEY is empty — set it in .env before making live calls.")
        self.base_url = self.settings.fmp_base_url.rstrip("/")
        self.force_refresh = force_refresh
        self._session = requests.Session()
        self._min_interval = 60.0 / max(self.settings.fmp_calls_per_minute, 1)
        self._last_call = 0.0
        self._lock = threading.Lock()

    # -- low level ---------------------------------------------------------
    def _throttle(self) -> None:
        """Sleep just long enough to respect ``fmp_calls_per_minute``."""
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type((FMPTransientError, requests.RequestException)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(default_settings.fmp_max_retries),
        reraise=True,
    )
    def _request(self, url: str, params: dict[str, Any]) -> Any:
        """Single HTTP GET with retry on transient failures."""
        self._throttle()
        resp = self._session.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning(f"FMP transient {resp.status_code} for {url} — retrying")
            raise FMPTransientError(f"HTTP {resp.status_code}")
        if resp.status_code in (401, 403):
            raise FMPError(f"Auth error {resp.status_code}: check FMP_API_KEY / subscription tier")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            raise FMPError(str(data["Error Message"]))
        return data

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        force_refresh: bool | None = None,
    ) -> Any:
        """GET an endpoint (relative to base URL) with transparent caching."""
        params = dict(params or {})
        key = hash_key(endpoint, sorted(params.items()))
        refresh = self.force_refresh if force_refresh is None else force_refresh
        if not refresh:
            cached = read_json_cache(key)
            if cached is not None:
                return cached
        params["apikey"] = self.settings.fmp_api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = self._request(url, params)
        write_json_cache(key, data)
        return data

    # -- prices ------------------------------------------------------------
    def historical_price_eod_adjusted(self, symbol: str) -> list[dict]:
        """Split- & dividend-adjusted daily OHLCV (use this for return series)."""
        data = self.get(
            "historical-price-eod/dividend-adjusted",
            params={"symbol": symbol},
        )
        return data or []

    def historical_price_eod_full(self, symbol: str) -> list[dict]:
        """Raw daily OHLCV plus change/changePercent/vwap (unadjusted prices)."""
        data = self.get(
            "historical-price-eod/full",
            params={"symbol": symbol},
        )
        return data or []

    # -- company profile ---------------------------------------------------
    def profile(self, symbol: str) -> list[dict]:
        """Company profile — sector, industry, mktCap, averageVolume, beta, ..."""
        data = self.get("profile", params={"symbol": symbol})
        return data or []

    # -- fundamentals ------------------------------------------------------
    def _statement(self, endpoint: str, symbol: str, period: str, limit: int) -> list[dict]:
        """Shared shape for the period+limit financial-statement endpoints."""
        data = self.get(
            endpoint,
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        return data or []

    def income_statement(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("income-statement", symbol, period, limit)

    def balance_sheet(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("balance-sheet-statement", symbol, period, limit)

    def cash_flow(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("cash-flow-statement", symbol, period, limit)

    def ratios(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("ratios", symbol, period, limit)

    def key_metrics(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("key-metrics", symbol, period, limit)

    def enterprise_values(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("enterprise-values", symbol, period, limit)

    def financial_growth(
        self, symbol: str, *, period: str = "quarter", limit: int = 80
    ) -> list[dict]:
        return self._statement("financial-growth", symbol, period, limit)

    # -- universe (S&P 500 — kept for cross-market parity with sister repo) -
    # NOTE: FMP does not publish a comparable point-in-time ASX 200 / All Ords
    # constituents endpoint; the ASX universe is built from ASIC short-position
    # reports in short_king.data.asic_universe. These wrappers are provided
    # because the rest of the codebase consumes them when benchmarking against
    # the qfr (S&P 500) reference implementation.
    def sp500_constituents_current(self) -> list[dict]:
        """Current S&P 500 members (symbol, name, sector, sub-sector, ...)."""
        return self.get("sp500-constituent") or []

    def sp500_constituents_changes(self) -> list[dict]:
        """Historical S&P 500 additions / removals with effective dates."""
        return self.get("historical-sp500-constituent") or []
