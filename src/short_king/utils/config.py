"""Application configuration via pydantic-settings.

Every setting can be overridden through an environment variable or a project-root
``.env`` file (see ``.env.example``). Filesystem paths are derived from the
project root so the package behaves identically from notebooks, scripts and
tests regardless of the current working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/short_king/utils/config.py  ->  parents[3] == repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Financial Modeling Prep ---
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    fmp_max_retries: int = 5
    fmp_calls_per_minute: int = 300

    # --- ASIC short-selling reports ---
    asic_base_url: str = "https://download.asic.gov.au/short-selling/RR{datestr}-001-SSDailyAggShortPos.pdf"
    asic_max_workers: int = 6

    # --- Logging ---
    log_level: str = "INFO"

    # --- Derived filesystem paths ---
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def asic_cache_dir(self) -> Path:
        return self.raw_dir / "asic_cache"

    @property
    def fmp_cache_dir(self) -> Path:
        return self.raw_dir / "fmp_cache"

    @property
    def charts_dir(self) -> Path:
        return PROJECT_ROOT / "charts"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    def ensure_dirs(self) -> None:
        for p in (
            self.raw_dir,
            self.processed_dir,
            self.external_dir,
            self.asic_cache_dir,
            self.fmp_cache_dir,
            self.charts_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
