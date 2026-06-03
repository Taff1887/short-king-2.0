"""Smoke-test the utils package wiring (config, logging, io, dates)."""

from short_king.utils.config import settings
from short_king.utils.dates import asof_date, fridays_back, most_recent_friday
from short_king.utils.io import hash_key
from short_king.utils.logging import logger

settings.ensure_dirs()
logger.info(f"FMP key set: {bool(settings.fmp_api_key)}")
logger.info(f"Project root: {settings.data_dir.parent}")
logger.info(f"Most recent Friday: {most_recent_friday()}")
logger.info(f"Last 3 Fridays: {fridays_back(3)}")
logger.info(f"As-of for today's Friday: {asof_date(most_recent_friday()).date()}")
logger.info(f"Cache key sample: {hash_key('foo', 1, 2)}")
