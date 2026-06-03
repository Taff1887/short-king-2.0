"""Re-apply the kept-universe filter to asic_long.parquet.

01_pull_asic overwrites asic_long.parquet whenever it's re-run; this small
helper restores the top-N universe filter so 04+ stay aligned with the FMP
fundamentals + prices that were pulled for the kept tickers.
"""

from short_king.utils.config import settings
from short_king.utils.io import read_parquet, write_parquet
from short_king.utils.logging import logger

asic_path = settings.processed_dir / "asic_long.parquet"
universe_path = settings.processed_dir / "universe_tickers.txt"

asic = read_parquet(asic_path)
keep = set(universe_path.read_text(encoding="utf-8").splitlines())
keep.discard("")
before = len(asic)
asic = asic[asic["Ticker"].isin(keep)].reset_index(drop=True)
write_parquet(asic, asic_path)
logger.info(
    f"_refilter_asic: kept {len(asic):,}/{before:,} rows | "
    f"{asic['Ticker'].nunique()} tickers | {asic['Date'].nunique()} dates"
)
