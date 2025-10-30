from datetime import datetime, timedelta
import logging
from app.data_fetch import fetch_data
from app.database import save_to_database, init_database
from app.logger import setup_logging
from config.config import AppConfig

logger = logging.getLogger(__name__)

def populate_historical_data(config: AppConfig, days: int = 60) -> None:
    """Fetch and store historical raw data for backtesting. No cleaning/filling to preserve gaps.
    Caps days at 730 for yfinance intraday limits, initializes DB, fetches via data_fetch.py,
    and saves to database.py. Used for initial backfill before full ETL runs.

    Args:
        config: Application configuration for ticker, logging, and database.
        days: Number of days to fetch (default: 60, capped at 730).

    Returns:
        None: Logs success or failure.
    """
    setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)

    # Cap days for intraday (precaution)
    max_days = 730
    if days > max_days:
        days = max_days
        logger.warning(f"Days capped at {max_days} due to yfinance intraday limits.")

    # Initialize database
    init_database(config)

    # Date range
    end_date = datetime.now().replace(tzinfo=None)
    start_date = end_date - timedelta(days=days)

    logger.info(f"Populating historical data for {days} days from {start_date} to {end_date}")

    # Fetch raw data
    df = fetch_data(config, start_date=start_date, end_date=end_date)
    if df is None:
        logger.error("Failed to fetch historical data")
        return

    # Save raw data
    success = save_to_database(config, df, config.trading.ticker)
    if success:
        logger.info(f"Successfully saved {len(df)} raw rows for {config.trading.ticker}")
    else:
        logger.error("Failed to save historical data to database")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Populate historical data for trading platform.")
    parser.add_argument("--days", type=int, default=60, help="Number of days to fetch (default: 60)")
    args = parser.parse_args()

    config = AppConfig()
    populate_historical_data(config, days=args.days)