import logging
from app.data_fetch import fetch_data
from app.database import init_database, save_to_database
from app.transform import transform_data
from app.backtest import run_backtest
from config.config import AppConfig
from app.logger import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Main entry point: Run ETL pipeline and backtest.
    Orchestrates the full workflow: Initialize DB, fetch and save raw data,
    transform with indicators, and execute backtest. Central coordinator
    for the ETL process, logging each step.

    Returns:
        None.
    """
    config = AppConfig()
    setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)
    logger.info("Starting Agentic AI Trading Platform with Gaussian + Kijun Strategy")

    init_database(config)

    # Fetch and save raw data
    df = fetch_data(config)
    if df is not None:
        save_to_database(config, df, config.trading.ticker)
        logger.info("ETL: Data fetched and saved")

    # Transform with indicators
    transformed_df = transform_data(config, config.trading.ticker)
    if transformed_df is not None:
        logger.info("ETL: Data transformed with indicators")

        # Run backtest (saves CSV and plot internally)
        try:
            summary = run_backtest(transformed_df, config=config)
            logger.info(f"Backtest completed: {summary}")
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")

if __name__ == "__main__":
    main()