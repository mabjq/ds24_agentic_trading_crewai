from typing import Optional
from datetime import datetime, timedelta
import logging
import pandas as pd
import yfinance as yf
from config.config import AppConfig

logger = logging.getLogger(__name__)

def fetch_data(config: AppConfig, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for the specified ticker and timeframe from yfinance.
    Focuses on extraction, detailed cleaning is handled in the transform module.

    Args:
        config: Application configuration containing ticker and timeframe.
        start_date: Start date for data fetch (default: lookback_days ago).
        end_date: End date for data fetch (default: now).

    Returns:
        DataFrame with OHLCV data, or None if fetch fails.
    """
    if end_date is None:
        end_date = datetime.now().replace(tzinfo=None)
    if start_date is None:
        start_date = end_date - timedelta(days=config.trading.lookback_days)

    try:
        logger.info(f"Fetching data for {config.trading.ticker} from {start_date} to {end_date} with timeframe {config.trading.timeframe}")

        ticker = yf.Ticker(config.trading.ticker)
        data = ticker.history(
            interval=config.trading.timeframe,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            actions=False
        )

        if data.empty:
            logger.error(f"No data retrieved for {config.trading.ticker}")
            return None

        # Handle yfinance's DatetimeIndex and ensure timezone-naive format
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            data.rename(columns={"Datetime": "Date"}, inplace=True)
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

        logger.info(f"Successfully fetched {len(data)} rows for {config.trading.ticker}")
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data for {config.trading.ticker}: {str(e)}")
        return None

if __name__ == "__main__":
    from config.config import AppConfig
    from app.logger import setup_logging
    config = AppConfig()
    setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)
    df = fetch_data(config)
    if df is not None:
        print(df.head())