import pandas as pd
from typing import Optional
from datetime import datetime
import logging
from app.database import fetch_from_database
from app.indicators import compute_all_indicators
from config.config import AppConfig

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV DataFrame: remove invalid rows, outliers, ensure correct data types, and sort by date.
    Removes NaN, rows where volume=0 or high=low, and outliers (>5 standard deviations). 
    Preserves gaps (no forward-filling) for authentic backtesting.
    Part of the Transform step in ETL, processing data fetched from database.py.

    Args:
        df: Input DataFrame with raw OHLCV data from database.py.

    Returns:
        pd.DataFrame: Cleaned DataFrame, sorted by Date with gaps preserved.
    """
    df = df.copy()
    initial_rows = len(df)

    # Convert to correct data types first
    df['Date'] = pd.to_datetime(df['Date'])
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with NaN or invalid values
    df = df.dropna(subset=numeric_columns + ["Date"])
    df = df[df['Volume'] > 0]
    df = df[df['High'] != df['Low']]
    logger.info(f"Dropped {initial_rows - len(df)} rows with NaN or invalid values (high=low or volume=0).")

    # Remove outliers (> 5 standard deviations)
    for col in ["Open", "High", "Low", "Close"]:
        mean = df[col].mean()
        std = df[col].std()
        if pd.notna(mean) and pd.notna(std) and std > 0:
            df = df[(df[col] >= mean - 5 * std) & (df[col] <= mean + 5 * std)]
    logger.info(f"Removed outliers, {len(df)} rows remain.")

    # Sort by date, preserving gaps
    df = df.sort_values("Date").reset_index(drop=True)
    logger.info(f"Data cleaning complete: {len(df)} rows after processing (raw with gaps preserved).")
    return df

def transform_data(config: AppConfig, ticker: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Full transformation pipeline: fetch from DB, clean, and calculate indicators.
    Fetches raw OHLCV data from database.py, cleans it with clean_data(), and adds
    indicators (Gaussian, Kijun, VAPI, ADX, ATR, SMMA, swing) via indicators.py.
    Part of the Transform step in ETL, preparing data for backtest.py.

    Args:
        config: Application configuration for database and trading parameters.
        ticker: Ticker symbol ('KC=F') to fetch data for.
        start_date: Optional start date for data range (default: None).
        end_date: Optional end date for data range (default: None).

    Returns:
        Optional[pd.DataFrame]: Transformed DataFrame with indicators, or None if no data or error.
    """
    raw_df = fetch_from_database(config=config, ticker=ticker, start_date=start_date, end_date=end_date)
    if raw_df is None or raw_df.empty:
        logger.error(f"No raw data found for {ticker}")
        return None

    df = clean_data(raw_df)
    df = compute_all_indicators(df, config)

    logger.info(f"Transformation complete: {len(df)} rows with indicators for {ticker}")
    return df

if __name__ == "__main__":
    from config.config import AppConfig
    from app.logger import setup_logging
    config = AppConfig()
    setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)
    transformed_df = transform_data(config, config.trading.ticker)
    if transformed_df is not None:
        print(transformed_df.tail())