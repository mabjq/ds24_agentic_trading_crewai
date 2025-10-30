import sqlite3
from typing import Optional
import logging
import pandas as pd
from datetime import datetime
from config.config import AppConfig

logger = logging.getLogger(__name__)

def init_database(config: AppConfig) -> None:
   """Initialize SQLite database and create trading data table.
    Creates the 'ohlcv_data' table with columns: date (TEXT, PRIMARY KEY),
    open, high, low, close (REAL), volume (INTEGER), and ticker (TEXT).
    Used as the Load step in the ETL pipeline to store raw OHLCV data
    fetched from yfinance.

    Args:
        config: Application configuration containing database path.

    Returns:
        None.
    """
   try:
       with sqlite3.connect(config.database.db_path) as conn:
           cursor = conn.cursor()
           cursor.execute("""
               CREATE TABLE IF NOT EXISTS ohlcv_data (
                   date TEXT PRIMARY KEY,
                   open REAL,
                   high REAL,
                   low REAL,
                   close REAL,
                   volume INTEGER,
                   ticker TEXT
               )
           """)
           conn.commit()
           logger.info(f"Initialized database at {config.database.db_path}")
   except sqlite3.Error as e:
       logger.error(f"Failed to initialize database: {str(e)}")
       raise

def save_to_database(config: AppConfig, df: pd.DataFrame, ticker: str) -> bool:
   """Save OHLCV data to SQLite database, ignoring duplicates.
    Converts Date to timezone-naive format and uses INSERT OR IGNORE to
    avoid duplicate entries. Part of the Load step in ETL, storing raw data
    fetched by data_fetch.py for later transformation.

    Args:
        config: Application configuration containing database path.
        df: Input DataFrame with OHLCV data from data_fetch.py.
        ticker: Ticker symbol (e.g., 'KC=F') for the data.

    Returns:
        bool: True if save succeeds, False on error or empty DataFrame.
   """
   if df.empty:
       logger.warning("Empty DataFrame provided, nothing to save")
       return False
  
   try:
       with sqlite3.connect(config.database.db_path) as conn:
           df = df.copy()
           df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
           df['ticker'] = ticker
          
           for _, row in df.iterrows():
               conn.execute("""
                   INSERT OR IGNORE INTO ohlcv_data (date, open, high, low, close, volume, ticker)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
               """, (
                   row['Date'], row['Open'], row['High'], row['Low'],
                   row['Close'], row['Volume'], row['ticker']
               ))
           conn.commit()
       logger.info(f"Saved {len(df)} rows to database for {ticker} (duplicates ignored)")
       return True
   except (sqlite3.Error, KeyError) as e:
       logger.error(f"Failed to save data to database: {str(e)}")
       return False

def fetch_from_database(config: AppConfig, ticker: str, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
   """Fetch OHLCV data from SQLite database for a given ticker and date range.
    Queries the 'ohlcv_data' table, filters by ticker and date, and renames
    columns to match pandas format (e.g., 'date' to 'Date'). Serves as the
    data source for the Transform step in ETL, providing raw data to transform.py.

    Args:
        config: Application configuration containing database path.
        ticker: Ticker symbol ('KC=F') to filter data.
        start_date: Optional start date for filtering (default: None).
        end_date: Optional end date for filtering (default: None).

    Returns:
        Optional[pd.DataFrame]: DataFrame with OHLCV data, or None if no data found or error.
   """
   try:
       with sqlite3.connect(config.database.db_path) as conn:
           query = "SELECT * FROM ohlcv_data WHERE ticker = ?"
           params = [ticker]
          
           if start_date:
               query += " AND date >= ?"
               params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
           if end_date:
               query += " AND date <= ?"
               params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))
          
           df = pd.read_sql_query(query, conn, params=params)
           if df.empty:
               logger.warning(f"No data found for {ticker} in specified range")
               return None
              
           df['Date'] = pd.to_datetime(df['date'])
           df = df.drop(columns=['date']).rename(columns={
               'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume', 'ticker': 'Ticker'
           })
       logger.info(f"Fetched {len(df)} rows from database for {ticker}")
       return df
   except sqlite3.Error as e:
       logger.error(f"Failed to fetch data from database: {str(e)}")
       return None

if __name__ == "__main__":
   from config.config import AppConfig
   from app.logger import setup_logging
   config = AppConfig()
   setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)
  
   init_database(config)
