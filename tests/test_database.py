"""Tests for database.py: Verify SQLite operations (init, save, fetch)."""

import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from app.database import init_database, save_to_database, fetch_from_database
from config.config import AppConfig


@pytest.fixture
def config(tmp_path) -> AppConfig:
    """Fixture for AppConfig with temporary DB path.

    Args:
        tmp_path: Pytest temporary path fixture for isolated testing.

    Returns:
        AppConfig: Configuration with temp DB path for testing.
    """
    db_file = tmp_path / "trading_test.db"
    db_file.touch()  # Create empty DB file to pass Pydantic validation
    cfg = AppConfig()
    cfg.database.db_path = db_file
    return cfg


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture for sample OHLCV data.

    Returns:
        pd.DataFrame: Sample DataFrame with 5 rows of mock OHLCV data.
    """
    dates = pd.date_range(start='2025-01-01', periods=5, freq='30min')
    return pd.DataFrame({
        'Date': dates,
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'High': [105.0, 106.0, 107.0, 108.0, 109.0],
        'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })


def test_init_database_success(config: AppConfig) -> None:
    """Test database initialization creates table without errors.
    Verifies DB file exists and 'ohlcv_data' table is created.
    """
    init_database(config)
    assert config.database.db_path.exists()

    with sqlite3.connect(config.database.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data'")
        assert cursor.fetchone() is not None


def test_save_to_database_success(config: AppConfig, sample_df: pd.DataFrame) -> None:
    """Test saving data to database returns True and stores rows.
    Verifies success flag and that 5 rows are inserted for ticker 'KC=F'.
    """
    init_database(config)
    success = save_to_database(config, sample_df, "KC=F")
    assert success is True

    with sqlite3.connect(config.database.db_path) as conn:
        df_saved = pd.read_sql_query("SELECT * FROM ohlcv_data WHERE ticker='KC=F'", conn)
        assert len(df_saved) == 5


def test_fetch_from_database_success(config: AppConfig, sample_df: pd.DataFrame) -> None:
    """Test fetching data from database returns correct DataFrame.
    Verifies non-None return, 5 rows, and renamed columns like 'Open'.
    """
    init_database(config)
    save_to_database(config, sample_df, "KC=F")
    df_fetched = fetch_from_database(config, "KC=F")
    assert df_fetched is not None
    assert len(df_fetched) == 5
    assert 'Open' in df_fetched.columns


def test_save_to_database_empty_df(config: AppConfig) -> None:
    """Test saving empty DataFrame returns False.
    Verifies warning log and False return for empty input.
    """
    empty_df = pd.DataFrame()
    success = save_to_database(config, empty_df, "KC=F")
    assert success is False


def test_save_to_database_no_date_column(config: AppConfig, sample_df: pd.DataFrame) -> None:
    """Test save fails gracefully if Date column is missing.
    Verifies False return for DataFrame without 'Date' column.
    """
    df_no_date = sample_df.drop(columns=['Date'])
    success = save_to_database(config, df_no_date, "KC=F")
    assert success is False


def test_fetch_from_database_no_data(config: AppConfig) -> None:
    """Test fetching non-existent ticker returns None.
    Verifies None return for unknown ticker after DB init.
    """
    init_database(config)
    df_fetched = fetch_from_database(config, "NONEXISTENT")
    assert df_fetched is None
