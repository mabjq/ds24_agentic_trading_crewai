"""Tests for data_fetch.py: Verify yfinance fetch and basic validation."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import pandas as pd
from app.data_fetch import fetch_data
from config.config import AppConfig

@pytest.fixture
def config() -> AppConfig:
    """Fixture for AppConfig.

    Returns:
        AppConfig: Default application configuration.
    """
    return AppConfig()

def test_fetch_data_success(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful data fetch with mocked yfinance.
    Verifies DataFrame is returned, has correct columns, and basic validation (high > low).
    """
    mock_data = pd.DataFrame({
        'Date': [datetime.now()],
        'Open': [100.0],
        'High': [105.0],
        'Low': [95.0],
        'Close': [102.0],
        'Volume': [1000]
    })
    
    def mock_history(*args, **kwargs):
        return mock_data.set_index('Date')
    
    monkeypatch.setattr('yfinance.Ticker.history', mock_history)
    
    df = fetch_data(config)
    assert df is not None
    assert len(df) == 1
    assert 'Date' in df.columns
    assert df['High'].iloc[0] > df['Low'].iloc[0]

def test_fetch_data_empty(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test empty data returns None.
    Verifies that an empty DataFrame from yfinance results in None return.
    """
    def mock_history(*args, **kwargs):
        return pd.DataFrame()
    
    monkeypatch.setattr('yfinance.Ticker.history', mock_history)
    
    df = fetch_data(config)
    assert df is None

