"""Tests for transform.py: Verify cleaning and indicator integration."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from app.transform import clean_data, transform_data
from app.database import fetch_from_database
from config.config import AppConfig
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    """Fixture for raw OHLCV data with some invalid rows (150 bars for Kijun 125).

    Returns:
        pd.DataFrame: Mock raw data with NaN, zero volume, and high=low rows for cleaning tests.
    """
    dates = pd.date_range(start='2025-01-01', periods=300, freq='30min')  # 300 bars for Kijun 125
    n = len(dates)
    
    # Generate data with some invalid rows
    np.random.seed(42)  # For reproducible tests
    open_prices = np.random.uniform(100, 110, n)
    high_prices = open_prices + np.random.uniform(0, 5, n)
    low_prices = open_prices - np.random.uniform(0, 5, n)
    close_prices = np.random.uniform(low_prices, high_prices)
    volumes = np.random.uniform(500, 2000, n)
    
    # Introduce invalid rows (10% of data)
    invalid_indices = np.random.choice(n, size=min(15, n//10), replace=False)
    for i in invalid_indices:
        open_prices[i] = np.nan
        volumes[i] = 0
        high_prices[i] = low_prices[i]  # high=low invalid
    
    return pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })

def test_clean_data_removes_invalid(mock_raw_df: pd.DataFrame) -> None:
    """Test clean_data removes NaN, volume=0, high=low rows.
    Verifies row reduction, no remaining invalid values, and no NaN after cleaning.
    """
    initial_len = len(mock_raw_df)
    cleaned_df = clean_data(mock_raw_df)
    final_len = len(cleaned_df)
    
    assert final_len < initial_len  # Some rows removed
    assert final_len >= initial_len - 20  # Reasonable removal (max 15 invalid + outliers)
    assert cleaned_df['Volume'].min() > 0
    assert (cleaned_df['High'] > cleaned_df['Low']).all()
    assert cleaned_df['Open'].isna().sum() == 0  # No NaN after cleaning

def test_transform_data_full_pipeline(mock_raw_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full transform: clean + indicators (with sufficient data for all indicators).
    Mocks fetch_from_database and verifies indicators added, non-None result, and NaN thresholds.
    """
    config = AppConfig()
    
    # Mock fetch_from_database to return mock_raw_df
    def mock_fetch(config, ticker, *args, **kwargs):
        return mock_raw_df
    
    monkeypatch.setattr('app.transform.fetch_from_database', mock_fetch)
    
    df_result = transform_data(config, "KC=F")
    assert df_result is not None
    assert len(df_result) > 0
    assert 'gauss' in df_result.columns  # Indicators added
    
    # Check all indicators exist and have reasonable non-NaN counts
    required_indicators = ['gauss', 'kijun', 'vapi', 'smma', 'adx', 'atr']
    total_rows = len(df_result)
    
    for indicator in required_indicators:
        assert indicator in df_result.columns
        non_null_count = df_result[indicator].notna().sum()
        
        # Different thresholds based on period length
        if indicator in ['kijun']:  # Period 125
            min_non_null = total_rows - 125  # Expect NaN for first 125 bars
            assert non_null_count >= min_non_null, f"{indicator} should have values after period, got {non_null_count}/{total_rows}"
        elif indicator in ['smma']:  # Period 200
            min_non_null = max(0, total_rows - 200)
            assert non_null_count >= min_non_null, f"{indicator} should have values after period, got {non_null_count}/{total_rows}"
        else:  # Shorter periods (gauss=34, adx=14, etc.)
            assert non_null_count > total_rows * 0.8, f"{indicator} should have >80% values, got {non_null_count}/{total_rows}"
        
        assert non_null_count > 0, f"{indicator} should have some non-NaN values"