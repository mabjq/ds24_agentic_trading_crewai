"""Tests for indicators.py: Verify calculations for Gaussian, VAPI, etc."""

import pytest
import pandas as pd
import numpy as np  
from app.indicators import compute_all_indicators
from config.config import AppConfig

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture for sample OHLCV data.

    Returns:
        pd.DataFrame: Random OHLCV data with 300 rows for indicator testing.
    """
    dates = pd.date_range(start='2025-01-01', periods=300, freq='30min') 
    return pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, 300),
        'High': np.random.uniform(105, 115, 300),
        'Low': np.random.uniform(95, 105, 300),
        'Close': np.random.uniform(100, 110, 300),
        'Volume': np.random.uniform(1000, 5000, 300)
    })

def test_compute_all_indicators(sample_df: pd.DataFrame) -> None:
    """Test all indicators are computed without errors and no excessive NaN.
    Verifies required columns exist and NaN counts are reasonable for periods.
    """
    config = AppConfig()
    df_result = compute_all_indicators(sample_df, config)
    
    required_cols = ['gauss', 'kijun', 'vapi', 'smma', 'adx', 'atr', 'swing_high', 'swing_low']
    for col in required_cols:
        assert col in df_result.columns
    
    # Check for reasonable NaN count 
    nan_counts = df_result[required_cols].isna().sum()
    # Allow up to 70% NaN for long-period indicators (SMMA 200 = 199/300 = 66%)
    assert nan_counts.max() < len(df_result) * 0.7, f"Too many NaN: {nan_counts}"

def test_gaussian_triple_ema(sample_df: pd.DataFrame) -> None:
    """Test Gaussian is smoother (triple EMA) by checking variance.
    Verifies reduced variance compared to raw Close prices.
    """
    config = AppConfig()
    df_result = compute_all_indicators(sample_df, config)
    
    gauss_var = df_result['gauss'].var()
    close_var = sample_df['Close'].var()
    assert gauss_var < close_var * 0.8  # Triple EMA should reduce variance

def test_vapi_price_scale(sample_df: pd.DataFrame) -> None:
    """Test VAPI values are in price scale (near close values).
    Verifies VAPI mean is within 10% of Close mean.
    """
    config = AppConfig()
    df_result = compute_all_indicators(sample_df, config)
    
    # VAPI should be close to price range (EMA version)
    close_mean = sample_df['Close'].mean()
    vapi_mean = df_result['vapi'].mean()
    assert abs(vapi_mean - close_mean) < close_mean * 0.1  # Within 10% of close mean