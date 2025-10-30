"""Tests for backtest.py: Verify run_backtest execution, metrics extraction, and plot generation."""

import pytest
import pandas as pd
from pathlib import Path
from app.backtest import run_backtest
from config.config import AppConfig
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Fixture for mock OHLCV + indicators.

    Returns:
        pd.DataFrame: Sample DataFrame with required columns for backtest.
    """
    dates = pd.date_range(start='2025-01-01', periods=10, freq='30min')
    return pd.DataFrame({
        'Date': dates,
        'Open': [100.0] * 10,
        'High': [105.0] * 10,
        'Low': [95.0] * 10,
        'Close': [102.0] * 10,
        'Volume': [1000] * 10,
        'gauss': [101.0] * 10,
        'kijun': [100.5] * 10,
        'vapi': [101.5] * 10,
        'smma': [100.0] * 10,
        'adx': [30.0] * 10,
        'atr': [2.0] * 10,
        'swing_high': [105.0] * 10,
        'swing_low': [95.0] * 10
    })

def test_run_backtest_success_isolated(mock_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test run_backtest completes without affecting production files.
    Mocks plot and CSV saves to isolate execution and verify metrics output.
    Ensures summary dict is returned and plot is called (fallback for no trades).
    """
    config = AppConfig()

    with patch('app.backtest.plot_with_trades') as mock_plot, \
         patch('pandas.DataFrame.to_csv') as mock_to_csv:

        summary = run_backtest(mock_df, config)

        # Verify summary dict
        assert 'final_value' in summary
        assert summary['total_trades'] >= 0
        assert summary['pnl_percent'] is not None

        # Plot should always be called, even with empty trades (fallback 150 bars)
        mock_plot.assert_called_once()

        # CSV writes should have been attempted
        assert mock_to_csv.call_count >= 1

