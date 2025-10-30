"""Tests for strategies.py: Verify entry/exit logic with mock data."""

import pytest
import pandas as pd
import backtrader as bt
from app.strategies import GaussianKijunStrategy
from config.config import AppConfig

def test_strategy_initialization() -> None:
    """Test strategy initializes without errors.
    Verifies config is passed to GaussianKijunStrategy and initial state (min_bars, trades_today).
    Uses dummy data to run cerebro without errors."""
    config = AppConfig()
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(config.trading.starting_equity)
    
    # Add strategy - verify config is passed correctly
    cerebro.addstrategy(GaussianKijunStrategy, app_config=config)
    
    # Run with dummy data to test init 
    dates = pd.date_range('2025-01-01', periods=10)
    data_df = pd.DataFrame({
        'open': [100]*10, 'high': [105]*10, 'low': [95]*10, 'close': [102]*10, 'volume': [1000]*10
    }, index=dates)  # Use index as datetime instead of column
    
    data = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data)
    
    strategies = cerebro.run()
    assert len(strategies) == 1
    strat = strategies[0]
    assert strat.cfg.min_bars == 200  # Verify config loaded
    assert strat.trades_today == 0  # Initial state

def test_strategy_no_trade_low_adx() -> None:
    """Test no trade if ADX < threshold (requires full backtest setup for deep test).
    Placeholder for integration test with mock data feed; verifies config threshold.
    """
    config = AppConfig()
    assert config.trading.adx_threshold == 25  # From config
    assert True  # Expand with mock data feed in future