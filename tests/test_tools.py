"""Tests for agent tools: Focus on configuration, fallbacks, and logic contracts."""
import pytest
import pandas as pd
from unittest.mock import patch
from config.config import AppConfig
from config.utils import safe_validate_config
from tools.load_csv import wrapper_load_csv
from tools.analyze_signals import wrapper_analyze_signals
from tools.trade_logic import wrapper_trade_logic
from tools.backtest_tool import run_backtest_tool 
from tools.optimize_params import optimize_params_tool 

# Mock Data (Simulate DataFrame-structure)
@pytest.fixture
def simple_mock_data():
    """Returns a minimal DataFrame dictionary structure."""
    return {'df_input': {'data': [
        {'Close': 400, 'gauss': 390, 'adx': 25, 'smma': 380, 'Date': '2025-10-01'},
        {'Close': 410, 'gauss': 405, 'adx': 18, 'smma': 385, 'Date': '2025-10-02'}
    ]}}

# Config tests
class TestSafeValidateConfig:
    """Tests the critical Pydantic validation and data contract enforcement."""
    
    def test_map_extra_fields_and_aliases(self):
        """Verify that LLM-friendly aliases (e.g., 'symbol') map correctly to Pydantic structure."""
        config_dict = {'symbol': 'KC=F', 'tp1': 2.5, 'trading': {'adx_threshold': 19}}
        config = safe_validate_config(config_dict)
        assert config.trading.ticker == 'KC=F'
        assert config.trading.tp_r_multiple == 2.5
        
    def test_fallback_defaults(self):
        """Verify that missing non-required fields receive correct default values."""
        config_dict = {'trading': {}}
        config = safe_validate_config(config_dict)
        # Check that default values are applied
        assert config.trading.atr_period == 14
        assert config.trading.gaussian_period == 26
        
# Tools and Fallback tests
class TestLoadCsv:
    """Tests data fetching and handling of missing files."""
    
    def test_load_missing_csv_fallback(self):
        """Test the robust logic when CSV is missing (should return 'empty')."""
        result = wrapper_load_csv._run(data={'kwargs': {'csv_path': 'nonexistent_file.csv'}})
        assert result['status'] == 'empty'
        assert len(result['data']) == 0
        assert "could not be loaded" in result['summary']

class TestAnalyzeSignals:
    """Tests the signal quantification logic."""
    
    def test_analyze_success_quantification(self, simple_mock_data):
        """Verify that the agent correctly quantifies signals based on input data."""
        result = wrapper_analyze_signals._run(data=simple_mock_data)
        assert 'signals' in result
        assert result['signals']['potential_entries'] == 2 
        assert result['signals']['adx_above_19'] == 0.5 # 1 of 2 rows
        assert result['status'] == 'success'
        
class TestTradeLogic:
    """Tests the simulated trade generation (mocking slippage logic)."""
    
    @patch('tools.trade_logic.random.uniform', return_value=1.5) 
    def test_generate_trades_count(self, mock_random_uniform):
        """Verify that the number of generated trades matches the input 'potential_entries'."""
        
        signals = {
            'potential_entries': 5, 
            'latest_atr': 5, 
            'latest_close': 400,
            'adx_above_19': 0.8 
        } 
    
        mock_data_rows = []
        for i in range(10):
            mock_data_rows.append({'Close': 400.0 + i, 'ATR': 5.0, 'Date': f'2025-10-{i+1:02d}'})
             
        mock_data_context = {
            'config_dict': {
                'app': {'project_name': 'Test Project'},
                'trading': {
                    'ticker': 'KC=F', 
                    'tp_r_multiple': 2.0, 
                    'sl_r_multiple': 1.0,
                    'contract_multiplier': 4.018 
                } 
            },
            'signals': signals,
            'df_input': {'data': mock_data_rows}
        }
        
        result = wrapper_trade_logic._run(data=mock_data_context)
        
        assert len(result['trades']) == 5 
        assert result['status'] == 'success'
        
# Optimizing tests
class TestOptimizationLogic:
    """Tests the conditional logic and fallback mechanisms."""
    
    def test_backtest_low_adx_fallback_metrics(self):
        """Test the explicit fallback metrics used when Backtrader fails or ADX is low."""

        signals = {'adx_above_19': 0.3} # Low ADX -> fallback
        trades = [{'entry': 'long@400'}, {'entry': 'short@390'}] # 2 trades
        
        result = run_backtest_tool._run({'signals': signals, 'trades': trades})
        
        assert result['status'] in ['fallback', 'success'] 
        assert 'winrate' in result['backtest_results']
        assert result['backtest_results']['total_trades'] == 14
        
    def test_optimize_params_fallback_mini_run(self):
        """Test the optimization tool's internal fallback logic on errors."""

        result = optimize_params_tool._run({'backtest_results': {'winrate': 0.46}})
        
        assert result['status'] == 'fallback'
        assert result['optimized_params']['gaussian_period'] in [26, 28, 30] 
        assert result['new_winrate'] > 0.46 