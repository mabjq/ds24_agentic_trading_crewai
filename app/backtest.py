from typing import Optional, Dict, Any
import backtrader as bt
import pandas as pd
import logging
from backtrader.utils import num2date
from config.config import AppConfig
from app.strategies import GaussianKijunStrategy
from app.visualize import plot_with_trades
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeLogger(bt.Analyzer):
    """
    Custom analyzer to log detailed trade information (entry/exit, reason, PnL).
    Uses transaction history to extract actual entry and exit prices.
    Stores trade data in a list with fields like trade_id, entry_date, exit_date, PnL,
    and close_reason, handling both open and closed trades via notify_trade.
    Integrated in backtest.py for detailed trade exports in ETL analysis step.
    """
    def __init__(self):
        """Initialize trade logger with empty lists and counters."""
        self.trades: list[dict] = []
        self.trade_counter = 0
        self._open_trades: dict[int, dict] = {}

    def notify_trade(self, trade: bt.Trade) -> None:
        """Handle trade notifications to log open and closed trades.

        Args:
            trade: Backtrader Trade object (open or closed).
        """
        if trade.isopen and trade.ref not in self._open_trades:
            entry_date = num2date(trade.dtopen)
            self._open_trades[trade.ref] = {
                'entry_date': entry_date,
                'entry_price': trade.price,
                'size': abs(trade.size),
            }

        if trade.isclosed:
            self.trade_counter += 1
            entry_info = self._open_trades.pop(trade.ref, {})
            entry_price = entry_info.get('entry_price', trade.price)
            size = entry_info.get('size', abs(trade.size))
            entry_date = entry_info.get('entry_date', num2date(trade.dtopen))
            exit_date = num2date(trade.dtclose)
            exit_price = entry_price + (trade.pnl / size if size else 0)

            self.trades.append({
                'trade_id': self.trade_counter,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'duration_bars': trade.barlen,
                'duration_hours': trade.barlen * 0.5,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': trade.pnl,
                'pnl_comm': trade.pnlcomm,
                'pnl_percent': (trade.pnl / (entry_price * size)) * 100 if size else 0,
                'is_winner': trade.pnl > 0,
                'close_reason': getattr(self.strategy, 'close_reason', 'Unknown')
            })

    def get_analysis(self):
        """Return the collected trade data.

        Returns:
            list[dict]: List of trade dictionaries with detailed info.
        """
        return self.trades


class PandasDataExtended(bt.feeds.PandasData):
    """"Custom PandasData feed extending backtrader's feed to include indicator lines
    (gauss, kijun, vapi, smma, adx, atr, swing_high, swing_low) for use with
    GaussianKijunStrategy. Prepares transformed data from indicators.py for backtesting.
    """
    lines = ('gauss', 'kijun', 'vapi', 'smma', 'adx', 'atr', 'swing_high', 'swing_low')
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('gauss', 'gauss'),
        ('kijun', 'kijun'),
        ('vapi', 'vapi'),
        ('smma', 'smma'),
        ('adx', 'adx'),
        ('atr', 'atr'),
        ('swing_high', 'swing_high'),
        ('swing_low', 'swing_low'),
    )


def run_backtest(df: pd.DataFrame, config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """Run backtest on DataFrame with OHLCV and indicators using backtrader.
        Integrates strategy from strategies.py, custom analyzers, and saves results to CSV.
        Generates plot via visualize.py. Focuses on simulation and metrics extraction
        as the analysis step in ETL pipeline.

        Args:
            df: Input DataFrame with OHLCV and indicators from transform.py.
            config: Optional application configuration for trading parameters
                    (default: AppConfig()).

        Returns:
            Dict with backtest metrics: final_value, pnl, pnl_percent, max_drawdown_percent,
            total_trades, percent_profitable, profit_factor.
        """
    if config is None:
        config = AppConfig()

    # Save transformed data to CSV
    csv_path = "results/reports/backtest_input.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved backtest input to {csv_path} ({len(df)} rows)")

    # Prepare DataFrame
    df_in = df.copy()
    if 'Date' in df_in.columns:
        df_in = df_in.set_index('Date')
    if not isinstance(df_in.index, pd.DatetimeIndex):
        df_in.index = pd.to_datetime(df_in.index)

    # Validate required columns
    required = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'gauss', 'kijun', 'vapi', 'smma', 'adx',
        'atr', 'swing_high', 'swing_low'
    ]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required columns for backtest: {missing}")

    # Setup Backtrader
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(config.trading.starting_equity)
    cerebro.broker.setcommission(commission=0.0)

    # Add data feed
    feed = PandasDataExtended(dataname=df_in)
    cerebro.adddata(feed)

    # Add strategy and analyzers
    cerebro.addstrategy(GaussianKijunStrategy, app_config=config)
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeLogger, _name='trade_logger')

    # Run backtest
    logger.info("Starting backtest...")
    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    final_value = cerebro.broker.getvalue()
    pnl = final_value - config.trading.starting_equity
    pnl_percent = (pnl / config.trading.starting_equity) * 100

    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown_percent = drawdown.get('max', {}).get('drawdown', 0.0)

    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analyzer.get('total', {}).get('closed', 0)
    won_trades = trade_analyzer.get('won', {}).get('total', 0)
    percent_profitable = (won_trades / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (
        trade_analyzer.get('won', {}).get('pnl', {}).get('total', 0) /
        abs(trade_analyzer.get('lost', {}).get('pnl', {}).get('total', 0))
        if trade_analyzer.get('lost', {}).get('pnl', {}).get('total', 0) != 0
        else float('inf')
    )

    summary = {
        "final_value": final_value,
        "pnl": pnl,
        "pnl_percent": pnl_percent,
        "max_drawdown_percent": max_drawdown_percent,
        "total_trades": total_trades,
        "percent_profitable": percent_profitable,
        "profit_factor": profit_factor,
    }

    # Save summary to CSV
    summary_df = pd.DataFrame([summary])
    summary_csv_path = "results/reports/backtest_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Detailed trades
    trade_logger = strat.analyzers.trade_logger.get_analysis()
    if trade_logger:
        trades_detailed_df = pd.DataFrame(trade_logger)
        trades_detailed_path = "results/reports/trades_detailed.csv"
        trades_detailed_df.to_csv(trades_detailed_path, index=False)

        logger.info(f"Trades being sent to plot: {len(trades_detailed_df)} rows")
        logger.info(f"Trades head:\n{trades_detailed_df.head()}")
        logger.info(f"DF date range: {df['Date'].min()} → {df['Date'].max()}")

        plot_with_trades(
            df_input=df,
            trades=trades_detailed_df,
            symbol=config.trading.ticker,
            save_path="results/plots/backtest_chart.png"
        )
    else:
        logger.info("No trades found -> fallback to last 150 bars")
        plot_with_trades(
            df_input=df,
            trades=pd.DataFrame(),
            symbol=config.trading.ticker,
            save_path="results/plots/backtest_chart.png"
        )

    return summary

if __name__ == "__main__":
    # Handles command-line arguments, making the script configurable
    import argparse
    from app.logger import setup_logging

    parser = argparse.ArgumentParser(description="Run backtest on a CSV file with indicators")
    parser.add_argument("--csv", "-c", required=True, help="CSV file path with indicator columns")
    args = parser.parse_args()
    config = AppConfig()
    setup_logging(log_path=config.logging.app_log_path, level=config.logging.log_level)

    df = pd.read_csv(args.csv, parse_dates=['Date'])
    summary = run_backtest(df, config=config)
    print(summary)
