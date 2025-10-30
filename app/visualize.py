import pandas as pd
import numpy as np
import mplfinance as mpf
import logging
from typing import List

logger = logging.getLogger(__name__)

def _nearest_index(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Timestamp:
    """Find nearest datetime index in df to given timestamp.

    Args:
        df: DataFrame with datetime index.
        ts: Timestamp to find nearest index for.

    Returns:
        pd.Timestamp: Nearest index timestamp, or None if df is empty.
    """
    if df.empty:
        return None
    return df.index[(df.index.get_indexer([ts], method="nearest"))[0]]

def plot_with_trades(df_input: pd.DataFrame, trades: pd.DataFrame, symbol: str, save_path: str):
    """
    Generate and save a candlestick chart with indicators, ATR bands, and trade signals.
    Zooms to the region covering trades (max 10 days), or falls back to last 150 bars if no trades.
    Part of the visualization step in ETL, called by backtest.py.

    Args:
        df_input: Input DataFrame with OHLCV, indicators, and 'date' column from transform.py.
        trades: DataFrame with trade details ( entry_date, exit_date, entry_price, exit_price).
        symbol: Ticker symbol ('KC=F') for plot title.
        save_path: File path to save the plot PNG.

    Returns:
        None: Saves plot to save_path.
    """
    df = df_input.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure datetime index without tz
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.set_index('date')

    trades = trades.copy()
    if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
        trades['entry_date'] = pd.to_datetime(trades['entry_date']).dt.tz_localize(None)
        trades['exit_date'] = pd.to_datetime(trades['exit_date']).dt.tz_localize(None)

    # Limit visible window
    if not trades.empty and 'entry_date' in trades.columns:
        visible_start = trades['entry_date'].min()
        visible_end = trades['exit_date'].max()
        if (visible_end - visible_start) > pd.Timedelta(days=10):
            visible_start = visible_end - pd.Timedelta(days=10)
        df = df[(df.index >= visible_start) & (df.index <= visible_end)]
        logger.info(f"Plot window set from {visible_start} to {visible_end}")
    else:
        df = df.tail(150)
        logger.info("No trades found -> fallback to last 150 bars")

    add_plots: List[dict] = []

    # Indicators
    indicator_colors = {'gauss': 'blue', 'kijun': 'orange', 'smma': 'purple'}
    for name, color in indicator_colors.items():
        if name in df.columns:
            add_plots.append(mpf.make_addplot(df[name], color=color, panel=0))

    if 'atr' in df.columns and 'close' in df.columns:
        atr_mult = 4.0
        df['atr_upper'] = df['close'] + df['atr'] * atr_mult
        df['atr_lower'] = df['close'] - df['atr'] * atr_mult
        add_plots.append(mpf.make_addplot(df['atr_upper'], color='red', alpha=0.3, panel=0))
        add_plots.append(mpf.make_addplot(df['atr_lower'], color='lime', alpha=0.3, panel=0))
        logger.info("Added ATR bands to plot (trailing stop levels)")

    if 'adx' in df.columns:
        add_plots.append(mpf.make_addplot(df['adx'], color='cyan', panel=2, ylabel='ADX'))

    # Markers
    df['long_entry'] = np.nan
    df['short_entry'] = np.nan
    df['partial_exit'] = np.nan
    df['full_exit'] = np.nan

    if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
        for _, row in trades.iterrows():
            entry_dt, exit_dt = row['entry_date'], row['exit_date']
            entry_idx = _nearest_index(df, entry_dt)
            exit_idx = _nearest_index(df, exit_dt)

            if entry_idx is not None and entry_idx in df.index:
                is_long = row['entry_price'] < row['exit_price']
                df.at[entry_idx, 'long_entry' if is_long else 'short_entry'] = (
                    df.at[entry_idx, 'low']*0.99 if is_long else df.at[entry_idx, 'high']*1.01
                )
            if exit_idx is not None and exit_idx in df.index:
                df.at[exit_idx, 'full_exit'] = df.at[exit_idx, 'close']

            logger.info(
                f"Trade {row.get('trade_id', '?')} "
                f"entry {entry_dt}→{entry_idx}, exit {exit_dt}→{exit_idx}"
            )

    # Addplots for trades
    if not pd.isna(df['long_entry']).all():
        add_plots.append(mpf.make_addplot(df['long_entry'], type='scatter', marker='^', color='lime', markersize=100))
    if not pd.isna(df['short_entry']).all():
        add_plots.append(mpf.make_addplot(df['short_entry'], type='scatter', marker='v', color='red', markersize=100))
    if not pd.isna(df['partial_exit']).all():
        add_plots.append(mpf.make_addplot(df['partial_exit'], type='scatter', marker='o', color='orange', markersize=40, alpha=0.6))
    if not pd.isna(df['full_exit']).all():
        add_plots.append(mpf.make_addplot(df['full_exit'], type='scatter', marker='o', color='orange', markersize=120))

    try:
        mpf.plot(
            df, type='candle', style='charles',
            title=f"{symbol} - Backtest (Zoomed) - ATR Bands",
            ylabel='Price ($)', volume=True, panel_ratios=(3, 1, 1),
            addplot=add_plots, figscale=1.5,
            savefig=dict(fname=save_path, dpi=150)
        )
        logger.info(
            f"Trade plot saved to: {save_path} "
            f"(Long^/Shortv/ExitO) | "
            f"Markers set -> Long: {df['long_entry'].count()}, "
            f"Short: {df['short_entry'].count()}, "
            f"Full exits: {df['full_exit'].count()}"
        )
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")
