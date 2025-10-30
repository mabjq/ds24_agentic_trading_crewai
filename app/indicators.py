import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from config.config import AppConfig

logger = logging.getLogger(__name__)

def compute_gaussian_channel(df: pd.DataFrame, period: int = 26) -> pd.DataFrame:
    """Compute Gaussian Channel: Triple EMA for mid-line, upper/lower as mid +/- ATR.
    Uses a custom triple EMA (26-period) for smoothing, with ATR-based bands.
    Part of the Transform step in ETL, used in GaussianKijunStrategy for entry signals.

    Args:
        df: Input DataFrame with OHLCV columns (High, Low, Close).
        period: Period for EMA calculation (default: 26 from config).

    Returns:
        pd.DataFrame: Updated DataFrame with 'gauss', 'gaussian_upper', and 'gaussian_lower' columns.
    """
    df = df.copy()
    ema1 = df['Close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    df['gauss'] = ema2.ewm(span=period, adjust=False).mean()
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['gaussian_upper'] = df['gauss'] + df['atr'] * 3.0
    df['gaussian_lower'] = df['gauss'] - df['atr'] * 3.0
    logger.info(f"Computed Gaussian Channel (triple EMA) with period {period}.")
    return df

def compute_kijun_sen(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    """Compute Kijun-Sen: Midpoint of highest high and lowest low over period.
    Calculated over 100 periods from config, used in GaussianKijunStrategy
    for trendbreak exit signals in the Transform step of ETL.

    Args:
        df: Input DataFrame with OHLCV columns (High, Low).
        period: Rolling window period (default: 100 from config).

    Returns:
        pd.DataFrame: Updated DataFrame with 'kijun' column.
    """
    df = df.copy()
    df['kijun'] = (df['High'].rolling(window=period).max() + df['Low'].rolling(window=period).min()) / 2
    logger.info(f"Computed Kijun-Sen with period {period}.")
    return df

def compute_vapi(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """Compute VAPI: 'EMA(close * volume) / EMA(volume)' (price scale).
    Custom implementation adding vapi_trend for direction.
    Used in GaussianKijunStrategy for entry signals, part of the Transform step.

    Args:
        df: Input DataFrame with OHLCV columns (Close, Volume).
        period: EMA period (default: 13 from config).

    Returns:
        pd.DataFrame: Updated DataFrame with 'vapi' and 'vapi_trend' columns.
    """
    df = df.copy()
    df['vapi'] = (df['Close'] * df['Volume']).ewm(span=period, adjust=False).mean() / df['Volume'].ewm(span=period, adjust=False).mean()
    df['vapi_trend'] = np.where(df['vapi'] > df['vapi'].shift(1), 'up', np.where(df['vapi'] < df['vapi'].shift(1), 'down', 'neutral'))
    logger.info(f"Computed VAPI (EMA version, price scale) with period {period}.")
    return df

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX using pandas_ta.
    Used in GaussianKijunStrategy with a threshold ( >19) for entry confirmation,
    calculated in the Transform step of ETL.

    Args:
        df: Input DataFrame with OHLCV columns (High, Low, Close).
        period: ADX calculation period (default: 14 from config).

    Returns:
        pd.DataFrame: Updated DataFrame with 'adx' column.
    """
    df = df.copy()
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=period)
    df['adx'] = adx_df['ADX_14']
    logger.info(f"Computed ADX with period {period}.")
    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ATR using pandas_ta.
    Used in GaussianKijunStrategy for volatility-based stops and trailing,
    calculated in the Transform step of ETL.

    Args:
        df: Input DataFrame with OHLCV columns (High, Low, Close).
        period: ATR calculation period (default: 14 from config).

    Returns:
        pd.DataFrame: Updated DataFrame with 'atr' column.
    """
    df = df.copy()
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    logger.info(f"Computed ATR with period {period}.")
    return df

def compute_smma(df: pd.DataFrame, period: int = 200, src: str = 'Close') -> pd.DataFrame:
    """Compute Smoothed Moving Average (SMMA): Recursive formula.
    Used in GaussianKijunStrategy to determine long-term trend,
    calculated in the Transform step of ETL.

    Args:
        df: Input DataFrame with source column (default: 'Close').
        period: SMMA period (default: 200 from config).
        src: Source column for calculation (default: 'Close').

    Returns:
        pd.DataFrame: Updated DataFrame with 'smma' column.
    """
    df = df.copy()
    length = period
    smma = pd.Series(np.nan, index=df.index)
    if len(df) >= length:
        smma.iloc[length-1] = df[src].iloc[:length].mean()
        for i in range(length, len(df)):
            smma.iloc[i] = (smma.iloc[i-1] * (length - 1) + df[src].iloc[i]) / length
    df['smma'] = smma
    logger.info(f"Computed SMMA with period {period} on {src}.")
    return df

def find_swing_high_low(df: pd.DataFrame, order: int = 55) -> pd.DataFrame:
    """Find recent swing high/low using rolling max/min for initial SL (no ffill).
    Used in GaussianKijunStrategy to set initial stop-loss levels,
    calculated in the Transform step of ETL.

    Args:
        df: Input DataFrame with OHLCV columns (High, Low).
        order: Rolling window for max/min (default: 55 from config).

    Returns:
        pd.DataFrame: Updated DataFrame with 'swing_high' and 'swing_low' columns.
    """
    df = df.copy()
    df['swing_high'] = df['High'].rolling(window=order, min_periods=1).max()
    df['swing_low'] = df['Low'].rolling(window=order, min_periods=1).min()
    logger.info(f"Computed swing high/low with order {order} (raw, no fill).")
    return df

def compute_all_indicators(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Apply all indicator calculations sequentially.
    Orchestrates the computation of Gaussian, Kijun, VAPI, ADX, ATR, SMMA, and swing
    levels for use in the Transform step of ETL, feeding into GaussianKijunStrategy.

    Args:
        df: Input DataFrame with OHLCV columns from clean_data in transform.py.
        config: Application configuration for indicator periods and parameters.

    Returns:
        pd.DataFrame: Updated DataFrame with all indicator columns.
    """
    df = compute_atr(df, config.trading.atr_period)
    df = compute_gaussian_channel(df, config.trading.gaussian_period)
    df = compute_kijun_sen(df, config.trading.kijun_period)
    df = compute_vapi(df, config.trading.vapi_period)
    df = compute_adx(df, config.trading.adx_period)
    df = compute_smma(df, config.trading.smma_period)
    df = find_swing_high_low(df, config.trading.swing_order)
    return df