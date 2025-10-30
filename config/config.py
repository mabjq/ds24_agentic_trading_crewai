import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict, FilePath
from dotenv import load_dotenv

class TradingConfig(BaseModel):
    """Configuration for trading parameters.
    Defines core strategy settings like ticker, periods for indicators,
    risk management, and backtest options. Used across ETL and strategy modules.
    """
    ticker: str = "KC=F"  # Coffee futures
    timeframe: str = "30m"  # 30-minute timeframe
    gaussian_period: int = 26
    kijun_period: int = 100
    vapi_period: int = 13
    adx_period: int = 14
    atr_period: int = 14
    smma_period: int = 200
    tp_r_multiple: float = 2.0  # For TP1 (long)
    trailing_atr_mult: float = 4.0  # For TP2
    lookback_days: int = 60  # Days for data fetch
    adx_threshold: int = 19  # ADX threshold 
    swing_order: int = 55  # Lookback for swing high/low (initial Stop Loss)
    risk_pct: float = 0.01  # 1% risk per trade
    max_trades_per_day: int = 5
    min_bars: int = 200  # Minimum bars before trading in backtest
    contract_multiplier: float = 3.768  # Value per price point per contract
    starting_equity: float = 100000.0
    fixed_position_size: float = 20000.0
    test_adx_threshold: int = 13  # Temporary lower threshold for testing low-trend data (default 19 for prod)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

class DatabaseConfig(BaseModel):
    """Configuration for database settings.
    Specifies SQLite path for storing raw OHLCV data in ETL Load step.

    Fields:
        db_path: FilePath = Path("data/trading.db") 
    """
    db_path: FilePath = Path("data/trading.db")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

class APIConfig(BaseModel):
    """Configuration for external APIs.
    Loads keys from .env or environment variables for security.

    Fields:
        xai_api_key: Optional[str] = None 
    """
    xai_api_key: Optional[str] = None

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

    @classmethod
    def load_from_env(cls) -> "APIConfig":
        """Load API config from .env file.
        Prioritizes .env for local dev security.

        Returns:
            APIConfig: Configured instance with loaded keys.
        """
        # Load .env if file exists
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        
        # Fallback to os.getenv
        instance = cls(
            xai_api_key=os.getenv("XAI_API_KEY"),
        )
        return instance

class LoggingConfig(BaseModel):
    """Configuration for logging.
    Sets up centralized logging for ETL and backtest traces.

    Fields:
        app_log_path: FilePath = Path("logs/app.log")  
        log_level: str = "INFO"  # Logging level (e.g., "DEBUG", "INFO", "ERROR").
    """
    app_log_path: FilePath = Path("logs/app.log")
    log_level: str = "INFO"  

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

class AppConfig(BaseModel):
    """Main application configuration.
    Aggregates sub-configs for trading, database, API, and logging.
    Used as central config in ETL pipeline and strategy.

    Fields:
        trading: TradingConfig = TradingConfig()
        database: DatabaseConfig = DatabaseConfig()
        api: APIConfig = APIConfig.load_from_env() 
        logging: LoggingConfig = LoggingConfig() 
    """
    trading: TradingConfig = TradingConfig()
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig.load_from_env()
    logging: LoggingConfig = LoggingConfig()

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )
