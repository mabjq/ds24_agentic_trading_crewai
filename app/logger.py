import logging
from pathlib import Path

def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    """Set up centralized logging configuration.
    Configures file and console handlers with timestamped format.
    Used across modules for ETL and backtest logging, based on config.logging.

    Args:
        log_path: Path to the log file (from config.logging.app_log_path).
        level: Logging level ( "INFO", "DEBUG" or "ERROR").

    Returns:
        logging.Logger: Configured logger instance for the current module.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)