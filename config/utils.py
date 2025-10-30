import logging
from typing import Any, Dict
from pathlib import Path

from config.config import AppConfig

logger = logging.getLogger(__name__)


def safe_dump_config(config: Any) -> Dict[str, Any]:
    """Safely dump AppConfig to dict, converting Path objects to str for CrewAI.
    Masks sensitive keys like API keys to prevent leaks in logs/console.
    Handles dict input by falling back to default AppConfig dump.
    """
    if isinstance(config, dict):
        logger.info("Fallback to default AppConfig dump")
        config = AppConfig()  # Create default if input empty or partial

    config_dict = config.model_dump()

    def convert_paths(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            # Mask API keys
            if "xai_api_key" in obj:
                obj["xai_api_key"] = "MASKED"
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj

    dumped = convert_paths(config_dict)
    return dumped


def safe_validate_config(config_dict: Dict[str, Any]) -> AppConfig:
    """Safely validate config_dict, mapping extra fields to 'trading' submodel and forbidding true extras."""
    # Map common extra fields to 'trading'
    trading_map = {
        "ticker": "ticker",
        "gaussian_period": "gaussian_period",
        "adx_threshold": "adx_threshold",
        "atr_period": "atr_period",
        "tp_r_multiple": "tp_r_multiple",
        "trailing_atr_mult": "trailing_atr_mult",
        "risk_pct": "risk_pct",
        "symbol": "ticker",
        "tp1": "tp_r_multiple",
    }

    # Extract and map to 'trading'
    trading_extra = {}
    for key, value in list(config_dict.items()):
        if key in trading_map:
            mapped_key = trading_map[key]
            trading_extra[mapped_key] = value
            config_dict.pop(key)
        elif key == "trading":
            # Merge with existing 'trading'
            if isinstance(value, dict):
                trading_extra.update(value)

    # Add mapped trading to config
    if trading_extra:
        config_dict["trading"] = trading_extra

    # Fallback defaults if missing
    if "trading" not in config_dict:
        config_dict["trading"] = {}
    trading = config_dict["trading"]
    defaults = {
        "atr_period": 14,
        "gaussian_period": 26,
        "adx_threshold": 19,
        "tp_r_multiple": 2.0,
        "trailing_atr_mult": 4.0,
        "risk_pct": 0.01,
    }
    for key, default in defaults.items():
        if key not in trading:
            trading[key] = default

    try:
        return AppConfig.model_validate(config_dict)
    except Exception as e:
        logger.error(f"Safe validate failed: {e}. Using defaults.")
        return AppConfig()