import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("Load pre-computed OHLCV + indicators from /results/reports/backtest_input.csv for KC=F. Call tool with data dict containing config_dict and return JSON summary (truncate to last 100 rows).")
def wrapper_load_csv(data: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Wrapper to load backtest CSV for analysis. Validates data quality and returns truncated JSON summary.

    Args:
        data: Dict with 'config_dict' (AppConfig), optional 'kwargs'.

    Returns:
        Dict[str, Any]: Summary JSON with truncated data (last 100 rows).
    """
    config_dict = data.get("config_dict", {})
    kwargs = data.get("kwargs", {})
    logger.info(
        f"Tool called: wrapper_load_csv with config_dict keys: {list(config_dict.keys())} and kwargs: {kwargs}"
    )
    try:
        csv_path = kwargs.get("csv_path", "results/reports/backtest_input.csv")
        if not Path(csv_path).exists():
            logger.warning(f"CSV not found at {csv_path}; returning empty dict.")
            return {"data": [], "status": "empty", "shape": (0, 0)}
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        if df.empty:
            logger.warning("CSV empty; returning empty summary.")
            return {"data": [], "status": "empty", "shape": (0, 0)}

        # Truncate to last 100 rows to avoid token bloat
        df_trunc = df.tail(100).copy()
        data_dict = df_trunc.to_dict("records")
        summary = {
            "total_rows": len(df),
            "recent_stats": {
                "mean_close": round(df["Close"].tail(10).mean(), 2) if "Close" in df else 0,
                "adx_mean": round(df["adx"].tail(10).mean(), 2) if "adx" in df else 0,
            },
        }
        logger.info(
            f"Loaded {len(df)} bars from {csv_path}, truncated to {len(data_dict)} for summary. "
            f"Config example: {list(config_dict.get('trading', {}).keys())}"
        )
        return {"data": data_dict, "summary": summary, "status": "success", "shape": df.shape}
    except Exception as e:
        logger.error(f"CSV load error: {e}")
        return {"data": [], "status": "error", "message": str(e)}