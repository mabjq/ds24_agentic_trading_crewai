import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("Analyze CSV data for entry/exit signals and suggest optimizations from summary CSVs. Call tool with data dict containing config_dict and df_input, return concise JSON.")
def wrapper_analyze_signals(data: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Analyze loaded DF for signals and metrics. Suggest parameter tweaks. Return concise JSON.

    Args:
        data: Dict with 'config_dict' (AppConfig), 'df_input' (from load tool), optional 'kwargs'.

    Returns:
        Dict[str, Any]: Concise analysis JSON with recommendations.
    """
    config_dict = data.get("config_dict", {})
    df_input = data.get("df_input", None)
    kwargs = data.get("kwargs", {})
    logger.info(
        f"Tool called: wrapper_analyze_signals with config_dict keys: {list(config_dict.keys())} "
        f"and df_input type/len: {type(df_input)} / {len(df_input) if hasattr(df_input, '__len__') else 'N/A'}"
    )
    try:
        if isinstance(df_input, dict):
            raw_data = df_input.get("data", df_input)  # Prefer 'data' key, else flat dict
        elif isinstance(df_input, list):
            raw_data = df_input
        else:
            raw_data = {}

        df = pd.DataFrame(raw_data)
        logger.info(f"Parsed DF shape: {df.shape}, columns: {list(df.columns)}")
        if df.empty or "Close" not in df or "gauss" not in df:
            logger.warning(f"DF missing key cols (e.g., 'Close'/'gauss'); using fallback signals=0")
            return {
                "signals": {"gauss_up_count": 0, "adx_above_19": 0, "potential_entries": 0},
                "recommendations": [],
                "status": "missing_cols",
            }

        # Concise signal analysis (limit to recent data)
        recent: pd.DataFrame = df.tail(20)
        signals = {
            "gauss_up_count": int((recent["gauss"] > recent["gauss"].shift(1)).sum())
            if "gauss" in recent.columns
            else 0,
            "adx_above_19": round((recent["adx"] > 19).mean(), 2) if "adx" in recent.columns else 0,
            "potential_entries": int((recent["Close"] > recent["smma"]).sum())
            if "Close" in recent.columns and "smma" in recent.columns
            else 0,
        }
        logger.info(f"Computed signals: {signals}")

        # Load summary CSVs
        summary_path = kwargs.get("summary_path", "results/reports/backtest_summary.csv")
        trades_path = kwargs.get("trades_path", "results/reports/trades_detailed.csv")
        summary = pd.read_csv(summary_path) if Path(summary_path).exists() else pd.DataFrame()
        trades = pd.read_csv(trades_path) if Path(trades_path).exists() else pd.DataFrame()

        winrate = round(summary["percent_profitable"].iloc[0], 2) if not summary.empty else 0
        recommendations = []
        if winrate < 0.71:
            recommendations.append(
                {
                    "parameter": "adx_threshold",
                    "current": 19,
                    "new": 14,
                    "reason": "Lower to capture more signals; backtest on trades_detailed.csv",
                }
            )
        if not trades.empty and trades["pnl"].sum() < 5000:
            recommendations.append(
                {
                    "parameter": "tp_r_multiple",
                    "current": 2.0,
                    "new": 1.5,
                    "reason": "Conservative TP for volatile KC=F; aim for higher winrate",
                }
            )

        analysis = {
            "signals": signals,
            "winrate": winrate,
            "total_trades": len(trades),
            "recommendations": recommendations,
            "status": "success",
        }
        logger.info(
            f"Analysis complete: {len(recommendations)} recommendations (concise). Config used: {config_dict.get('trading', {}).get('adx_threshold', 'N/A')}"
        )
        return analysis
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"signals": {}, "recommendations": [], "status": "error", "message": str(e)}