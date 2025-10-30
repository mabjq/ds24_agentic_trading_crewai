import logging
from datetime import datetime
from pathlib import Path
from app.main import main

# Log directory and file
LOG_DIR = Path("/home/projects/agentic-tradingplatform/logs")
LOG_FILE = LOG_DIR / "automate.log"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__)


def run_job() -> None:
    """Executes the full ETL pipeline including data fetch, transform, and backtest.
    Logs start and end times, handles exceptions, and triggers main() to process
    the entire workflow. Intended for daily execution via cron.
    """
    start_time = datetime.now()
    logger.info(f"ETL job started at {start_time}")
    try:
        main()  # Calls init_db, fetch, save, transform, backtest
        logger.info("ETL job completed successfully")
    except Exception as e:
        logger.exception(f"ETL job failed: {str(e)}")
    end_time = datetime.now()
    logger.info(
        f"ETL job ended at {end_time} (duration: {end_time - start_time})"
    )


if __name__ == "__main__":
    run_job()
