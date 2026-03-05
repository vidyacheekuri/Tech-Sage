"""
TechSage — Standalone ingestion scheduler.

Run this process to fetch articles automatically WITHOUT starting the web server.
Keeps running in the background: news every 5 min, GitHub every 6 hours.

Usage:
    python -m scheduler.run_standalone

Run in background:
    nohup python -m scheduler.run_standalone &
    # or: screen -dmS techsage python -m scheduler.run_standalone
"""

import logging
import signal
import sys
import time

from config.model_registry import model_registry
from config.settings import get_settings
from database.session import init_db
from scheduler.ingestion_job import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Received signal %s, shutting down gracefully...", signum)
    _shutdown = True


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("TechSage Ingestion Scheduler (standalone)")
    logger.info("=" * 60)

    logger.info("Loading ML models...")
    model_registry.load_all()

    logger.info("Initializing database...")
    init_db()

    logger.info("Starting ingestion schedulers...")
    start_scheduler()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        while not _shutdown:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_scheduler()
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
