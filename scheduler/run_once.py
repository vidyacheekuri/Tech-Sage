"""
TechSage — One-shot ingestion (for cron).

Run a single news + optional GitHub ingestion pass, then exit.
Use with system cron for automatic fetching without a long-running process.

Usage:
    python -m scheduler.run_once           # News + GitHub (if due)
    python -m scheduler.run_once --news    # News only
    python -m scheduler.run_once --github  # GitHub only

Crontab example (every 5 minutes):
    */5 * * * * cd /path/to/TechSage && python -m scheduler.run_once --news >> /tmp/techsage-ingest.log 2>&1

Crontab example (every 6 hours for GitHub):
    0 */6 * * * cd /path/to/TechSage && python -m scheduler.run_once --github >> /tmp/techsage-github.log 2>&1
"""

import argparse
import asyncio
import logging
import sys

from config.model_registry import model_registry
from config.settings import get_settings
from database.session import get_session_factory, init_db
from scheduler.ingestion_job import run_github_pipeline, run_ingestion_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="TechSage one-shot ingestion")
    parser.add_argument("--news", action="store_true", help="Run news ingestion only")
    parser.add_argument("--github", action="store_true", help="Run GitHub ingestion only")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if not args.news and not args.github:
        args.news = args.github = True

    settings = get_settings()
    level = logging.WARNING if args.quiet else getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    logger.info("Loading models...")
    model_registry.load_all()
    init_db()
    factory = get_session_factory()
    session = factory()

    exit_code = 0
    try:
        if args.news:
            logger.info("Running news ingestion...")
            result = asyncio.run(run_ingestion_pipeline(session))
            session.commit()
            logger.info("News: %s", result)

        if args.github:
            logger.info("Running GitHub ingestion...")
            result = asyncio.run(run_github_pipeline(session))
            session.commit()
            logger.info("GitHub: %s", result)
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        session.rollback()
        exit_code = 1
    finally:
        session.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
