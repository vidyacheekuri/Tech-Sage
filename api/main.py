"""
TechSage — FastAPI application entry point.

Startup:
1. Configure logging
2. Load all ML models (embedding, classifier, summarizer) once
3. Initialize database (create tables, install pgvector)
4. Start background schedulers (news + GitHub trending)
5. Mount API routes + serve frontend
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config.model_registry import model_registry
from config.settings import get_settings
from database.session import init_db
from scheduler.ingestion_job import start_scheduler, stop_scheduler

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _configure_logging() -> None:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger(__name__)

    _configure_logging()
    logger.info("=" * 60)
    logger.info("TechSage — AI Ecosystem Intelligence Engine")
    logger.info("=" * 60)

    logger.info("Loading ML models (this may take a minute on first run)...")
    model_registry.load_all()

    logger.info("Initializing database...")
    init_db()

    logger.info("Starting ingestion schedulers...")
    start_scheduler()

    logger.info("Startup complete. TechSage is ready.")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down TechSage...")
    stop_scheduler()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TechSage",
        description=(
            "AI Ecosystem Intelligence Engine — monitors official AI labs, "
            "frameworks, infrastructure providers, research papers, and trending "
            "GitHub repositories using transformer-based NLP models for semantic "
            "deduplication, release detection, summarization, and hybrid ranking."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(_STATIC_DIR / "index.html"))

    return app


app = create_app()
