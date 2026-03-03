"""
Database session management.

Uses SQLAlchemy 2.0 async-compatible session factory.
The engine and session factory are created once and reused throughout the application.
"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from database.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def get_engine():
    """Return a singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Database engine created: %s", settings.database_url.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker:
    """Return a singleton session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


def get_db() -> Session:
    """
    Dependency-injectable session generator for FastAPI.
    Usage: db: Session = Depends(get_db)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Create all tables and install pgvector extension.
    Called once at application startup.
    """
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")
