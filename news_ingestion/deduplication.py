"""
TechSage — Duplicate detection with source tier preference.

Strategy:
- URL deduplication (fast, exact match)
- Semantic deduplication (cosine similarity >= 0.85)
- Tier-aware: when a duplicate is found, prefer the higher-tier (lower number) source.
  If the new article has a better tier, it replaces the existing one and the old
  article is marked as duplicate_of the new one. Otherwise the new article is skipped.
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import Article

logger = logging.getLogger(__name__)


def is_duplicate(
    new_embedding: np.ndarray,
    db: Session,
    threshold: float | None = None,
) -> tuple[bool, float, str | None]:
    """
    Check if an article with the given embedding is a duplicate.

    Returns:
        Tuple of (is_duplicate, max_similarity, duplicate_article_id).
    """
    if threshold is None:
        threshold = get_settings().duplicate_threshold

    existing_articles = (
        db.query(Article.id, Article.embedding)
        .filter(Article.embedding.isnot(None))
        .all()
    )

    if not existing_articles:
        return False, 0.0, None

    ids = []
    embeddings = []
    for article_id, emb in existing_articles:
        if emb is not None:
            ids.append(str(article_id))
            embeddings.append(np.array(emb, dtype=np.float32))

    if not embeddings:
        return False, 0.0, None

    existing_matrix = np.vstack(embeddings)
    new_vec = new_embedding.reshape(1, -1)

    similarities = cosine_similarity(new_vec, existing_matrix)[0]
    max_idx = int(np.argmax(similarities))
    max_sim = float(similarities[max_idx])

    if max_sim >= threshold:
        logger.info(
            "Duplicate detected (similarity=%.3f, threshold=%.2f) — matches article %s",
            max_sim,
            threshold,
            ids[max_idx],
        )
        return True, max_sim, ids[max_idx]

    return False, max_sim, None


def handle_tiered_duplicate(
    new_source_tier: int,
    duplicate_article_id: str,
    db: Session,
) -> str:
    """
    Decide what to do when a semantic duplicate is found.

    Returns:
        'skip' — new article is lower priority, don't insert
        'replace' — new article is higher priority, mark old as duplicate
    """
    existing = db.query(Article).filter(Article.id == duplicate_article_id).first()
    if not existing:
        return "skip"

    existing_tier = existing.source_tier or 3

    if new_source_tier < existing_tier:
        return "replace"

    return "skip"


def mark_as_duplicate(
    subordinate_id: str,
    primary_id: str,
    db: Session,
) -> None:
    """Mark an article as a duplicate of a higher-tier source."""
    subordinate = db.query(Article).filter(Article.id == subordinate_id).first()
    if subordinate:
        subordinate.duplicate_of = primary_id
        db.flush()
        logger.info("Marked article %s as duplicate of %s", subordinate_id, primary_id)


def is_duplicate_url(url: str, db: Session) -> bool:
    """Quick check: does this URL already exist in the database?"""
    return db.query(Article.id).filter(Article.url == url).first() is not None
