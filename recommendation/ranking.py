"""
TechSage — Personalized ranking engine.

Scoring formula:
    Score = 0.40 * ReleasePriority
          + 0.25 * EmbeddingSimilarity
          + 0.20 * RecencyScore
          + 0.10 * SourceTierWeight
          + 0.05 * CategoryAffinity

Prioritizes official model releases and high-signal framework launches
over generic news coverage.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import desc
from sqlalchemy.orm import Session

from config.settings import get_settings
from database.models import Article
from recommendation.user_profile import get_category_vector, get_embedding_centroid, get_user_profile

logger = logging.getLogger(__name__)

_RECENCY_HALF_LIFE_HOURS = 24.0

# release_type -> priority score
RELEASE_PRIORITY = {
    "model": 1.0,
    "framework": 0.9,
    "agent_platform": 0.85,
    "infra": 0.8,
    "github_repo": 0.75,
    "research": 0.7,
    "general_news": 0.5,
}

# source_tier -> weight
SOURCE_TIER_WEIGHT = {
    1: 1.0,
    2: 0.9,
    3: 0.7,
    4: 0.8,
}


def compute_recency_score(published_at: datetime | None) -> float:
    if published_at is None:
        return 0.5

    now = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    hours_old = max(0, (now - published_at).total_seconds() / 3600.0)
    decay_rate = math.log(2) / _RECENCY_HALF_LIFE_HOURS
    return math.exp(-decay_rate * hours_old)


def compute_release_priority(release_type: str | None) -> float:
    return RELEASE_PRIORITY.get(release_type or "general_news", 0.5)


def compute_source_tier_weight(source_tier: int | None) -> float:
    return SOURCE_TIER_WEIGHT.get(source_tier or 3, 0.7)


def compute_category_affinity(
    user_category_weights: dict[str, float],
    article_category: str | None,
) -> float:
    if not article_category:
        return 0.5
    return user_category_weights.get(article_category, 0.0)


def compute_embedding_similarity(
    user_centroid: np.ndarray | None,
    article_embedding: np.ndarray | None,
) -> float:
    if user_centroid is None or article_embedding is None:
        return 0.5

    user_vec = user_centroid.reshape(1, -1)
    article_vec = article_embedding.reshape(1, -1)
    sim = cosine_similarity(user_vec, article_vec)[0][0]
    return float((sim + 1.0) / 2.0)


def rank_articles(
    db: Session,
    user_id: UUID,
    limit: int = 20,
    category_filter: str | None = None,
    release_type_filter: str | None = None,
    content_type_filter: str | None = None,
    max_age_days: int | None = None,
) -> list[dict]:
    """
    Rank articles using the 5-component TechSage formula.

    Supports filtering by category, release_type, content_type, and max age.
    """
    settings = get_settings()

    profile = get_user_profile(db, user_id)
    user_categories = get_category_vector(profile)
    user_centroid = get_embedding_centroid(profile)

    query = (
        db.query(Article)
        .filter(Article.duplicate_of.is_(None))
        .order_by(desc(Article.created_at))
    )
    if category_filter:
        query = query.filter(Article.category == category_filter)
    if release_type_filter:
        query = query.filter(Article.release_type == release_type_filter)
    if content_type_filter:
        query = query.filter(Article.content_type == content_type_filter)
    if max_age_days is not None:
        from sqlalchemy import case
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        effective_date = case(
            (Article.content_type == "github_repo", Article.created_at),
            else_=Article.published_at,
        )
        query = query.filter(
            (effective_date >= cutoff) | (effective_date.is_(None))
        )

    candidates = query.limit(300).all()

    if not candidates:
        logger.info("No candidate articles found for ranking.")
        return []

    scored_articles = []
    for article in candidates:
        recency = compute_recency_score(article.published_at or article.created_at)
        release_pri = compute_release_priority(article.release_type)
        tier_weight = compute_source_tier_weight(article.source_tier)
        cat_affinity = compute_category_affinity(user_categories, article.category)

        article_emb = (
            np.array(article.embedding, dtype=np.float32)
            if article.embedding is not None
            else None
        )
        emb_similarity = compute_embedding_similarity(user_centroid, article_emb)

        score = (
            settings.w_release_priority * release_pri
            + settings.w_embedding * emb_similarity
            + settings.w_recency * recency
            + settings.w_source_tier * tier_weight
            + settings.w_category * cat_affinity
        )

        scored_articles.append(
            {
                "article_id": str(article.id),
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "category": article.category,
                "category_confidence": article.category_confidence,
                "summary": article.summary,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "created_at": article.created_at.isoformat() if article.created_at else None,
                "content_type": article.content_type,
                "release_type": article.release_type,
                "is_release": article.is_release,
                "source_tier": article.source_tier,
                "context_length_tokens": article.context_length_tokens,
                "model_size_params": article.model_size_params,
                "license": article.license,
                "open_source": article.open_source,
                "benchmark_claims": article.benchmark_claims,
                "cost_changes": article.cost_changes,
                "api_changes": article.api_changes,
                "hardware_requirements": article.hardware_requirements,
                "fine_tuning_supported": article.fine_tuning_supported,
                "engineering_impact": article.engineering_impact,
                "github_stars": article.github_stars,
                "github_language": article.github_language,
                "github_topics": article.github_topics,
                "repo_quality_score": article.repo_quality_score,
                "star_velocity": article.star_velocity,
                "repo_quality_tag": article.repo_quality_tag,
                "score": round(score, 4),
                "score_breakdown": {
                    "release_priority": round(release_pri, 4),
                    "embedding_similarity": round(emb_similarity, 4),
                    "recency": round(recency, 4),
                    "source_tier_weight": round(tier_weight, 4),
                    "category_affinity": round(cat_affinity, 4),
                },
            }
        )

    scored_articles.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "Ranked %d articles for user %s (top score=%.4f).",
        len(scored_articles),
        user_id,
        scored_articles[0]["score"] if scored_articles else 0,
    )

    return scored_articles[:limit]
